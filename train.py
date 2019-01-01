#!/usr/bin/python3

import argparse
import itertools
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
from PIL import Image
import torch
import cv2
import numpy as np

from matplotlib import pyplot as plt

from modules import RevUNetGenerator, Discriminator
from utils import weights_init_normal, tensor2image
from datasets import ImageDataset, CelebA
from time import time
from utils import Logger, ReplayBuffer

###### Definition of variables ######
# Networks
input_nc = 3

netG = RevUNetGenerator(input_nc)#.double()
netD = Discriminator(input_nc)

resume = False

if resume:
    print("Loading checkpoint")
    netG = torch.load('output/netG_A2B.pth')
    netD = torch.load('output/netD.pth')
else:
    netG.apply(weights_init_normal)
    netD.apply(weights_init_normal)

cuda = torch.cuda.is_available()

if cuda:
    netG.cuda()
    netD.cuda()

# Lossess
criterion_GAN = torch.nn.MSELoss()
criterion_cycle = torch.nn.L1Loss()
criterion_identity = torch.nn.L1Loss()

# Inputs & targets memory allocation
Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

###################################
# currently must be a factor of 16
# epoch, size, batch_size
current_scale = 64
max_scale = 128
max_scale_batch_size = 4
epochs_per_scale = 20
n_epochs = 100
iterations_per_epoch = 100
n_cpu = 4

# Loss plot
logger = Logger(n_epochs, iterations_per_epoch)

lr = 3e-4
ii = 0
###### Training ######
for epoch in range(0, n_epochs):

    if epoch % epochs_per_scale == 0:
        if current_scale < max_scale:

            batch_size = max_scale_batch_size * int(max_scale//current_scale)**2

            print("Current scale: {}, batch size: {}".format(current_scale, batch_size))
            # Dataset loader
            transforms_ = [
                            transforms.Resize(int(current_scale*1.12), Image.BICUBIC),
                            transforms.RandomCrop(current_scale),
                            #GuidedFilter(),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ]

            #dataset = ImageDataset(opt.dataroot, transforms_=transforms_, unaligned=True)
            dataset = CelebA("/home/msu/Data/celeba", transforms_=transforms_, unaligned=True, attribute = "Blond_Hair")

            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                                    num_workers=n_cpu,drop_last=True)
            input_A = Tensor(batch_size, input_nc, current_scale, current_scale)
            input_B = Tensor(batch_size, input_nc, current_scale, current_scale)

            _ones = Tensor(batch_size).fill_(1.0)
            _half = Tensor(batch_size).fill_(0.0)
            _zeros = Tensor(batch_size).fill_(-1.0)

            target_A_real = Variable(_ones,requires_grad=False)
            target_A_fake = Variable(_half,requires_grad=False)
            target_B_fake = Variable(_half,requires_grad=False)
            target_B_real = Variable(_zeros,requires_grad=False)
         
            fake_A_buffer = ReplayBuffer()
            fake_B_buffer = ReplayBuffer()

            optimizer_G = torch.optim.Adam(netG.parameters(), lr=lr, betas=(0.5, 0.999))
            optimizer_D = torch.optim.Adam(netD.parameters(), lr=lr, betas=(0.5, 0.999))
            lr /= 2
            current_scale += 16
    
    for i, batch in enumerate(dataloader):
        #import pdb; pdb.set_trace()
        #with torch.autograd.profiler.profile() as prof:
        tic = time()
        # Set model input
        real_A = Variable(input_A.copy_(batch['A']))
        real_B = Variable(input_B.copy_(batch['B']))

        ###### Generators A2B and B2A ######
        optimizer_G.zero_grad()

        # Identity loss
        # G_A2B(B) should equal B if real B is fed

        same_B, _ = netG(real_B)
        loss_identity_B = criterion_identity(F.tanh(same_B), real_B)*5.0

        # G_B2A(A) should equal A if real A is fed
        same_A, _ = netG(real_A, reverse=True)
        loss_identity_A = criterion_identity(F.tanh(same_A), real_A)*5.0

        # GAN loss
        fake_B, zB = netG(real_A)
        pred_fake_B = netD(F.tanh(fake_B))
        loss_GAN_A2B = criterion_GAN(pred_fake_B, target_B_real)

        fake_A, zA = netG(real_B, reverse=True)
        pred_fake_A = netD(F.tanh(fake_A))
        loss_GAN_B2A = criterion_GAN(pred_fake_A, target_A_real)

        # Cycle loss
        recovered_A, zA = netG(fake_B, zB, reverse=True)
        loss_cycle_ABA = criterion_cycle(recovered_A, real_A)*10.0
        
        recovered_B, zB = netG(fake_A, zA)
        loss_cycle_BAB = criterion_cycle(recovered_B, real_B)*10.0

        # Total loss
        loss_G = loss_identity_A + loss_identity_B + loss_GAN_A2B + loss_GAN_B2A # + loss_cycle_ABA + loss_cycle_BAB
        loss_G.backward()

        optimizer_G.step()
        #import pdb; pdb.set_trace()

        ###################################

        ###### Discriminator A ######
        optimizer_D.zero_grad()

        # Real loss
        pred_real_A = netD(real_A)
        loss_D_real_A = criterion_GAN(F.tanh(pred_real_A), target_A_real)

        # Fake loss
        fake_A = fake_A_buffer.push_and_pop(fake_A)
        pred_fake_A = netD(fake_A.detach())
        loss_D_fake_A = criterion_GAN(F.tanh(pred_fake_A), target_A_fake)

        # Real loss
        pred_real_B = netD(real_B)
        loss_D_real_B = criterion_GAN(F.tanh(pred_real_B), target_B_real)

        # Fake loss
        fake_B = fake_B_buffer.push_and_pop(fake_B)
        pred_fake_B = netD(fake_B.detach())
        loss_D_fake_B = criterion_GAN(F.tanh(pred_fake_B), target_B_fake)

        # Total loss
        loss_D = (loss_D_real_A + loss_D_fake_A + loss_D_real_B + loss_D_fake_B)*0.25

        loss_D.backward()

        optimizer_D.step()
        ###################################
        print("time", time() - tic)
        # Progress report (http://localhost:8097)
        logger.log({'loss_G': loss_G, 'loss_G_identity': (loss_identity_A + loss_identity_B), 'loss_cycle_GAN': (loss_cycle_ABA + loss_cycle_BAB), 'loss_G_GAN': (loss_GAN_A2B + loss_GAN_B2A), 'loss_D': loss_D},
                    images={'real_A': real_A, 'real_B': real_B, 'fake_A': fake_A, 'fake_B': fake_B})

        if i > iterations_per_epoch:
            print("EPOCH FINISH =======================================")
            print("save checkpoint")
            torch.save(netG, 'output/netG.pth')
            torch.save(netD, 'output/netD.pth')
            #import pdb; pdb.set_trace()
            plt.imsave("log/epoch_{}_A_real.png".format(epoch), np.moveaxis(tensor2image(real_A.data),0,2))
            plt.imsave("log/epoch_{}_B_real.png".format(epoch), np.moveaxis(tensor2image(real_B.data),0,2))
            plt.imsave("log/epoch_{}_A_fake.png".format(epoch), np.moveaxis(tensor2image(F.tanh(fake_A).data),0,2))
            plt.imsave("log/epoch_{}_B_fake.png".format(epoch), np.moveaxis(tensor2image(F.tanh(fake_B).data),0,2))
            break



###################################
