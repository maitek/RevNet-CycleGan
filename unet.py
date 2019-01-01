#!/usr/bin/python
# full assembly of the sub-parts to form the complete net

import torch
import torch.nn as nn
import torch.nn.functional as F
from time import time

class UNet(nn.Module):
    def __init__(self, n_channels, n_output, nf = 32):
        super(UNet, self).__init__()
        self.inc = inconv(n_channels, nf)
        self.down1 = down(nf, nf*2)
        self.down2 = down(nf*2, nf*4)
        self.down3 = down(nf*4, nf*8)
        self.down4 = down(nf*8, nf*8)
        self.up1 = up(nf*16, nf*4)
        self.up2 = up(nf*8, nf*2)
        self.up3 = up(nf*4, nf)
        self.up4 = up(nf*2, nf)
        self.outc = outconv(nf, n_output)

    def forward(self, x):
        tic = time()
        x1 = self.inc(x) # 32
        x2 = self.down1(x1) # 64
        x3 = self.down2(x2) # 128
        x4 = self.down3(x3) # 256
        x5 = self.down4(x4) # 256
        x = self.up1(x5, x4) #128
        x = self.up2(x, x3) # 128
        x = self.up3(x, x2) # 64
        x = self.up4(x, x1) # 32
        x = self.outc(x)
        return x

class UNetDiscriminator(nn.Module):
    def __init__(self, n_channels, n_output, nf = 64):
        super(UNet, self).__init__()
        self.inc = inconv(n_channels, nf)
        self.down1 = down(nf, nf*2)
        self.down2 = down(nf*2, nf*4)
        self.down3 = down(nf*4, nf*8)
        self.down4 = down(nf*8, nf*8)

        self.out1 = outconv(nf*4, n_output)
        self.out2 = outconv(nf*2, n_output)
        self.out3 = outconv(nf, n_output)
        self.out4 = outconv(nf, n_output)
        self.outc = outconv(nf, n_output)

    def forward(self, x):
        tic = time()
        x1 = self.inc(x) # 32
        x2 = self.down1(x1) # 64
        x3 = self.down2(x2) # 128
        x4 = self.down3(x3) # 256
        x5 = self.down4(x4) # 256
        x = self.up1(x5, x4) #128
        x = self.up2(x, x3) # 128
        x = self.up3(x, x2) # 64
        x = self.up4(x, x1) # 32
        x = self.outc(x)
        return x

class PyramidUNet(nn.Module):
    def __init__(self, n_channels, n_output, nf = 32):
        super(PyramidUNet, self).__init__()
        self.inc = inconv(n_channels, nf)
        self.down1 = down(nf, nf*2)
        self.down2 = down(nf*2, nf*4)
        self.down3 = down(nf*4, nf*8)
        self.down4 = down(nf*8, nf*8)
        self.up1 = up(nf*16, nf*4)
        self.up2 = up(nf*8, nf*2)
        self.up3 = up(nf*4, nf)
        self.up4 = up(nf*2, nf)

        self.out1 = outconv(nf*4, n_output)
        self.out2 = outconv(nf*2, n_output)
        self.out3 = outconv(nf, n_output)
        self.out4 = outconv(nf, n_output)

    def forward(self, x):
        tic = time()
        x1 = self.inc(x) # 32
        x2 = self.down1(x1) # 64
        x3 = self.down2(x2) # 128
        x4 = self.down3(x3) # 256
        x5 = self.down4(x4) # 256

        x = self.up1(x5, x4) #128
        o1 = self.out1(x)

        x = self.up2(x, x3) # 128
        o2 = self.out2(x)

        x = self.up3(x, x2) # 64
        o3 = self.out3(x)

        x = self.up4(x, x1) # 32
        o4 = self.out4(x)

        return [o4, o3, o2, o1]

class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()

        self.skip_path = nn.Conv2d(in_ch, out_ch, 1, padding=0, bias=False)

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch//4, 1, padding=0, bias=False), #squeeze
            nn.Conv2d(out_ch//4, out_ch//4, 3, padding=1),
            nn.InstanceNorm2d(out_ch//4),
            nn.ELU(inplace=True),
            #nn.Dropout(p=0.2),
            nn.Conv2d(out_ch//4, out_ch//4, 3, padding=1, bias=False),
            nn.InstanceNorm2d(out_ch//4),
            nn.ELU(inplace=True),
            #nn.Dropout(p=0.2),
            nn.Conv2d(out_ch//4, out_ch, 1, padding=0, bias=False) #expand
        )

    def forward(self, x):
        residual = x
        x = self.conv(x)
        x += self.skip_path(residual)
        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, ):
        super(up, self).__init__()

        self.up = nn.Upsample(scale_factor=2, mode='bilinear')
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffX = x1.size()[2] - x2.size()[2]
        diffY = x1.size()[3] - x2.size()[3]
        x2 = F.pad(x2, (diffX // 2, int(diffX / 2),
                        diffY // 2, int(diffY / 2)))
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1, bias=False)


    def forward(self, x):
        x = self.conv(x)
        return F.tanh(x)
