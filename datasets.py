import glob
import random
import os
import csv

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, unaligned=False, mode='train'):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned

        self.files_A = sorted(glob.glob(os.path.join(root, '%s/A' % mode) + '/*.*'))
        self.files_B = sorted(glob.glob(os.path.join(root, '%s/B' % mode) + '/*.*'))

    def __getitem__(self, index):
        item_A = self.transform(Image.open(self.files_A[index % len(self.files_A)]))

        if self.unaligned:
            item_B = self.transform(Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)]))
        else:
            item_B = self.transform(Image.open(self.files_B[index % len(self.files_B)]))

        return {'A': item_A, 'B': item_B}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))

class InverseTransformImageDataset(Dataset):

    def __init__(self, root, pre_transforms_=None, transforms_=None, target_transforms_=None):
        self.pre_transform = transforms.Compose(pre_transforms_) # common for both image and target
        self.transform = transforms.Compose(transforms_)
        self.target_transform = transforms.Compose(target_transforms_)

        self.files = sorted(os.listdir(root))

        self.files = [os.path.join(root,x) for x in self.files if x.split(".")[-1] in ["jpg","png"]]
        if len(self.files) <= 0:
            raise ValueError("No images found in {}".format(root))


    def __getitem__(self, index):

        im = Image.open(self.files[index % len(self.files)])
        im.load()
        im = self.pre_transform(im)

        image = self.transform(im)

        target = self.target_transform(im)



        return {'image': image, 'target': target}

    def __len__(self):
        return len(self.files)

class CelebA(Dataset):
    """"

    Possible attributes:
        5_o_Clock_Shadow
        Arched_Eyebrows
        Attractive
        Bags_Under_Eyes
        Bald Bangs
        Big_Lips
        Big_Nose
        Black_Hair
        Blond_Hair
        Blurry Brown_Hair
        Bushy_Eyebrows Chubby
        Double_Chin
        Eyeglasses
        Goatee
        Gray_Hair
        Heavy_Makeup
        High_Cheekbones
        Male
        Mouth_Slightly_Open
        Mustache
        Narrow_Eyes
        No_Beard
        Oval_Face Pale_Skin
        Pointy_Nose
        Receding_Hairline
        Rosy_Cheeks
        Sideburns Smiling
        Straight_Hair
        Wavy_Hair
        Wearing_Earrings
        Wearing_Hat
        Wearing_Lipstick
        Wearing_Necklace
        Wearing_Necktie
        Young
    """

    def __init__(self, root, transforms_=None, unaligned=False, attribute = "Eyeglasses"):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned
        self.files = sorted(glob.glob(os.path.join(root, 'img_align_celeba') + '/*.*'))
        attribute_file = os.path.join(root, 'list_attr_celeba.txt')
        image_path = os.path.join(root, 'img_align_celeba')



        self.files_A = []
        self.files_B = []

        with open(attribute_file, "r") as f:
            reader = csv.reader(f, delimiter=" ")
            num_files = next(reader, None)
            attribute_list = next(reader, None)
            attribute_list = [x.lower() for x in attribute_list]
            attribute = attribute.lower()
            if attribute.lower() in attribute_list:
                attribute_idx = attribute_list.index(attribute)

                for row in reader:
                    row = [x for x in row if x != '']
                    if row[attribute_idx+1] == '1':
                        self.files_A.append(os.path.join(image_path,row[0]))
                    else:
                        self.files_B.append(os.path.join(image_path,row[0]))
            else:
                raise AttributeError("Attribute {} not in attribute list: {}".format(attribute, attribute_list))
        
    def __getitem__(self, index):
        #import pdb; pdb.set_trace()
        item_A = self.transform(Image.open(self.files_A[index % len(self.files_A)]))

        if self.unaligned:
            item_B = self.transform(Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)]))
        else:
            item_B = self.transform(Image.open(self.files_B[index % len(self.files_B)]))

        return {'A': item_A, 'B': item_B}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))

if __name__ == "__main__":
    current_scale = 128
    transforms_ = [
                    transforms.Resize(int(current_scale), Image.BICUBIC),
                    GuidedFilter(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ]

    target_transforms_ = [
                    transforms.Resize(int(current_scale), Image.BICUBIC),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ]

    #dataset = CelebA("/home/msu/Data/celeba", unaligned=True, attribute = "male")
    dataset = InverseTransformImageDataset("datasets/rome/train/B", transforms_=transforms_, target_transforms_=target_transforms_)
    print(dataset[0])
