import torch
import torch.nn as nn
import torch.nn.functional as F
from unet import UNet

image_shape = (64,64,3)
in_channels = 3
hidden_channels = 16
K = 16
L = 3

def squeeze2d(input, factor=2):
    assert factor >= 1 and isinstance(factor, int)
    if factor == 1:
        return input
    size = input.size()
    B = size[0]
    C = size[1]
    H = size[2]
    W = size[3]
    assert H % factor == 0 and W % factor == 0, "{}".format((H, W))
    x = input.view(B, C, H // factor, factor, W // factor, factor)
    x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
    x = x.view(B, C * factor * factor, H // factor, W // factor)
    return x


def unsqueeze2d(input, factor=2):
    assert factor >= 1 and isinstance(factor, int)
    factor2 = factor ** 2
    if factor == 1:
        return input
    size = input.size()
    B = size[0]
    C = size[1]
    H = size[2]
    W = size[3]
    assert C % (factor2) == 0, "{}".format(C)
    x = input.view(B, C // factor2, factor, factor, H, W)
    x = x.permute(0, 1, 4, 2, 5, 3).contiguous()
    x = x.view(B, C // (factor2), H * factor, W * factor)
    return x


class ResNetBlock(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch):
        super(ResNetBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x) + x
        return x

class SqueezeLayer(nn.Module):
    def __init__(self, factor):
        super().__init__()
        self.factor = factor

    def forward(self, in_tensor, reverse=False):
        if not reverse:
            out = squeeze2d(in_tensor, self.factor)
            return out
        else:
            out = unsqueeze2d(in_tensor, self.factor)
            return out

class UnsqueezeLayer(nn.Module):
    def __init__(self, factor):
        super().__init__()
        self.factor = factor

    def forward(self, in_tensor, reverse=False):
        if not reverse:
            out = unsqueeze2d(in_tensor, self.factor)
            return out
        else:
            out = squeeze2d(in_tensor, self.factor)
            return out

class RevNetBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        assert in_channels % 2 == 0
        self.F = ResNetBlock(in_channels // 2, in_channels // 2)
        self.G = ResNetBlock(in_channels // 2, in_channels // 2)
        
        #self.F = nn.Conv2d(in_channels // 2, in_channels // 2,kernel_size=3,padding=1)
        #self.G = nn.Conv2d(in_channels // 2, in_channels // 2,kernel_size=3,padding=1)

    def forward(self,in_tensor,reverse=False):
        # split
        c = in_tensor.shape[1]
        assert c % 2 == 0
        in1, in2 = in_tensor[:, :c // 2, ...], in_tensor[:, c // 2:, ...]
        
        if not reverse:
            out = self.encode(in1,in2)
        else:
            out = self.decode(in1,in2)
        return out

    def encode(self,x1,x2):
        #import pdb; pdb.set_trace()
        y1 = self.F(x2) + x1
        y2 = self.G(y1) + x2
        return torch.cat((y1, y2), dim=1)

    def decode(self,y1,y2):
        x2 = y2 - self.G(y1)
        x1 = y1 - self.F(x2)
        return torch.cat((x1, x2), dim=1)



class RevNetGenerator(nn.Module):
    def __init__(self, image_shape):
        super().__init__()

        
        layers = [
            # encoder
            SqueezeLayer(2),
            RevNetBlock(in_channels*4),
            RevNetBlock(in_channels*4),
            SqueezeLayer(2),
            RevNetBlock(in_channels*4*4),
            RevNetBlock(in_channels*4*4),
            SqueezeLayer(2),

            RevNetBlock(in_channels*4*4*4),
            
            #decoder
            UnsqueezeLayer(2),
            RevNetBlock(in_channels*4*4),
            RevNetBlock(in_channels*4*4),
            UnsqueezeLayer(2),
            RevNetBlock(in_channels*4),
            RevNetBlock(in_channels*4),
            UnsqueezeLayer(2)
        ]
        self.layers = nn.ModuleList(layers)
        
    def forward(self,x, reverse=False):
        #print(x.shape)
        if not reverse:
            for layer in self.layers:
                x = layer(x)
                #print(x.shape)
        else:
            for layer in reversed(self.layers):
                x = layer(x,reverse=True)
                #print(x.shape)
        return x

class Discriminator(nn.Module):
    def __init__(self, in_channels):
        super(Discriminator, self).__init__()

        # A bunch of convolutions one after another
        layers = [   nn.Conv2d(in_channels, 64, 4, stride=2, padding=1),
                    nn.LeakyReLU(0.2, inplace=True) ]

        layers += [  nn.Conv2d(64, 128, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(128),
                    nn.LeakyReLU(0.2, inplace=True) ]

        layers += [  nn.Conv2d(128, 256, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(256),
                    nn.LeakyReLU(0.2, inplace=True) ]

        layers += [  nn.Conv2d(256, 512, 4, padding=1),
                    nn.InstanceNorm2d(512),
                    nn.LeakyReLU(0.2, inplace=True) ]

        # FCN classification layer
        layers += [nn.Conv2d(512, 1, 4, padding=1)]

        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        # Average pooling and flatten
        return torch.squeeze(F.avg_pool2d(x, x.size()[2:]))

class RevUnet(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        assert in_channels % 2 == 0
        self.F = UNet(in_channels // 2, in_channels // 2)
        self.G = UNet(in_channels // 2, in_channels // 2)
        
        #self.F = nn.Conv2d(in_channels // 2, in_channels // 2,kernel_size=3,padding=1)
        #self.G = nn.Conv2d(in_channels // 2, in_channels // 2,kernel_size=3,padding=1)

    def forward(self,in_tensor,reverse=False):
        # split
        c = in_tensor.shape[1]
        assert c % 2 == 0
        in1, in2 = in_tensor[:, :c // 2, ...], in_tensor[:, c // 2:, ...]
        
        if not reverse:
            out = self.encode(in1,in2)
        else:
            out = self.decode(in1,in2)
        return out

    def encode(self,x1,x2):
        #import pdb; pdb.set_trace()
        y1 = self.F(x2) + x1
        y2 = self.G(y1) + x2
        return torch.cat((y1, y2), dim=1)

    def decode(self,y1,y2):
        x2 = y2 - self.G(y1)
        x1 = y1 - self.F(x2)
        return torch.cat((x1, x2), dim=1)

class RevUNetGenerator(nn.Module):
    def __init__(self, image_shape):
        super().__init__()

        
        layers = [
            # encoder
            SqueezeLayer(2),
            RevUnet(in_channels*4),
            UnsqueezeLayer(2)
        ]
        self.layers = nn.ModuleList(layers)
        
    def forward(self,x, reverse=False):
        #print(x.shape)
        if not reverse:
            for layer in self.layers:
                x = layer(x)
                #print(x.shape)
        else:
            for layer in reversed(self.layers):
                x = layer(x,reverse=True)
                #print(x.shape)
        return x

if __name__ == "__main__":
    in_channels = 3
    revnet = RevUNetGenerator(in_channels)#.double()
    #flow_enc = FlowStep(in_channels, hidden_channels,
    #                 actnorm_scale=1.0,
    #                 flow_permutation="invconv",
    #                 flow_coupling="additive",
    #                 LU_decomposed=False)

    #flow_dec = FlowStep(image_shape, hidden_channels,
    #                 actnorm_scale=1.0,
    #                 flow_permutation="invconv",
    #                 flow_coupling="additive",
    #                 LU_decomposed=False)

    A = torch.randn((1,in_channels,64,64),dtype=torch.float32)
    #import pdb; pdb.set_trace()
    print(A.shape)
    z = revnet(A)
    print(z.shape)
    B = revnet(z,reverse=True)
    print(B.shape)

    #A_r = flow_net(B)

    print(torch.abs(B-A).sum())
    import pdb; pdb.set_trace()