import torch
import torch.nn as nn
import torch.nn.functional as F
from model.resnet import resnet18

__all__ = ['resnet_tiny_refine']

def conv3x3(in_planes, out_planes, strd=1, padding=1, bias=True):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3,
                     stride=strd, padding=padding, bias=bias)

class ConvBlock(nn.Module):
    def __init__(self, in_planes, mid_planes, out_planes, stride=1):
        super(ConvBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = conv3x3(in_planes, mid_planes, strd=stride)
        self.bn2 = nn.BatchNorm2d(mid_planes)
        self.conv2 = conv3x3(mid_planes, out_planes)

        if stride == 2:
            self.downsample = nn.Sequential(
                nn.BatchNorm2d(in_planes),
                nn.ReLU(True),
                nn.Conv2d(in_planes, out_planes,
                          kernel_size=3, stride=stride, padding=1, bias=True),
            )
        else:
            self.downsample = None

    def forward(self, x):
        residual = x

        out1 = self.bn1(x)
        out1 = F.relu(out1, True)
        out1 = self.conv1(out1)

        out2 = self.bn2(out1)
        out2 = F.relu(out2, True)
        out2 = self.conv2(out2)

        if self.downsample is not None:
            residual = self.downsample(residual)

        out2 += residual

        return out2


class Resnet_Tiny_Refine(nn.Module):

    def __init__(self,class_num = 5):
        super(Resnet_Tiny_Refine, self).__init__()

        # Base part
        self.conv11 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1)
        self.bn11 = nn.BatchNorm2d(32)
        self.conv12 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn12 = nn.BatchNorm2d(64)
        self.relu12 = nn.ReLU()

        self.conv21 = ConvBlock(64, 32, 64)
        self.conv22 = ConvBlock(64, 32, 64)

        self.conv31 = ConvBlock(64, 32, 64, stride=2)
        self.conv32 = ConvBlock(64, 32, 64)

        self.conv41 = ConvBlock(64, 32, 64,stride=2)
        self.conv42 = ConvBlock(64, 32, 64)

        self.conv51 = ConvBlock(64, 32, 64,stride=2)
        self.conv52 = ConvBlock(64, 32, 64)

        self.refine1 = ConvBlock(64, 32, 64)
        #self.refine1_1x1 = nn.Conv2d(192, 128, kernel_size=1, stride=1)

        self.refine2 = ConvBlock(64, 32, 64)
        # self.refine2_1x1 = nn.Conv2d(128, 96, kernel_size=1, stride=1)

        self.refine3 = ConvBlock(64, 32, 64)
        self.refine3_1x1 = nn.Conv2d(64, 96, kernel_size=1, stride=1)

        self.refine4 = ConvBlock(96, 96, 96)
        self.deconv1_bn=nn.BatchNorm2d(96)

        self.conv_feat_P = nn.Conv2d(96, class_num, kernel_size=3, stride=1, padding=1)
        self.feat_bn_P = nn.BatchNorm2d(class_num)

    def forward(self, x):
        x = F.relu(self.bn11(self.conv11(x)), True)
        x = F.relu(self.bn12(self.conv12(x)), True)
        x = self.conv21(x)
        x = self.conv22(x)
        refine3_pre = x

        x = self.conv31(x)
        x = self.conv32(x)
        refine2_pre = x

        x = self.conv41(x)
        x = self.conv42(x)
        refine1_pre = x

        x = self.conv51(x)
        x = self.conv52(x)

        x=self.refine1(x)
        # x=self.refine1_1x1(x)
        x = refine1_pre + F.upsample(x, scale_factor=2, mode='bilinear')

        x=self.refine2(x)
        # x=self.refine2_1x1(x)
        x = refine2_pre + F.upsample(x, scale_factor=2, mode='bilinear')

        x=self.refine3(x)
        
        x = refine3_pre + F.upsample(x, scale_factor=2, mode='bilinear')
        
        x=self.refine3_1x1(x)
        x=self.refine4(x)

        x = F.relu(self.deconv1_bn(x), True)

        x_P = self.feat_bn_P(self.conv_feat_P(x))
        x_P = F.upsample(x_P, scale_factor=4, mode='bilinear')

        outputs = []
        outputs.append(x_P)
        return outputs

def resnet_tiny_refine(**kwargs):
    model = Resnet_Tiny_Refine()
    return model









if __name__ == '__main__':
    import time
    from tqdm import *
    model = resnet_tiny_refine().cuda()
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    number = count_parameters(model)
    print(number)

    t1 = time.time()
    for i in tqdm( range(1000) ) :
        img = torch.rand(1,3,256,128).cuda()   #416 320     800 608
        out =  model(img)
        # print(out[0].size())
    t2 = time.time()
    print(1000 / (t2 - t1))
