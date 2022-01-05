import torch
from torch import nn
from torch.nn.modules.conv import Conv2d


class Func(nn.Module):
    def __init__(self, in_ch, out_ch, initial_stride=1):
        super(Func, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=initial_stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return out

        
def downsample(in_channels, out_channels, stride):
    return nn.Sequential(
                Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

class RKBlock(nn.Module):
    def __init__(self, in_channels, out_channels, down=None, initial_stride=1):
        super(RKBlock, self).__init__()
        self.h = 1
        self.down = down
        self.block0 = Func(in_channels, out_channels, initial_stride=initial_stride)
        self.block1 = Func(out_channels, out_channels)
        self.block2 = Func(out_channels, out_channels)
        self.block3 = Func(out_channels, out_channels)
        self.block4 = Func(out_channels, out_channels)
        self.block5 = Func(out_channels, out_channels)
        self.block6 = Func(out_channels, out_channels)
        self.block7 = Func(out_channels, out_channels)



        self.act = nn.ReLU()

    def forward(self,x):
        h = self.h
        x0 = x

        if self.down is not None:
            x0down = self.down(x0)
        else:
            x0down = x0

        # x1 ---------------------
        
        x1 = x0down + h*self.block0(x0)

        # x2 ---------------------     
        k1 = self.block0(x0) 
        k2 = self.block1(x1 + h*k1)
        k3 = self.block2(x1 - h*k1 +2*h*k2)
        
        x2 = x0down + (h/3)*(k1+4*k2 + k3)     
        
        # x3 ---------------------
        k1 = self.block1(x1) 
        k2 = self.block2(x2 + h*k1) 
        k3 = self.block3(x2 - h*k1 +2*h*k2)

        x3 = x1 + (h/3)*(k1+4*k2 + k3)     
        
        # x4 ---------------------
        k1 = self.block2(x2) 
        k2 = self.block3(x3 + h*k1) 
        k3 = self.block4(x3 - h*k1 +2*h*k2)

        x4 = x2 + (h/3)*(k1+4*k2 + k3)     
        
        # x5 ---------------------
        k1 = self.block3(x3) 
        k2 = self.block4(x4 + h*k1) 
        k3 = self.block5(x4 - h*k1 +2*h*k2)

        x5 = x3 + (h/3)*(k1+4*k2 + k3)     
        
        # x6 ---------------------
        k1 = self.block4(x4) 
        k2 = self.block5(x5 + h*k1) 
        k3 = self.block6(x5 - h*k1 +2*h*k2)

        x6 = x4 + (h/3)*(k1+4*k2 + k3)     
        
        # x7 ---------------------
        k1 = self.block5(x5) 
        k2 = self.block6(x6 + h*k1) 
        k3 = self.block7(x6 - h*k1 +2*h*k2)

        x7 = x5 + (h/3)*(k1+4*k2 + k3) 


        return x7



class rkNet(nn.Module):
    def __init__(self, num_classes=10):
        super(rkNet, self).__init__()
        self.name = rkNet
        self.depth = 49
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.down64to128 = downsample(64, 128, stride=2)
        self.down128to256 = downsample(128, 256, stride=2)
        self.down256to512 = downsample(256, 512, stride=2)

        self.layer1 = RKBlock(64, 64)
        self.layer2 = RKBlock(64, 128, self.down64to128, initial_stride = 2)
        self.layer3 = RKBlock(128, 256, self.down128to256, initial_stride = 2)
        # self.layer4 = RKBlock(256, 512, self.down256to512, initial_stride = 2)

        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)
        self.act = nn.ReLU()

    def forward(self, x):
        h = 1
        out = self.conv1(x)  #x1: 64x32x32 -> 64x16x16  
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)
        
        #layers 
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)

        # FCN
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)

        return out
