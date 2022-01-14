import torch
from torch import nn
from torch.nn.modules.conv import Conv2d


class MyBasicBlock(nn.Module):

    def __init__(self, in_ch, out_ch,  downsample, initial_stride, h, fun):
        super().__init__()
        self.h = h
        self.relu = nn.ReLU()
        self.fun = fun(in_ch, out_ch, initial_stride)
        self.downsample = downsample
        
    def forward(self, x):
        identity = x
        out = self.fun(x)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = identity + self.h*out
        out = self.relu(out)

        return out


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

class Func2(nn.Module):
    def __init__(self, in_ch, out_ch, initial_stride=1):
        super(Func2, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, in_ch, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(in_ch)
    
        self.conv2 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=initial_stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)

        self.conv3 = nn.Conv2d(out_ch, out_ch, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(out_ch)

        self.act1 = nn.ReLU()

    def forward(self, x):

        out = self.act1(self.bn1(self.conv1(x)))
        out = self.act1(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        return out



class Sequential2(nn.Sequential):
    def __init__(self, *args):
        super(Sequential2, self).__init__(*args)

    def forward(self, x, y):
        for model in self:
            x,y = model(x,y)
        return x,y 

        
def downsample(in_channels, out_channels, stride):
    return nn.Sequential(
                Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )




class MiddleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, down1=None, down2=None, initial_stride=1, h=1, fun=Func):
        super(MiddleBlock, self).__init__()
        self.down = down1
        self.down2 = down2
        self.block = fun(in_channels, out_channels, initial_stride=initial_stride)
        self.act = nn.ReLU()
        self.h = h


    def forward(self, x1, x2):
        if self.down is not None:
            x1 = self.down(x1)

        y2 = self.h*self.act(x1 + self.block(x2))


        if self.down is not None:
          x2 = self.down2(x2)
        return x2, y2

class MiddleNet(nn.Module):
    def __init__(self, in_channels, layers, h=1, num_classes=10, fun=Func):
        super(MiddleNet, self).__init__()

        self.name = 'middleNet'
        self.depth = sum(layers)*2 + 2
        self.h  = h
        self.conv1 = nn.Conv2d(3, in_channels, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.block = Func(in_channels, in_channels)
        self.act = nn.ReLU()

        #makelayers 
        self.part1 = self.make_layer(MiddleBlock, in_channels, 64, initial_stride=1, num_blocks = layers[0], h=self.h, fun=fun)
        self.part2 = self.make_layer(MiddleBlock, 64, 128, initial_stride=2, num_blocks = layers[1], h=self.h, fun=fun)
        self.part3 = self.make_layer(MiddleBlock, 128, 256, initial_stride=2, num_blocks = layers[2], h=self.h, fun=fun)
        self.part4 = self.make_layer(MiddleBlock, 256, 512, initial_stride=2, num_blocks = layers[3], h=self.h, fun=fun)
        
        #FCN
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        

    def forward(self, x):
        x1 = self.conv1(x)  #x1: 64x32x32 -> 64x16x16  
        x1 = self.bn1(x1)
        x1 = self.relu(x1)
        x1 = self.maxpool(x1)

        x2 = self.block(x1) #x2: 64x16x16-> 64x16x16

        # layers
        x1, x2 = self.part1(x1,x2)
        x1, x2 = self.part2(x1,x2)
        x1, x2 = self.part3(x1,x2)
        x1, x2 = self.part4(x1,x2)
            
        
        # FCN
        x2 = self.avgpool(x2)
        x2 = torch.flatten(x2, 1)
        x2 = self.fc(x2)
       
        return x2

    def make_layer(self, block, in_channels, out_channels, initial_stride, num_blocks, h, fun):

        #Create downsample instance
        downsample = None 
        downsample2 = None 
        if initial_stride != 1:
            downsample = nn.Sequential(
                Conv2d(in_channels, out_channels, kernel_size=1, stride=initial_stride),
                nn.BatchNorm2d(out_channels)
            )

            downsample2 = nn.Sequential(
                Conv2d(in_channels, out_channels, kernel_size=1, stride=initial_stride),
                nn.BatchNorm2d(out_channels)
            )
             
        # Create Layers list
        layers = []

        # make first block
        initial_block = block(in_channels, out_channels, downsample, downsample2, initial_stride, h, fun)
        layers.append(initial_block)

        # create multiple layers
        for _ in range(1, num_blocks):
            new_block = block(out_channels, out_channels, h=h, fun=fun)
            layers.append(new_block) 
        
        return Sequential2(*layers)

class MyResNet(nn.Module):
    def __init__(self, in_channels, layers, h=1, num_classes=10, fun=Func):
        super(MyResNet, self).__init__()

        self.name = 'resNet'
        self.depth = sum(layers)*2 + 2
        self.h  = h
        self.conv1 = nn.Conv2d(3, in_channels, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.block = Func(in_channels, in_channels)
        self.act = nn.ReLU()

        #makelayers 
        self.part1 = self.make_layer(MyBasicBlock, in_channels, 64, initial_stride=1, num_blocks = layers[0], h=self.h, fun=fun)
        self.part2 = self.make_layer(MyBasicBlock, 64, 128, initial_stride=2, num_blocks = layers[1], h=self.h, fun=fun)
        self.part3 = self.make_layer(MyBasicBlock, 128, 256, initial_stride=2, num_blocks = layers[2], h=self.h, fun=fun)
        self.part4 = self.make_layer(MyBasicBlock, 256, 512, initial_stride=2, num_blocks = layers[3], h=self.h, fun=fun)
        
        #FCN
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        

    def forward(self, x):
        out = self.conv1(x)  #x1: 64x32x32 -> 64x16x16  
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)

        out= self.block(out) #x2: 64x16x16-> 64x16x16

        # layers
        out = self.part1(out)
        out = self.part2(out)
        out = self.part3(out)
        out = self.part4(out)
            
        
        # FCN
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
       
        return out

    def make_layer(self, block, in_channels, out_channels, initial_stride, num_blocks, h, fun):

        #Create downsample instance
        downsample = None 
        downsample2 = None 
        if initial_stride != 1:
            downsample = nn.Sequential(
                Conv2d(in_channels, out_channels, kernel_size=1, stride=initial_stride),
                nn.BatchNorm2d(out_channels)
            )
             
        # Create Layers list
        layers = []

        # make first block
        initial_block = block(in_channels, out_channels, downsample, initial_stride, h, fun)
        layers.append(initial_block)

        # create multiple layers
        for _ in range(1, num_blocks):
            new_block = block(out_channels, out_channels, downsample=None, initial_stride=1, h=h, fun=fun)
            layers.append(new_block) 
        
        return nn.Sequential(*layers)


class RkNet34(nn.Module):
    def __init__(self, num_classes=10, h=1, fun = Func):
        super(RkNet34, self).__init__()
        self.name = 'rkNet'
        self.h = h
        if fun == Func:
            self.depth = 34
        else:
            self.depth = 50

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.down64to128 = downsample(64, 128, stride=2)
        self.down128to256 = downsample(128, 256, stride=2)
        self.down256to512 = downsample(256, 512, stride=2)

        self.layer1 = RKBlock3(64, 64, h=1)
        self.layer2 = RKBlock4(64, 128, self.down64to128, h=self.h, initial_stride = 2, fun=fun)
        self.layer3 = RKBlock6(128, 256, self.down128to256, h=self.h, initial_stride = 2, fun=fun)
        self.layer4 = RKBlock3(256, 512, self.down256to512, h=self.h, initial_stride = 2, fun= fun)

        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
        self.act = nn.ReLU()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
       
    def forward(self, x):    
        out = self.conv1(x)  #x1: 64x32x32 -> 64x16x16  
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)
        
        #layers 
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        # FCN
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)

        return out

class RkNet68(nn.Module):
    def __init__(self, num_classes=10, h=1, fun = Func):
        super(RkNet68, self).__init__()
        self.name = 'rkNet'
        self.h = h
        if fun == Func:
            self.depth = 68
        else:
            self.depth = 101

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.down64to128 = downsample(64, 128, stride=2)
        self.down128to256 = downsample(128, 256, stride=2)
        self.down256to512 = downsample(256, 512, stride=2)

        self.layer1 = RKBlock3(64, 64, h=1)
        self.layer2 = RKBlock4(64, 128, self.down64to128, h=self.h, initial_stride = 2, fun=fun)
        self.layer3 = RKBlock23(128, 256, self.down128to256, h=self.h, initial_stride = 2, fun=fun)
        self.layer4 = RKBlock3(256, 512, self.down256to512, h=self.h, initial_stride = 2, fun= fun)

        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
        self.act = nn.ReLU() 

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        

    def forward(self, x):
        
        out = self.conv1(x)  #x1: 64x32x32 -> 64x16x16  
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)
        
        #layers 
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        # FCN
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)

        return out

class RKBlock3(nn.Module):
    def __init__(self, in_channels, out_channels, down=None, h = 1, initial_stride=1, fun=Func):
        super(RKBlock3, self).__init__()
        self.h = h
        self.down = down
        self.block0 = fun(in_channels, out_channels, initial_stride=initial_stride)
        self.block1 = fun(out_channels, out_channels)
        self.block2 = fun(out_channels, out_channels)
        self.block3 = fun(out_channels, out_channels)
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

        return x3

class RKBlock4(nn.Module):
    def __init__(self, in_channels, out_channels, down=None, h = 1, initial_stride=1, fun=Func):
        super(RKBlock4, self).__init__()
        self.h = h
        self.down = down
        self.block0 = fun(in_channels, out_channels, initial_stride=initial_stride)
        self.block1 = fun(out_channels, out_channels)
        self.block2 = fun(out_channels, out_channels)
        self.block3 = fun(out_channels, out_channels)
        self.block4 = fun(out_channels, out_channels)
    
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
        
        return x4

class RKBlock6(nn.Module):
    def __init__(self, in_channels, out_channels, down=None, h = 1, initial_stride=1, fun=Func):
        super(RKBlock6, self).__init__()
        self.h = h
        self.down = down
        self.block0 = fun(in_channels, out_channels, initial_stride=initial_stride)
        self.block1 = fun(out_channels, out_channels)
        self.block2 = fun(out_channels, out_channels)
        self.block3 = fun(out_channels, out_channels)
        self.block4 = fun(out_channels, out_channels)
        self.block5 = fun(out_channels, out_channels)
        self.block6 = fun(out_channels, out_channels)

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
        
        return x6


class RKBlock10(nn.Module):
    def __init__(self, in_channels, out_channels, down=None, h = 1, initial_stride=1, fun=Func):
        super(RKBlock10, self).__init__()
        self.h = h
        self.down = down
        self.block0 = fun(in_channels, out_channels, initial_stride=initial_stride)
        self.block1 = fun(out_channels, out_channels)
        self.block2 = fun(out_channels, out_channels)
        self.block3 = fun(out_channels, out_channels)
        self.block4 = fun(out_channels, out_channels)
        self.block5 = fun(out_channels, out_channels)
        self.block6 = fun(out_channels, out_channels)
        self.block7 = fun(out_channels, out_channels)
        self.block8 = fun(out_channels, out_channels)
        self.block9 = fun(out_channels, out_channels)
        self.block10 = fun(out_channels, out_channels)

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

        # x8 ---------------------
        k1 = self.block6(x6) 
        k2 = self.block7(x7 + h*k1) 
        k3 = self.block8(x7 - h*k1 +2*h*k2)

        x8 = x6 + (h/3)*(k1+4*k2 + k3) 

        # x9 ---------------------
        k1 = self.block7(x7) 
        k2 = self.block8(x8 + h*k1) 
        k3 = self.block9(x8 - h*k1 +2*h*k2)

        x9 = x7 + (h/3)*(k1+4*k2 + k3) 

        # x10 ---------------------
        k1 = self.block8(x8) 
        k2 = self.block9(x9 + h*k1) 
        k3 = self.block10(x9 - h*k1 +2*h*k2)

        x10 = x7 + (h/3)*(k1+4*k2 + k3) 


        return x10

class RKBlock23(nn.Module):
    def __init__(self, in_channels, out_channels, down=None, h = 1, initial_stride=1, fun=Func):
        super(RKBlock23, self).__init__()
        self.h = h
        self.down = down
        self.block0 = fun(in_channels, out_channels, initial_stride=initial_stride)
        self.block1 = fun(out_channels, out_channels)
        self.block2 = fun(out_channels, out_channels)
        self.block3 = fun(out_channels, out_channels)
        self.block4 = fun(out_channels, out_channels)
        self.block5 = fun(out_channels, out_channels)
        self.block6 = fun(out_channels, out_channels)
        self.block7 = fun(out_channels, out_channels)
        self.block8 = fun(out_channels, out_channels)
        self.block9 = fun(out_channels, out_channels)
        self.block10 = fun(out_channels, out_channels)
        self.block11 = fun(out_channels, out_channels)
        self.block12 = fun(out_channels, out_channels)
        self.block13 = fun(out_channels, out_channels)
        self.block14 = fun(out_channels, out_channels)
        self.block15 = fun(out_channels, out_channels)
        self.block16 = fun(out_channels, out_channels)
        self.block17 = fun(out_channels, out_channels)
        self.block18 = fun(out_channels, out_channels)
        self.block19 = fun(out_channels, out_channels)
        self.block20 = fun(out_channels, out_channels)
        self.block21 = fun(out_channels, out_channels)
        self.block22 = fun(out_channels, out_channels)
        self.block23 = fun(out_channels, out_channels)

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

        # x8 ---------------------
        k1 = self.block6(x6) 
        k2 = self.block7(x7 + h*k1) 
        k3 = self.block8(x7 - h*k1 +2*h*k2)

        x8 = x6 + (h/3)*(k1+4*k2 + k3) 

        # x9 ---------------------
        k1 = self.block7(x7) 
        k2 = self.block8(x8 + h*k1) 
        k3 = self.block9(x8 - h*k1 +2*h*k2)

        x9 = x7 + (h/3)*(k1+4*k2 + k3) 

        # x10 ---------------------
        k1 = self.block8(x8) 
        k2 = self.block9(x9 + h*k1) 
        k3 = self.block10(x9 - h*k1 +2*h*k2)

        x10 = x8 + (h/3)*(k1+4*k2 + k3) 

        # x11 ---------------------
        k1 = self.block9(x9) 
        k2 = self.block10(x10 + h*k1) 
        k3 = self.block11(x10 - h*k1 +2*h*k2)

        x11 = x9 + (h/3)*(k1+4*k2 + k3) 

        # x12 ---------------------
        k1 = self.block10(x10) 
        k2 = self.block11(x11 + h*k1) 
        k3 = self.block12(x11 - h*k1 +2*h*k2)

        x12 = x10 + (h/3)*(k1+4*k2 + k3) 

        # x13 ---------------------
        k1 = self.block11(x11) 
        k2 = self.block12(x12 + h*k1) 
        k3 = self.block13(x12 - h*k1 +2*h*k2)

        x13 = x11 + (h/3)*(k1+4*k2 + k3) 

        # x14 ---------------------
        k1 = self.block12(x12) 
        k2 = self.block13(x13 + h*k1) 
        k3 = self.block14(x13 - h*k1 +2*h*k2)

        x14 = x12 + (h/3)*(k1+4*k2 + k3) 

        # x15 ---------------------
        k1 = self.block13(x13) 
        k2 = self.block14(x14 + h*k1) 
        k3 = self.block15(x14 - h*k1 +2*h*k2)

        x15 = x13 + (h/3)*(k1+4*k2 + k3) 

        # x16 ---------------------
        k1 = self.block14(x14) 
        k2 = self.block15(x15 + h*k1) 
        k3 = self.block16(x15 - h*k1 +2*h*k2)

        x16 = x14 + (h/3)*(k1+4*k2 + k3) 

        # x17 ---------------------
        k1 = self.block15(x15) 
        k2 = self.block16(x16 + h*k1) 
        k3 = self.block17(x16 - h*k1 +2*h*k2)

        x17 = x15 + (h/3)*(k1+4*k2 + k3) 
        
        # x18 ---------------------
        k1 = self.block16(x16) 
        k2 = self.block17(x17 + h*k1) 
        k3 = self.block18(x17 - h*k1 +2*h*k2)

        x18 = x16 + (h/3)*(k1+4*k2 + k3) 

        # x19 ---------------------
        k1 = self.block17(x17) 
        k2 = self.block18(x18 + h*k1) 
        k3 = self.block19(x18 - h*k1 +2*h*k2)

        x19 = x17 + (h/3)*(k1+4*k2 + k3)

        # x20 ---------------------
        k1 = self.block18(x18) 
        k2 = self.block19(x19 + h*k1) 
        k3 = self.block20(x19 - h*k1 +2*h*k2)

        x20 = x18 + (h/3)*(k1+4*k2 + k3)

        # x21 ---------------------
        k1 = self.block19(x19) 
        k2 = self.block20(x20 + h*k1) 
        k3 = self.block21(x20 - h*k1 +2*h*k2)

        x21 = x19 + (h/3)*(k1+4*k2 + k3)

        # x22 ---------------------
        k1 = self.block20(x20) 
        k2 = self.block21(x21 + h*k1) 
        k3 = self.block22(x21 - h*k1 +2*h*k2)

        x22 = x20 + (h/3)*(k1+4*k2 + k3)
         
        # x23 ---------------------
        k1 = self.block21(x21) 
        k2 = self.block22(x22 + h*k1) 
        k3 = self.block23(x22 - h*k1 +2*h*k2)

        x23 = x21 + (h/3)*(k1+4*k2 + k3)
         
        return x23