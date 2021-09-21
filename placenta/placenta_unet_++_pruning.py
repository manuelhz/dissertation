import torch
import torch.nn as nn
import torch.nn.functional as F


def double_conv2(in_ch, out_ch):
    return nn.Sequential(        
        nn.Conv2d(in_ch, out_ch, 3, padding=2, dilation=2),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),  # True means cover the origin input
        nn.Conv2d(out_ch, out_ch, 3, padding=3, dilation=3),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True)
        )

def up4(x1, x2, x3, x4):
    x = torch.cat([x4, x3, x2, x1], dim=1)
    return x        

def up3(x1, x2, x3):
    x = torch.cat([x3, x2, x1], dim=1)
    return x    

def up(x1, x2):
    # x1--up , x2 ---down        
    diffX = x1.size()[2] - x2.size()[2]
    diffY = x2.size()[3] - x1.size()[3]
    x1 = F.pad(x1, (
        diffY // 2, diffY - diffY // 2,
        diffX // 2, diffX - diffX // 2,))
    x = torch.cat([x2, x1], dim=1)    
    return x

def down(in_ch, out_ch):    
    return nn.Sequential(
        
        # double conv
        nn.Conv2d(in_ch, in_ch, 3, padding=2, dilation=2),
        nn.BatchNorm2d(in_ch),
        nn.ReLU(inplace=True),  # True means cover the origin input
        nn.Conv2d(in_ch, out_ch, 3, padding=3, dilation=3),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True)
        )

def double_conv_in(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, in_ch, 5, padding=2),
        nn.BatchNorm2d(in_ch),
        nn.ReLU(inplace=True),  # True means cover the origin input
        nn.Conv2d(in_ch, out_ch, 3, padding=1, stride=2),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True)       
        )


cc = 64


class UNetPlusPlus(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNetPlusPlus, self).__init__()
        self.inconv = double_conv_in(n_channels, cc)
        self.down1 = down(cc, 2 * cc)
        self.down2 = down(2 * cc, 4 * cc)
        self.down3 = down(4 * cc, 8 * cc)
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding=1)
        self.maxpool2 = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)        
        #self.up1 = up(12 * cc, 4 * cc)
        self.conv1 = double_conv2(12 * cc, 4 * cc)
        
        #self.up20 = up(6 * cc, 2 * cc)
        self.conv20 = double_conv2(6 * cc, 2 * cc)
        
        #self.up2 = up3(8 * cc, 2 * cc)
        self.conv2 = double_conv2(8 * cc, 2 * cc)
        
        #self.up30 = up(3 * cc, cc)
        self.conv30 = double_conv2(3 * cc, cc)
        
        #self.up31 = up3(4 * cc, cc)
        self.conv31 = double_conv2(4 * cc, cc)
        
        #self.up3 = up4(5 * cc, cc)
        self.conv3 = double_conv2(5 * cc, cc)
        
        
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.outconv0 = nn.Conv2d(cc, n_classes, 1)
        self.outconv1 = nn.Conv2d(cc, n_classes, 1)
        self.outconv2 = nn.Conv2d(cc, n_classes, 1)
        

    def forward(self, x, pruning):        
        if pruning == 0:
            x1 = self.inconv(x)
            x = self.maxpool(x1)
            
            x2 = self.down1(x)
            x = self.maxpool2(x2) 
            
            x2 = self.upsample(x2)
            x11 = up(x2, x1) # up30
            x11 = self.conv30(x11)
            
            x11 = self.upsample2(x11)
            y0 = self.outconv0(x11)            
         
            return [y0]
        
        if pruning == 1:
            x1 = self.inconv(x)
            x = self.maxpool(x1)
            
            x2 = self.down1(x)
            x = self.maxpool2(x2)
            
            x3 = self.down2(x)
            
            
            x3 = self.upsample(x3)
            x21 = up(x3, x2) # up20
            x21 = self.conv20(x21)
            
            
            
            
            x2 = self.upsample(x2)
            x11 = up(x2, x1) # up30
            x11 = self.conv30(x11)
            
            x21 = self.upsample(x21)
            x12 = up3(x21, x11, x1) # up31
            x12 = self.conv31(x12)
            
           
            
            #output 0 1 2           
            
            x11 = self.upsample2(x11)
            y0 = self.outconv0(x11)
            
            x12 = self.upsample2(x12)
            y1 = self.outconv1(x12)
            return y0, y1
            
        
        else:
            x1 = self.inconv(x)
            x = self.maxpool(x1)
            
            x2 = self.down1(x)
            x = self.maxpool2(x2)
            
            x3 = self.down2(x)
            x = self.maxpool2(x3)
            
            
            x4 = self.down3(x)
            #x = self.maxpool2(x4)
            
            x4 = self.upsample(x4)
            x = up(x4, x3) # up1
            x = self.conv1(x)
            
            x3 = self.upsample(x3)
            x21 = up(x3, x2) # up20
            x21 = self.conv20(x21)
            
            x = self.upsample(x)
            x = up3(x, x21, x2) # up2
            x = self.conv2(x)
            
            
            x2 = self.upsample(x2)
            x11 = up(x2, x1) # up30
            x11 = self.conv30(x11)
            
            x21 = self.upsample(x21)
            x12 = up3(x21, x11, x1) # up31
            x12 = self.conv31(x12)
            
            x = self.upsample(x)
            x = up4(x, x12, x11, x1) # up3
            x = self.conv3(x)        
            
            #output 0 1 2
            x = self.upsample2(x)
            y2 = self.outconv2(x)
            
            x11 = self.upsample2(x11)
            y0 = self.outconv0(x11)
            
            x12 = self.upsample2(x12)
            y1 = self.outconv1(x12)
            return y0, y1, y2
