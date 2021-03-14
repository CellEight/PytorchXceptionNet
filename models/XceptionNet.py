import torch
import torch.nn as nn
import torch.nn.functional as F
from utility.SepConv2d import SepConv2d

class XceptionNet(nn.Module):
    """ """
    def __init__(self, n_classes):
        super().__init__()
        # Entry Flow
        self.conv1 = nn.Conv2d(3,32,kernel_size=3,stride=2,padding=1,padding_mode='reflect')
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32,64,kernel_size=3,stride=1,padding=1,padding_mode='reflect')
        self.bn2 = nn.BatchNorm2d(64)
        self.convblk1 = ConvResBlock(64,128,relu=False)
        self.convblk2 = ConvResBlock(128,256)
        self.convblk3 = ConvResBlock(256,728)
        # Middle Flow
        self.idenblk1 = IndentiyResBlock(728,728)
        self.idenblk2 = IndentiyResBlock(728,728)
        self.idenblk3 = IndentiyResBlock(728,728)
        self.idenblk4 = IndentiyResBlock(728,728)
        self.idenblk5 = IndentiyResBlock(728,728)
        self.idenblk6 = IndentiyResBlock(728,728)
        self.idenblk7 = IndentiyResBlock(728,728)
        self.idenblk8 = IndentiyResBlock(728,728)
        # Exit Flow
        self.convblk4 = ConvResBlock(728,1024,chan_mid=728)
        self.sep1 = SepConv2d(1024,1536)
        self.bn3 = nn.BatchNorm2d(1536)
        self.sep2 = SepConv2d(1536,2048)
        self.bn4 = nn.BatchNorm2d(2048)
        self.avgpool = nn.AvgPool2d(9)
        self.fc1 = nn.Linear(2048,n_classes)

    def forward(self, x):
        # Entry Flow
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.convblk1(x)
        x = self.convblk2(x)
        x = self.convblk3(x)
        # Middle Flow
        x = self.idenblk1(x)
        x = self.idenblk2(x)
        x = self.idenblk3(x)
        x = self.idenblk4(x)
        x = self.idenblk5(x)
        x = self.idenblk6(x)
        x = self.idenblk7(x)
        x = self.idenblk8(x)
        # Exit Flow
        x = self.convblk4(x)
        x = F.relu(self.bn3(self.sep1(x)))
        x = F.relu(self.bn4(self.sep2(x)))
        x = self.avgpool(x)
        x = x.view(-1,2048)
        x = F.relu(self.fc1(x))
        return x

class ConvResBlock(nn.Module):
    """ Realizes the repeated convolutional units with convolutional residual connections 
        that are present in the entry flow of the Xception Net architecture """
    def __init__(self, chan_in, chan_out, chan_mid = None, relu=True):
        super().__init__()
        self.relu = relu
        if not chan_mid:
            chan_mid = chan_out
        self.res1 = nn.Conv2d(chan_in,chan_out,kernel_size=1,stride=2)
        self.sep1 = SepConv2d(chan_in,chan_mid)
        self.bn1 = nn.BatchNorm2d(chan_mid)
        self.sep2 = SepConv2d(chan_mid,chan_out)
        self.bn2 = nn.BatchNorm2d(chan_out)
        self.max_pool = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)

    def forward(self, x):
        x0 = self.res1(x)
        if self.relu:
            x = F.relu(x)
        x1 = F.relu(self.bn1(self.sep1(x)))
        x1 = self.max_pool(self.bn2(self.sep2(x1)))
        return x0 + x1
        
class IndentiyResBlock(nn.Module):
    """ Realizes the repeated convolutional blocks with identity residual connections 
        that are present in the middle flow of the Xception Net architecture. """
    def __init__(self, chan_in, chan_out):
        super().__init__()
        self.sep1 = SepConv2d(chan_in,chan_out)
        self.bn1 = nn.BatchNorm2d(chan_out)
        self.sep2 = SepConv2d(chan_out,chan_out)
        self.bn2 = nn.BatchNorm2d(chan_out)
        self.sep3 = SepConv2d(chan_out,chan_out)
        self.bn3 = nn.BatchNorm2d(chan_out)
        
    def forward(self, x):
        x0 = x
        x1 = self.bn1(self.sep1(F.relu(x))) 
        x1 = self.bn2(self.sep2(F.relu(x))) 
        x1 = self.bn3(self.sep3(F.relu(x))) 
        return x0 + x1


