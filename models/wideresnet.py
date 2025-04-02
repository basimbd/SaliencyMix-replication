import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable

import sys
import os
import numpy


def conv3x3(in_channels, out_channels, stride=1, dilation=1):   
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride,
                     padding=dilation, dilation=dilation, bias=True)

def conv_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        init.constant_(m.weight, 1)
        if m.bias is not None:
            init.constant_(m.bias, 0)
    elif classname.find('Linear') != -1:
        init.normal_(m.weight, 0, 0.01)
        if m.bias is not None:
            init.constant_(m.bias, 0)


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dor=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)

        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)

        self.dropout = dor
        self.equalInputAndOutput = (in_channels == out_channels)
        self.convShortcut = None
        if not self.equalInputAndOutput:
            self.convShortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False)

    def forward(self, x):
        if not self.equalInputAndOutput:
            x = self.relu1(self.bn1(x))
            out = self.relu2(self.bn2(self.conv1(x)))            
        else:
            out =  self.relu2(self.bn2(self.conv1(self.relu1(self.bn1(x))))) 
    
        if self.dropout > 0.0:
            out = F.dropout(out, p=self.dropout, training=self.training, inplace=True)

        out = self.conv2(out)

        return torch.add(x if self.equalInputAndOutput else self.convShortcut(x) , out) 

class NetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, blocks, stride=1, dor=0.0):
        super(NetBlock, self).__init__()
        self.layer = self._make_layer(in_channels, out_channels, blocks, stride, dor)

    def _make_layer(self, in_channels, out_channels, blocks, stride=1, dor=0.0):
        layers = []
        layers.append(BasicBlock(in_channels, out_channels, stride=stride, dor=dor))
        for i in range(1, blocks):
            layers.append(BasicBlock(out_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)
    
class WideResNet(nn.Module):    
    def __init__(self, depth, widen_factor, num_classes, input_channels=3, dor=0.0):
        super(WideResNet, self).__init__()
        self.num_classes = num_classes
        self.input_channels = input_channels
        self.depth = depth
        self.widen_factor = widen_factor
        self.dropout = dor

        n = int((depth - 4) // 6)
        k = widen_factor

        # wide resnet architecture
        in_channels = 16
        out_channels = 16 * k

        self.conv1 = nn.Conv2d(input_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)

        self.layer1 = NetBlock(in_channels, out_channels, n, stride=1, dor=self.dropout)
        in_channels = out_channels

        out_channels *= 2
        self.layer2 = NetBlock(in_channels, out_channels, n, stride=2, dor=self.dropout)
        in_channels = out_channels

        out_channels *= 2
        self.layer3 = NetBlock(in_channels, out_channels, n, stride=2, dor=self.dropout)

        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

        
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(out_channels, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.bn2(x)
        x = self.relu2(x)

        x = self.avgpool(x)
        
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = self.fc(x)
        return x

