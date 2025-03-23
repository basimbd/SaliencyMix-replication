import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

class ConvBn(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=0
    ):
        super(ConvBn, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return self.bn(self.conv(x))

class BottleneckBlock(nn.Module):
    in_out_ratio = 4
    def __init__(
        self, 
        in_channels, 
        firstBlock=False, 
        firstLayer=False
    ):
        super(BottleneckBlock, self).__init__()
        out_channels = in_channels*self.in_out_ratio

        self.convBn1 = ConvBn(out_channels, in_channels)
        self.convBn2 = ConvBn(in_channels, in_channels, 3, padding=1)
        if firstBlock:
            self.convBn1 = ConvBn(out_channels // 2, in_channels)
            if firstLayer:
                self.convBn1 = ConvBn(in_channels, in_channels)
            else:
                self.convBn2 = ConvBn(in_channels, in_channels, 3, stride=2, padding=1)
        self.convBn3 = ConvBn(in_channels, out_channels)

        self.projection = nn.Sequential()
        if firstBlock:
            self.projection = nn.Sequential(OrderedDict([
                ('convP', nn.Conv2d(
                    in_channels if firstLayer else out_channels // 2,
                    out_channels,
                    kernel_size=1,
                    stride=1 if firstLayer else 2,)),
                ('bnP', nn.BatchNorm2d(out_channels))
            ]))

    def forward(self, x):
        out = F.relu(self.convBn1(x))
        out = F.relu(self.convBn2(out))
        out = self.convBn3(out)
        out = out + self.projection(x)
        return F.relu(out)


class ResNet50(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet50, self).__init__()
        self.conv0 = nn.Conv2d(3, 64, 3, 1, padding=1, bias=False)
        self.bn0 = nn.BatchNorm2d(64)
        self.relu0 = nn.ReLU()
        self.maxpool0 = nn.MaxPool2d(3, 1, padding=1)

        in_channels = 64
        self.layer1 = nn.Sequential(
            BottleneckBlock(in_channels, True, True),
            BottleneckBlock(in_channels),
            BottleneckBlock(in_channels),
        )
        in_channels = 128
        self.layer2 = nn.Sequential(
            BottleneckBlock(in_channels, True),
            BottleneckBlock(in_channels),
            BottleneckBlock(in_channels),
            BottleneckBlock(in_channels),
        )
        in_channels = 256
        self.layer3 = nn.Sequential(
            BottleneckBlock(in_channels, True),
            BottleneckBlock(in_channels),
            BottleneckBlock(in_channels),
            BottleneckBlock(in_channels),
            BottleneckBlock(in_channels),
            BottleneckBlock(in_channels),
        )
        in_channels = 512
        self.layer4 = nn.Sequential(
            BottleneckBlock(in_channels, True),
            BottleneckBlock(in_channels),
            BottleneckBlock(in_channels),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        out = self.maxpool0(self.relu0(self.bn0(self.conv0(x))))
        for layer in [self.layer1, self.layer2, self.layer3, self.layer4]:
            out = layer(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        return self.fc(out)
