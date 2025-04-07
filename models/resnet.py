import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import math

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

class BottleneckBlockINet(nn.Module):
    in_out_ratio = 4
    def __init__(
        self, 
        in_channels, 
        out_channels,
        stride=1,
        downsample = None
    ):
        super(BottleneckBlockINet, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels* BottleneckBlockINet.in_out_ratio, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels* BottleneckBlockINet.in_out_ratio)
        self.relu = nn.ReLU(inplace=True)
        
        self.downsample = downsample
        self.stride = stride
        

    def forward(self, x):
        res = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        
        
        if self.downsample is not None:
            res = self.downsample(x)
                
        out = out + res
        
        return F.relu(out)

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

class ResNet101(nn.Module):
    # 3 4 23 3 architecture
    def __init__(self, num_classes=10):
        super(ResNet101, self).__init__()
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
            BottleneckBlock(in_channels),
            BottleneckBlock(in_channels),
            BottleneckBlock(in_channels),
            BottleneckBlock(in_channels),
            BottleneckBlock(in_channels),
            BottleneckBlock(in_channels),
            BottleneckBlock(in_channels), 
            BottleneckBlock(in_channels),
            BottleneckBlock(in_channels),
            BottleneckBlock(in_channels),
            BottleneckBlock(in_channels),
            BottleneckBlock(in_channels),
            BottleneckBlock(in_channels),
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
    

class ResNet_imagenet(nn.Module):
    def __init__(self, num_classes=200):
        super(ResNet_imagenet, self).__init__()

        layers = [3, 4, 6, 3]
        block = BottleneckBlockINet

        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2) 
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(512 * 4, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, math.sqrt(2.0 / (m.kernel_size[0] * m.kernel_size[1] *m.out_channels)))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.in_out_ratio:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.in_out_ratio, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.in_out_ratio),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.in_out_ratio
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x