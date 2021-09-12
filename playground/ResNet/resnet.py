import torch
import torch.nn as nn
from torch import Tensor


class ResBlock(nn.Module):
    expansion = 1

    def __init__(self ,inplanes, planes ,downsample = None ,stride = 1):
        super(ResBlock,self).__init__()
        self.conv1 = nn.Conv2d(inplanes,planes,kernel_size=3,stride=stride,padding=1,bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes,planes,kernel_size=3,padding=1,bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample


    def forward(self,x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)


        out += identity
        out = self.relu(out)
        return out


class Bottleneck(nn.Module) :
    expansion = 4

    def __init__(self,inplanes,planes,downsample = None,stride = 1):
        super(Bottleneck,self).__init__()
        self.conv1 = nn.Conv2d(inplanes,planes,kernel_size=1,bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes,planes,kernel_size=3,stride=stride,bias=False,padding=1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes,planes*self.expansion,kernel_size=1,bias=False)
        self.bn3 = nn.BatchNorm2d(planes*self.expansion)
        self.downsample = downsample

    def forward(self,x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)

        
        out += identity
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    
    def __init__(self,block,blocks,n_classes=10):
        super(ResNet,self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3,self.inplanes,kernel_size=7,stride=2,padding=3,bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv2 = self._make_layer(block,64,blocks[0])
        self.conv3 = self._make_layer(block,128,blocks[1],stride=2)
        self.conv4 = self._make_layer(block,256,blocks[2],stride=2)
        self.conv5 = self._make_layer(block,512,blocks[3],stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512*block.expansion,n_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self,x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)

        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)

        out = self.avgpool(out)
        out = torch.flatten(out,1)
        out = self.fc(out)
        return out

    def _make_layer(self,block,planes,blocks,stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes , planes * block.expansion , kernel_size=1 , stride=stride,bias=False),
                nn.BatchNorm2d(planes*block.expansion)
            )

        layers = []
        layers.append(block(self.inplanes,planes,downsample,stride))
        self.inplanes = planes * block.expansion
        for _ in range(1,blocks):
            layers.append(block(self.inplanes,planes))

        return nn.Sequential(*layers)

def ResNet18():
    return ResNet(ResBlock,[2,2,2,2])


def ResNet34():
    return ResNet(ResBlock,[3,4,6,3])


def ResNet50():
    return ResNet(Bottleneck,[3,4,6,3])

def test():
    net = ResNet50()
    y = net(torch.randn(1, 3, 224, 224))
    print(y.size())


if __name__ == "__main__":
    test()