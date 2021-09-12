import math
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from Inception import Conv2d, Inception
from torch import Tensor


class GoogLeNet(nn.Module):
    def __init__(self, n_classes=1000, aux_logits=False):
        super(GoogLeNet, self).__init__()
        self.conv1 = Conv2d(3, 64,kernel_size=7,stride=2,padding=3)
        self.max_pool1 = nn.MaxPool2d(kernel_size=3, stride=2,ceil_mode=True)
        self.conv2 = Conv2d(64, 64, kernel_size=1)
        self.conv3 = Conv2d(64, 192, kernel_size=3, padding=1)
        self.max_pool2 = nn.MaxPool2d(kernel_size=3, stride=2,ceil_mode=True)

        self.inception_3a = Inception(192, 64, 96, 128, 16, 32, 32)
        self.inception_3b = Inception(256, 128, 128, 192, 32, 96, 64)
        self.max_pool3 = nn.MaxPool2d(kernel_size=3, stride=2,ceil_mode=True)

        self.inception_4a = Inception(480, 192, 96, 208, 16, 48, 64)
        self.inception_4b = Inception(512, 160, 112, 224, 24, 64, 64)
        self.inception_4c = Inception(512, 128, 128, 256, 24, 64, 64)
        self.inception_4d = Inception(512, 112, 144, 288, 32, 64, 64)
        self.inception_4e = Inception(528, 256, 160, 320, 32, 128, 128)
        self.max_pool4 = nn.MaxPool2d(kernel_size=2, stride=2,ceil_mode=True)

        self.inception_5a = Inception(832, 256, 160, 320, 32, 128, 128)
        self.inception_5b = Inception(832, 384, 192, 384, 48, 128, 128)
        self.aux1 = None
        self.aux2 = None
        if aux_logits:
            self.aux1 = InceptionAux(512, n_classes)
            self.aux2 = InceptionAux(528, n_classes)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(1024, n_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))


    def forward(self, x):
        x = self.conv1(x)
        x = self.max_pool1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.max_pool2(x)

        x = self.inception_3a(x)
        x = self.inception_3b(x)
        x = self.max_pool3(x)

        x = self.inception_4a(x)

        aux1 = torch.jit.annotate(Optional[Tensor], None)
        if self.aux1 is not None:
            if self.training:
                aux1 = self.aux1(x)

        x = self.inception_4b(x)
        x = self.inception_4c(x)
        x = self.inception_4d(x)
        aux2 = torch.jit.annotate(Optional[Tensor], None)
        if self.aux2 is not None:
            if self.training:
                aux2 = self.aux2(x)

        x = self.inception_4e(x)
        x = self.max_pool4(x)
        x = self.inception_5a(x)
        x = self.inception_5b(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x, aux1, aux2



class InceptionAux(nn.Module):
    def __init__(self, in_channels, n_classes, **kwargs):
        super(InceptionAux, self).__init__()
        self.conv = Conv2d(in_channels, 128, kernel_size=1)
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, n_classes)

    def forward(self, x):
        x = F.adaptive_avg_pool2d(x, (4, 4))
        x = self.conv(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x), inplace=True)
        x = F.dropout(x, 0.7, training=self.training)
        x = self.fc2(x)
        return x
