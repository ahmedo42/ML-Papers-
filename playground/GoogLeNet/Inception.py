import torch
import torch.nn as nn
import torch.nn.functional as F


class Inception(nn.Module):
    def __init__(self, in_channels, n1x1, n3x3_reduce, n3x3, n5x5_reduce, n5x5, pool):
        super(Inception, self).__init__()
        self.b1 = Conv2d(in_channels,n1x1,kernel_size=1)

        self.b2 = nn.Sequential(
            Conv2d(in_channels,n3x3_reduce,kernel_size=1),
            Conv2d(n3x3_reduce,n3x3,kernel_size=3,padding=1)
        )

        self.b3 = nn.Sequential(
            Conv2d(in_channels,n5x5_reduce,kernel_size=1),
            Conv2d(n5x5_reduce, n5x5, kernel_size=3, padding=1)
        )

        self.b4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            Conv2d(in_channels, pool, kernel_size=1)
        )

    def forward(self, x):
        x1 = self.b1(x)
        x2 = self.b2(x)
        x3 = self.b3(x)
        x4 = self.b4(x)
        out = torch.cat([x1, x2, x3, x4], 1)
        return out

class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(Conv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)


