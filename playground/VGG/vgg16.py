import math

import torch
import torch.nn as nn


def conv_layer(in_channels, out_channels, kernel, p):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel, padding=p),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
    )


def conv_block(inputs, outputs, kernels, paddings, pooling):
    layers = [
        conv_layer(inputs[i], outputs[i], kernels[i], paddings[i])
        for i in range(len(inputs))
    ]
    layers.append(nn.MaxPool2d(kernel_size=pooling[0], stride=pooling[1]))
    return nn.Sequential(*layers)


def fcn_layer(input_size, output_size):
    return nn.Sequential(nn.Linear(input_size, output_size), nn.Dropout(0.5), nn.ReLU())


class VGG16(nn.Module):
    def __init__(self, n_classes=1000):
        super(VGG16, self).__init__()
        self.conv_layer1 = conv_block([3, 64], [64, 64], [3, 3], [1, 1], [2, 2])
        self.conv_layer2 = conv_block([64, 128], [128, 128], [3, 3], [1, 1], [2, 2])
        self.conv_layer3 = conv_block(
            [128, 256, 256], [256, 256, 256], [3, 3, 3], [1, 1, 1], [2, 2]
        )
        self.conv_layer4 = conv_block(
            [256, 512, 512], [512, 512, 512], [3, 3, 3], [1, 1, 1], [2, 2]
        )
        self.conv_layer5 = conv_block(
            [512, 512, 512], [512, 512, 7 * 7 * 512], [3, 3, 3], [1, 1, 1], [2, 2]
        )
        self.fcn1 = fcn_layer(7 * 7 * 512, 4096)
        self.fcn2 = fcn_layer(4096, 4096)
        self.final_layer = nn.Linear(4096, n_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                m.bias.data.zero_()

    def forward(self, x):
        out = self.conv_layer1(x)
        out = self.conv_layer2(out)
        out = self.conv_layer3(out)
        out = self.conv_layer4(out)
        vgg16_features = self.conv_layer5(out)
        out = vgg16_features.view(out.size(0), -1)
        out = self.fcn1(out)
        out = self.fcn2(out)
        out = self.final_layer(out)
        return out
