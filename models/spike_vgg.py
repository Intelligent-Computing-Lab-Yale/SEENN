'''
Modified from https://github.com/pytorch/vision.git
'''
import math

import torch
import torch.nn as nn
import torch.nn.init as init
from .layers import *

__all__ = [
    'VGG', 'vgg16',
]


class VGG(nn.Module):
    '''
    VGG model
    '''

    def __init__(self, cfg, num_classes=10, batch_norm=True, width_mult=1, in_c=3, **lif_parameters):
        super(VGG, self).__init__()

        self.features, out_c = make_layers(cfg, batch_norm, width_mult, in_c, **lif_parameters)
        self.avgpool = SeqToANNContainer(nn.AdaptiveAvgPool2d((1, 1)))
        self.classifier = nn.Sequential(
            # nn.Dropout(0.25),
            SeqToANNContainer(nn.Linear(out_c, num_classes)),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()

        self.add_dim = lambda x: add_dimention(x, self.T)

    def forward(self, x, return_feature=False):
        if len(x.shape) == 4:
            x = add_dimention(x, self.T)
        x = self.features(x)
        x = self.avgpool(x)
        feature_output = torch.flatten(x, 1) if len(x.shape) == 4 else torch.flatten(x, 2)
        output = self.classifier(feature_output)
        if return_feature:
            return output, feature_output
        else:
            return output


def make_layers(cfg, batch_norm=False, width_mult=1, in_c=3, **lif_parameters):
    layers = []
    in_channels = in_c
    for v in cfg:
        if v == 'M':
            layers += [SeqToANNContainer(nn.AvgPool2d(kernel_size=2, stride=2))]
        else:
            conv2d = SeqToANNContainer(nn.Conv2d(in_channels, int(v * width_mult / 4), kernel_size=3, padding=1))

            lif = LIFSpike(**lif_parameters)

            if batch_norm:
                bn = tdBatchNorm(int(v * width_mult / 4))
                layers += [conv2d, bn, lif]
            else:
                layers += [conv2d, lif]

            in_channels = int(v * width_mult / 4)
    return nn.Sequential(*layers), in_channels


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 512, 512],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 512, 512],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 512, 512, 512, 512],
}


def vgg16(*args, **kwargs):
    """VGG 16-layer model (configuration "D") with batch normalization"""
    return VGG(cfg['D'], *args, **kwargs)


if __name__ == '__main__':
    model = vgg16(num_classes=10, width_mult=1)
    model.T = 2
    x = torch.rand(2, 3, 32, 32)
    y = model(x)
    y.sum().backward()
    print(y.shape)