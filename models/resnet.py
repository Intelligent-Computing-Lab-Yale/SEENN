import random
from models.layers import *
from torch.nn import functional as F


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class PreActBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, mode='normal'):
        super(PreActBasicBlock, self).__init__()
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = SeqToANNContainer(conv3x3(inplanes, planes, stride))
        self.bn1 = norm_layer(planes)
        self.conv2 = SeqToANNContainer(conv3x3(planes, planes))
        self.bn2 = norm_layer(planes)
        self.stride = stride
        self.spike1 = LIFSpike(dspike=dspike, gama=temp)
        self.spike2 = LIFSpike(dspike=dspike, gama=temp)
        self.mode = mode

        if (stride != 1 or inplanes != planes) and self.mode != 'none':
            # Projection also with pre-activation according to paper.
            self.downsample = nn.Sequential(tdLayer(nn.AvgPool2d(stride)),
                                            SeqToANNContainer(conv1x1(inplanes, planes)),
                                            norm_layer(planes))

    def forward(self, x):
        residual = self.downsample(x) if hasattr(self, 'downsample') else x

        out = self.spike1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + residual
        return self.spike2(out)


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10, width_mult=1, T=4, in_c=3, mode='normal',
                 use_dspike=False, gamma=1):
        super(ResNet, self).__init__()

        global norm_layer
        norm_layer = tdBatchNorm
        global dspike
        dspike = use_dspike
        global temp
        temp = gamma

        self.inplanes = 16 * width_mult

        self.conv1 = nn.Conv2d(in_c, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = tdBatchNorm(self.inplanes)
        self.spike = LIFSpike(dspike=dspike, gama=temp)
        self.layer1 = self._make_layer(block, self.inplanes, layers[0], mode=mode)
        self.layer2 = self._make_layer(block, self.inplanes * 2, layers[1], stride=2, mode=mode)
        self.layer3 = self._make_layer(block, self.inplanes * 2, layers[2], stride=2, mode=mode)
        self.avgpool = tdLayer(nn.AdaptiveAvgPool2d((1, 1)))
        self.fc1 = SeqToANNContainer(nn.Linear(self.inplanes, num_classes))
        self.T = T
        self.add_dim = lambda x: add_dimention(x, self.T)

    def _make_layer(self, block, planes, blocks, stride=1, mode='normal'):

        layers = []
        layers.append(block(self.inplanes, planes, stride, mode))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, mode=mode))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        x = self.conv1(x)
        if len(x.shape) == 4:
            x = self.add_dim(x)
        x = self.bn(x)
        x = self.spike(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1) if len(x.shape) == 4 else torch.flatten(x, 2)
        y = self.fc1(x)
        return y

    def forward(self, x):

        return self._forward_impl(x)


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = {}
        model.load_state_dict(state_dict)
    return model


def resnet19(pretrained=False, progress=True, **kwargs):
    return _resnet('resnet19', PreActBasicBlock, [3, 3, 2], pretrained, progress, **kwargs)


if __name__ == '__main__':

    a = torch.randn(1, 3, 32, 32)

    model = resnet19(num_classes=10, width_mult=1, mode='normal')
    # print(model)
    model.T = 3
    x = torch.rand(2, 3, 32, 32)
    y = model(x)
    print(y.shape)
    y.mean().backward()