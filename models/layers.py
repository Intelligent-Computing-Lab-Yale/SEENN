import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class TensorNormalization(nn.Module):
    def __init__(self,mean, std):
        super(TensorNormalization, self).__init__()
        if not isinstance(mean, torch.Tensor):
            mean = torch.tensor(mean)
        if not isinstance(std, torch.Tensor):
            std = torch.tensor(std)
        self.mean = mean
        self.std = std

    def forward(self, X):
        return normalizex(X,self.mean,self.std)


def normalizex(tensor, mean, std):
    mean = mean[None, :, None, None]
    std = std[None, :, None, None]
    if mean.device != tensor.device:
        mean = mean.to(tensor.device)
        std = std.to(tensor.device)
    return tensor.sub(mean).div(std)


class SeqToANNContainer(nn.Module):
    # This code is form spikingjelly https://github.com/fangwei123456/spikingjelly
    def __init__(self, *args):
        super().__init__()
        if len(args) == 1:
            self.module = args[0]
        else:
            self.module = nn.Sequential(*args)

    def forward(self, x_seq: torch.Tensor):
        y_shape = [x_seq.shape[0], x_seq.shape[1]]
        y_seq = self.module(x_seq.flatten(0, 1).contiguous())
        y_shape.extend(y_seq.shape[1:])
        return y_seq.view(y_shape)


class Layer(nn.Module):
    def __init__(self,in_plane,out_plane,kernel_size,stride,padding,spiking=True):
        super(Layer, self).__init__()
        self.fwd = SeqToANNContainer(
            nn.Conv2d(in_plane,out_plane,kernel_size,stride,padding),
            nn.BatchNorm2d(out_plane)
        )
        self.act = LIFSpike() if spiking else nn.ReLU(True)

    def forward(self,x):
        x = self.fwd(x)
        x = self.act(x)
        return x


class APLayer(nn.Module):
    def __init__(self,kernel_size):
        super(APLayer, self).__init__()
        self.fwd = SeqToANNContainer(
            nn.AvgPool2d(kernel_size),
        )
        self.act = LIFSpike()

    def forward(self,x):
        x = self.fwd(x)
        x = self.act(x)
        return x


class ZIF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, gama):
        out = (input > 0).float()
        L = torch.tensor([gama])
        ctx.save_for_backward(input, out, L)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (input, out, others) = ctx.saved_tensors
        gama = others[0].item()
        grad_input = grad_output.clone()
        tmp = (1 / gama) * (1 / gama) * ((gama - input.abs()).clamp(min=0))
        # tmp = torch.ones_like(input)
        # tmp = torch.where(input.abs() < 0.5, 1., 0.)
        grad_input = grad_input * tmp
        return grad_input, None


class DynamicLIF(nn.Module):
    def __init__(self, channels: int, reduction=4, thresh=1., tau=0.5, gama=1.0, dspike=False):
        super(DynamicLIF, self).__init__()
        if not dspike:
            self.act = ZIF.apply
        else:
            self.act = DSPIKE(region=gama)
        self.thresh = thresh
        self.tau = tau
        self.gama = gama

        self.fc1 = nn.Linear(channels, channels // reduction)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(channels // reduction, 1)

    def get_lifp(self, x):
        x_mean = torch.mean(x, dim=[-1, -2])
        embed = self.fc1(x_mean)
        embed = self.relu(embed)
        embed = self.fc2(embed)
        lifp = torch.sigmoid(embed)
        return lifp

    def forward(self, x):
        tau = 0.5
        mem = 0
        spike_pot = []
        B, T = x.shape[:2]
        for t in range(T):
            mem = mem * tau + x[:, t, ...]
            tau = self.get_lifp(mem).view(B, 1, 1, 1)
            spike = self.act(mem - self.thresh, self.gama)
            mem = (1 - spike) * mem
            spike_pot.append(spike)
        return torch.stack(spike_pot, dim=1)


class PLIF(nn.Module):
    def __init__(self, channel, thresh=1., tau=0.5, gama=1.0, dspike=False):
        super(PLIF, self).__init__()
        if not dspike:
            self.act = ZIF.apply
        else:
            self.act = DSPIKE(region=gama)
        self.thresh = thresh
        self.tau = torch.nn.Parameter(torch.zeros(channel))
        self.gama = gama

    def forward(self, x):
        mem = 0
        spike_pot = []
        T = x.shape[1]
        tau = torch.sigmoid(self.tau.view(1, -1, 1, 1))
        for t in range(T):
            mem = mem * tau + x[:, t, ...]
            spike = self.act(mem - self.thresh, self.gama)
            mem = (1 - spike) * mem
            spike_pot.append(spike)
        return torch.stack(spike_pot, dim=1)


class LIFSpike(nn.Module):
    def __init__(self, channel=1, thresh=1., tau=0.5, gama=1.0, dspike=False):
        super(LIFSpike, self).__init__()
        if not dspike:
            self.act = ZIF.apply
        else:
            self.act = DSPIKE(region=gama)
        self.thresh = thresh
        self.tau = tau
        self.gama = gama

    def forward(self, x):
        mem = 0
        spike_pot = []
        T = x.shape[1]
        for t in range(T):
            mem = mem * self.tau + x[:, t, ...]
            spike = self.act(mem - self.thresh, self.gama)
            mem = (1 - spike) * mem
            spike_pot.append(spike)
        return torch.stack(spike_pot, dim=1)


class DSPIKE(nn.Module):
    def __init__(self, region=1.0):
        super(DSPIKE, self).__init__()
        self.region = region

    def forward(self, x, temp):
        out_bp = torch.clamp(x, -self.region, self.region)
        out_bp = (torch.tanh(temp * out_bp)) / \
                 (2 * np.tanh(self.region * temp)) + 0.5
        out_s = (x >= 0).float()
        return (out_s.float() - out_bp).detach() + out_bp


def add_dimention(x, T):
    # x.unsqueeze_(1)
    # x = x.repeat(1, T, 1, 1, 1)
    x = torch.stack([x]*T, dim=1)
    return x


# ----- For ResNet19 code -----


class tdLayer(nn.Module):
    def __init__(self, layer, bn=None):
        super(tdLayer, self).__init__()
        self.layer = SeqToANNContainer(layer)
        self.bn = bn

    def forward(self, x):
        x_ = self.layer(x)
        if self.bn is not None:
            x_ = self.bn(x_)
        return x_


class tdBatchNorm(nn.Module):
    def __init__(self, out_panel):
        super(tdBatchNorm, self).__init__()
        self.bn = nn.BatchNorm2d(out_panel)
        self.seqbn = SeqToANNContainer(self.bn)

    def forward(self, x):
        y = self.seqbn(x)
        return y