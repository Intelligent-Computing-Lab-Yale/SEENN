import torch
import torch.nn as nn
from .layers import SeqToANNContainer, LIFSpike, tdLayer, tdBatchNorm


class RecurrentLIF(nn.Module):

    def __init__(self, lifspike: LIFSpike):
        super(RecurrentLIF, self).__init__()

        self.thresh = lifspike.thresh
        self.tau = lifspike.tau
        self.gama = lifspike.gama
        self.mem = 0

    def forward(self, x):
        self.mem = self.mem * self.tau + x
        spike = ((self.mem - self.thresh) > 0).float()
        self.mem = self.mem * (1 - spike)
        return spike

    def re_init(self):
        self.mem = 0


class RecurrentSpikeModel(nn.Module):

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.T = model.T
        self.model.add_dim = lambda x: x
        self.spike_module_refactor(self.model)

    def spike_module_refactor(self, module: nn.Module):
        """
        Recursively replace the normal conv2d and Linear layer to SpikeLayer
        """
        for name, child_module in module.named_children():

            if isinstance(child_module, SeqToANNContainer):
                setattr(module, name, child_module.module)

            elif isinstance(child_module, LIFSpike):
                setattr(module, name, RecurrentLIF(child_module))

            else:
                self.spike_module_refactor(child_module)

    def forward(self, x):
        return self.model(x)

    def re_init(self):
        for m in self.modules():
            if isinstance(m, RecurrentLIF):
                m.re_init()

