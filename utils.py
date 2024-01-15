import torch
import torch.nn as nn
import random
import os
import numpy as np
import logging
from models.snn_recurrent import RecurrentSpikeModel


def seed_all(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


def TET_loss(outputs, labels, criterion, means, lamb):
    T = outputs.size(1)
    Loss_es = 0
    for t in range(T):
        Loss_es += criterion(outputs[:, t, ...], labels)
    Loss_es = Loss_es / T  # L_TET
    if lamb != 0:
        MMDLoss = torch.nn.MSELoss()
        y = torch.zeros_like(outputs).fill_(means)
        Loss_mmd = MMDLoss(outputs, y)  # L_mse
    else:
        Loss_mmd = 0
    return (1 - lamb) * Loss_es + lamb * Loss_mmd  # L_Total


def mix_loss(outputs, labels, criterion, means, lamb):
    T = outputs.size(1)
    Loss_es = 0
    for t in range(T):
        if t == 0:
            Loss_es = criterion(outputs[:, 0, ...], labels)
        else:
            Loss_es += criterion(outputs[:, :t, ...].sum(1), labels)
    Loss_es = Loss_es / T
    if lamb != 0:
        MMDLoss = torch.nn.MSELoss()
        y = torch.zeros_like(outputs).fill_(means)
        Loss_mmd = MMDLoss(outputs, y)  # L_mse
    else:
        Loss_mmd = 0
    return (1 - lamb) * Loss_es + lamb * Loss_mmd


class AvgMeter:

    def __init__(self):
        self.num = 0
        self.value = 0

    def update(self, x, n=1):
        self.num += n
        self.value += x

    def avg(self):
        return self.value / self.num



def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def rand_bbox(size, lam):
    W = size[3]
    H = size[4]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def cutmix_data(input, target, alpha=1.0):
    lam = np.random.beta(alpha, alpha)
    rand_index = torch.randperm(input.size()[0]).cuda()

    target_a = target
    target_b = target[rand_index]

    # generate mixed sample
    bbx1, bby1, bbx2, bby2 = rand_bbox(input.size(), lam)
    input[:, :, :, bbx1:bbx2, bby1:bby2] = input[rand_index, :, :, bbx1:bbx2, bby1:bby2]
    # adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (input.size()[-1] * input.size()[-2]))
    return input, target_a, target_b, lam



class Energy:

    def __init__(self, model: RecurrentSpikeModel, test_loader, device, model2=None):

        self.model = model
        self.model2 = model2
        self.device = device
        self.loader = test_loader
        self.energy = 0
        self.sparsity = AvgMeter()

    def hook_function(self):

        def get_artificial_energy(layer, inputs, outputs):
            FLOPs = outputs.numel() * np.prod(layer.weight.shape[1:])
            energy = 0.9 * FLOPs + 4.6 * FLOPs
            self.energy += energy / (10 ** 9)

        def get_spike_energy(layer, inputs, outputs):
            FLOPs = outputs.numel() * np.prod(layer.weight.shape[1:])
            density = torch.sum(torch.nonzero(inputs[0])).item() / inputs[0].numel()
            energy = 0.9 * FLOPs * density
            self.energy += energy / (10 ** 9)
            self.sparsity.update(density)

        first_conv = False
        for name, module in self.model.named_modules():

            if isinstance(module, nn.Conv2d):
                if not first_conv:
                    first_conv = True
                    module.register_forward_hook(get_artificial_energy)
                else:
                    module.register_forward_hook(get_spike_energy)
                print("{} registered!".format(name))

        if self.model2 is not None:
            for name, module in self.model2.named_modules():
                if isinstance(module, nn.Conv2d):
                    module.register_forward_hook(get_artificial_energy)
                    print("{} registered!".format(name))


class FLOPs:

    def __init__(self, model):

        self.model = model
        self.flops = 0

    def test(self, x):

        def get_flops(layer, inputs, outputs):
            flops = outputs.numel() * np.prod(layer.weight.shape[1:])
            self.flops += flops

        for name, module in self.model.named_modules():

            if isinstance(module, nn.Conv2d):
                module.register_forward_hook(get_flops)
                print("{} registered!".format(name))

        _ = self.model(x)
        print(self.flops / (10 ** 6))

