import argparse
import shutil
import os
import time
import torch
import logging as logger
import torch.nn as nn
from torch import autocast
from torch.cuda.amp import GradScaler
from models.spike_vgg import vgg16
from models.spike_vgg_dvs import VGGSNNwoAP
from models.resnet import resnet19
import data_loaders
from utils import TET_loss, seed_all, mix_loss


parser = argparse.ArgumentParser(description='PyTorch Temporal Efficient Training')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 10)')
parser.add_argument('--epochs', default=300, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--dset', default='c10', type=str, metavar='N', choices=['c10', 'c100', 'ti', 'c10dvs'],
                    help='dataset')
parser.add_argument('--model', default='res19', type=str, metavar='N', choices=['vgg16', 'res19', 'vgg', 'res20'],
                    help='neural network architecture')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch_size', default=1024, type=int, metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning_rate', default=0.05, type=float, metavar='LR', help='initial learning rate',
                    dest='lr')
parser.add_argument('--seed', default=1000, type=int,
                    help='seed for initializing training. ')
parser.add_argument('-T', '--time', default=3, type=int, metavar='N',
                    help='snn simulation time (default: 2)')
parser.add_argument('--means', default=1.0, type=float, metavar='N',
                    help='make all the potential increment around the means (default: 1.0)')
parser.add_argument('--TET', action='store_false',
                    help='if use Temporal Efficient Training, default: on.')
parser.add_argument('--lamb', default=1e-3, type=float, metavar='N',
                    help='adjust the norm factor to avoid outlier (default: 0.0)')
parser.add_argument('--amp', action='store_false',
                    help='if use amp training, default: on.')
args = parser.parse_args()


def train(model, device, train_loader, criterion, optimizer, epoch, scaler, args):
    running_loss = 0
    model.train()
    M = len(train_loader)
    total = 0
    correct = 0
    s_time = time.time()
    for i, (images, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        labels = labels.to(device)
        images = images.to(device).float()

        if args.amp:
            with autocast(device_type='cuda', dtype=torch.float16):
                outputs = model(images)
                mean_out = outputs.mean(1)
                if args.TET:
                    loss = TET_loss(outputs, labels, criterion, args.means, args.lamb)
                else:
                    loss = criterion(mean_out, labels)
                scaler.scale(loss.mean()).backward()
                scaler.step(optimizer)
                scaler.update()
        else:
            outputs = model(images)
            mean_out = outputs.mean(1)
            if args.TET:
                loss = TET_loss(outputs, labels, criterion, args.means, args.lamb)
            else:
                loss = criterion(mean_out, labels)
            loss.mean().backward()
            optimizer.step()

        running_loss += loss.item()
        total += float(labels.size(0))
        _, predicted = mean_out.cpu().max(1)
        correct += float(predicted.eq(labels.cpu()).sum().item())
    e_time = time.time()
    return running_loss / i, 100 * correct / total, (e_time-s_time)/60


@torch.no_grad()
def test(model, test_loader, device):
    correct = 0
    total = 0
    model.eval()
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        inputs = inputs.to(device).float()
        outputs = model(inputs)
        mean_out = outputs.mean(1)
        _, predicted = mean_out.cpu().max(1)
        total += float(targets.size(0))
        correct += float(predicted.eq(targets).sum().item())

    correct = torch.tensor([correct]).cuda()
    total = torch.tensor([total]).cuda()
    final_acc = 100 * correct / total
    return final_acc.item()


if __name__ == '__main__':

    seed_all(args.seed)

    if args.dset == 'c10':
        train_dataset, val_dataset = data_loaders.build_cifar(use_cifar10=True, cutout=False, auto_aug=False)
        num_cls = 10
        wd = 5e-4
        in_c = 3
    elif args.dset == 'c100':
        train_dataset, val_dataset = data_loaders.build_cifar(use_cifar10=False, cutout=True, auto_aug=True)
        num_cls = 100
        wd = 5e-4
        in_c = 3
    elif args.dset == 'c10dvs':
        train_dataset, val_dataset = data_loaders.build_dvscifar('data/cifar-dvs', transform=1)
        num_cls = 10
        wd = 1e-4
        in_c = 2
    else:
        raise NotImplementedError

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                              num_workers=args.workers, pin_memory=True)

    if args.model == 'vgg16':
        model = vgg16(width_mult=4, in_c=in_c, num_classes=num_cls, dspike=True, gama=3)
    elif args.model == 'vgg':
        model = VGGSNNwoAP(in_c=in_c, num_classes=num_cls)
    elif args.model == 'res19':
        model = resnet19(width_mult=8, in_c=in_c, num_classes=num_cls, use_dspike=True, gamma=3)
    else:
        raise NotImplementedError

    model.T = args.time
    model.cuda()
    device = next(model.parameters()).device

    scaler = GradScaler() if args.amp else None

    model_save_name = 'raw/{}_{}'.format(args.dset, args.model) if args.TET else 'raw/{}_{}_wotet'.format(args.dset, args.model)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr/256 * args.batch_size, weight_decay=wd, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=0, T_max=args.epochs)
    best_acc = 0
    best_epoch = 0
    print('start training!')
    for epoch in range(args.epochs):

        loss, acc, t_diff = train(model, device, train_loader, criterion, optimizer, epoch, scaler, args)
        print('Epoch:[{}/{}]\t loss={:.5f}\t acc={:.3f},\t time elapsed: {}'.format(epoch, args.epochs, loss, acc,
                                                                                    t_diff))
        scheduler.step()
        facc = test(model, test_loader, device)
        print('Epoch:[{}/{}]\t Test acc={:.3f}'.format(epoch, args.epochs, facc))

        if best_acc < facc:
            best_acc = facc
            best_epoch = epoch + 1
            torch.save(model.state_dict(), model_save_name)
        print('Best Test acc={:.3f}'.format(best_acc))
