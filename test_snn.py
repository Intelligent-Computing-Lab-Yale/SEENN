import argparse
import shutil
import os
import time as timep
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torchvision
import math
from models.resnet import resnet19
from models.spike_vgg import vgg16
from models.spike_vgg_dvs import VGGSNNwoAP
import data_loaders
from utils import TET_loss, seed_all, Energy
from models.snn_recurrent import RecurrentSpikeModel


parser = argparse.ArgumentParser(description='PyTorch Temporal Efficient Training')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 10)')
parser.add_argument('--dset', default='c10', type=str, metavar='N', choices=['c10', 'c100', 'ti', 'c10dvs'],
                    help='dataset')
parser.add_argument('--model', default='res19', type=str, metavar='N', choices=['vgg16', 'res19'],
                    help='neural network architecture')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch_size', default=256, type=int, metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--seed', default=1000, type=int,
                    help='seed for initializing training. ')
parser.add_argument('-T', '--time', default=4, type=int, metavar='N',
                    help='snn simulation time (default: 4)')
parser.add_argument('--test-all', action='store_true',
                    help='if test all time steps.')
parser.add_argument('--t', default=0.9, type=float, metavar='N',
                    help='threshold for entropy or confidence.')
parser.add_argument('--TET', action='store_false',
                    help='if use Temporal Efficient Training.')
parser.add_argument('--gpu-test', action='store_true',
                    help='if use GPU runtime test.')
parser.add_argument('--aet-test', action='store_true',
                    help='if test the empirical and theorectical AET of an SNN.')
args = parser.parse_args()


@torch.no_grad()
def throughput_test(model: RecurrentSpikeModel, test_loader, device, dynamic=True,
                    threshold=0.2, metric='entropy', timesteps=4, measure_energy=False):
    correct = 0
    total = 0
    model.eval()
    overall_time = 0

    for batch_idx, (inputs, targets) in enumerate(test_loader):
        inputs = inputs.to(device)
        s_time = timep.time()
        model.re_init()
        if not dynamic:
            logits = 0
            for ts in range(timesteps):
                logits += model(inputs)
            _, predicted = logits.cpu().max(1)
            total += 1
            correct += float(predicted.eq(targets).sum().item())
        else:
            partial_logits = 0
            # evaluate sample-by-sample
            for s_i in range(args.time):
                t = s_i + 1
                partial_logits += model(inputs)
                if metric == 'entropy':
                    probs = torch.log_softmax(partial_logits, dim=1)
                    entropy = - torch.sum(probs * torch.exp(probs), dim=1) / math.log(partial_logits.shape[1])
                    # Reverse the entropy for consistent thresholding implementation
                    entropy = 1 - entropy
                elif metric == 'confidence':
                    probs = torch.softmax(partial_logits, dim=1)
                    confidence, _ = torch.sort(probs, dim=1, descending=True)
                    confidence = confidence[:, 0]
                    entropy = confidence
                else:
                    raise NotImplementedError

                if entropy > threshold or t == args.time:
                    _, predicted = partial_logits.cpu().max(1)
                    total += 1
                    correct += float(predicted.eq(targets).sum().item())
                    break
        e_time = timep.time()
        overall_time += (e_time - s_time)
    final_acc = (100 * correct / total)

    print("Throughput: {}, accuracy: {}".format(total/overall_time, final_acc))


@torch.no_grad()
def test(model, test_loader, device, dynamic=True, threshold=0.2, metric='entropy', save_image=False):
    correct = 0
    total = 0
    time = 0
    time_vec = np.array([0 for t in range(args.time)])
    model.eval()

    num_saved_hard_images = 0
    num_saved_easy_images = 0

    # Creating lists to store examples of easy and hard images and their corresponding labels
    score_list = []

    for batch_idx, (inputs, targets) in enumerate(test_loader):
        inputs = inputs.to(device).float()
        outputs = model(inputs)

        if not dynamic:
            mean_out = outputs.mean(1)
            _, predicted = mean_out.cpu().max(1)
            total += float(targets.size(0))
            correct += float(predicted.eq(targets).sum().item())
        else:
            metric_list = []
            batch_dim = targets.size(0)
            total += float(batch_dim)
            # calculate entropy
            for s_i in range(args.time):
                t = s_i + 1
                partial_logits = outputs[:, :t].sum(1) if s_i > 0 else outputs[:, 0]
                if metric == 'entropy':
                    probs = torch.log_softmax(partial_logits, dim=1)
                    entropy = - torch.sum(probs * torch.exp(probs), dim=1) / math.log(partial_logits.shape[1])
                    metric_list += [1 - entropy]
                elif metric == 'confidence':
                    probs = torch.softmax(partial_logits, dim=1)
                    confidence, _ = torch.sort(probs, dim=1, descending=True)
                    confidence = confidence[:, 0]
                    metric_list += [confidence]
                else:
                    raise NotImplementedError
            score_list += [torch.stack(metric_list, dim=0)]
            # compute accuracy sample-by-sample
            final_logits = []
            for b_i in range(batch_dim):
                for s_i in range(args.time):
                    if metric_list[s_i][b_i] > threshold or s_i == args.time - 1:
                        t = s_i + 1             # current time step
                        time_vec[s_i] = time_vec[s_i] + 1
                        final_logits += [outputs[b_i, :t].sum(0) if s_i > 0 else outputs[b_i, 0]]
                        break
            final_logits = torch.stack(final_logits, dim=0)
            _, predicted = final_logits.cpu().max(1)
            correct += float(predicted.eq(targets).sum().item())

            if num_saved_easy_images >= 25 and num_saved_hard_images >= 25:
                break

    if save_image:
        score_list = torch.cat(score_list, dim=1)
        print(score_list.shape)
        np.save("raw/conf_dist.npy", score_list.cpu().numpy())

    final_acc = (100 * correct / total)
    if dynamic:
        time = np.dot(time_vec, np.array(range(1, args.time + 1)))
        avg_time = (time / total)
        time_ratio = time_vec / total
        return final_acc, avg_time, time_ratio
    else:
        return final_acc


@torch.no_grad()
def test_all_time_steps(model, test_loader, device):
    correct = [0 for i in range(args.time)]
    total = 0
    model.eval()
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        inputs = inputs.float().to(device)
        outputs = model(inputs)
        total += float(targets.size(0))
        for t in range(args.time):
            mean_out = outputs[:, :(t+1)].mean(1) if t > 0 else outputs[:, 0]
            _, predicted = mean_out.cpu().max(1)
            correct[t] = correct[t] + float(predicted.eq(targets).sum().item())

    correct = torch.tensor(correct).cuda()
    total = torch.tensor([total]).cuda()
    final_accs = (100 * correct / total)
    print('Accuracy on each timestep: {}'.format(final_accs))


@torch.no_grad()
def test_aet(model, test_loader, device):
    correct = [0 for i in range(args.time)]
    total = 0
    emp_aet = 0
    model.eval()
    for batch_idx, (inputs, targets) in enumerate(tqdm(test_loader)):
        inputs = inputs.to(device)
        outputs = model(inputs)
        total += float(targets.size(0))
        sample_wise_correct_sign = [False for i in range(targets.size(0))]
        for t in range(args.time):
            mean_out = outputs[:, :(t+1)].mean(1) if t > 0 else outputs[:, 0]
            _, predicted = mean_out.cpu().max(1)
            c = predicted.eq(targets).float()
            for i in range(targets.size(0)):
                if c[i] == 1:
                    correct[t] += 1
                    if sample_wise_correct_sign[i] is False:
                        emp_aet += (t + 1)
                        sample_wise_correct_sign[i] = True
                else:
                    if sample_wise_correct_sign[i] is False and t == args.time - 1:
                        emp_aet += (t + 1)
    # compute aet
    aet = 0
    for t in range(args.time):
        if t == 0:
            aet += correct[t]
        else:
            aet += (t + 1) * (correct[t] - correct[t-1])
    aet += (total - correct[-1]) * args.time

    aet = aet / total
    emp_aet = emp_aet / total
    print("AET: {}, Empirical AET: {}".format(aet, emp_aet))


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

    # setting the batch size to 1 for GPU throughput test
    batch_size = 1 if args.gpu_test else args.batch_size
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                               num_workers=args.workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True,
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
    model_save_name = 'raw/{}_{}'.format(args.dset, args.model) if args.TET else 'raw/{}_{}_wotet'.format(args.dset, args.model)

    model.load_state_dict(torch.load(model_save_name))

    if args.gpu_test:
        model = RecurrentSpikeModel(model)
        throughput_test(model, test_loader, device, dynamic=False, threshold=args.t, metric='entropy',
                        timesteps=args.time, measure_energy=True)
        exit()

    print('start testing!')
    if args.test_all:
        test_all_time_steps(model, test_loader, device)

    if args.aet_test:
        test_aet(model, test_loader, device)

    facc, time, time_ratio = test(model, test_loader, device, dynamic=True, threshold=args.t, metric='confidence',
                                  save_image=False)
    print('Threshold: {}, time: {}, acc: {}, portion: {}'.format(args.t, time, facc, time_ratio))
