import numpy as np
import torch
from torch import nn
import math
from tqdm import tqdm
from utils import *
from modules import LabelSmoothing
import torch.distributed as dist
import random
import os


def seed_all(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def eval_ann(test_dataloader, model, loss_fn, device, rank=0):
    epoch_loss = 0
    tot = torch.tensor(0.).cuda(device)
    model.eval()
    model.cuda(device)
    length = 0
    with torch.no_grad():
        for img, label in test_dataloader:
            img = img.cuda(device)
            label = label.cuda(device)
            out = model(img)
            loss = loss_fn(out, label)
            epoch_loss += loss.item()
            length += len(label)    
            tot += (label==out.max(1)[1]).sum().data
    return tot/length, epoch_loss/length


def train_ann(train_dataloader, test_dataloader, model, epochs, device, loss_fn, lr=0.1, wd=5e-4, save=None, parallel=False, rank=0):
    model.cuda(device)
    para1, para2, para3 = regular_set(model)
    optimizer = torch.optim.SGD([
                                {'params': para1, 'weight_decay': wd}, 
                                {'params': para2, 'weight_decay': wd}, 
                                {'params': para3, 'weight_decay': wd}
                                ],
                                lr=lr, 
                                momentum=0.9)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    best_acc = 0
    for epoch in range(epochs):
        epoch_loss = 0
        length = 0
        model.train()
        for img, label in train_dataloader:
            img = img.cuda(device)
            label = label.cuda(device)
            optimizer.zero_grad()
            out = model(img)
            loss = loss_fn(out, label)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            length += len(label)
        tmp_acc, val_loss = eval_ann(test_dataloader, model, loss_fn, device, rank)
        if parallel:
            dist.all_reduce(tmp_acc)
        print('Epoch {} -> Val_loss: {}, Acc: {}'.format(epoch, val_loss, tmp_acc), flush=True)
        if rank == 0 and save != None and tmp_acc >= best_acc:
            torch.save(model.state_dict(), './saved_models/' + save + '.pth')
        best_acc = max(tmp_acc, best_acc)
        print('best_acc: ', best_acc)
        scheduler.step()
    return best_acc, model


def eval_snn(test_dataloader, model, device, sim_len=8, rank=0, fp16=False):
    tot = torch.zeros(sim_len).cuda(device)
    length = 0
    model = model.cuda(device)
    model.eval()
    # valuate
    with torch.no_grad():
        for idx, (img, label) in enumerate(tqdm(test_dataloader)):
            spikes = 0
            length += len(label)
            img = img.cuda()
            label = label.cuda()
            for t in range(sim_len):
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    out = model(img)
                spikes += out
                tot[t] += (label == spikes.max(1)[1]).sum()
            reset_net(model)
    return tot/length


@torch.no_grad()
def ee_test(model, test_loader, device, T, threshold=0.2, metric='confidence', save_image=False):
    correct = 0
    total = 0
    time = 0
    time_vec = np.array([0 for t in range(T)])
    model.eval()

    # for saving images
    num_saved_hard_images =0
    num_saved_easy_images = 0

    # Creating lists to store examples of easy and hard images and their corresponding labels
    hard_images = []
    hard_images_target = []
    easy_images = []
    easy_images_target = []

    reset_net(model)
    for batch_idx, (inputs, targets) in enumerate(tqdm(test_loader)):
        inputs = inputs.to(device)

        metric_list = []
        batch_dim = targets.size(0)
        total += float(batch_dim)
        outputs = 0
        stacked_outputs = []

        for s_i in range(T):
            t = s_i + 1

            with torch.autocast(device_type='cuda', dtype=torch.float16):
                cur_out = model(inputs)

                outputs = outputs + cur_out
                stacked_outputs += [cur_out]
                if metric == 'entropy':
                    probs = torch.log_softmax(outputs, dim=1)
                    entropy = - torch.sum(probs * torch.exp(probs), dim=1) / math.log(outputs.shape[1])
                    metric_list += [1 - entropy]
                elif metric == 'confidence':
                    probs = torch.softmax(outputs / 8, dim=1)
                    confidence, _ = torch.sort(probs, dim=1, descending=True)
                    # confidence = torch.sum(confidence[:, :3], dim=1)
                    confidence = confidence[:, 0]
                    metric_list += [confidence]
                else:
                    raise NotImplementedError

        reset_net(model)
        stacked_outputs = torch.stack(stacked_outputs)
        stacked_outputs = torch.permute(stacked_outputs, (1, 0, 2))  # change dim to Batch, Time, Hidden
        # compute accuracy sample-by-sample
        final_logits = []
        for b_i in range(batch_dim):
            for s_i in range(T):
                if metric_list[s_i][b_i] > threshold or s_i == T - 1:
                    t = s_i + 1             # current time step
                    time_vec[s_i] = time_vec[s_i] + 1
                    final_logits += [stacked_outputs[b_i, :t].sum(0) if s_i > 0 else stacked_outputs[b_i, 0]]
                    if save_image:
                        if t == 4 and num_saved_hard_images < 25:
                            num_saved_hard_images += 1
                            hard_images.append(inputs[b_i].squeeze())
                            hard_images_target.append(int(targets[b_i]))
                        elif t == 1 and num_saved_easy_images < 25:
                            num_saved_easy_images += 1
                            easy_images.append(inputs[b_i].squeeze())
                            easy_images_target.append(int(targets[b_i]))
                    break
        final_logits = torch.stack(final_logits, dim=0)
        _, predicted = final_logits.cpu().max(1)
        correct += float(predicted.eq(targets).sum().item())

        if num_saved_easy_images >= 25 and num_saved_hard_images >= 25:
            break

    if save_image:
        print("Hard Images Labels: ", hard_images_target)
        print("Easy Images Labels: ", easy_images_target)
        # torchvision.utils.save_image(hard_images, "raw/hard_images.png", normalize=True, nrow=5)
        # torchvision.utils.save_image(easy_images, "raw/easy_images.png", normalize=True, nrow=5)

    final_acc = (100 * correct / total)
    time = np.dot(time_vec, np.array(range(1, T + 1)))
    avg_time = (time / total)
    time_ratio = time_vec / total
    print('Threshold: {}, time: {}, acc: {}, portion: {}'.format(threshold, avg_time, final_acc, time_ratio))
    return final_acc, avg_time, time_ratio


@torch.no_grad()
def test_aet(model, test_loader, device, T):
    correct = [0 for i in range(T)]
    total = 0
    emp_aet = 0
    model.eval()
    for batch_idx, (inputs, targets) in enumerate(tqdm(test_loader)):
        inputs = inputs.to(device)
        outputs = model(inputs)
        total += float(targets.size(0))
        sample_wise_correct_sign = [False for i in range(targets.size(0))]
        for t in range(T):
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
    for t in range(T):
        if t == 0:
            aet += correct[t]
        else:
            aet += (t + 1) * (correct[t] - correct[t-1])
    aet += (total - correct[-1]) * T

    aet = aet / total
    emp_aet = emp_aet / total
    print("AET: {}, Empirical AET: {}".format(aet, emp_aet))


def ee_ii_train(train_dataloader, test_dataloader, model, policy_model, epochs, device, T, beta,
                lr=0.01, wd=5e-4):
    model.cuda(device)
    model.eval()
    policy_model.cuda(device)
    optimizer = torch.optim.SGD(policy_model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * len(train_dataloader))
    ce = nn.CrossEntropyLoss()

    print('compute sample rate')
    correct = 0
    for img, label in train_dataloader:
        img = img.cuda(device)
        label = label.cuda(device)
        for k in range(T):
            with torch.no_grad():
                output = model(img)
            correct += (label == output.max(1)[1]).float().sum()
            reset_net(model)
            break
    rate = (correct / len(train_dataloader.dataset)).item()
    print('1-timestep training accuracy: {}'.format(rate))

    for epoch in range(epochs):
        epoch_loss = 0
        total = 0
        policy_model.train()

        for img, label in train_dataloader:
            img = img.cuda(device)
            label = label.cuda(device)
            optimizer.zero_grad()

            # compute v: b x T
            v = torch.softmax(policy_model(img), dim=1)
            # compute reward
            outputs = 0
            rewards = 0
            correct_0 = 0

            for k in range(T):
                with torch.no_grad():
                    outputs += model(img)
                correct = (label == outputs.max(1)[1]).float()
                if k == 0:
                    correct_0 = correct + rate / (1 - rate) * (1 - correct)
                rewards = rewards + (beta * (1 - correct) - (2 ** (-k) * correct)) * v[:, k] * correct_0

            rewards = rewards.mean()
            rewards.backward()
            optimizer.step()
            scheduler.step()
            epoch_loss += rewards.item() * len(label)
            total += len(label)
            reset_net(model)

        print('epoch: {}, rewards: {}'.format(epoch+1, epoch_loss/total))
        # evaluate
        policy_model.eval()
        epoch_loss = 0
        total = 0
        times = 0
        correct = 0
        for img, label in test_dataloader:
            img = img.cuda(device)
            label = label.cuda(device)
            outputs = 0

            with torch.no_grad():
                v = torch.max(policy_model(img), dim=1)[1]
                for k in range(T):
                    outputs += model(img)
                    correct += ((label == outputs.max(1)[1]).float() * (v == k).float()).sum()

                reset_net(model)
                times += (v+1).sum()
                total += len(label)

        acc = correct.item() / total
        avg_time = times.item() / total
        print('test results: Acc: {}, Time: {}'.format(acc, avg_time))
