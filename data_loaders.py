import torch
import random
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import CIFAR10, CIFAR100, ImageFolder
import warnings
import os
from autoaugment import CIFAR10Policy, Cutout


warnings.filterwarnings('ignore')


def build_cifar(cutout=False, use_cifar10=True, download=True, normalize=True, auto_aug=False):
    aug = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]
    if auto_aug:
        aug.append(CIFAR10Policy())

    aug.append(transforms.ToTensor())

    if cutout:
        aug.append(Cutout(n_holes=1, length=16))

    if use_cifar10:
        if normalize:
            aug.append(
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), )

        transform_train = transforms.Compose(aug)
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        train_dataset = CIFAR10(root='data',
                                train=True, download=download, transform=transform_train)
        val_dataset = CIFAR10(root='data',
                              train=False, download=download, transform=transform_test)

    else:
        if normalize:

            aug.append(
                transforms.Normalize(
                    (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
            )
        transform_train = transforms.Compose(aug)
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        train_dataset = CIFAR100(root='data',
                                 train=True, download=download, transform=transform_train)
        val_dataset = CIFAR100(root='data',
                               train=False, download=download, transform=transform_test)

    return train_dataset, val_dataset


class DVSCifar10(Dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train
        self.resize = transforms.Resize(size=(48, 48))
        self.rotate = transforms.RandomRotation(degrees=30)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        data, target = torch.load(self.root + '/{}.pt'.format(index))
        data = self.resize(data.permute([3, 0, 1, 2]))

        if self.transform:

            flip = random.random() > 0.5
            if flip:
                data = torch.flip(data, dims=(2,))

            choices = ['roll', 'rotate']
            aug = random.choice(choices)
            if aug == 'roll':
                off1 = random.randint(-5, 5)
                off2 = random.randint(-5, 5)
                data = torch.roll(data, shifts=(off1, off2), dims=(2, 3))
            elif aug == 'rotate':
                data = self.rotate(data)

        return data, target.long().squeeze(-1)

    def __len__(self):
        return len(os.listdir(self.root))


def build_dvscifar(path='/mnt/lustre/liyuhang1/data/cifar-dvs', transform=None):
    train_path = path + '/train'
    val_path = path + '/test'
    train_dataset = DVSCifar10(root=train_path, transform=transform)
    val_dataset = DVSCifar10(root=val_path, transform=False)

    return train_dataset, val_dataset


def build_imagenet():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    root = '/data_smr/dataset/ImageNet'
    train_root = os.path.join(root,'train')
    val_root = os.path.join(root,'val')
    train_dataset = ImageFolder(
        train_root,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    )
    val_dataset = ImageFolder(
        val_root,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])
    )
    return train_dataset, val_dataset



if __name__ == '__main__':
    pass
