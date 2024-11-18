from .getdataloader import *

def datapool(DATANAME, batchsize, **kwargs):
    if DATANAME.lower() == 'cifar10':
        return GetCifar10(batchsize, **kwargs)
    elif DATANAME.lower() == 'cifar100':
        return GetCifar100(batchsize)
    elif DATANAME.lower() == 'imagenet':
        return GetImageNet(batchsize)
    else:
        print("still not support this model")
        exit(0)