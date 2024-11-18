from .ResNet import *
from .VGG import *


def modelpool(MODELNAME, DATANAME, **kwargs):
    if type(DATANAME) == int:
        num_classes = DATANAME
    elif 'imagenet' in DATANAME.lower():
        num_classes = 1000
    elif '100' in DATANAME.lower():
        num_classes = 100
    else:
        num_classes = 10
    if MODELNAME.lower() == 'vgg16':
        return vgg16(num_classes=num_classes)
    elif MODELNAME.lower() == 'resnet18':
        return resnet18(num_classes=num_classes, **kwargs)
    elif MODELNAME.lower() == 'resnet20':
        return resnet20(num_classes=num_classes, **kwargs)
    elif MODELNAME.lower() == 'resnet34':
        return resnet34(num_classes=num_classes, **kwargs)
    elif MODELNAME.lower() == 'resnet8':
        return resnet34(num_classes=num_classes)
    else:
        print("still not support this model")
        exit(0)