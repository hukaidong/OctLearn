from collections import namedtuple

import numpy as np
import torch


class RectangleRepTBLR(namedtuple('RectangleRep', ['top', 'bottom', 'left', 'right'])):
    pass


def RandSeeding(seed=None):
    import random
    import torch

    seed = seed or random.randint(0, 100000000)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    return seed


def WeightInitializer(m):
    classname = m.__class__.__name__
    if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
        torch.nn.init.xavier_normal_(m.weight.data)
    if hasattr(m, 'bias') and m.bias is not None:
        torch.nn.init.constant_(m.bias.data, 0.0)
    if classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0)
        torch.nn.init.constant_(m.bias.data, 0.0)

def DataLoaderCollate(batch):
    return tuple(zip(*batch))