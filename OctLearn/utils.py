import torch
import numpy as np
from collections import namedtuple


class RectangleRepTBLR(namedtuple('RectangleRep', ['top', 'bottom', 'left', 'right'])):
    pass


class RectangleRepXY(namedtuple('RectangleRep', ['x', 'y'])):
    def __mul__(self, other):
        return RectangleRepXY(self.x * other, self.y * other)


def ImageTranslateCrop(array, shift, target_size=None):
    def aInt(x): return np.array(x, dtype=int)

    # image from parameter: origin image
    # image extend for cropping: padded image
    # image to return: cropped image
    origin = "ORIGIN"
    cropped = "CROPPED"
    padded = "PADDED"

    size = dict()
    size[origin] = aInt(array.shape)
    size[cropped] = aInt(target_size or size[origin])
    size[padded] = size[origin] + size[cropped]

    coord_shift = dict()  # coord_shift[(from, to)]
    coord_shift[(origin, cropped)] = aInt(shift)
    coord_shift[(origin, padded)] = size[cropped] // 2
    coord_shift[(padded, cropped)] = -coord_shift[(origin, padded)] + coord_shift[(origin, cropped)]

    center_point = dict()  # center_point[(name, coord)]
    center_point[(origin, origin)] = size[origin] // 2
    center_point[(padded, origin)] = center_point[(origin, origin)]
    center_point[(cropped, origin)] = center_point[(origin, origin)] + coord_shift[(origin, cropped)]
    center_point[(padded, padded)] = size[padded] // 2
    center_point[(cropped, cropped)] = size[cropped] // 2
    center_point[(cropped, padded)] = center_point[(cropped, origin)] + coord_shift[(origin, padded)]

    min_point = dict()  # min_point[(name, coord)]
    min_point[(origin, padded)] = coord_shift[(origin, padded)]
    min_point[(cropped, padded)] = center_point[(cropped, padded)] - center_point[(cropped, cropped)]

    max_point = dict()  # max_point[(name, coord)]
    max_point[(origin, padded)] = min_point[(origin, padded)] + size[origin]
    max_point[(cropped, padded)] = min_point[(cropped, padded)] + size[cropped]

    def sRect(target): return tuple(slice(min_point[target][x], max_point[target][x]) for x in (0, 1))

    img_padded = np.zeros(size[padded])
    img_padded[sRect((origin, padded))] = array
    return img_padded[sRect((cropped, padded))]


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