from collections import namedtuple

import numpy as np


class RectangleRepTBLR(namedtuple('RectangleRep', ['top', 'bottom', 'left', 'right'])):
    pass


class RectangleRepXY(namedtuple('RectangleRep', ['x', 'y'])):
    def __mul__(self, other):
        return RectangleRepXY(self.x * other, self.y * other)


def ImageTranslateCrop(array, shift, bleeding_already=False):
    if bleeding_already:
        shape: tuple = (int(array.shape[0] / 2), int(array.shape[1] / 2))
    else:
        shape: tuple = array.shape

    paddingT = int(np.floor_divide(shape[0], 2))
    origEdgeB = int(paddingT + shape[0])
    targEdgeT = int(paddingT - shift[0])
    targEdgeB = int(targEdgeT + shape[0])

    paddingL = int(np.floor_divide(shape[1], 2))
    origEdgeR = int(paddingL + shape[1])
    targEdgeL = int(paddingL - shift[1])
    targEdgeR = int(targEdgeL + shape[1])

    if bleeding_already:
        newarray = array
    else:
        ywidth = paddingT * 2 + shape[1]
        xwidth = paddingL * 2 + shape[0]
        newarray = np.zeros((ywidth, xwidth))
        newarray[paddingT: origEdgeB, paddingL: origEdgeR] = array

    targArray = newarray[targEdgeT: targEdgeB, targEdgeL: targEdgeR]
    return targArray.copy()
