from collections import namedtuple

import numpy as np


def translate_and_crop(array, shift, target_size=None, debug_ax=None):
    # array: canvas
    # shift: canvas translation

    def aInt(x): return np.array(x, dtype=int)

    # image from parameter: origin image
    # image extend for cropping: padded image
    # image to return: cropped image
    origin = "ORIGIN"
    cropped = "CROPPED"
    padded = "PADDED"

    size = dict()
    size[origin] = aInt(array.shape)
    size[cropped] = aInt(target_size if target_size is not None else size[origin])
    size[padded] = size[origin] + size[cropped]

    # view translation is negative of canvas translation
    view_shift = dict()  # coord_shift[(from, to)]
    view_shift[(origin, cropped)] = -aInt(shift)
    view_shift[(origin, padded)] = -size[cropped] // 2
    view_shift[(padded, cropped)] = -view_shift[(origin, padded)] + view_shift[(origin, cropped)]

    center_point = dict()  # center_point[(name, coord)]
    center_point[(origin, origin)] = size[origin] // 2
    center_point[(padded, origin)] = center_point[(origin, origin)]
    center_point[(cropped, origin)] = center_point[(origin, origin)] + view_shift[(origin, cropped)]
    center_point[(padded, padded)] = size[padded] // 2
    center_point[(cropped, cropped)] = size[cropped] // 2
    center_point[(cropped, padded)] = center_point[(cropped, origin)] + view_shift[(origin, padded)]

    min_point = dict()  # min_point[(name, coord)]
    min_point[(origin, padded)] = -view_shift[(origin, padded)]
    min_point[(cropped, padded)] = center_point[(cropped, padded)] + center_point[(cropped, cropped)]

    max_point = dict()  # max_point[(name, coord)]
    max_point[(origin, padded)] = min_point[(origin, padded)] + size[origin]
    max_point[(cropped, padded)] = min_point[(cropped, padded)] + size[cropped]

    def sRect(target): return tuple(slice(min_point[target][x], max_point[target][x]) for x in (1, 0))

    img_padded = np.zeros(size[padded])
    img_padded[sRect((origin, padded))] = array

    def drawRect(pmin, pmax, ax, c):
        x1, y1 = pmin
        x2, y2 = pmax
        ax.plot([x1, x1, x2, x2, x1], [y1, y2, y2, y1, y1], linestyle=':', c=c)

    def debug_draw():
        ax = debug_ax
        ax.scatter(*center_point[(cropped, padded)], c='blue', label='cropped')
        ax.scatter(*center_point[(padded, padded)], c='orange', label='padded')
        drawRect(min_point[(origin, padded)], max_point[(origin, padded)], ax, c='gray')
        drawRect(min_point[(cropped, padded)], max_point[(cropped, padded)], ax, c='blue')
        drawRect([0, 0], size[padded], ax, c='orange')
        ax.imshow(img_padded, vmin=0, vmax=1, origin='lower', alpha=0.5)
        ax.set_title('shift {}'.format(shift))

    if debug_ax is not None:
        debug_draw()

    srect = sRect((cropped, padded))
    return img_padded[sRect((cropped, padded))]


def position_to_array_index(position, array_shape, resolution=1, center_point=np.array((0, 0))):
    origin_point = center_point - np.array(array_shape) / 2 / resolution
    return np.floor((position - origin_point) * resolution).astype(int)


class RectangleRepXY(namedtuple('RectangleRep', ['x', 'y'])):
    def __mul__(self, other):
        return RectangleRepXY(self.x * other, self.y * other)