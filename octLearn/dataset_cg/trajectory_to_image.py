import numpy as np


def trajectory_to_image_slow(trajectory, rect, resolution):
    pixXsize = np.ceil(np.abs(rect["xmax"]-rect["xmin"])/resolution).astype(int)
    pixYsize = np.ceil(np.abs(rect["ymax"]-rect["ymin"])/resolution).astype(int)
    image = np.zeros([pixYsize, pixXsize])
    for x, y in trajectory:
        xidx = np.round((x-rect["xmin"])/resolution).astype(int)
        yidx = np.round((y-rect["ymin"])/resolution).astype(int)
        if (pixXsize<xidx or xidx<0 or pixYsize<yidx or yidx<0):
            print(x, y, xidx, yidx)
        image[yidx, xidx] = 1
    return image


def trajectory_to_image(trajectory, rect, resolution):
    pixXsize = np.ceil(np.abs(rect["xmax"]-rect["xmin"])/resolution).astype(int)
    pixYsize = np.ceil(np.abs(rect["ymax"]-rect["ymin"])/resolution).astype(int)
    image = np.zeros([pixYsize, pixXsize])

    ground_zero = [rect["xmin"], rect["ymin"]]
    point_index = np.round((trajectory - ground_zero)/resolution).astype(int)
    image_index = np.matmul(point_index, [1, pixXsize])
    np.put(image, image_index, 1, mode='raise')
    return image