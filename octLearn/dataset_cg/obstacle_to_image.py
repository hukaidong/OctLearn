import numpy as np
import logging

logger = logging.getLogger(__name__)


def obstacle_to_image_slow(obstacleType, obstacleInfo, canvas_rect, resolution):
    pixXsize = np.ceil(np.abs(canvas_rect["xmax"] - canvas_rect["xmin"]) / resolution).astype(int)
    pixYsize = np.ceil(np.abs(canvas_rect["ymax"] - canvas_rect["ymin"]) / resolution).astype(int)
    image = np.zeros([pixYsize, pixXsize])

    obstacle_info_cursor = 0
    for obsT in obstacleType:
        if obsT == 0:  # Box: xmin, xmax, ymin, ymax
            draw_box_in_image(box_param=obstacleInfo[obstacle_info_cursor:obstacle_info_cursor + 4],
                              canvas_rect=canvas_rect,
                              resolution=resolution,
                              image_out=image)
            obstacle_info_cursor += 4
        elif obsT == 1:  # Circ: xpos, ypos, rad
            draw_circ_in_image(circ_param=obstacleInfo[obstacle_info_cursor:obstacle_info_cursor + 3],
                               canvas_rect=canvas_rect,
                               resolution=resolution,
                               image_out=image)
            obstacle_info_cursor += 3
        elif obsT == 2:  # Orie: xpos, ypos, xlen, ylen, theta
            draw_orie_in_image(orie_param=obstacleInfo[obstacle_info_cursor:obstacle_info_cursor + 5],
                               canvas_rect=canvas_rect,
                               resolution=resolution,
                               image_out=image)
            obstacle_info_cursor += 5
        else:
            print("???")

    return image


def draw_box_in_image(box_param, canvas_rect, resolution, image_out):
    xmin, xmax, ymin, ymax = box_param
    pixYsize, pixXsize = image_out.shape
    logger.debug("drawing box obstacle, canvas size %dx%d", pixXsize, pixYsize)
    for xidx in range(pixXsize):
        for yidx in range(pixYsize):
            xpos = canvas_rect["xmin"] + resolution * xidx
            ypos = canvas_rect["ymin"] + resolution * yidx

            if xmin <= xpos <= xmax and ymin <= ypos <= ymax:
                image_out[yidx, xidx] = 1


def draw_circ_in_image(circ_param, canvas_rect, resolution, image_out):
    xcen, ycen, rad = circ_param
    pixYsize, pixXsize = image_out.shape
    logger.debug("drawing circ obstacle, canvas size %dx%d", pixXsize, pixYsize)
    for xidx in range(pixXsize):
        for yidx in range(pixYsize):
            xpos = canvas_rect["xmin"] + resolution * xidx
            ypos = canvas_rect["ymin"] + resolution * yidx

            if (xcen - xpos) ** 2 + (ycen - ypos) ** 2 <= rad ** 2:
                image_out[yidx, xidx] = 1


def draw_orie_in_image(orie_param, canvas_rect, resolution, image_out):
    xcen, ycen, xlen, ylen, deg = orie_param
    pixYsize, pixXsize = image_out.shape
    logger.debug("drawing orie obstacle, canvas size %dx%d", pixXsize, pixYsize)
    rad = deg / 180 * np.pi
    rot_mtx = np.array([[np.cos(rad), -np.sin(rad)], [np.sin(rad), np.cos(rad)]])
    for xidx in range(pixXsize):
        for yidx in range(pixYsize):
            xpos = canvas_rect["xmin"] + resolution * xidx
            ypos = canvas_rect["ymin"] + resolution * yidx

            alg_canvas_pos = np.array([xpos, ypos])
            alg_obs_pos = np.matmul(rot_mtx, alg_canvas_pos - [xcen, ycen])
            if np.abs(alg_obs_pos[0]) < xlen / 2 and np.abs(alg_obs_pos[1]) < ylen / 2:
                image_out[yidx, xidx] = 1
