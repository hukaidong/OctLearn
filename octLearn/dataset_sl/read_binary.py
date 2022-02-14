import logging

import numpy

from ..dataset_cg import read_binary

logger = logging.getLogger(__name__)


def get_max_frame_from_trajectory(filename: str):
    *_, agent_array = read_binary.read_trajectory_binary(filename)
    return max([len(x) for x in agent_array])


def get_trajectory_slice(filename: str, start: int, end: int):
    """
    :param filename:
    :param start:
    :param end:
    :return: a list of numpy 2d array with length of (num_agent),
        each numpy array, first dimension has length of available frames, (end-start) at maximum,
        second dimension is 3, contains [agent_id, position_x, position_y]
    """
    *_, agent_array = read_binary.read_trajectory_binary(filename)
    agent_traj = [[] for _ in range(start, end)]
    for aid, arr in enumerate(agent_array):
        for frameid, data in enumerate(arr[start:end]):
            agent_traj[frameid].append([aid, *data])

    return [numpy.array(x) for x in agent_traj], len(agent_array)
