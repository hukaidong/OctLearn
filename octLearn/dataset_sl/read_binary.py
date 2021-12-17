import logging

import numpy

from ..dataset_cg import read_binary

logger = logging.getLogger(__name__)


def get_max_frame_from_trajectory(filename: str):
    *_, agent_array = read_binary.read_trajectory_binary(filename)
    return max([len(x) for x in agent_array])


def get_trajectory_slice(filename: str, start: int, end: int):
    *_, agent_array = read_binary.read_trajectory_binary(filename)
    agent_traj = [[] for _ in range(start, end)]
    for aid, arr in enumerate(agent_array):
        for frameid, data in enumerate(arr[start:end]):
            agent_traj[frameid].append([aid, *data])

    return [numpy.array(x) for x in agent_traj]
