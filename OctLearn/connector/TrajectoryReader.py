import numpy as np

from os import environ as ENV
from os import path



def readTrajectory(objId):
    TRAJECTORY_ROOT = ENV['TrajRoot']
    filename = path.join(TRAJECTORY_ROOT, str(objId)[-2:], str(objId))
    return prepare_trajectories(filename)


def prepare_trajectories(filename):
    with open(filename, 'rb') as file:
        eof = file.seek(0, 2)
        file.seek(0, 0)
        agentData = []

        while file.tell() < eof:
            trajectoryLength, agentId = np.fromfile(file, np.int32, 2)
            # offset by one because agentId has been read before
            trajMtx = np.fromfile(file, np.float32, trajectoryLength - 1).reshape((-1, 2, 2))
            agentData.append([agentId, trajMtx])

        agentData.sort(key=lambda x: x[0])
        assert agentData[-1][0] == len(agentData) - 1  # Check if agent id matches index

        trajectories = [x[1][:, 0] for x in agentData]
        forwards = [x[1][:, 1] for x in agentData]

        return trajectories, forwards
