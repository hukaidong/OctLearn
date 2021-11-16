import os

import numpy as np
from octLearn.g_config.config import get_config


def prepare_trajectories(filename):
    with open(filename, 'rb') as file:
        eof = file.seek(0, 2)
        file.seek(0, 0)
        agentData = []

        while file.tell() < eof:
            trajectoryLength, agentId = np.fromfile(file, np.int32, 2)
            # offset by one because agentId has been read before
            trajectory_matrix = np.fromfile(file, np.float32, trajectoryLength - 1).reshape((-1, 2, 2))
            agentData.append([agentId, trajectory_matrix])

        IndexAgentID = 0
        IndexTrajectoryMatrix = 1
        agentData.sort(key=lambda x: x[IndexAgentID])
        assert agentData[-1][IndexAgentID] == len(agentData) - 1  # Check if agent id matches index

        trajectories = [x[IndexTrajectoryMatrix][:, 0] for x in agentData]
        forwards = [x[IndexTrajectoryMatrix][:, 1] for x in agentData]

        return {'trajectories': trajectories, 'forwards': forwards}


class RawData:
    def __init__(self):
        self._init_variables_()

    def _init_variables_(self):
        self.objectId = None
        self.num_agent = None
        self.document = None
        self.trajectories = None
        self.forwards = None
        self.obstacle_map = None
        self.agent_init_location = None
        self.agent_goal_location = None
        self.agent_parameters = None

    def prepare_all(self):
        self.get_scene_parameters()
        self.get_agent_parameters()
        self.get_trajectory_and_forwards()

    def load_document(self, document):
        self._init_variables_()
        self.document = document
        self.objectId = str(document['_id'])

    def get_trajectory_and_forwards(self):
        configs = get_config()
        traj_root = configs['misc']['traj_root']
        filename = os.path.join(traj_root, str(self.objectId)[-2:], str(self.objectId))
        result = prepare_trajectories(filename)
        self.trajectories = result['trajectories']
        self.forwards = result['forwards']

    # Require load document first
    def get_scene_parameters(self):
        sceneParamIter = iter(self.document['scene parameters'])
        worldParam = np.fromiter(sceneParamIter, np.float32, 400)
        numAgent, *_ = np.fromiter(sceneParamIter, np.int32, 1)
        agentInitConfig = np.fromiter(sceneParamIter, np.float32, numAgent * 4).reshape((-1, 2, 2))

        self.num_agent = numAgent
        self.obstacle_map = worldParam.reshape((20, 20)).T
        self.agent_init_location = agentInitConfig[:, 0]
        self.agent_goal_location = agentInitConfig[:, 1]

    # Require load document first
    def get_agent_parameters(self):
        agentParam = self.document['agent parameters']
        params = np.zeros([len(agentParam), len(agentParam[0]['agent parameters'])])
        for agent in agentParam:
            aid = agent['agent id']
            par = agent['agent parameters']
            params[aid, :] = par
        self.agent_parameters = params
