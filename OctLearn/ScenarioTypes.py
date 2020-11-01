import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from typing import List, Optional


class ScenarioType:
    SCENE_PARAM = 'scene parameters'
    AGENT_PARAM = 'scene parameters'

    world: np.ndarray
    world_id: str
    num_agent = int
    agent_start: np.ndarray
    agent_target: np.ndarray
    agent_trajectories: List[np.ndarray]
    agent_forwards: List[np.ndarray]

    def __init__(self):
        pass


class ScenarioType1(ScenarioType):
    def __init__(self, doc: dict, record_path=''):
        super().__init__()
        self.prepare_parameters(doc)
        self.prepare_trajectories(Path(record_path) / str(doc['_id']))

    def prepare_parameters(self, doc):
        sceneParamIter = iter(doc[self.SCENE_PARAM])
        worldParam = np.fromiter(sceneParamIter, np.float32, 400)
        self.world = worldParam.reshape((20, 20)).T  # Make X axes second level
        numAgent, *_ = np.fromiter(sceneParamIter, np.int32, 1)
        agentInitConfig = np.fromiter(sceneParamIter, np.float32, numAgent * 4).reshape((-1, 2, 2))
        self.num_agent = numAgent
        self.agent_start = agentInitConfig[:, 0]
        self.agent_target = agentInitConfig[:, 1]

    def PlotWorld(self, ax):
        worldBoundary = [-10.5, 9.5, -10.5, 9.5]
        ax.imshow(1 - self.world, cmap='gray', origin='lower', extent=worldBoundary)

    def PlotAgentTask(self, ax, aid):
        ax.scatter(self.agent_start[aid, 0], self.agent_start[aid, 1], color='r')
        ax.scatter(self.agent_target[aid, 0], self.agent_target[aid, 1], color='y')

    def PlotAgentTrajectory(self, ax, aid):
        ax.plot(self.agent_trajectories[aid][0], self.agent_trajectories[aid][1])

    def prepare_trajectories(self, trajectory_path):
        trajectories: List[Optional[np.ndarray]]
        forwards: List[Optional[np.ndarray]]
        trajectories = [None for _ in range(self.num_agent)]
        forwards = [None for _ in range(self.num_agent)]
        with open(trajectory_path, 'rb') as file:
            eof = file.seek(0, 2)
            file.seek(0, 0)

            while file.tell() < eof:
                trajectoryLength, agentId = np.fromfile(file, np.int32, 2)
                t = np.fromfile(file, np.float32, trajectoryLength - 1).reshape((-1, 2, 2))
                # offset by one because agentId has been read before
                trajectories[agentId] = t[:, 0].T
                forwards[agentId] = t[:, 1].T

            self.agent_trajectories = trajectories
            self.agent_forwards = forwards


__all__ = [ScenarioType1]
