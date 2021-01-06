if __name__ == '__main__':
    import matplotlib as mpl
    mpl.use('qt5agg')

from typing import List

import matplotlib.pyplot as plt
import numpy as np
from octLearn.connector.TrajectoryReader import readTrajectory

from octLearn.f.graphic import translate_and_crop, RectangleRepXY


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
    def __init__(self, doc: dict):
        super().__init__()
        self.doc = doc
        self.prepare_parameters(doc)

    def prepare_parameters(self, doc):
        sceneParamIter = iter(doc[self.SCENE_PARAM])
        worldParam = np.fromiter(sceneParamIter, np.float32, 400)
        self.world = worldParam.reshape((20, 20))  # Make X axes second level
        numAgent, *_ = np.fromiter(sceneParamIter, np.int32, 1)
        agentInitConfig = np.fromiter(sceneParamIter, np.float32, numAgent * 4).reshape((-1, 2, 2))
        self.num_agent = numAgent
        self.agent_start = agentInitConfig[:, 0]
        self.agent_target = agentInitConfig[:, 1]


# --------  class for demonstration  ----------
class ScenarioType2(ScenarioType1):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prepare_from_raw_trajectory()

    def prepare_from_raw_trajectory(self):
        t, f = readTrajectory(self.doc['_id'])
        self.agent_trajectories = t
        self.agent_forwards = f

    def PlotWorld(self, ax):
        worldBoundary = [-10.5, 9.5, -9.5, 10.5]
        ax.imshow(1 - self.world, cmap='gray', origin='upper', extent=worldBoundary)

    def PlotAgentTask(self, ax, aid):
        ax.scatter(self.agent_start[aid, 1], -self.agent_start[aid, 0], color='r')
        ax.scatter(self.agent_target[aid, 1], -self.agent_target[aid, 0], color='y')

    def PlotAgentTrajectory(self, ax, aid):
        ax.plot(self.agent_trajectories[aid][:, 1], -self.agent_trajectories[aid][:, 0])

    def FillByAgentTraj(self, image, aid, size_per_grid=1):
        traj = self.agent_trajectories[aid]
        shift = (np.array(image.shape) / 2).astype(np.int)
        for num_traj in range(traj.shape[0]):
            locate = traj[num_traj, :] / size_per_grid + shift
            image[int(locate[1]), int(locate[0])] = 1


def WorldLocationToGridIdx(loc):
    world_topleft = np.array((-10, -10))
    loc_world = (loc - world_topleft).astype(int)
    return loc_world

# --------  class for feature synthesis  ----------
class ScenarioType3(ScenarioType1):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.shift = []
        self.mapShape = RectangleRepXY(20, 20)
        for i in range(self.num_agent):
            self.shift.append(self.GetAgentShift(i))

    def GetAgentShift(self, aid):
        start = self.agent_start[aid]
        end = self.agent_target[aid]
        center = np.floor_divide(start + end, 2)
        return -center

    def GetAgentCubeMapVision(self):
        visionData = np.empty([self.num_agent, 1, 20, 20])
        for i in range(self.num_agent):
            shift = self.shift[i]
            visionData[i, 0] = translate_and_crop(self.world, shift)
        return visionData

    def GetAgentTaskVision(self):
        visionData = np.zeros(
            [self.num_agent, 2, self.mapShape.x, self.mapShape.y])
        for i in range(self.num_agent):
            start: np.ndarray = self.agent_start[i]
            end: np.ndarray = self.agent_target[i]
            center = (start + end) / 2
            wStart = WorldLocationToGridIdx(start - center)
            visionData[i, :, wStart[0], wStart[1]] = (end - start) / 20
        return visionData


if __name__ == '__main__':
    from octLearn.connector.dbRecords import MongoInstance


    cc = MongoInstance('learning', 'complete')

    doc = cc.Case_By_id('')
    st = ScenarioType3(doc)
    st.prepare_from_raw_trajectory()
    vis = st.GetAgentTrajVision()
    print(vis)
    plt.imshow(vis[0][0])
    plt.show()
