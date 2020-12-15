import numpy as np
import os

from functools import lru_cache
from os import path
from os import environ as ENV
from OctLearn.utils import RectangleRepXY, ImageTranslateCrop
from OctLearn.connector.dbRecords import MongoInstance
from OctLearn.connector.TrajectoryReader import readTrajectory
from OctLearn.scenariomanage.ScenarioTypes import ScenarioType3



@lru_cache(maxsize=None)
def ObjectId2Feature(objectId: str):
    FeatRoot = ENV['FeatRoot']
    objTail = objectId[-2:]
    dirTarget = path.join(FeatRoot, objTail)
    fileTarget = path.join(dirTarget, objectId+'.npz')
    if os.path.exists(fileTarget):
        target = dict(np.load(fileTarget))
    else:
        mongo = MongoInstance('learning', 'completed')
        doc = mongo.Case_By_id(objectId)
        target = Trajectory2Feature(doc)
    return target


def Trajectory2Feature(doc, proto=ScenarioType3, save_result=True):
    FeatRoot = ENV['FeatRoot']
    scenario = proto(doc)

    objId = str(doc['_id'])
    traj, _ = readTrajectory(objId)

    feature = dict(
        aid=objId,
        agtparm=GetAgentParameters(doc),
        cubevis=scenario.GetAgentCubeMapVision(),
        taskvis=scenario.GetAgentTaskVision(),
        trajvis=GetAgentTrajVision(scenario, traj)
    )

    if save_result:
        objTail = objId[-2:]
        dirTarget = path.join(FeatRoot, objTail)
        fileTarget = path.join(dirTarget, objId)

        os.makedirs(dirTarget, mode=0o755, exist_ok=True)
        np.savez_compressed(fileTarget, **feature)

    return feature


def GetAgentParameters(doc):
    apdoc = doc['agent parameters']
    params = np.zeros([len(apdoc), len(apdoc[0]['agent parameters'])])
    for d in apdoc:
        aid, par = d.values()
        params[aid, :] = par
    return params


def GetAgentTrajVision(scenario, trajs):
    Resolution = 5  # ceil per meter
    MapSize = RectangleRepXY(20, 20)
    GridManhttanSize = MapSize * Resolution
    GridBleedingSize = GridManhttanSize * 2

    visionData = np.zeros([scenario.num_agent, 1, *GridManhttanSize])

    for i in range(scenario.num_agent):
        shift = scenario.shift[i]
        # world coordinate, origin located at [0, 0]
        # image coordinate, origin located at top left (with bleeding)
        # manhattan coordinate, scaled by Resolution
        trajSeq_Cworld = trajs[i]
        trajSeq_Cimage = trajSeq_Cworld + [-MapSize.x, MapSize.y]
        trajSeq_Cmanht = np.floor(trajSeq_Cimage * Resolution).astype(np.int)

        trajIdx_Cmanht = trajSeq_Cmanht[:, 0] * GridBleedingSize.x + trajSeq_Cmanht[:, 1]
        trajUIdx = np.unique(trajIdx_Cmanht, axis=0)  # reduce computation time

        trajVis = np.zeros([*GridBleedingSize])
        np.put(trajVis, trajUIdx, 1, mode='wrap')  # override all trajectory space with 1

        visionData[i, 0] = ImageTranslateCrop(trajVis, shift * Resolution, target_size=GridManhttanSize)

    return visionData


if __name__ == '__main__':

    objId = '5f85aee9767dae76c6c9bf1b'
    print(ObjectId2Feature(objId))


