import math
import torch
from torch.utils.data import IterableDataset

from OctLearn.connector.dbRecords import MongoCollection
from OctLearn.scenariomanage.ScenarioTypes import ScenarioType3


class HopDataset(IterableDataset):
    def __init__(self, device):
        self.device = device

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            for x in self.ids:
                m, t, j = self.Id2Feat(x)
                for i in range(m.shape[0]):
                    yield m[i], t[i], j[i]
        else:
            num_workers = worker_info.num_workers
            worker_id = worker_info.id
            for idx in range(math.ceil(len(self.ids - worker_id) / num_workers)):
                m, t, j = self.Id2Feat(self.ids[x * num_workers + worker_id])
                for i in range(m.shape[0]):
                    yield m[i], t[i], j[i]


from OctLearn.scenariomanage.ScenarioTypes import ScenarioType4


class HopDataWorker:
    from os import environ
    TrajRoot = environ['TrajRoot']
    FeatRoot = environ['FeatRoot']

    def __init__(self):
        self.db = MongoCollection('learning', 'complete')
        self.ids = self.db.Case_Ids()

    def PrepareLearningScenario(self, objectId):
        doc = self.db.Case_By_id(objectId)
        scenario = ScenarioType4(doc, self.TrajRoot)
        trajs = scenario.GetAgentTrajVisionList()
        for agentId, traj in enumerate(trajs):
            crit = {
                '_id': objectId,
                'agent parameters': {
                    '$elemMatch': {'agent id': {'$eq': agentId}}
                }
            }

            new_val = {'$set': {'agent parameters.$.agent vtraj': traj.tolist()}}
            self.db.col.update_one(crit, new_val, upsert=True)

    def Id2Feat(self, objectId):
        doc = self.db.Case_By_id(objectId)
        scenario = ScenarioType3(doc, self.TrajRoot)

        raw_map = scenario.GetAgentCubeMapVision()
        raw_tsk = scenario.GetAgentTaskVision()
        raw_trj = scenario.GetAgentTrajVision1()

        map = torch.Tensor(raw_map).to(self.device)
        tsk = torch.Tensor(raw_tsk).to(self.device)
        trj = torch.Tensor(raw_trj).to(self.device)

        return map, tsk, trj



if __name__ == '__main__':
    wk = HopDataWorker()
    for id in wk.db.Case_Ids():
        wk.PrepareLearningScenario(id)