import math
import torch
from torch.utils.data import IterableDataset

from OctLearn.connector.dbRecords import MongoCollection
from OctLearn.connector.TrajectoryEncodes import ObjectId2Feature
from OctLearn.scenariomanage.ScenarioTypes import ScenarioType3


class HopDataset(IterableDataset):
    def __init__(self, device, case_list):
        super(HopDataset).__init__()
        self.device = device
        self.case_list = case_list

    def distIds(self, data, num_workers, index):
        it = iter(data)
        try:
            for i in range(index):
                next(it)
            while True:
                yield next(it)
                for i in range(num_workers-1):
                    next(it)
        except StopIteration:
            return

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            for x in self.case_list:
                m, t, j = self.Id2Feat(x)
                for i in range(m.shape[0]):
                    yield m[i], t[i], j[i]
        else:
            num_workers = worker_info.num_workers
            worker_id = worker_info.id

            for objId in self.distIds(self.case_list, num_workers, worker_id):
                m, t, j = self.Id2Feat(objId)
                for i in range(m.shape[0]):
                    yield m[i], t[i], j[i]


    def Id2Feat(self, objectId):
        feat = ObjectId2Feature(objectId)

        raw_map = feat['cubevis']
        raw_tsk = feat['taskvis']
        raw_trj = feat['trajvis']

        map = torch.Tensor(raw_map)
        tsk = torch.Tensor(raw_tsk)
        trj = torch.Tensor(raw_trj)

        return map, tsk, trj

