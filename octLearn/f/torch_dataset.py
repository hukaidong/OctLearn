import torch
from torch.utils.data.dataset import IterableDataset

from octLearn.f.feature_cache import ObjectId2Feature


def distributeIds(data, num_workers, index):
    it = iter(data)
    try:
        for i in range(index):
            next(it)
        while True:
            yield next(it)
            for i in range(num_workers - 1):
                next(it)
    except StopIteration:
        return


def ObjectId2Tensors(objectId):
    feat = ObjectId2Feature(objectId)

    raw_map = feat['cubevis']
    raw_tsk = feat['taskvis']
    raw_trj = feat['trajvis']
    raw_prm = feat['agtparm']

    tensors = list(map(torch.Tensor, [raw_map, raw_tsk, raw_trj, raw_prm]))
    for i in range(raw_map.shape[0]):
        yield [t[i] for t in tensors]


class HopDataset(IterableDataset):
    def __getitem__(self, index):
        if index != 0:
            raise ValueError
        return next(ObjectId2Tensors(self.case_list[0]))

    def __init__(self, case_list):
        super(HopDataset).__init__()
        self.case_list = case_list

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            for objId in self.case_list:
                yield from ObjectId2Tensors(objId)
        else:
            num_workers = worker_info.num_workers
            worker_id = worker_info.id

            for objId in distributeIds(self.case_list, num_workers, worker_id):
                yield from ObjectId2Tensors(objId)

