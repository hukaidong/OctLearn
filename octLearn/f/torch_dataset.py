import torch
from torch.utils.data.dataset import IterableDataset

from octLearn.f.feature_cache import ObjectId2Feature


def distributeIds(data, num_workers, worker_index, data_limit=0):
    it = enumerate(iter(data))
    try:
        for i in range(worker_index):
            next(it)
        while True:
            data_index, data = next(it)
            if data_limit > 0 and data_index > data_limit:
                return
            yield data
            for i in range(num_workers - 1):
                next(it)
    except StopIteration:
        return


def ObjectId2Tensors(objectId, db=None):
    feat = ObjectId2Feature(objectId, db)

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

    def __len__(self):
        bucket = len(self.case_list)*50
        if self.data_limit > 0:
            return min(self.data_limit, bucket)
        return bucket


    def __init__(self, case_list, from_database=None):
        super(HopDataset).__init__()
        self.case_list = case_list
        self.db = from_database
        self.data_limit = 0

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            for idx, objId in enumerate(self.case_list):
                if idx > self.data_limit:
                    return
                yield from ObjectId2Tensors(objId, db=self.db)
        else:
            num_workers = worker_info.num_workers
            worker_id = worker_info.id

            for objId in distributeIds(self.case_list, num_workers, worker_id, self.data_limit):
                yield from ObjectId2Tensors(objId)

