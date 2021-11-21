import gzip
from os import environ
from pathlib import Path

import numpy as np
import torch
import logging


from octLearn.dataset_cg.read_binary import get_trajectory_feature_from_file


logger = logging.getLogger()


class HopDataset(torch.utils.data.dataset.Dataset):
    def __init__(self, resolution):
        super(HopDataset, self).__init__()
        self.resolution = resolution
        self.base_path = Path(environ["SteersimRecordPath"])
        self.feature_cache = {}

    def __len__(self):
        return len(self.keys())

    def __getitem__(self, index):
        if index not in self.feature_cache.keys():
            self.prepare_source_binary(index.split("+")[0])
        fdict = self.feature_cache[index]
        traj = torch.Tensor(np.frombuffer(
            gzip.decompress(fdict["traj"])).reshape(fdict["fshape"]))
        param = torch.Tensor(np.frombuffer(
            gzip.decompress(fdict["param"]), dtype=np.float32))
        return traj[[0, 3]], traj[[3, ]], param

    def keys(self):
        key_list = []
        feature_files = self.base_path.glob("*.bin")
        for fname in feature_files:
            key_list.extend([f"{fname}+{k}" for k in range(250)])
        return key_list

    def sample(self):
        return self[self.keys()[0]]

    def prepare_source_binary(self, filename):
        feature = get_trajectory_feature_from_file(filename, self.resolution)
        traj = feature[0]
        parm = feature[1]
        feat_shape = traj[0].shape
        for i in range(250):
            self.feature_cache[f"{filename}+{i}"] = {
                "fshape": feat_shape,
                "traj": gzip.compress(traj[i].tobytes()),
                "param": gzip.compress(parm.tobytes())
            }


class HopTestDataset(HopDataset):
    def __init__(self, *args, **kwargs):
        super(HopTestDataset, self).__init__(*args, **kwargs)
        self.base_path = self.base_path / "test"


if __name__ == "__main__":
    dataset = HopDataset(resolution=5)
    for k in dataset.keys():
        print(dataset[k][1])
