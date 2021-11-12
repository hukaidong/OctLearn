from os import environ
from pathlib import Path

import torch
from torch.utils.data.dataset import Dataset

from octLearn.dataset_cg.read_binary import get_trajectory_feature_from_file


class HopDataset(Dataset):
    def __init__(self, resolution):
        super(HopDataset).__init__()
        self.feature_cache = {}
        self.resolution = resolution
        self.base_path = Path(environ["SteersimRecordPath"])

    def __len__(self):
        return len(self.keys())

    def __getitem__(self, index):
        if index not in self.feature_cache.keys():
            self.prepare_source_binary(index[0])
        return self.feature_cache[index]

    def keys(self):
        key_list = []
        feature_files = self.base_path.glob("*.bin")
        for fname in feature_files:
            key_list.extend([(str(fname), k) for k in range(250)])
        return key_list

    def sample(self):
        return self[self.keys()[0]]

    def prepare_source_binary(self, filename):
        feature = get_trajectory_feature_from_file(filename, self.resolution)
        traj = torch.Tensor(feature[0])
        parm = torch.Tensor(feature[1])
        for i in range(250):
            self.feature_cache[(filename, i)] = [traj[i], traj[i][[3,]], parm]


if __name__ == "__main__":
    dataset = HopDataset(resolution=5)
    for k in dataset.keys():
        print(dataset[k][1])
