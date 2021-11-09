import torch
from torch.utils.data.dataset import Dataset

from os import environ
from pathlib import Path
from octLearn.dataset_cg.read_binary import get_trajectory_feature_from_file

class HopDataset(Dataset):
    def __getitem__(self, index):
        if index not in self.feature_cache.keys():
            feat = get_trajectory_feature_from_file(
                    self.base_path / index, self.resolution)
            self.feature_cache[index] = feat

        return self.feature_cache[index]

    def __len__(self):
        return len(self.keys)

    def keys(self):
        feature_files = self.base_path.glob("*.bin")
        return [str(f).lstrip(".bin") for f in feature_files]

    def __init__(self, resolution):
        super(HopDataset).__init__()
        self.feature_cache = {}
        self.resolution = resolution
        self.base_path = Path(environ["SteersimPath"])


if __name__ == "__main__":
    dataset = HopDataset(resolution=5)
    for k in dataset.keys():
        print(dataset[k][1])

