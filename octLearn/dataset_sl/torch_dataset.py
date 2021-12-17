import logging
from os import environ
from pathlib import Path

import torch

from .read_binary import get_max_frame_from_trajectory, get_trajectory_slice

logger = logging.getLogger(__name__)


class HopDataset(torch.utils.data.dataset.Dataset):
    def __init__(self, slice_size=20):
        super(HopDataset, self).__init__()
        self.base_path = Path(environ["SteersimRecordPath"])
        self.feature_cache = {}
        self.slice_size = slice_size

    def __len__(self):
        return len(self.keys())

    def __getitem__(self, index):
        filename, start_key_str = index.split("+")
        start_key = int(start_key_str)
        xseq = get_trajectory_slice(filename, start_key, start_key + self.slice_size)
        return xseq

    def keys(self):
        key_list = []
        feature_files = list(self.base_path.glob("*.bin"))
        logger.debug("There are %d files in basepath", len(feature_files))
        for fname in feature_files:
            frame_num = get_max_frame_from_trajectory(fname)
            key_list.extend([f"{fname}+{k}" for k in range(frame_num - self.slice_size)])
        return key_list

    def sample(self):
        return self[self.keys()[0]]


class HopTestDataset(HopDataset):
    def __init__(self, *args, **kwargs):
        super(HopTestDataset, self).__init__(*args, **kwargs)
        self.base_path = self.base_path / "test"
