import logging
from os import environ
from pathlib import Path

import torch

from .read_binary import get_max_frame_from_trajectory, get_trajectory_slice
from .trajectory_process import position_frame_from_trajectory_slices, \
    hidden_state_masking_table_from_trajcetory_slices

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

        sequence_length = self.slice_size
        agent_sequences = get_trajectory_slice(filename, start_key, start_key + sequence_length)
        num_agent_avail = len(agent_sequences)
        agent_position_matrix = position_frame_from_trajectory_slices(agent_sequences, sequence_length, num_agent_avail)
        masking_tables = hidden_state_masking_table_from_trajcetory_slices(agent_sequences, sequence_length, num_agent_avail)
        return {
            "num_agent_avail": num_agent_avail,
            "agent_position_matrix": agent_position_matrix,
            "masking_tables": masking_tables,
        }

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
