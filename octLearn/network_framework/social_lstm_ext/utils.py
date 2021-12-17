# from https://github.com/quancore/social-lstm/blob/master/model.py
import numpy as np

import torch
from torch.autograd import Variable


def convert_proper_array(x_seq, pedlist, seq_length):
    # converter function to appropriate format. Instead of direcly use ped ids, we are mapping ped ids to
    # array indices using a lookup table for each sequence -> speed
    # output: seq_lenght (real sequence lenght+1)*max_ped_id+1 (biggest id number in the sequence)*2 (x,y)

    # get unique ids from sequence
    unique_ids = np.unique(np.concatenate(pedlist).ravel().tolist()).astype(int)
    # create a lookup table which maps ped ids -> array indices
    lookup_table = dict(zip(unique_ids, range(0, len(unique_ids))))

    seq_data = np.zeros(shape=(seq_length, len(lookup_table), 2))

    # create new structure of array
    for ind, frame in enumerate(x_seq):
        corr_index = [lookup_table[int(x)] for x in frame[:, 0]]
        seq_data[ind, corr_index, :] = frame[:, 1:3]

    return_arr = Variable(torch.from_numpy(np.array(seq_data)).float())

    return return_arr, lookup_table


class WriteOnceDict(dict):
    def __setitem__(self, key, value):
        if not key in self:
            super(WriteOnceDict, self).__setitem__(key, value)


def vectorize_seq(x_seq, PedsList_seq, lookup_seq):
    # substract first frame value to all frames for a ped.Therefore, convert absolute pos. to relative pos.
    first_values_dict = WriteOnceDict()
    vectorized_x_seq = x_seq.clone()
    for ind, frame in enumerate(x_seq):
        for ped in PedsList_seq[ind]:
            first_values_dict[ped] = frame[lookup_seq[ped], 0:2]
            vectorized_x_seq[ind, lookup_seq[ped], 0:2] = frame[lookup_seq[ped], 0:2] - first_values_dict[ped][0:2]

    return vectorized_x_seq, first_values_dict
