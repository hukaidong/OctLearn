import itertools

import numpy as np



env1_dim = [140, 200]
def getSequenceGridMask(sequence, pedlist_seq, dimensions=None, neighborhood_size=32, grid_size=4, is_occupancy=False):
    '''
    Get the grid masks for all the frames in the sequence
    params:
    sequence : A numpy matrix of shape SL x MNP x 3
    dimensions : This will be a list [width, height]
    neighborhood_size : Scalar value representing the size of neighborhood considered
    grid_size : Scalar value representing the size of the grid discretization
    using_cuda: Boolean value denoting if using GPU or not
    is_occupancy: A flag using for calculation of accupancy map
    '''
    sl = len(sequence)
    sequence_mask = []
    if dimensions is None:
        dimensions = env1_dim

    for i in range(sl):
        mask = getGridMask(sequence[i], dimensions, len(pedlist_seq[i]), neighborhood_size, grid_size, is_occupancy)
        sequence_mask.append(mask)

    return sequence_mask
