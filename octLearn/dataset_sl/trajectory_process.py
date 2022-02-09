import numpy as np
from itertools import permutations


def position_frame_from_trajectory_slices(agent_sequences, length_of_sequence):
    """
    :param num_agent: number of agents included in this slice
    :param length_of_sequence: length of sequence in this slice
    :param agent_sequences <= .dataset_sl.read_binary.get_trajectory_slice(*)
    :return: A list of numpy matrix of length "length of sequence".
        The first dimension is the number of agent engaged. The second dimension
        is 2, contains a tuple of (position_x, position_y). Agent positions will be
        evaluated relative to its start frame, If agent is not engaged in a
        certain frame , it's value will be (0, 0).
    """
    sequence_matrix = [[] for _ in length_of_sequence]

    for agent_index, agent_seq in enumerate(agent_sequences):
        agent_seq = agent_seq - agent_seq[(0,), :]
        sequence_matrix[agent_index].append(agent_seq)

    return [np.array(x) for x in sequence_matrix]


def hidden_state_masking_table_from_trajectory_slices(agent_sequences, length_of_sequence, num_agent):
    """

    :param agent_sequences:
    :param length_of_sequence:
    :param num_agent:
    :return: A masking table used to filter out active hidden state (num_active, hidden_size)
        from all hidden_state (num_agent, hidden_size) or update hidden state after frame is processed.
        A left multiply was expect, i.e.:
            hidden_state_active = masking_table * hidden_state_all
            hidden_state_updated = masking_unchanged * hidden_state_all + masking_table_inv * hidden_state_active
        for each frame, all masks (masking_table, masking_table_inv, masking_unchanged) are given in a tuple
    """
    agent_ids_by_frame = [[] for _ in range(length_of_sequence)]
    for agent_index, agent_sequence in enumerate(agent_sequences):
        current_agent_frame_length = agent_sequence.shape[0]
        for frame_index in range(current_agent_frame_length):
            agent_ids_by_frame[frame_index].append(agent_index)

    all_agent_id_set = set(range(num_agent))
    masking_table = []

    for agent_ids_set in agent_ids_by_frame:
        active_agent_length = len(agent_ids_set)
        masking_matrix = np.zeros((active_agent_length, num_agent))
        masking_unchanged = np.zeros((num_agent, num_agent))

        for agent_active_id, agent_id in enumerate(agent_ids_set):
            masking_matrix[agent_id, agent_active_id] = 1

        for agent_id in all_agent_id_set.difference(agent_ids_set):
            masking_unchanged[agent_id, agent_id] = 1

        masking_table.append((masking_matrix, masking_matrix.T, masking_unchanged))

    return masking_table


# dimensions = {-70, 70, -100, 100}
def get_grid_mask_single_frame(agent_sequence, neighborhood_size, grid_size, is_occupancy=False):
    """
    :param agent_sequence: agents position (x, y) in unit of meters
    :param neighborhood_size: agent neighborhood manhatton distance in unit of meters
    :param grid_size: size of grid in each dimension
    :param is_occupancy:
    :return: frame_mask is a 3d tensor that aggregate agents' hidden state according to
        their position.
    """

    agent_number = agent_sequence.shape[0]

    if is_occupancy:
        frame_mask = np.zeros((agent_number, grid_size ** 2))
    else:
        frame_mask = np.zeros((agent_number, agent_number, grid_size ** 2))

    width_bound, height_bound = neighborhood_size * 2, neighborhood_size * 2  # ??

    for agent_index, other_index in permutations(range(agent_number), 2):
        current_x, current_y = agent_sequence[agent_index]
        other_x, other_y = agent_sequence[other_index]

        width_low, width_high = current_x - width_bound / 2, current_x + width_bound / 2
        height_low, height_high = current_y - height_bound / 2, current_y + height_bound / 2

        cell_x = int(np.floor(((other_x - width_low) / width_bound) * grid_size))
        cell_y = int(np.floor(((other_y - height_low) / height_bound) * grid_size))
        grid_index = cell_x + cell_y * grid_size

        if cell_x >= grid_size or cell_x < 0 or cell_y >= grid_size or cell_y < 0:
            continue

        if is_occupancy:
            frame_mask[agent_index, grid_index] = 1
        else:
            frame_mask[agent_index, other_index, grid_index] = 1

    return frame_mask
