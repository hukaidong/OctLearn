import numpy as np


def position_frame_from_trajectory_slices(agent_sequences, length_of_sequence, num_agent):
    """
    :param num_agent: number of agents included in this slice
    :param length_of_sequence: length of sequence in this slice
    :param agent_sequences <= .dataset_sl.read_binary.get_trajectory_slice(*)
    :return: A 3d numpy tensor. The first dimension is the length of sequence.
        The second dimension is the number of agent engaged. The third dimension
        is 2, contains a tuple of (position_x, position_y). Agent positions will be
        evaluated relative to its start frame, If agent is not engaged in a
        certain frame , it's value will be (0, 0).
    """
    sequence_matrix = np.zeros(shape=(length_of_sequence, num_agent, 2))

    for agent_index, agent_seq in enumerate(agent_sequences):
        agent_seq = agent_seq - agent_seq[(0,), :]
        sequence_matrix[0:agent_seq.shape(0), agent_index, :] = agent_seq

    return sequence_matrix

def hidden_state_masking_table_from_trajcetory_slices(agent_sequences, length_of_sequence, num_agent):
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

