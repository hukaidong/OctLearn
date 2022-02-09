# from https://github.com/quancore/social-lstm/blob/master/model.py

import torch
import torch.nn as nn

from torch.autograd import Variable
from .social_lstm_ext.grid import getSequenceGridMask
from .social_lstm_ext.utils import convert_proper_array, vectorize_seq


class SocialModel(nn.Module):

    def __init__(self, hidden_size=128, grid_size=4, output_size=5, input_size=2, embedding_size=64):
        super(SocialModel, self).__init__()

        # Linear layer to embed the input position
        self.input_embedding_layer = nn.Sequential(
            nn.Linear(input_size, embedding_size),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        # Linear layer to embed the social tensor
        self.tensor_embedding_layer = nn.Sequential(
            nn.Linear(grid_size * grid_size * hidden_size, embedding_size),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        # Two embedded layer are merged together and fed into The LSTM cell
        self.cell = nn.LSTMCell(2 * embedding_size, hidden_size)

        # Linear layer to map the hidden state of LSTM to output
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, data_dict, initial_states):
        agent_position_matrix = data_dict["agent_position_matrix"]
        masking_tables = data_dict["masking_tables"]
        grid_masks_interact = data_dict["grid_masks_interact"]

        hidden_state, cell_state = initial_states
        hidden_state.cuda()
        cell_state.cuda()

        num_frame = agent_position_matrix.shape[0]

        outputs = []
        for frame_index in range(num_frame):
            # Extract tensors used in program and deliver to CUDA
            agent_position = agent_position_matrix[frame_index].cuda()

            masking_table, masking_table_inv, masking_table_unchanged = \
                masking_tables[frame_index]
            masking_table.cuda()
            masking_table_inv.cuda()
            masking_table_unchanged.cuda()

            grid_mask_interact = grid_masks_interact[frame_index].cuda()

            hidden_state_frame = masking_table * hidden_state
            cell_state_frame = masking_table * cell_state

            input_embedded = self.input_embedding_layer(agent_position)
            social_tensor = torch.tensordot(grid_mask_interact.permute(0, 2, 1), hidden_state_frame)
            tensor_embedded = self.tensor_embedding_layer(social_tensor)
            concat_embedded = torch.cat((input_embedded, tensor_embedded), 1)

            hidden_next_frame, cell_next_frame = self.cell(concat_embedded, hidden_state_frame, cell_state_frame)
            outputs.append(self.output_layer(hidden_next_frame))
            hidden_state = masking_table_unchanged * hidden_state + masking_table_inv * hidden_next_frame
            cell_state = masking_table_unchanged * cell_state + masking_table_inv * cell_next_frame

        return outputs
