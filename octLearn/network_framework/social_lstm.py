# from https://github.com/quancore/social-lstm/blob/master/model.py

import torch
import torch.nn as nn

from torch.autograd import Variable
from .social_lstm_ext.grid import getSequenceGridMask
from .social_lstm_ext.utils import convert_proper_array, vectorize_seq


class SocialModel(nn.Module):

    def __init__(self, seq_length=20, rnn_size=128, grid_size=4, embedding_size=64, input_size=2, output_size=5, infer=False):
        super(SocialModel, self).__init__()

        self.infer = infer
        self.device = "cuda:0"
        # Store required sizes
        self.rnn_size = rnn_size
        self.grid_size = grid_size
        self.embedding_size = embedding_size
        self.input_size = input_size
        self.output_size = output_size
        self.seq_length = seq_length

        # The LSTM cell
        self.cell = nn.LSTMCell(2 * self.embedding_size, self.rnn_size)
        # Linear layer to embed the input position
        self.input_embedding_layer = nn.Sequential(
            nn.Linear(self.input_size, self.embedding_size),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        # Linear layer to embed the social tensor
        self.tensor_embedding_layer = nn.Linear(self.grid_size * self.grid_size * self.rnn_size, self.embedding_size)
        # Linear layer to map the hidden state of LSTM to output
        self.output_layer = nn.Linear(self.rnn_size, self.output_size)

        # ReLU and dropout unit
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def getSocialTensor(self, grid, hidden_states):
        numNodes = grid.size()[0]

        # Construct the variable
        social_tensor = Variable(torch.zeros(numNodes, self.grid_size * self.grid_size, self.rnn_size)).to(self.device)

        # For each ped
        for node in range(numNodes):
            # Compute the social tensor
            social_tensor[node] = torch.mm(torch.t(grid[node]), hidden_states)

        # Reshape the social tensor
        social_tensor = social_tensor.view(numNodes, self.grid_size * self.grid_size * self.rnn_size)
        return social_tensor

    def forward(self, pedxy, hidden_states, cell_states, outputs):

        peds_list = [x[:, 0] for x in pedxy]
        xvec, look_up = convert_proper_array(pedxy, peds_list, self.seq_length)
        xseq_data, _ = vectorize_seq(xvec, peds_list, look_up)
        grids= getSequenceGridMask(xvec, peds_list)
        grids_tensor = [torch.tensor(x, requires_grad=True, dtype=torch.float32, device=self.device) for x in grids]


        # For each frame in the sequence
        for framenum, frame in enumerate(xseq_data):

            # print("now processing: %s base frame number: %s, in-frame: %s"%(dataloader.get_test_file_name(), dataloader.frame_pointer, framenum))
            # print("list of nodes")
            frame.cuda()
            nodeIDs = [int(nodeID) for nodeID in peds_list[framenum]]
            if len(nodeIDs) == 0:  # If no peds, then go to the next frame
                continue

            # List of nodes
            # print("lookup table :%s"% look_up)
            list_of_nodes = [look_up[x] for x in nodeIDs]

            corr_index = Variable(torch.LongTensor(list_of_nodes)).to(self.device)
            nodes_current = frame[list_of_nodes, :]
            # Get the corresponding grid masks
            grid_current = grids_tensor[framenum]

            # Get the corresponding hidden and cell states
            hidden_states_current = torch.index_select(hidden_states, 0, corr_index)
            cell_states_current = torch.index_select(cell_states, 0, corr_index)

            # Compute the social tensor
            social_tensor = self.getSocialTensor(grid_current, hidden_states_current)

            input_embedded = self.input_embedding_layer(nodes_current)
            # Embed inputs
            # Embed the social tensor
            tensor_embedded = self.dropout(self.relu(self.tensor_embedding_layer(social_tensor)))

            # Concat input
            concat_embedded = torch.cat((input_embedded, tensor_embedded), 1)

            h_nodes, c_nodes = self.cell(concat_embedded, (hidden_states_current, cell_states_current))

            # Compute the output
            outputs[framenum * len(look_up) + corr_index.data] = self.output_layer(h_nodes)

            # Update hidden and cell states
            hidden_states[corr_index.data] = h_nodes
            cell_states[corr_index.data] = c_nodes

        # Reshape outputs
        outputs_return = Variable(torch.zeros(self.seq_length, len(look_up), self.output_size)).to(self.device)
        for framenum in range(self.seq_length):
            for node in range(len(look_up)):
                outputs_return[framenum, node, :] = outputs[framenum * len(look_up) + node, :]

        return outputs_return, hidden_states, cell_states
