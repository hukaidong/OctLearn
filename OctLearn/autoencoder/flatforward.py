import torch
from torch import nn


class ForwardNetwork(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super().__init__()
        modules = [nn.Linear(input_size, hidden_size),
                   nn.ReLU(),
                   nn.Linear(hidden_size, hidden_size),
                   nn.ReLU(),
                   nn.Linear(hidden_size, output_size)]
        self.net = nn.Sequential(*modules)

    def forward(self, _input):
        return self.net(_input)