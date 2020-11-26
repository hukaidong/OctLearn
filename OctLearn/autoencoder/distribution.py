import torch
from torch import nn
import numpy as np


class NormalLogProb(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, loc, scale, z):
        var = torch.pow(scale, 2)
        return -0.5 * torch.log(2 * np.pi * var) - torch.pow(z - loc, 2) / (2 * var)


class BernoulliLogProb(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce_with_logits = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, logits, target):
        # bernoulli log prob is equivalent to negative binary cross entropy
        return -self.bce_with_logits(logits, target)
