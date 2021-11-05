import torch
from torch import nn


class Decoder(nn.Module):
    def __init__(self, base_network, distort_network):
        super().__init__()
        self.generative_network = base_network
        self.postprocess_network = distort_network

    def forward(self, z, compute_dist=False):
        xp = torch.clip(self.generative_network(z), 0, 1)
        if compute_dist:
            eps = torch.randn(z.shape, device=z.device)
            xd_pred = self.generative_network(z + eps)
            xd = self.postprocess_network(xd_pred)
            return xp, xd
        else:
            return xp