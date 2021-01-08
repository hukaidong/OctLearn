import torch
from torch import nn


class Encoder(nn.Module):
    """Approximate posterior parameterized by an inference network."""

    def __init__(self, base_network):
        super().__init__()
        self.inference_network = base_network
        self.softplus = nn.Softplus()

    def forward(self, x):
        inf = self.inference_network(x)
        loc, scale_arg = torch.chunk(inf, chunks=2, dim=-1)
        scale = self.softplus(scale_arg)
        eps = torch.randn(loc.shape, device=loc.device)
        z = loc + scale * eps  # Reparameterization
        return z