import torch
from torch import nn

from OctLearn.autoencoder.flatforward import ForwardNetwork
from OctLearn.autoencoder.convnet import CNNForwardNetwork, CNNBackwardNetwork
from OctLearn.autoencoder.distribution import BernoulliLogProb


class Encoder(nn.Module):
    """Approximate posterior parameterized by an inference network."""

    def __init__(self, latent_size):
        super().__init__()
        self.inference_network = CNNForwardNetwork(input_channels=4, latent_size=latent_size*2)
        self.softplus = nn.Softplus()

    def forward(self, x, n_samples=1):
        loc, scale_arg = torch.chunk(self.inference_network(x), chunks=2, dim=-1)
        scale = self.softplus(scale_arg)
        eps = torch.randn(loc.shape, device=loc.device)
        z = loc + scale * eps  # reparameterization
        return z


class Decoder(nn.Module):
    def __init__(self, latent_size):
        super().__init__()
        self.log_p_x = BernoulliLogProb()
        self.generative_network = CNNBackwardNetwork(latent_size=latent_size, output_channels=1)

    def forward(self, z, compute_dist=False):
        xp = torch.clip(self.generative_network(z), 0, 1)
        if compute_dist:
            eps = torch.randn(z.shape, device=z.device)
            xd = torch.clip(self.generative_network(z + eps), 0, 1)
            return xp, xd
        else:
            return xp
