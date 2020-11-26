import torch
from torch import nn

from OctLearn.autoencoder.distribution import NormalLogProb


class RdAutoencoder(nn.Module):
    def __init__(self, latent_size, lambda1=1, lambda2=1):
        super().__init__()
        self.register_buffer('p_z_loc', torch.zeros(latent_size))
        self.register_buffer('p_z_scale', torch.ones(latent_size))
        self.lambda0 = lambda1
        self.lambda1 = lambda2
        self.log_p_z = NormalLogProb()

    def forward(self, z, x, xpred, xdist):
        d_x_xp = torch.pow(x - xpred, 2).mean((-3, -2, -1))
        d_xp_xd = torch.pow(xpred - xdist, 2).mean((-3, -2, -1))
        log_p_z = self.log_p_z(self.p_z_loc, self.p_z_scale, z).mean(-1)
        return -log_p_z + self.lambda0 * torch.log(d_x_xp) + self.lambda1 * d_xp_xd