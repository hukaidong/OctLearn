import torch
from torch import nn

from octLearn.autoencoder.distribution import NormalLogProb


def isnan(tensor):
    return torch.any(torch.isnan(tensor))


EPSILON = 1e-6


# Referred from paper: https://arxiv.org/abs/1910.04329
class RateDistortionAutoencoder(nn.Module):
    def __init__(self, latent_size, lambda0=1, lambda1=1):
        super().__init__()
        self.register_buffer('p_z_loc', torch.zeros(latent_size))
        self.register_buffer('p_z_scale', torch.ones(latent_size))
        self.lambda0 = lambda0
        self.lambda1 = lambda1
        self.log_p_z = NormalLogProb()
        self.last_states = None

    def forward(self, z, x, xpred, xdist):
        dims = [-3, -2, -1]
        s_x_xp = torch.abs(torch.mean(x, dims) - torch.mean(torch.pow(xpred, 2), dims))
        d_x_xp = torch.pow(x - xpred, 2).mean(dims)
        d_xp_xd = torch.pow(xpred - xdist, 2).mean(dims)
        log_p_z = self.log_p_z(self.p_z_loc, self.p_z_scale, z).mean(-1)
        log_d_x = torch.log(d_x_xp + 0.01 * s_x_xp + EPSILON)

        self.last_states = {'d_x_xp': d_x_xp, 'log_d_x': log_d_x, 'd_xp_xd': d_xp_xd, 'log_p_z': log_p_z, 'x': x,
                            'xpred': xpred, 'z':z}

        d_xp_xd = torch.clip(d_xp_xd, 0.1, 1)
        log_p_z = torch.clip(log_p_z, -3, 0)
        loss = -0.01 * log_p_z + self.lambda0 * log_d_x + self.lambda1 * d_xp_xd

        if isnan(log_p_z) or isnan(log_d_x) or isnan(loss):
            for k, v in self.last_states.items():
                print(k, ': ', v)
            raise Exception('nan occurs')

        self.last_states['loss'] = loss
        return loss

    def record_sample(self, writer, prefix, step):
        for k, v in self.last_states.items():
            writer.add_scalar(prefix % ('variational component ' + k), torch.mean(v), step)
