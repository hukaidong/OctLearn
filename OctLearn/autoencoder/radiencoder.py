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

    def forward(self, z, x, xpred, xdist, *, out=None):
        dims = [-3, -2, -1]
        s_x_xp = torch.abs(torch.mean(x, dims) - torch.mean(torch.pow(xpred, 2), dims))
        d_x_xp = torch.pow(x - xpred, 2).mean(dims)
        d_xp_xd = torch.pow(xpred - xdist, 2).mean(dims)
        log_p_z = self.log_p_z(self.p_z_loc, self.p_z_scale, z).mean(-1)
        log_d_x = torch.log(d_x_xp + 0.01 * s_x_xp + 0.00001)

        if isinstance(out, dict):
            out['d_x_xp'] = d_x_xp
            out['d_xp_xd'] = d_xp_xd
            out['log_d_x'] = log_d_x
            out['log_p_z'] = log_p_z

        d_xp_xd = torch.clip(d_xp_xd, 0.1, 1)
        log_p_z = torch.clip(log_p_z, -3, 0)
        loss = -0.01*log_p_z + self.lambda0 * (log_d_x) + self.lambda1 * d_xp_xd

        def isnan(tensor):
            return torch.any(torch.isnan(tensor))

        if isnan(log_p_z) or isnan(log_d_x) or isnan(loss):
            valtoshow = {
                'd_x_xp': d_x_xp,
                'log_d_x': log_d_x,
                'd_xp_xd': d_xp_xd,
                'log_p_z': log_p_z,
                'x': x[0],
                'xpred': xpred[0],
            }
            for k, v in valtoshow.items():
                print(k, ': ', v)
            raise Exception('nan occurs')

        return loss


    def record_sample(self, writer, z, x, xpred, xdist, *args, **kwargs):
        out = {}
        self.forward(z, x, xpred, xdist, out=out)
        for k, v in out.items():
            writer.add_scalar('loss component/'+k, torch.mean(v), *args, **kwargs)
