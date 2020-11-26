import numpy as np
import torch
from torch import nn


class Synthezier(nn.Module):
    def __init__(self):
        super().__init__()
        self.upsamp = nn.UpsamplingNearest2d(size=(100, 100))

    def forward(self, map, agentTask, agentTraj):
        if len(map.shape) == 3:
            map = map.unsqueeze(1)
        if len(agentTraj.shape) == 3:
            agentTraj = map.unsqueeze(1)

        m20x20stack = torch.hstack((map, agentTask))
        m100x100stack = self.upsamp(m20x20stack)
        return torch.hstack((agentTraj, m100x100stack)), agentTraj


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    syn = Synthezier()
    map = torch.Tensor(np.load('../agentcubemap.npy')).unsqueeze(0)
    aTa = torch.Tensor(np.load('../agenttask.npy')).unsqueeze(0)
    aTj = torch.Tensor(np.load('../trajcentered.npy')).unsqueeze(0)

    result = syn(map, aTa, aTj)

    r = result.numpy()
    for i in range(4):
        plt.imshow(r[0, i], 'gray_r', origin='lower')
        plt.show()
