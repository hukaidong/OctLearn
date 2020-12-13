import torch
from torch import nn


class Features2TaskTensors(nn.Module):
    def __init__(self):
        super().__init__()
        self.upsampling = nn.UpsamplingNearest2d(size=(100, 100))

    def forward(self, vmap, agentTask, atraj):
        if len(vmap.shape) == 3:
            vmap = vmap.unsqueeze(1)
        if len(atraj.shape) == 3:
            atraj = vmap.unsqueeze(1)

        m20x20stack = torch.hstack((vmap, agentTask))
        m100x100stack = self.upsampling(m20x20stack)
        return torch.hstack((atraj, m100x100stack)), atraj


