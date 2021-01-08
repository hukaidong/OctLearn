import torch
from torch import nn
from torch.nn.functional import interpolate


class Features2TaskTensors(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, data):
        vmap, agentTask, atraj, aparms = data
        m20x20stack = torch.cat((vmap, agentTask), -3)
        m100x100stack = interpolate(m20x20stack, size=(100, 100))
        return torch.cat((atraj, m100x100stack), -3), atraj


class Features2ParamTensors(nn.Module):
    def __init__(self):
        super().__init__()
        self.upsampling = nn.UpsamplingNearest2d(size=(100, 100))

    def forward(self, data):
        vmap, agentTask, atraj, aparms = data

        m20x20stack = torch.hstack((vmap, agentTask))
        m100x100stack = self.upsampling(m20x20stack)
        return torch.hstack((atraj, m100x100stack)), aparms


