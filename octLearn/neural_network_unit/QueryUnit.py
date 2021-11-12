import torch
from octLearn.dataset_cubes.agent_parameters import DenormalizeAgentParameters


class QueryUnit:
    def __init__(self, active_learning_network):
        self.network = active_learning_network

    def sample(self, num_sample):
        self.network.eval()
        with torch.no_grad():
            sample_t = self.network(num_sample)
        return sample_t.cpu().numpy()
