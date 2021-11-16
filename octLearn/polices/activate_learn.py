from torch import nn
from torch import normal


class QueryNetwork(nn.Module):
    def __init__(self, latent_size, decipher_network, device=None):
        super().__init__()
        self.latent_size = latent_size
        self.decipher = decipher_network
        self.device = device

    def to(self, device):
        self.device = device
        return super().to(device)

    def train(self, mode, **kwargs):
        if mode is True:
            raise RuntimeError("Module is not trainable")

    def eval(self):
        super().eval()
        self.decipher.eval()

    def forward(self, num_sample):
        latent = normal(0, 1, size=[num_sample, self.latent_size], device=self.device)
        sample = self.decipher(latent)
        return sample
