import torch
from torch import nn


class Decipher(nn.Module):
    def __init__(self, encoder, decipher):
        super().__init__()
        self.encoder = encoder
        self.decipher = decipher
        self.last_states = None

    def train(self, *args, **kwargs):
        super().train(*args, **kwargs)
        self.encoder.eval()
        self.decipher.train()

    def eval(self):
        super().eval()
        self.encoder.eval()
        self.decipher.eval()

    def forward(self, img_input):
        latent = self.encoder(img_input)
        return self.decipher(latent)

    def compute_loss(self, img_input, parms_output):
        latent = self.encoder(img_input)
        pred = self.decipher(latent)
        loss = torch.pow(pred - parms_output, 2)
        mean_loss = loss.mean()
        self.last_states = {
            'img_input': img_input,
            'loss': loss,
            'mean_loss': mean_loss
        }
        return mean_loss