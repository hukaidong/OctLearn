import torch
from torch import nn


class Encoder(nn.Module):
    """Approximate posterior parameterized by an inference network."""

    def __init__(self, base_network):
        super().__init__()
        self.inference_network = base_network
        self.softplus = nn.Softplus()

    def forward(self, x):
        inf = self.inference_network(x)
        loc, scale_arg = torch.chunk(inf, chunks=2, dim=-1)
        scale = self.softplus(scale_arg)
        eps = torch.randn(loc.shape, device=loc.device)
        z = loc + scale * eps  # Reparameterization
        return z


class Decoder(nn.Module):
    def __init__(self, base_network, distort_network):
        super().__init__()
        self.generative_network = base_network
        self.postprocess_network = distort_network

    def forward(self, z, compute_dist=False):
        xp = torch.clip(self.generative_network(z), 0, 1)
        if compute_dist:
            eps = torch.randn(z.shape, device=z.device)
            xd_pred = self.generative_network(z + eps)
            xd = self.postprocess_network(xd_pred)
            return xp, xd
        else:
            return xp


class Autoencoder(nn.Module):
    def __init__(self, encoder, decoder, policy):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.policy = policy

    def forward(self, img_input):
        latent = self.encoder(img_input)
        img_output = self.decoder(latent)
        return img_output

    def compute_loss(self, img_input, img_output):
        latent = self.encoder(img_input)
        img_pred, img_dist = self.decoder(latent, compute_dist=True)
        loss = self.policy(latent, img_output, img_pred, img_dist)
        return loss.mean()


class Decipher(nn.Module):
    def __init__(self, encoder, decipher):
        super().__init__()
        self.encoder = encoder
        self.decipher = decipher
        self.last_states = None

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
