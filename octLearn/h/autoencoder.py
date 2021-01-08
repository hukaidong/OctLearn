from torch import nn


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


