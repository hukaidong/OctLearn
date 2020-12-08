import torch
from torch import nn


class CNNForwardNetwork(nn.Module):
    def __init__(self, input_channels, latent_size):
        hidden_channels = 10
        super().__init__()
        modules = (
            nn.Conv2d(input_channels, hidden_channels, kernel_size=7, padding=2),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(True),
            nn.Flatten(),
            nn.Linear(10 * 13 * 13, 10 * 13 * 13),
            nn.ReLU(True),
            nn.Linear(10 * 13 * 13, latent_size)
        )
        self.net = nn.Sequential(*modules)

    def forward(self, _input):
        return self.net(_input)


class CNNBackwardNetwork(nn.Module):
    def __init__(self, latent_size, output_channels):
        hidden_channels = 10
        kernel_size = 3
        padding = 1
        strides = 0
        super().__init__()
        modules = [
            nn.Linear(latent_size, 5 * 13 * 13),
            nn.Linear(5 * 13 * 13, 10 * 13 * 13),
            nn.Unflatten(1, (10, 13, 13)),
            nn.ConvTranspose2d(hidden_channels, hidden_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(True),
            nn.ConvTranspose2d(hidden_channels, hidden_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(True),
            nn.ConvTranspose2d(hidden_channels, hidden_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(True),
            nn.Conv2d(hidden_channels, output_channels, kernel_size=7, stride=1, padding=1),
            nn.BatchNorm2d(output_channels),
            nn.Sigmoid()
        ]
        self.net = nn.Sequential(*modules)

    def forward(self, _input):
        return self.net(_input)


class CNNPostNetwork(nn.Module):
    def __init__(self, channels):
        super().__init__()
        modules = [
            nn.Conv2d(channels, channels, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(channels),
            nn.Sigmoid()
        ]
        self.net = nn.Sequential(*modules)

    def forward(self, _input):
        return self.net(_input)


if __name__ == '__main__':
    net = CNNForwardNetwork(4, 50)
    input = torch.randn([10, 4, 100, 100])
    result = net(input)
    print(input.shape)
    print(result.shape)
    # net = CNNBackwardNetwork(100, 1)
    # input = torch.randn([50, 100])
    # result = net(input)
    # print(input.shape)
    # print(result.shape)
