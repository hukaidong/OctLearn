import torch
from torch import nn


class CNNForwardNetwork(nn.Module):
    def __init__(self, input_channels, latent_size):
        hidden_channels = 10
        kernel_size = 3
        padding = 1
        strides = 1
        super().__init__()
        modules = [
            nn.Conv2d(input_channels, hidden_channels, kernel_size, strides, padding),
            nn.BatchNorm2d(hidden_channels), nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size, strides, padding),
            nn.BatchNorm2d(hidden_channels), nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size, strides, padding),
            nn.BatchNorm2d(hidden_channels), nn.AvgPool2d(5),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size, strides, padding),
            nn.BatchNorm2d(hidden_channels), nn.AvgPool2d(4),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size, strides, padding),
            nn.Flatten(1),
            nn.Linear(250, 250), nn.ReLU(),
            nn.Linear(250, 250), nn.ReLU(),
            nn.Linear(250, latent_size), nn.ReLU(),
        ]
        self.net = nn.Sequential(*modules)

    def forward(self, _input):
        return self.net(_input)


class CNNBackwardNetwork(nn.Module):
    def __init__(self, latent_size, output_channels):
        hidden_channels = 10
        kernel_size = 3
        padding = 1
        strides = 1
        super().__init__()
        modules = [
            nn.Linear(latent_size, 250), nn.ReLU(),
            nn.Linear(250, 250), nn.ReLU(),
            nn.Unflatten(1, (hidden_channels, 5, 5)),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size, strides, padding),
            nn.BatchNorm2d(hidden_channels), nn.UpsamplingNearest2d(scale_factor=4),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size, strides, padding),
            nn.BatchNorm2d(hidden_channels), nn.UpsamplingNearest2d(scale_factor=5),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size, strides, padding),
            nn.BatchNorm2d(hidden_channels), nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size, strides, padding),
            nn.BatchNorm2d(hidden_channels), nn.ReLU(),
            nn.Conv2d(hidden_channels, output_channels, kernel_size, strides, padding),
        ]
        self.net = nn.Sequential(*modules)

    def forward(self, _input):
        return self.net(_input)

if __name__ == '__main__':
    # net = CNNForwardNetwork(4, 50)
    # input = torch.randn([10, 4, 100, 100])
    # result = net(input)
    # print(input.shape)
    # print(result.shape)
    net = CNNBackwardNetwork(100, 1)
    input = torch.randn([50, 100])
    result = net(input)
    print(input.shape)
    print(result.shape)
