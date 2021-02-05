from torch import nn


class ImgToFlatNetwork(nn.Module):
    # TODO: Make image network independent to its size
    def __init__(self, input_channels, output_size):
        super().__init__()
        hidden_channels = 10
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
            nn.Linear(10 * 13 * 13, output_size),
            # nn.ReLU(True)
        )
        self.net = nn.Sequential(*modules)

    def forward(self, _input):
        return self.net(_input)


class FlatToImgNetwork(nn.Module):
    # TODO: Make image network independent to its size
    def __init__(self, input_size, output_channels):
        super().__init__()
        hidden_channels = 10
        modules = [
            nn.Linear(input_size, 5 * 13 * 13),
            nn.ReLU(True),
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


class ImgToImgDisturbNetwork(nn.Module):
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


class FlatToFlatNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        hidden_size = 1024
        modules = [
            nn.Linear(input_size, hidden_size), nn.ReLU(True),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(True),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(True),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(True),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(True),
            nn.Linear(hidden_size, output_size)
        ]
        self.net = nn.Sequential(*modules)

    def forward(self, _input):
        return self.net(_input)


if __name__ == '__main__':
    import torch
    net = FlatToFlatNetwork(10, 10)
    tensor = torch.normal(0, 1, size=[10, 10])
    print(net(tensor))
