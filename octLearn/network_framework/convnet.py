from torch import nn


class ImgToFlatNetwork(nn.Module):
    # TODO: Make image network independent to its size
    def __init__(self, input_channels, output_size):
        super().__init__()
        modules = (
            nn.Conv2d(input_channels, 5, kernel_size=7, padding=3),
            nn.BatchNorm2d(5),
            nn.ReLU(True),
            nn.Conv2d(5, 10, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(10),
            nn.ReLU(True),
            nn.Conv2d(10, 20, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(20),
            nn.ReLU(True),
            nn.Conv2d(20, 40, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(40),
            nn.ReLU(True),
            nn.Flatten(),
            nn.Linear(40 * 25 * 18, 10 * 13 * 13),
            nn.ReLU(True),
            nn.Linear(10 * 13 * 13, output_size),
        )
        self.net = nn.Sequential(*modules)

    def forward(self, _input):
        return self.net(_input)


class FlatToImgNetwork(nn.Module):
    # TODO: Make image network independent to its size
    def __init__(self, input_size, output_channels):
        super().__init__()
        modules = [
            nn.Linear(input_size, 5 * 13 * 13),
            nn.ReLU(True),
            nn.Linear(5 * 13 * 13, 20 * 25 * 18), nn.Unflatten(1, (20, 25, 18)),
            nn.ReLU(True),
            nn.ConvTranspose2d(20, 10, kernel_size=3, stride=2, padding=(1, 1), output_padding=(0, 1)),
            nn.BatchNorm2d(10),
            nn.ReLU(True),
            nn.ConvTranspose2d(10, 5, kernel_size=3, stride=2, padding=(1, 2), output_padding=(1, 0)),
            nn.BatchNorm2d(5),
            nn.ReLU(True),
            nn.ConvTranspose2d(5, 3, kernel_size=3, stride=2, padding=(1, 2), output_padding=(1, 1)),
            nn.BatchNorm2d(3),
            nn.ReLU(True),
            nn.ConvTranspose2d(3, output_channels, kernel_size=7, stride=1, padding=(1, 1)),
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
    def __init__(self, input_size, output_size, dropout=False):
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
