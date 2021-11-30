import functools
import logging
from torch import nn

logger = logging.getLogger()

class ImgToFlatNetwork(nn.Module):
    # TODO: Make image network independent to its size
    def __init__(self, *, input_shape, output_size, **_):
        super().__init__()
        mulfn = lambda x, y: x * y
        logger.debug(f"Image input shape is {input_shape}")
        input_size = functools.reduce(mulfn, iter(input_shape), 1)
        hidden_size = 4000
        modules = (
            nn.Flatten(),
            nn.Linear(input_size, hidden_size), nn.ReLU(True),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(True),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(True),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(True),
            nn.Linear(hidden_size, output_size),
        )
        self.net = nn.Sequential(*modules)

    def forward(self, _input):
        return self.net(_input)


class FlatToImgNetwork(nn.Module):
    # TODO: Make image network independent to its size
    def __init__(self, *, input_size, output_shape, **_):
        super().__init__()
        logger.debug(f"Image output shape is {output_shape}")
        mulfn = lambda x, y: x * y
        output_size = functools.reduce(mulfn, iter(output_shape), 1)
        hidden_size = 4000
        modules = (
            nn.Linear(input_size, hidden_size), nn.ReLU(True),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(True),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(True),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(True),
            nn.Linear(hidden_size, output_size), nn.Sigmoid(),
            nn.Unflatten(1, output_shape),
        )
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
