import torch
from torch import nn
from torch.nn.functional import interpolate


class PartialDataExtract(nn.Module):
    def __init__(self, partial_num):
        super(PartialDataExtract, self).__init__()
        self.partial_num = partial_num

    def forward(self, data):
        return data[self.partial_num]


def define_configs():
    from octLearn.g_config import config

    config.reset()
    config.update_config({
        "device": "cuda:0",
        "latent_size": 400,
        "num_workers": 8,
        "step_per_epoch": 1000,
        "batch_size": 128,
        "infile_path": ".",
        "outfile_path": ".",
    })


def define_components():
    from octLearn.polices.encoder import Encoder
    from octLearn.polices.decoder import Decoder

    import octLearn.network_framework.convnet as net
    from octLearn.network_framework.radiencoder import RateDistortionAutoencoder
    from octLearn.utils import WeightInitializer

    from torch.optim import SGD
    from torch.optim.lr_scheduler import ExponentialLR

    components = {
        "image_preprocessor": lambda x: PartialDataExtract(0),
        "param_preprocessor": lambda x: PartialDataExtract(1),
        "image_encoder": (Encoder, net.ImgToFlatNetwork),
        "image_decoder": (Decoder, net.FlatToImgNetwork, net.ImgToImgDisturbNetwork),
        "param_decipher": (net.FlatToFlatNetwork,),
        "autoencoder_policy": (RateDistortionAutoencoder,
                               {"lambda0": 1, "lambda1": 0.001, "lambda2": 0.01, "lambda3": 0.5}),
        "weight_initializer": (WeightInitializer,),
        "autoencoder_optimizer": (SGD, dict(lr=0.01, weight_decay=0.002)),
        "autoencoder_lr_scheduler": (ExponentialLR, dict(gamma=0.95)),
        "decipher_optimizer": (SGD, dict(lr=0.01, weight_decay=1e-5)),
    }

    return components

def steersim_call(query):
    from os import environ
    from subprocess import Popen, PIPE, STDOUT

    steersim_command_path = environ["SteersimCommandPath"]
    steersim_command_exec = environ["SteersimCommandExec"]
    p = Popen(steersim_command_exec.split(), cwd=steersim_command_path,
              stdout=None, stdin=PIPE, stderr=STDOUT)
    p.communicate(input=query.encode())
    p.wait()

def steersim_require(queries):
    from multiprocessing import Pool

    with Pool() as p:
        p.map(steersim_call, queries)


def main():
    from octLearn.dataset_cg.torch_dataset import HopDataset
    from octLearn.neural_network_unit.TrainingHost import TrainingHost
    define_configs()
    components = define_components()
    train_dataset = HopDataset(resolution=1)
    trainer = TrainingHost()
    trainer.build_network(train_dataset, **components)
    # trainer.load()

if __name__ == "__main__":
    from os import environ

    steersim_require(("Random" for i in range(5)))