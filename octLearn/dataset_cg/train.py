import torch


class AEDataExtract(torch.nn.Module):
    def forward(self, data):
        return data[0], data[1]

class DCDataExtract(torch.nn.Module):
    def forward(self, data):
        return data[0], data[2]

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
        "image_preprocessor": lambda : AEDataExtract(),
        "param_preprocessor": lambda : DCDataExtract(),
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

def main():
    from torch.utils.tensorboard import SummaryWriter
    from octLearn.dataset_cg.torch_dataset import HopDataset
    from octLearn.neural_network_unit.TrainingHost import TrainingHost
    define_configs()
    components = define_components()
    train_dataset = HopDataset(resolution=1)
    trainer = TrainingHost()
    trainer.build_network(train_dataset, **components)
    writer = torch.utils.tensorboard.SummaryWriter()
    train_task = trainer.autoencoder.loopTrain(writer)
    print(next(train_task))


if __name__ == "__main__":
    import numpy as np

    #numbers = np.random.uniform(0, 1, (10, 43))
    #steersim_call_parallel(numbers)
    main()