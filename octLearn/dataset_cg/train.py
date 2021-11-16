import torch


def float_next(val):
    return float(next(val))


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
        "step_per_epoch": 200,
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
        "image_preprocessor": lambda: AEDataExtract(),
        "param_preprocessor": lambda: DCDataExtract(),
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
    from octLearn.dataset_cg.torch_dataset import HopDataset, HopTestDataset
    from octLearn.dataset_cg.torch_batch_sample import CgBatchSampler
    from octLearn.neural_network_unit.TrainingHost import TrainingHost
    from octLearn.dataset_cg.steersim_quest import steersim_call_parallel
    from octLearn.neural_network_unit.summary_writer import SummaryWriter

    define_configs()
    components = define_components()
    dataset_train = HopDataset(resolution=1)
    trainer = TrainingHost()
    trainer.build_network(dataset_train, **components)
    loader_train = CgBatchSampler(dataset_train, 128, 8)
    summary_writer = SummaryWriter()
    task_ae_train = trainer.autoencoder.loop_train(loader_train, summary_writer=summary_writer)
    task_dc_train = trainer.decipher.loop_train(loader_train, summary_writer=summary_writer)
    loader_train = CgBatchSampler(dataset_train, 128, 8)
    dataset_test = HopTestDataset(resolution=1)
    loader_test = CgBatchSampler(dataset_test, 128, 8)
    task_ae_test = trainer.autoencoder.score(loader_test, summary_writer=summary_writer)
    task_dc_test = trainer.decipher.score(loader_test, summary_writer=summary_writer)

    try:
        for step in range(800):
            print(f"Training Step {step}")
            summary_writer.global_step = step
            print(f"\tAutoencoder: ")

            loss_ae_train = float_next(task_ae_train)
            print(f"\t\tTraining loss: {loss_ae_train}")
            loss_ae_test = float_next(task_ae_test)
            print(f"\t\tTest loss: {loss_ae_test}")
            summary_writer.add_scalars("main-loss/autoencoder", {"train": loss_ae_train, "test": loss_ae_test})

            print(f"\tDecipher: ")
            loss_dc_train = float_next(task_dc_train)
            print(f"\t\tTraining loss: {loss_dc_train}")
            loss_dc_test = float_next(task_dc_test)
            print(f"\t\tTest loss: {loss_dc_test}")
            summary_writer.add_scalars("main-loss/decipher", {"train": loss_dc_train, "test": loss_dc_test})

            # numbers = np.random.uniform(0, 1, (10, 43))
            # steersim_call_parallel(numbers)
            sample_list = trainer.requester.sample(2)
            steersim_call_parallel(sample_list)
            sample_list = trainer.requester.sample(1)
            steersim_call_parallel(sample_list, generate_for_testcases=True)
            loader_train.update_keys()
            loader_test.update_keys()

    except KeyboardInterrupt:
        pass


def initial_sample_steersim():
    import os
    import shutil
    import numpy as np
    from octLearn.dataset_cg.steersim_quest import steersim_call_parallel

    ask_for_regenerate = input("Remove steersim record path and regen?")
    if ask_for_regenerate == "n":
        return
    shutil.rmtree(os.environ["SteersimRecordPath"], ignore_errors=True)
    os.makedirs(os.environ["SteersimRecordPath"], exist_ok=True)
    numbers = np.random.uniform(0, 1, (10, 43))
    steersim_call_parallel(numbers)
    numbers = np.random.uniform(0, 1, (5, 43))
    steersim_call_parallel(numbers, generate_for_testcases=True)


if __name__ == "__main__":
    initial_sample_steersim()
    main()
