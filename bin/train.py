#!/bin/env python3

from os import environ as ENV

from torch.optim import SGD
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from octLearn.processing.image_process import Features2TaskTensors, Features2ParamTensors
from octLearn.network_framework.convnet import FlatToFlatNetwork, FlatToImgNetwork, ImgToFlatNetwork, ImgToImgDisturbNetwork
from octLearn.network_framework.radiencoder import RateDistortionAutoencoder
from octLearn.mongo_stuff.mongo_instance import MongoInstance
from octLearn.mongo_stuff.mongo_offline import MongoOffline
from octLearn.e.config import update_config
from octLearn.e.config import reset as reset_config
from octLearn.dataset_cubes.torch_dataset import HopDataset
from octLearn.neural_network_unit.TrainingHost import TrainingHost
from octLearn.polices.decoder import Decoder
from octLearn.polices.encoder import Encoder

from octLearn.utils import RandSeeding, WeightInitializer

num_train_data = -1  # amount of scenarios
num_test_data = -1  # amount of scenarios

#train_data_inc = 500  # 10 scenarios * 50 trajectories

CUDA = "cuda:0"
CPU = "cpu"

reset_config()
configs = dict(device=CUDA, latent_size=400, num_workers=8, step_per_epoch=300, batch_size=125, 
        database='normal', 
        collection='completed', 
        load_pretrained_mask=(1, 1, 0), 
        mongo_adapter=MongoInstance,  # MongoOffline, 
        feat_root='/home/kaidong/normal/feature',
        traj_root='/home/kaidong/normal/trajectory',
        mongo_root='/home/kaidong/normal/database', 
        infile_path='/home/kaidong/sej21/p2_1',
        outfile_path='.'
       )

components = dict(image_preprocessor=Features2TaskTensors,
                  param_preprocessor=Features2ParamTensors,
                  image_encoder=(Encoder, ImgToFlatNetwork),
                  image_decoder=(Decoder, FlatToImgNetwork, ImgToImgDisturbNetwork),
                  param_decipher=(FlatToFlatNetwork,),
                  autoencoder_policy=(RateDistortionAutoencoder,
                                      dict(lambda0=1, lambda1=0.001, lambda2=0.01, lambda3=0.5)),
                  weight_initializer=(WeightInitializer,),
                  autoencoder_optimizer=(SGD, dict(lr=0.01, weight_decay=0.002)),
                  autoencoder_lr_scheduler=(ExponentialLR, dict(gamma=0.95)),
                  # autoencoder_lr_scheduler=(CyclicLR, dict(base_lr=0.01, max_lr=0.1, step_size_up=EPOCH_MAX // 4)),
                  decipher_optimizer=(SGD, dict(lr=1e-3, weight_decay=1e-5)),
                  # decipher_lr_scheduler=(StepLR, dict(step_size=1, gamma=0.99))
                  )



def main():
    global num_train_data
    update_config(configs)
    RandSeeding()

    db = configs['mongo_adapter']()
    train_dataset = HopDataset(db.Case_Ids()[:num_train_data])
    trainer = TrainingHost()
    trainer.build_network(train_dataset, **components)

    #  writer = SummaryWriter()
    #  autoencoder_train(trainer, writer, train_dataset)

    trainer.load(load_mask=configs.get('load_pretrained_mask', None))
    print('Verify autoencoder training loss: ', float(next(trainer.autoencoder.score())))
    writer = SummaryWriter()
    decipher_train(trainer, writer)



def autoencoder_train(trainer, writer, train_dataset):
    global num_test_data
    data = configs['mongo_adapter']('normal', 'cross_valid')
    test_dataset = HopDataset(data.Case_Ids()[:num_test_data])

    train_task = trainer.autoencoder.loop_train(writer)

    with trainer.extern_dataset(test_dataset):
        test_task = trainer.autoencoder.score(writer)

    max_epoch = 600

    try:
        for step in range(max_epoch):
            #train_dataset.data_limit = (step + 1) * train_data_inc
            train_loss = next(train_task)
            test_loss = next(test_task)
            print("Train step {} ends, loss: {}, {}".format(step, float(train_loss), float(test_loss)))
            writer.add_scalars("autoencoder/loss",  {'train': train_loss, 'test': test_loss}, trainer.ae_step)
    except KeyboardInterrupt:
        pass
    finally:
        trainer.dump(dump_mask=configs.get('dump_mask', None))


def decipher_train(trainer, writer, train_dataset=None):
    global num_test_data
    data = configs['mongo_adapter']('normal', 'cross_valid')
    test_dataset = HopDataset(data.Case_Ids()[:num_test_data])

    train_task = trainer.decipher.loop_train(writer)

    with trainer.extern_dataset(test_dataset):
        test_task = trainer.decipher.score(writer)

    max_epoch = 2000

    try:
        for step in range(max_epoch):
            #train_dataset.data_limit = (step + 1) * train_data_inc
            train_loss = next(train_task)
            test_loss = next(test_task)
            print("Train step {} ends, loss: {}, {}".format(step, float(train_loss), float(test_loss)))
            writer.add_scalars("decipher/loss",  {'train': train_loss, 'test': test_loss},  trainer.de_step)
    except KeyboardInterrupt:
        pass
    finally:
        trainer.dump(dump_mask=configs.get('dump_mask', None))

if __name__ == '__main__':
    main()
