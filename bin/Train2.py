#!/bin/env python3

from os import environ as ENV

from torch.optim import SGD
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.tensorboard import SummaryWriter

from octLearn.m.image_process import Features2TaskTensors, Features2ParamTensors
from octLearn.n.convnet import FlatToFlatNetwork, FlatToImgNetwork, ImgToFlatNetwork, ImgToImgDisturbNetwork
from octLearn.n.radiencoder import RateDistortionAutoencoder
from octLearn.c.mongo_instance import MongoInstance
from octLearn.c.mongo_offline import MongoOffline
from octLearn.e.config import update_config
from octLearn.e.config import reset as reset_config
from octLearn.f.torch_dataset import HopDataset
from octLearn.g.TrainingHost import TrainingHost
from octLearn.h.decoder import Decoder
from octLearn.h.encoder import Encoder

from octLearn.utils import RandSeeding, WeightInitializer

# ENV['FeatRoot'] = '/path/to/features'
# ENV['TrajRoot'] = '/path/to/trajectories'
# ENV['MongoRoot'] = '/path/to/MongoDB/dumps'


EPOCH_MAX = 50

CUDA = "cuda:0"
CPU = "cpu"

reset_config()
configs = dict(device=CUDA, latent_size=400, num_workers=8, step_per_epoch=1000, 
        batch_size=125, database='easy', collection='completed', load_pretrained_mask=(1, 1, 1), 
        mongo_adapter=MongoInstance, feat_root='/media/kaidong/Shared/easy/feature',
        traj_root=None, mongo_root=None, infile_path=None, outfile_path=None
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
                  decipher_optimizer=(SGD, dict(lr=0.01, weight_decay=1e-5)),
                  # decipher_lr_scheduler=(StepLR, dict(step_size=1, gamma=0.99))
                  )


def main():
    update_config(configs)

    RandSeeding()

    db = configs['mongo_adapter']()
    dataset = HopDataset(db.Case_Ids())
    trainer = TrainingHost(configs)
    trainer.build_network(dataset, **components)
    trainer.load(load_mask=configs.get('load_pretrained_mask', None), _format="%s.torchfile")

    writer = SummaryWriter()
    print("Training begin.")

    print(trainer.autoencoder.score())
    print(trainer.decipher.score())
    # autoencoder_train(trainer, writer)
    # trainer.dump(dump_mask=configs.get('dump_mask', None))

    # decipher_train(trainer, writer)
    # trainer.dump(dump_mask=configs.get('dump_mask', None))


def autoencoder_train(trainer, writer):
    train_task = trainer.autoencoder.loopTrain(writer)
    for step in range(EPOCH_MAX):
        try:
            reward = next(train_task)
            print("Train step {} ends, loss: {}".format(step, float(reward)))
            if step % 10 == 0:
                trainer.dump("%s-ez-step{}.torchfile".format(step))
        except KeyboardInterrupt:
            continue


def decipher_train(trainer, writer):
    train_task = trainer.decipher.loopTrain(writer)
    for step in range(EPOCH_MAX):
        try:
            reward = next(train_task)
            print("Train step {} ends, loss: {}".format(step, float(reward)))
        except KeyboardInterrupt:
            continue


def make_quest(trainer):
    mongo = MongoInstance('learning', 'queued')
    for i in range(40):
        sample = trainer.requester.sample(50)
        mongo.queue_agent_request(sample)


def exam():
    import subprocess
    import itertools

    update_config(configs)

    db = configs['mongo_adapter']()

    db.db["pending"].drop()
    db.db["queued"].drop()
    db.db["completed_test"].drop()

    col_src = db.db["completed"]
    col_dst = db.db["completed_test"]
    docs = itertools.islice(col_src.find({}), 1000)
    col_dst.insert_many(docs)

    dataset = HopDataset(db.Case_Ids())
    trainer = TrainingHost(configs)
    trainer.build_network(dataset, **components)

    writer = SummaryWriter()
    autoencoder_train(trainer, writer)
    decipher_train(trainer, writer)

    # configs['step_per_epoch'] = configs['step_per_epoch'] // 10
    # update_config(configs)
    for i in range(10):
        make_quest(trainer)
        subprocess.check_call(
            [r"C:\Users\Kaidong Hu\Desktop\Octlearn-cycle\client.exe", '-batchmode', '-logfile', 'Debug.log', '-nographics'])
        dataset = HopDataset(db.Case_Ids())
        trainer.refresh_dataset(dataset)
        autoencoder_train(trainer, writer)
        decipher_train(trainer, writer)
        trainer.dump(_format="%s-test.torchfile")

    writer = None


# Disable annoying unused code truncate
# noinspection PyStatementEffect
def implicit_used():
    ENV, MongoInstance, MongoOffline


if __name__ == '__main__':
    main()
