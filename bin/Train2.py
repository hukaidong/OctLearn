from os import environ as ENV

from torch.optim import SGD, Adam, RMSprop
from torch.optim.lr_scheduler import ExponentialLR, CyclicLR, StepLR
from torch.utils.tensorboard import SummaryWriter

from octLearn.autoencoder.TaskSynthesis import Features2TaskTensors, Features2ParamTensors
from octLearn.autoencoder.convnet import FlatToFlatNetwork, FlatToImgNetwork, ImgToFlatNetwork, ImgToImgDisturbNetwork
from octLearn.autoencoder.radiencoder import RateDistortionAutoencoder
from octLearn.connector.mongo_instance import MongoInstance
from octLearn.connector.mongo_offline import MongoOffline
from octLearn.e.config import update_config
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

configs = dict(device=CUDA, latent_size=400, num_workers=8, step_per_epoch=1000, load_pretrained=False,
               batch_size=125, database='learning', collection='completed',
               load_pretrained_mask=(1, 1, 1),
               mongo_adapter=MongoInstance,  # [ MongoInstance, MongoOffline ]
               )
config_disabled = dict(model_path=None, collate_fn=None, load_pretrained_mask=(1, 1, 0))

components = dict(image_preprocessor=Features2TaskTensors, param_preprocessor=Features2ParamTensors,
                  image_encoder=(Encoder, ImgToFlatNetwork),
                  image_decoder=(Decoder, FlatToImgNetwork, ImgToImgDisturbNetwork),
                  param_decipher=(FlatToFlatNetwork,),
                  autoencoder_policy=(RateDistortionAutoencoder, dict(lambda0=1, lambda1=0.001)),
                  weight_initializer=(WeightInitializer,),
                  autoencoder_optimizer=(SGD, dict(lr=0.01, weight_decay=0.002)),
                  autoencoder_lr_scheduler=(ExponentialLR, dict(gamma=0.99)),
                  # autoencoder_lr_scheduler=(CyclicLR, dict(base_lr=0.01, max_lr=0.1, step_size_up=EPOCH_MAX // 4)),
                  decipher_optimizer=(SGD, dict(lr=0.01, weight_decay=1e-5)),
                  # decipher_lr_scheduler=(StepLR, dict(step_size=1, gamma=0.99))
                  )


def main():
    update_config(configs)

    RandSeeding()

    db = configs['mongo_adapter']('learning', 'completed')
    dataset = HopDataset(db.Case_Ids()[:50])
    trainer = TrainingHost(configs)
    trainer.build_network(dataset, **components)
    if configs['load_pretrained']:
        print("loading pretrained networks.")
        trainer.load(load_mask=configs.get('load_pretrained_mask', None), _format="%s-test.torchfile")

    writer = SummaryWriter()
    # writer = None
    print("Training begin.")
    print(trainer.autoencoder.score())

    autoencoder_train(trainer, writer)
    trainer.dump(dump_mask=configs.get('dump_mask', None))

    # trainer.decipher.score()
    decipher_train(trainer, writer)
    print(trainer.decipher.score())
    trainer.dump(dump_mask=configs.get('dump_mask', None))


def autoencoder_train(trainer, writer):
    train_task = trainer.autoencoder.loopTrain(writer)
    for step in range(EPOCH_MAX):
        try:
            reward = next(train_task)
            print("Train step {} ends, loss: {}".format(step, float(reward)))
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
    exam()
