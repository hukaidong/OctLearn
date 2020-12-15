import torch
from typing import Dict
from torch.optim import SGD
from torch.optim.lr_scheduler import CyclicLR
from torch.utils.tensorboard import SummaryWriter

from OctLearn.connector.dbRecords import MongoInstance, MongoOffline
from OctLearn.utils import RandSeeding, WeightInitializer, DataLoaderCollate
from OctLearn.connector.hopdataset import HopDataset
from OctLearn.autoencoder.TrainingTaskHost import TrainingTaskHost
from OctLearn.autoencoder.convnet import FlatToFlatNetwork, FlatToImgNetwork, ImgToFlatNetwork, ImgToImgDisturbNetwork
from OctLearn.autoencoder.autoencoder import Encoder, Decoder
from OctLearn.autoencoder.radiencoder import RateDistortionAutoencoder
from OctLearn.autoencoder.TaskSynthesis import Features2TaskTensors, Features2ParamTensors

from os import environ as ENV
# ENV['FeatRoot'] = '/path/to/features'
# ENV['TrajRoot'] = '/path/to/trajectories'
# ENV['MongoRoot'] = '/path/to/MongoDB/dumps'

EPOCH_MAX = 800
SHOULD_LOAD = False
MONGO_ADAPTER = MongoOffline  # [ MongoInstance, MongoOffline ]

CUDA = "cuda:0"
CPU = "cpu"


def main():
    configs: Dict[str, object] = dict(device=torch.device(CUDA), latent_size=1000, load_pretrained=False,
                                      model_path=None,
                                      batch_size=128, num_workers=8, step_per_epoch=100, collate_fn=DataLoaderCollate)

    components: Dict[str, object] = dict(image_preprocessor=Features2TaskTensors,
                                         param_preprocessor=Features2ParamTensors,
                                         image_encoder=(Encoder, ImgToFlatNetwork),
                                         image_decoder=(Decoder, FlatToImgNetwork, ImgToImgDisturbNetwork),
                                         param_decipher=(FlatToFlatNetwork,),
                                         autoencoder_policy=(RateDistortionAutoencoder, dict(lambda0=1, lambda1=0.001)),
                                         weight_initializer=(WeightInitializer,),
                                         optimizer=(SGD, dict(lr=0.01, weight_decay=0.002)),
                                         lr_scheduler=(CyclicLR, dict(
                                             base_lr=0.01, max_lr=0.1, step_size_up=EPOCH_MAX // 10)), )

    RandSeeding()
    db = MONGO_ADAPTER('learning', 'completed')
    dataset = HopDataset(db.Case_Ids())

    trainer = TrainingTaskHost(configs)
    trainer.build_network(dataset, **components)
    if SHOULD_LOAD:
        trainer.load()

    writer = SummaryWriter()
    autoencoder_train_task = trainer.autoencoder.loopTrain(writer)
    for step in range(EPOCH_MAX):
        try:
            next(autoencoder_train_task)
        except KeyboardInterrupt:
            continue

    trainer.dump()


if __name__ == '__main__':
    main()
