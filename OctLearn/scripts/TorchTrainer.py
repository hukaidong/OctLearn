import numpy as np
import random
import torch
import matplotlib.pyplot as plt

from torch.utils.data.dataloader import DataLoader

from OctLearn.connector.dbRecords import MongoCollection
from OctLearn.autoencoder.TaskSynthesis import Synthezier
from OctLearn.autoencoder.autoencoder import Encoder, Decoder
from OctLearn.autoencoder.radiencoder import RdAutoencoder
from OctLearn.connector.hopdataset import HopDataset

global_seed = 74513
use_gpu = True
latent_size = 1000

device = torch.device("cuda:0" if use_gpu else "cpu")

random.seed(global_seed)
np.random.seed(global_seed)
torch.manual_seed(global_seed)

db = MongoCollection('learning', 'complete')


def cycled_training_cases():
    dataset = HopDataset(device)
    loader = DataLoader(dataset, batch_size=50)
    while True:
        for x in loader:
            yield x


TrajRoot = r"C:\Users\Kaidong Hu\Desktop\5f8"
syn = Synthezier()
enc = Encoder(latent_size).to(device)
dec = Decoder(latent_size).to(device)

def demo():
    enc.load_state_dict(torch.load("enc4001.torchfile"))
    dec.load_state_dict(torch.load('dec4001.torchfile'))
    enc.eval()
    dec.eval()

    map, tsk, trj = next(cycled_training_cases())

    xi, xo = syn(map, tsk, trj)
    z = enc(xi)
    xp = dec(z, compute_dist=False)

    for i in range(xo.shape[0]):
        plt.imshow(xo.detach().cpu().numpy()[i, 0], 'gray_r', origin='lower')
        plt.title(str(i)+"truth")
        plt.savefig(str(i)+'truth.jpg')
        plt.show()
        plt.imshow(xp.detach().cpu().numpy()[i, 0], 'gray_r', origin='lower')
        plt.title(str(i)+"pred")
        plt.savefig(str(i)+'pred.jpg')
        plt.show()

    raise KeyboardInterrupt



rd = RdAutoencoder(latent_size, lambda1=1, lambda2=1).to(device)
par = list() + list(enc.parameters()) + list(dec.parameters())
opt = torch.optim.RMSprop(par, lr=0.001, centered=True)

try:
    for step, (map, tsk, trj) in enumerate(cycled_training_cases()):
        enc.zero_grad()
        dec.zero_grad()

        xi, xo = syn(map, tsk, trj)
        z = enc(xi)
        xp, xd = dec(z, compute_dist=True)
        loss = rd(z, xo, xp, xd).mean()

        if step % 100 == 1:
            plt.imshow(xo[0][0].detach().cpu().numpy())
            plt.title('xo_0')
            plt.show()
            plt.imshow(xp[0][0].detach().cpu().numpy())
            plt.title('xp_0')
            plt.show()
            plt.imshow(xo[1][0].detach().cpu().numpy())
            plt.title('xo_1')
            plt.show()
            plt.imshow(xp[1][0].detach().cpu().numpy())
            plt.title('xp_1')
            plt.show()
            plt.imshow(xp[2][0].detach().cpu().numpy())
            plt.title('xp_2')
            plt.show()
            plt.pause(0)
            torch.save(enc.state_dict(), "enc%d.torchfile"%step)
            torch.save(dec.state_dict(), "dec%d.torchfile"%step)
            print('step:', step, ', loss:', float(loss))

        loss.backward()
        opt.step()

        if step == 20000:
            break

except KeyboardInterrupt:
    pass

