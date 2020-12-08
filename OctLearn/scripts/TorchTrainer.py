import numpy as np
import random
import torch
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.dataloader import DataLoader

from OctLearn.connector.dbRecords import MongoCollection
from OctLearn.autoencoder.TaskSynthesis import Synthezier
from OctLearn.autoencoder.autoencoder import Encoder, Decoder
from OctLearn.autoencoder.radiencoder import RdAutoencoder
from OctLearn.connector.hopdataset import HopDataset

if __name__ == '__main__':
    step = 0
    global_seed = random.randint(0, 100000000)
    print('Seeded with num: %d' % global_seed)
    use_gpu = True
    latent_size = 1000

    writer = SummaryWriter()
    device = torch.device("cuda:0" if use_gpu else "cpu")

    random.seed(global_seed)
    np.random.seed(global_seed)
    torch.manual_seed(global_seed)


    def cycled_training_cases():
        db = MongoCollection('learning', 'completed')
        dataset = HopDataset(device, case_list=db.Case_Ids())
        loader = DataLoader(dataset, batch_size=125, num_workers=8, pin_memory=True)
        del db
        batchnum = 0
        while True:
            batchnum += 1
            print('starting %d batch' % batchnum)
            for x in loader:
                yield x


    # TrajRoot = r"C:\Users\Kaidong Hu\Desktop\5f8"
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
            plt.title(str(i) + "truth")
            plt.savefig(str(i) + 'truth.jpg')
            plt.show()
            plt.imshow(xp.detach().cpu().numpy()[i, 0], 'gray_r', origin='lower')
            plt.title(str(i) + "pred")
            plt.savefig(str(i) + 'pred.jpg')
            plt.show()

        raise KeyboardInterrupt


    def init_weight(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            torch.nn.init.xavier_normal_(m.weight.data)
        if hasattr(m, 'bias') and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)
        if classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0)
            torch.nn.init.constant_(m.bias.data, 0.0)


    enc.apply(init_weight)
    dec.apply(init_weight)

    rd = RdAutoencoder(latent_size, lambda1=1, lambda2=0.001).to(device)
    par = list() + list(enc.parameters()) + list(dec.parameters())
    opt = torch.optim.SGD(par, lr=0.01,
                          # centered=True,
                          weight_decay=0.002)
    sch = torch.optim.lr_scheduler.CyclicLR(opt, 0.01, 0.1, 80)
    try:
        print('READY TO GO')
        for step, (map, tsk, trj) in enumerate(cycled_training_cases()):
            map = map.to(device)
            tsk = tsk.to(device)
            trj = trj.to(device)

            enc.zero_grad()
            dec.zero_grad()

            xi, xo = syn(map, tsk, trj)
            z = enc(xi)
            xp, xd = dec(z, compute_dist=True)
            loss = rd(z, xo, xp, xd).mean()
            if torch.isnan(loss):
                raise ValueError('nan occured')

            if step % 1000 == 1:
                sch.step()
                writer.add_scalar('loss', loss, step)
                writer.add_scalar('lr', sch.get_last_lr()[-1], step)
                rd.record_sample(writer, z[0], xo, xp, xd, step)
                for i in range(3):
                    sh0, *_ = xo.shape
                    idx0 = np.random.randint(sh0)
                    predG = xp[idx0]
                    pmin = torch.min(predG)
                    pmax = torch.max(predG)
                    normedG = (predG - pmin) / (pmax - pmin)
                    writer.add_image('truth/%d' % i, xo[idx0], step)
                    writer.add_image('pred/%d' % i, predG, step)
                    writer.add_image('pred/%d normed' % i, normedG, step)

            loss.backward()
            opt.step()

            if step == 200000:
                break

    except KeyboardInterrupt:
        pass

    finally:
        torch.save(enc.state_dict(), "enc%d.torchfile" % step)
        torch.save(dec.state_dict(), "dec%d.torchfile" % step)
