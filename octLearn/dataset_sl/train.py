import logging

import torch

logger = logging.getLogger(__name__)


def main():
    from octLearn.dataset_sl.torch_dataset import HopDataset
    from octLearn.network_framework.social_lstm import SocialModel
    from torch.utils.data.dataloader import DataLoader
    from torch.utils.data.sampler import SubsetRandomSampler
    dataset = HopDataset()
    dataloader = DataLoader(dataset,
                            batch_size=None,
                            sampler=SubsetRandomSampler(dataset.keys()),
                            pin_memory=True,
                            num_workers=2,
                            persistent_workers=False)
    model = SocialModel().cuda()
    optimizer = torch.optim.Adagrad(model.parameters(), lr=3e-3)
    file = open("losses.log", "w")
    for data in dataloader:
        optimizer.zero_grad()
        hidden_states = torch.zeros([250, 128], dtype=torch.float32)
        cell_states = torch.zeros([250, 128], dtype=torch.float32)
        output, losses = model.forward(data, [hidden_states, cell_states])
        total_loss = sum([x.mean() for x in losses]) / len(losses)
        file.write(f"{float(total_loss)}\n")
        file.flush()
        print(float(total_loss))
        total_loss.backward()
    file.close()


def initial_sample_steersim():
    import os
    import shutil
    import numpy as np
    from octLearn.dataset_cg.steersim_quest import steersim_call_parallel

    # ask_for_regenerate = input("Remove steersim record path and regen?")
    # if ask_for_regenerate == "n":
    #     return
    shutil.rmtree(os.environ["SteersimRecordPath"], ignore_errors=True)
    os.makedirs(os.environ["SteersimRecordPath"], exist_ok=True)
    numbers = np.random.uniform(0, 1, (225-30, 43))
    steersim_call_parallel(numbers)
    #numbers = np.random.uniform(0, 1, (1, 43))
    #steersim_call_parallel(numbers, generate_for_testcases=True)


if __name__ == "__main__":
    logging_format = "%(asctime)s - %(name)s - %(levelname)s - %(process)d: %(message)s"
    logging.basicConfig(format=logging_format, level=logging.INFO)
    initial_sample_steersim()
    # main()
