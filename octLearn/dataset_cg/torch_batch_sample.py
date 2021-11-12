from torch.utils.data.dataloader import DataLoader
from torch.utils.data import BatchSampler, SubsetRandomSampler


class CgBatchSampler:
    def __init__(self, dataset, batch_size, num_workers):
        self.dataset = dataset
        self.sampler = SubsetRandomSampler(dataset.keys())
        self.batch_sampler = BatchSampler(self.sampler, batch_size, drop_last=False)
        self.data_loader = DataLoader(dataset,
                                      batch_sampler=self.batch_sampler,
                                      persistent_workers=True,
                                      num_workers=num_workers,
                                      pin_memory=True)

    def update_keys(self):
        self.sampler.indices = self.dataset.keys()

    def get_data_iter(self):
        while True:
            yield from self.data_loader
