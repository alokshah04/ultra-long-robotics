import os
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Sampler


class InfiniteRandomSamplerDDP(Sampler):
    def __init__(self, dataset, rank=0, world_size=1, seed=0):
        self.dataset = dataset
        self.rank = rank
        self.world_size = world_size
        self.seed = seed
        self.epoch = 0

    def __iter__(self):
        while True:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
            indices = indices[self.rank::self.world_size]  # shard for DDP
            for idx in indices:
                yield idx
            self.epoch += 1

    def __len__(self):
        return 2**31
    

class InfiniteRandomSampler(Sampler):
    def __init__(self, dataset, seed=0):
        self.dataset = dataset
        self.seed = seed
        self.epoch = 0

    def __iter__(self):
        while True:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
            for idx in indices:
                yield idx
            self.epoch += 1

    def __len__(self):
        return 2**31
    
    