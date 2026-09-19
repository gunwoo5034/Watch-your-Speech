# this file is an adapated version https://github.com/albertfgu/diffwave-sashimi, licensed
# under https://github.com/albertfgu/diffwave-sashimi/blob/master/LICENSE


import torch
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from .dataset_lipvoicer import LipVoicerDataset

def dataloader(dataset_cfg, batch_size, num_gpus):

    dataset = LipVoicerDataset(split='train', **dataset_cfg)

    # distributed sampler
    train_sampler = DistributedSampler(dataset) if num_gpus > 1 else None

    trainloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle= False if num_gpus > 1 else True,
        sampler=train_sampler,
        num_workers=8,
        pin_memory=False,
        drop_last=True,

    )
    return trainloader
