# Copyright (c) 2020, CoolFong. All rights reserved.
# @Time    : 2020/9/10
# @Author  : CoolFong
"""This is the data for project.
dataloads: dataload.
transforms: some function to transform
 """
import logging
import importlib
import torch.utils.data
import torch.distributed
from data.dataloads.base_dataset import BaseDataset

ddp_logger = logging.getLogger('ddp_logger')


def find_dataset_using_name(dataset_name):
    """Import the module "data/[dataset_name]_dataset.py".

    In the file, the class called DatasetNameDataset() will
    be instantiated. It has to be a subclass of BaseDataset,
    and it is case-insensitive.
    """
    dataset_filename = "data.dataloads." + dataset_name + "_dataset"
    datasetlib = importlib.import_module(dataset_filename)

    dataset = None
    target_dataset_name = dataset_name.replace('_', '') + 'dataset'
    for name, cls in datasetlib.__dict__.items():
        if name.lower() == target_dataset_name.lower() and issubclass(cls, BaseDataset):
            dataset = cls

    if dataset is None:
        raise NotImplementedError("In %s.py, there should be a subclass of BaseDataset with class name "
                                  "that matches %s in lowercase." % (dataset_filename, target_dataset_name))
    return dataset


def create_dataset(opt):
    """Create a dataset given the option.

    This function wraps the class CustomDatasetDataLoader.
        This is the main interface between this package and 'train.py'/'test.py'

    Example:
        >>> from data import create_dataset
        >>> dataset = create_dataset(opt)
    """
    data_loader = CustomDatasetDataLoader(opt)
    dataset = data_loader.load_data()
    return dataset


class CustomDatasetDataLoader:
    """Wrapper class of Dataset class that performs multi-threaded data loading"""

    def __init__(self, opt):
        """Initialize this class

        Step 1: create a dataset instance given the name [dataset_mode]
        Step 2: create a multi-threaded data loader.
        """
        self.opt = opt
        dataset_class = find_dataset_using_name(opt.dataset_name)
        self.dataset = dataset_class(opt)
        # print("dataset [%s] was created" % type(self.dataset).__name__)
        ddp_logger.warning("dataset [%s] was created" % type(self.dataset).__name__)
        sampler = torch.utils.data.distributed.DistributedSampler(self.dataset,
                                                                  num_replicas=opt.world_size,
                                                                  rank=opt.rank,
                                                                  shuffle=not opt.serial_batches,
                                                                  seed=0,
                                                                  drop_last=False) if opt.use_distribute_sample else None
        # torch.distributed.get_world_size()  torch.distributed.get_rank()
        # batch_sample = torch.utils.data.BatchSampler(sampler=sampler, batch_size=opt.batch_size, drop_last=False)
        # collate_fn = None

        self.sample = sampler
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batch_size,
            shuffle=(sampler is None) and (not opt.serial_batches),
            sampler=sampler,        #
            batch_sampler=None,     #
            num_workers=int(opt.num_threads),
            collate_fn=None,        #
            pin_memory=True,
            drop_last=False,        #
            prefetch_factor=2       #
        )

    def load_data(self):
        return self

    def get_true_loader(self):
        return self.dataloader

    def set_epoch(self, epoch):
        if self.sample:
            self.sample.set_epoch(epoch)

    def __len__(self):
        """Return the number of data in the dataset"""
        return min(len(self.dataset), self.opt.max_dataset_size)

    def __iter__(self):
        """Return a batch of data"""
        for i, data in enumerate(self.dataloader):
            if i * self.opt.batch_size >= self.opt.max_dataset_size:
                ddp_logger.warning('max_dataset_size:{}'.format(self.opt.max_dataset_size))
                break
            yield data
