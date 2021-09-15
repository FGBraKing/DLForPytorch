import time
import cv2
import os
import torch
import imageio
import torch.utils.data
import argparse
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from data import CustomDatasetDataLoader
from torchvision import transforms
from torchvision.datasets.folder import default_loader
from configs.options.promise_3dunet import TrainOptions
from data import create_dataset
from models import create_model
from utils.forLogs.visualizer import Visualizer
from utils.others.utils import Timer, convert_str_to_list
from torch.nn.parallel import DistributedDataParallel as DDP
from utils.others.metrics import BinaryMetrics


def test_fun1(random_state):
    print(random_state.rand())


def test_fun2(random_state):
    print(random_state.rand())


def test():

    # parser = argparse.ArgumentParser(description='for the test of trus dataset')
    #
    # parser.add_argument('--dataroot', type=str,
    #                     default='/raid/lf/PROJECT/DLForPytorch/traces/datasets/prostate_daf3d_pre')
    # parser.add_argument('--phase', type=str, default='train')
    # parser.add_argument('--serial_batches', action='store_true')
    # parser.add_argument('--custom', action='store_true')
    # parser.add_argument('--preprocess', type=str, default='randomscale_randomcrop_ranomrotate_centercrop_rot90_mirror'
    #                                                       '_gaussianNoise_GaussianBlur_'
    #                                                       'BrightnessMultiplicative_contrast_simulate_gammatransform')
    # parser.add_argument('--crop_size', type=list, default=[128, 128, 128])
    # parser.add_argument('--order_data', type=int, default=3)
    # parser.add_argument('--order_seg', type=int, default=0)
    #
    # parser.add_argument('--dataset_name', type=str, default='trus')
    # parser.add_argument('--seed', type=int, default=1008)
    #
    # parser.add_argument('--num_threads', default=8, type=int,
    #                     help='# threads for loading data')
    # parser.add_argument('--batch_size', type=int,
    #                     default=5, help='input batch size')
    # parser.add_argument('--max_dataset_size', type=int, default=float("inf"),
    #                     help='Maximum number of samples allowed per dataset. '
    #                          'If the dataset directory contains more than max_dataset_size, '
    #                          'only a subset is loaded.')
    #
    # parser.add_argument('--DP', action='store_true', help='use torch.nn.DataParallel')
    # parser.add_argument('--DDP', action='store_true', help='torch.nn.parallel.DistributedDataParallel')
    # parser.add_argument('--world_size', default=3, type=int, help='number of distributed processes')
    # parser.add_argument('--dist_url', default='tcp://172.21.141.4:30303', type=str,
    #                     help='url used to set up distributed training')
    # parser.add_argument('--dist_backend', default='nccl', type=str, help='distributed backend')
    # parser.add_argument('--local_rank', type=int, help='rank of distributed processes')
    #
    # opt = parser.parse_args(args=['--serial_batches', '--custom'])
    #
    # data_loader = CustomDatasetDataLoader(opt).get_true_loader()
    # print(type(data_loader))
    # print(len(data_loader))
    # for data in data_loader:
    #     print(data['volume'].size())
    #     print(len(data['volume_path']))
    #
    #     # dict_keys(['volume', 'label', 'volume_path', 'label_path'])
    np.random.seed(seed=1008)
    print(np.random.rand())

    tt = np.random.RandomState(seed=1008)
    test_fun1(tt)
    tt = np.random.RandomState(seed=1008)
    test_fun2(tt)


if __name__ == '__main__':
    test()


