# -*- coding: utf-8 -*-
import time
import cv2
import os
import re
import sys
import yaml
import torch
import logging
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
from pprint import pprint
from configs.utils_config import get_config
from horovod.runner.launch import run_commandline


def print_visible(obj):
    pprint([a for a in dir(obj) if not a.startswith('_') and not a.endswith('_')])


def test():
    '''
['DEPRECATED_KEYS',
 'IMMUTABLE',
 'NEW_ALLOWED',
 'RENAMED_KEYS',
 'clear',
 'clone',
 'copy',
 'defrost',
 'dump',
 'freeze',
 'fromkeys',
 'get',
 'is_frozen',
 'is_new_allowed',
 'items',
 'key_is_deprecated',
 'key_is_renamed',
 'keys',
 'load_cfg',
 'merge_from_file',
 'merge_from_list',
 'merge_from_other_cfg',
 'pop',
 'popitem',
 'raise_key_rename_error',
 'register_deprecated_key',
 'register_renamed_key',
 'set_new_allowed',
 'setdefault',
 'update',
 'values']
'''
    from configs.default_config import _C as cfg

    default_dir = '/raid/lf/PROJECT/DLForPytorch/configs/defaults/'
    config_path = os.path.join(default_dir, 'trus_unet3d.yaml')
    print(cfg)  # yacs.config.CfgNode, dict
    print(len(cfg))     # 89
    print_visible(cfg)
    config_yaml = get_config(config_path)   # dict

    cfg.merge_from_file(config_path)

    print('config_yaml\n', config_yaml)
    print('yaml len:', len(config_yaml))        # 72
    # for k, v in config_yaml.items():
    #     print(k, v)
    # print('after merge:\n', cfg)


if __name__ == '__main__':
    test()

