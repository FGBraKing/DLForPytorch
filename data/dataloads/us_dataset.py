import torchvision.transforms as transforms
from data.transforms.transformOnArray import normalize, NormalizeRange, get_transform
from data.transforms.transforms import ToArray

import os
import re
import h5py
import torch
import random
import argparse
import numpy as np
import torch.utils.data

from data.utils_data import nii_loader, h5_loader
from data.transforms import get_transform, get_pre_transform, get_post_transform
from data.dataloads.base_dataset import BaseDataset
from data.transforms.transformOnArray import Normalize, random_scale, agent_resize
from utils.others.utils import print_numpy, clip_array, slim_array, convert_str_to_list
from utils.others.img_io import show_array_3d, show_volume_label, show_array_histogram, show_pired_histogram


def get_trus_path(dataroot, data_phase, model=None):
    root = os.path.join(dataroot, model+data_phase)
    return [{'volume': os.path.join(root, name), 'label': os.path.join(root, name.replace('image', 'label'))}
            for name in os.listdir(root) if 'image' in name]


def get_pre_transform():
    transform_list = []
    transform_list.append(NormalizeRange(dtype=np.float32))
    # transform_list.append(transforms.ToPILImage())
    return transforms.Compose(transform_list)


def get_post_transform():
    transform_list = []
    # transform_list.append(ToArray(normalize=False))
    # transform_list.append(NormalizeRange(dtype=np.float32))
    # transform_list.append(transforms.ToTensor())
    # transform_list.append(transforms.Normalize((0.5,), (0.5,)))
    return transforms.Compose(transform_list)


class UsDataset(BaseDataset):
    pass