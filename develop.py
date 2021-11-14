# -*- coding: utf-8 -*-

import os
import re
import sys
import time
import yaml
import torch
import logging
import imageio
import argparse
import numpy as np
import torch.nn as nn
import nibabel as nib
import matplotlib.pyplot as plt
import torch.nn.functional as F
import h5py
import torch.optim
import torch.distributed
import torch.utils.data
import horovod.torch as hvd
import torch.distributed as dist
from data import CustomDatasetDataLoader
from torchvision import transforms
from torchvision.datasets.folder import default_loader
# from configs.options.promise_3dunet import TrainOptions
from data import create_dataset
from models import create_model
from utils.forLogs import Visualizer, get_logger
from utils.forLogs.visualizer import Visualizer
from utils.others.utils import Timer, convert_str_to_list
from torch.nn.parallel import DistributedDataParallel as DDP
from utils.others.metrics import BinaryMetrics
from pprint import pprint
from configs.utils_config import get_config

from data.dataloads.trus_dataset import TrusDataset
from data.utils_data import nii_loader
from utils.others.utils import init_seed, init_torch
from models.modules.segmentation.three_d.unet3d_gn import UNet3D
from models.loss.region_based import BinaryDiceLoss
from configs.simple_options import get_opt
from configs.utils_config import pretty_print_opt
from models.auxiliary_funs import get_init_func, get_activation
from models.loss import get_loss_criterion
from models.optim import create_optimizer, create_optimizer_v2
from models.scheduler import create_scheduler
from horovod.runner.launch import run_commandline
from utils.others.distributed_utils_horovod import reduce_mean, metric_average
import torch.distributed.launch
from utils.others.img_io import show_volume_label_predict


def print_visible(obj):
    pprint([a for a in dir(obj) if not a.startswith('_') and not a.endswith('_')])


def test_config():
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


def debug():
    opt = get_opt(args=['--config_path=configs/defaults/trus_unet3d.yaml', '--use_config'])
    opt.horovod = True
    pretty_print_opt(opt)

    init_torch(gpu_id=opt.visible_gpu, deterministic=True)

    # 1.Run hvd.init().
    hvd.init()
    # 2. Pin each GPU to a single process.
    if torch.cuda.is_available():
        torch.cuda.set_device(hvd.local_rank())

    opt.gradient_predivide_factor = 1

    # 3. Define dataset and dataloader
    dataset = TrusDataset(opt, loader=nii_loader)
    print('dataset created!')

    sampler = torch.utils.data.distributed.DistributedSampler(dataset,
                                                              num_replicas=hvd.size(),
                                                              rank=hvd.rank(),
                                                              shuffle=not opt.serial_batches,
                                                              seed=0,
                                                              drop_last=False) if opt.horovod else None
    print('size:{},rank:{}, local_rank:{}'.format(hvd.size(), hvd.rank(), hvd.local_rank()))
    dataloader = torch.utils.data.DataLoader(
        dataset,
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

    # 4. define model and optimizer
    model = UNet3D(in_channels=opt.input_nc, out_channels=opt.output_nc, final_sigmoid=False,
                   conv_layer_order=opt.conv_order, init_channel_number=opt.init_channel_number)
    init_func = get_init_func(init_type=opt.init_type, init_gain=opt.init_gain)
    model.apply(init_func)
    model = model.cuda()
    criterion = get_loss_criterion(name='bdc', ignore_index=None, reducetion='mean',
                                   use_batch=True, use_sigmoid=True, smooth=0.).cuda()

    lr_scaler = hvd.size() if not opt.use_adasum else 1
    opt.lr = lr_scaler * opt.lr
    optimizer = create_optimizer_v2(model.parameters(), opt='adam', lr=opt.lr, betas=(opt.beta1, 0.999))

    optimizer = hvd.DistributedOptimizer(optimizer,
                                         named_parameters=model.named_parameters(),
                                         compression=hvd.Compression.fp16 if opt.fp16_allreduce else hvd.Compression.none,
                                         backward_passes_per_step=1,
                                         op=hvd.Adasum if opt.use_adasum else hvd.Average,
                                         gradient_predivide_factor=opt.gradient_predivide_factor)
    schedulers = create_scheduler(opt, optimizer)[0]

    # 5 broadcast the initial variable states from rank 0 to all other processes:
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(optimizer, root_rank=0)

    # 6. Modify your code to save checkpoints only on worker 0 to prevent other workers from corrupting them.
    for epoch in range(3):
        for batch_idx, data in enumerate(dataloader):
            volume = data['volume'].cuda(non_blocking=True)   # bs C D H W, C=1
            label = data['label'].cuda(non_blocking=True)     # bs C D H W, C=1
            volume_path = data['volume_path']
            label_path = data['label_path']
            output = model(volume)

            loss = criterion(output, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print('batch_idx:', batch_idx)
        schedulers.step(epoch)
        print('epoch:', epoch)


def test_generator():
    print('entering test_generator')
    for i in range(10):
        print('before yield', i)
        yield i
        print('after yield', i)


def test_get_item():
    pass


class TestGetItem:
    def __init__(self):
        self.num = 10
        self.test_list = list(range(self.num))

    def __getitem__(self, item):
        return self.test_list[item]


def test_generator_code():
    aa = test_generator()

    print_visible(aa)

    print(aa)
    print(next(aa))
    print('**'*50)
    print(next(aa))
    print('**'*50)


def test_val_dataset():
    import numpy as np
    from data.dataloads.base_dataset import TestOnePatientDataset
    from data.dataloads.trus_dataset import TestTrusDataset
    from yacs.config import CfgNode as CN
    from torch.utils.data import DataLoader
    from utils.others.img_io import show_image, show_array_3d
    opt = CN(new_allowed=True)

    opt.dataroot = './traces/datasets/prostate_daf3d_pre'
    opt.phase = 'test'
    opt.crop_size = 96
    opt.stride = 96
    opt.no_augment = False

    test_dataset = TestTrusDataset(opt)
    #  {'volume': volume, 'label': label, 'volume_path': volume_path, 'label_path': label_path}
    print('test_dataset:{}'.format(len(test_dataset)))
    for data in test_dataset:
        print(data['volume'].shape)     # (175, 224, 224)
        show_image(data['volume'][:, :, 100], title='origin image')

        one_patient_dataset = TestOnePatientDataset(data['volume'][:, :, 100], opt)
        print('one_patient_dataset:{}'.format(len(one_patient_dataset)))

        dataset_info = one_patient_dataset.get_info()   # 'crop_size' 'stride' 'origin_shape'  'pad_shape'
        dataset_volumes = one_patient_dataset.get_volume()   # 'origin_volume'  'pad_volume'
        row, column = one_patient_dataset.get_crop_num_list()

        test_dataloader = DataLoader(one_patient_dataset,
                                     batch_size=len(one_patient_dataset),
                                     shuffle=False,
                                     num_workers=8,
                                     drop_last=False)
        print('test_dataloader:{}'.format(len(test_dataloader)))
        for test_data in test_dataloader:
            print(test_data.shape)      # N C ...
            data_to_show = test_data[:, 0, ...].numpy()
            show_array_3d(data_to_show, row, column, title='crop_image')
            # 还原的时候，axis的顺序是由大到小，2D先1后0，3D是210。也就是从循环的最深层开始，逐层还原
            # concat_array = [np.concatenate(data_to_show[i*column:i*column+column], axis=1) for i in range(row)]
            # show_image(concat_array[1], title='partly concat image')
            # concat_array = np.concatenate(concat_array, 0)
            # show_image(concat_array, title='concat image')
            for kk in range(test_data.shape[1]):
                data_to_show = test_data[kk].numpy()
                show_array_3d(data_to_show, 2, 2, title='crop_image')
                break

            pass
        break


def main():
    # test_val_dataset()
    data_path = r'/home/lf/raid_lf/PROJECT/DLForPytorch/traces/results/' \
                r'trus_unet3d_DDP_SynBN_crop128_bs3x4_ch32_dc_adam_1e-4/test/' \
                r'slide_test_pad_noaug/65_net_trus_unet3d_DDP_SynBN_crop128_bs3x4_ch32_dc_adam_1e-4id-3.h5'
    patient_id = re.match(r'^/(?:.+/)*((\d+).*)\.h5$', data_path).groups()[-1]
    fr = h5py.File(data_path, 'r')
    label = fr.get('label')[:]
    segment = fr.get('segment')[:]
    volume = fr.get('pad_volume')[:]
    fr.close()
    show_volume_label_predict(volume.transpose((2, 1, 0)),
                              label.transpose((2, 1, 0)),
                              segment.transpose((2, 1, 0)),
                              True,
                              row=3, col=2, title=f'test on patient: {patient_id} ')

    pass


if __name__ == '__main__':
    main()

