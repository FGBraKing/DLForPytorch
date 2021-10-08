'''
horovod的local_rank是由程序提供并使用的，因此不能利用双重映射使用gpu_ids,应该在使用前通过环境变量指定可以使用的gpu_id。
使用horovodrun 启动器启动脚本
'''

import os
import sys
import time
import logging
import argparse
import torch
import torch.optim
import torch.utils.data
import torch.distributed
import numpy as np
import horovod.torch as hvd

from configs.options.trus_unet3d import ProjectOptions
from data import create_dataset
from models import create_model
from utils.forLogs import Visualizer, get_logger
from utils.others.utils import init_seed, init_torch
from data.dataloads.trus_dataset import TrusDataset
from data.utils_data import nii_loader
from models.modules.segmentation.three_d.unet3d_gn import UNet3D
from models.loss.region_based import BinaryDiceLoss
from configs.simple_options import get_opt
from configs.utils_config import pretty_print_opt
from models.auxiliary_funs import get_init_func, get_activation
from models.loss import losses, get_loss_criterion
from models.optim import create_optimizer, create_optimizer_v2
from models.scheduler import create_scheduler


#  与DDP异同
#  init方式不同， data相同，DDP封装模型horovod封装优化器
def metric_average(val, name):
    tensor = torch.tensor(val)
    avg_tensor = hvd.allreduce(tensor, name=name)
    return avg_tensor.item()


def reduce_mean(tensor, nprocs):
    rt = tensor.clone()
    hvd.allreduce(rt, name='barrier')
    # # horovod.allreduce calculates the average value by default
    # # https://github.com/tczhangzhi/pytorch-distributed/issues/14
    # rt /= nprocs
    return rt


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

    # TODO: 需要在配置文件修改的参数
    # opt.use_adasum = False
    # opt.fp16_allreduce = False
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


def train():
    pass


class MainProcess:
    def __init__(self):
        pass

    def do_task(self):
        opt = self._get_option()
        dataloader = self._get_dataloader(opt)
        model = self._get_model(opt)
        visualizer = self._get_visualizer(opt)

    def __del__(self):
        self.visualizer.close()




def main():
    processer = MainProcess()
    processer.do_task()


if __name__ == '__main__':
    train()
    debug()


#  horovod,53
# ['Adasum',
#  'Average',
#  'Compression',
#  'DistributedOptimizer',
#  'Sum',
#  'SyncBatchNorm',
#  'allgather',
#  'allgather_async',
#  'allgather_object',
#  'allreduce',
#  'allreduce_async',
#  'alltoall',
#  'alltoall_async',
#  'broadcast',
#  'broadcast_async',
#  'broadcast_object',
#  'broadcast_optimizer_state',
#  'broadcast_parameters',
#  'ccl_built',
#  'check_extension',
#  'compression',
#  'cross_rank',
#  'cross_size',
#  'cuda_built',
#  'ddl_built',
#  'elastic',
#  'functions',
#  'gloo_built',
#  'gloo_enabled',
#  'grouped_allreduce',
#  'grouped_allreduce_async',
#  'init',
#  'is_initialized',
#  'join',
#  'local_rank',
#  'local_size',
#  'mpi_built',
#  'mpi_enabled',
#  'mpi_lib_v2',
#  'mpi_ops',
#  'mpi_threads_supported',
#  'nccl_built',
#  'optimizer',
#  'poll',
#  'rank',
#  'rocm_built',
#  'shutdown',
#  'size',
#  'sparse_allreduce_async',
#  'start_timeline',
#  'stop_timeline',
#  'sync_batch_norm',
#  'synchronize']
