'''
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

from data import create_dataset
from models import create_model
from utils.forLogs import Visualizer, get_logger
from utils.others.utils import init_seed, init_torch
from data.dataloads.trus_dataset import TrusDataset
from data.utils_data import nii_loader
from models.modules.segmentation.three_d.unet3d_V0 import UNet3D
from models.loss.region_based import BinaryDiceLoss
from configs.simple_options import get_opt
from configs.utils_config import pretty_print_opt
from models.auxiliary_funs import get_init_func, get_activation
from models.loss import get_loss_criterion
from models.optim import create_optimizer, create_optimizer_v2
from models.scheduler import create_scheduler
from utils.others.distributed_utils_horovod import reduce_mean, metric_average


def train_on_batch():
    '''
    :return:返回训练的指标
    '''
    pass


def test_on_batch():
    '''
    :return: 返回测试的指标
    '''
    pass


def predict_on_batch():
    '''
    :return: 返回测试值
    '''
    pass


def train_epoch(epoch, dataloader, model, visualizer, opt):
    ddp_logger = logging.getLogger('ddp_logger')

    epoch_iter = 0
    total_iters = (epoch - 1) * opt.train_size
    if not opt.serial_batches:
        dataloader.set_epoch(epoch)
    model.update_learning_rate(epoch)   # update learning rates in the beginning/ending of every epoch.

    iter_data_time = time.time()
    for batch_idx, data in enumerate(dataloader):
        iter_start_time = time.time()

        total_iters += opt.batch_size
        epoch_iter += opt.batch_size

        model.set_input(data)
        model.optimize_parameters()

        if total_iters % opt.print_freq == 0 or total_iters % opt.plot_freq == 0:

            t_data = iter_start_time - iter_data_time
            t_comp = (time.time() - iter_start_time) / opt.batch_size

            # TODO: reduce metrics
            model.compute_metrics()
            metrics = model.get_current_metrics()
            # print(str(metrics).replace('basic_metrics', 'tp fn tn fp'))

            losses = model.get_current_losses()

            if hvd.rank() == 0:
                lrs = model.get_current_lrs()

                if total_iters % opt.print_freq == 0:

                    visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                    visualizer.print_current_metrics(metrics, epoch, epoch_iter)
                    visualizer.add_hparams({'lr': lrs[0]}, dict(metrics),
                                           name=f'result on epoch{epoch}', global_step=total_iters)

                if total_iters % opt.plot_freq == 0:
                    for lr_i, lr in enumerate(lrs):
                        visualizer.plot_one_scalar(lr, total_iters, name=str(lr_i+1), tag='lrs')

                    visualizer.plot_current_losses(epoch, float(epoch_iter)/opt.train_size, losses, total_iters)
                    # for key, value in losses.items():
                    #     visualizer.plot_one_scalar(value, total_iters, key)
                    for key, value in metrics.items():
                        if isinstance(value, tuple):
                            metrics.pop(key)
                    visualizer.plot_current_losses(epoch, float(epoch_iter)/opt.train_size, metrics, total_iters,
                                                   tag='metrics over time')

        if hvd.rank() == 0:
            # don't need to reduce
            if total_iters % opt.display_freq == 0:
                model.compute_visuals()
                visuals = model.get_current_visuals()
                # ['predict', 'label']
                if opt.display_histogram:
                    for name, image in visuals.items():
                        visualizer.add_histogram(name, image, total_iters)

                for name, image in visuals.items():
                    if image.ndim == 5:  # N C D H W
                        N, C, D, H, W = image.shape
                        for c in range(C):
                            if opt.play_video:
                                visualizer.play_current_video(torch.unsqueeze(image[:, c], dim=2),
                                                              total_iters, tag=name+'video')
                            for d in range(D):
                                for n in range(N):
                                    visualizer.show_current_images({name+'N:{} C:{} D:{}'.format(n, c, d): image[n, c, d]}, total_iters)

            if total_iters > opt.save_iter_start and (total_iters-opt.save_iter_start) % opt.save_iter_freq == 0 and not opt.DEBUG:
                ddp_logger.warning('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)
                model.save_optimizer(save_suffix)

        iter_data_time = time.time()


def test_epoch(epoch, dataloader, model, visualizer, opt):
    pass


def predict_epoch():
    pass


class MainProcess:
    def __init__(self, opt):
        self.ddp_logger = logging.getLogger('ddp_logger')
        self.opt = opt
        self.train_loader = self._get_train_loader()
        self.opt.train_size = len(self.train_loader)
        self.test_loader = self._get_test_loader()
        self.opt.test_size = len(self.test_loader)
        self.model = create_model(opt)

        self.model.warp_horovod_optimizer()     # create optimizer
        if hvd.rank == 0:
            self.model.setup(opt)   # load weights
            self.model.load_optimizer()
        self.model.broadcast_horovod_parameters()   # broadcast optimizer and network

        # self.optimizer = None
        # hvd.synchronize()
        if hvd.rank() == 0:
            self.visualizer = Visualizer(opt)
            if opt.draw_model:
                [self.visualizer.draw_model_graph(net, shape=[4, 1, 128, 128, 128]) for net in self.model.get_models()]
        else:
            self.visualizer = None

    def do_task(self):
        self.ddp_logger.warning('start training!')
        total_iters = 0
        for epoch in range(self.opt.epoch_start, self.opt.num_epochs + 1):
            epoch_start_time = time.time()

            train_epoch(epoch, self.train_loader, self.model, self.visualizer, self.opt)
            if self.opt.test_on_train and epoch % self.opt.val_epoch_freq == 0:
                test_epoch(epoch, self.test_loader, self.model, self.visualizer, self.opt)

            self.ddp_logger.info('End of epoch %d / %d \t Time Taken: %d sec' %
                                 (epoch, self.opt.num_epochs, time.time() - epoch_start_time))

            total_iters += self.opt.train_size

            if epoch > self.opt.save_epoch_start \
                    and (epoch-self.opt.save_epoch_start) % self.opt.save_epoch_freq == 0 \
                    and hvd.rank() == 0 and not self.opt.DEBUG:
                # cache our model every <save_epoch_freq> epochs
                self.ddp_logger.warning('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
                self.model.save_networks('latest')
                self.model.save_networks(epoch)
                self.model.save_optimizer('latest')
                self.model.save_optimizer(epoch)
                self.visualizer.add_text(self.opt.name, f'saving checkpoint on {epoch}', total_iters)

        self.ddp_logger.warning('end training!')

    def _get_train_loader(self):
        dataloader = create_dataset(self.opt)
        return dataloader

    def _get_test_loader(self):
        dataset = TrusDataset(self.opt)
        self.opt.test_size = len(dataset)
        sampler = torch.utils.data.distributed.DistributedSampler(dataset,
                                                                  num_replicas=hvd.size(),
                                                                  rank=hvd.rank(),
                                                                  shuffle=not self.opt.serial_batches,
                                                                  seed=0,
                                                                  drop_last=False) if self.opt.horovod else None
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.opt.batch_size,
            shuffle=(sampler is None) and (not self.opt.serial_batches),
            sampler=sampler,        #
            batch_sampler=None,     #
            num_workers=int(self.opt.num_threads),
            collate_fn=None,        #
            pin_memory=True,
            drop_last=False,        #
            prefetch_factor=2       #
        )
        return dataloader

    def __del__(self):
        if self.visualizer:
            self.visualizer.close()


def repair_opt_horovod(opt):
    # disable the DP and DDP
    opt.DP = False
    opt.DDP = False
    opt.horovod = True
    opt.use_adasum = False
    opt.fp16_allreduce = False
    opt.gradient_predivide_factor = 1

    # restore the local_rank
    opt.world_size = hvd.size()
    opt.local_size = hvd.local_size()
    opt.rank = hvd.rank()
    opt.local_rank = hvd.local_rank()
    opt.local_gpu = opt.gpu_ids[opt.local_rank] if opt.gpu_ids else -1

    return opt


def main():
    opt = get_opt(args=['--config_path=configs/defaults/trus_unet3d_horovod.yaml', '--use_config'])

    init_torch(gpu_id=opt.visible_gpu, deterministic=True)
    hvd.init()
    print('hvd nccl_built:', hvd.nccl_built())
    # torch.cuda.empty_cache()

    opt = repair_opt_horovod(opt)
    pretty_print_opt(opt)

    if torch.cuda.is_available():
        torch.cuda.set_device(opt.local_gpu)

    ddp_logger = get_logger(logname='ddp_logger',  is_save=False,
                            level=logging.INFO if hvd.rank() == 0 else logging.WARNING,
                            fmt="[%(process)d][%(filename)s][%(funcName)s]%(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    ddp_logger.warning('local_rank:{}, rank:{}, world_size:{}'.format(hvd.local_rank(),
                                                                      hvd.rank(),
                                                                      hvd.size()))
    init_seed(opt.seed + hvd.rank())

    processer = MainProcess(opt)
    processer.do_task()


def debug():
    pass


if __name__ == '__main__':
    main()
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
