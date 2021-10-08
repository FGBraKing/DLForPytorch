'''
当没有使用DDP时，使用gpu_ids[0]；当用了DDP时，使用命令行中的local_rank或者环境变量中的local_rank；
这个是旧版本，没有很好地利用到gpu_ids。但不影响使用，需要使用torch.distributed.launch从shell运行
'''
import os
import contextlib
import sys
import time
import tqdm
import logging
import numpy as np
import torch.distributed

# from configs.options.promise_3dunet import TrainOptions
from configs.options.trus_unet3d import ProjectOptions
from data import create_dataset
from models import create_model
from utils.forLogs import Visualizer, get_logger
from utils.others.utils import Timer
from utils.others.utils import init_seed, init_torch
# from contextlib import nullcontext

# try:
#     from contextlib import nullcontext
# except ModuleNotFoundError as e:
#     from contextlib import suppress as nullcontext


@contextlib.contextmanager
def torch_distributed_zero_first(rank: int):
    if rank not in [-1, 0]:
        torch.distributed.barrier()
    yield
    if rank == 0:
        torch.distributed.barrier()


def repair_local_rank(args):
    # 需要维护的参数：local_rank

    if not args.DDP:
        args.local_rank = args.gpu_ids[0] if args.gpu_ids else -1
    elif args.dist_url == 'env://':
        args.local_rank = int(os.environ["LOCAL_RANK"])
        assert args.local_rank >= 0, 'LOCAL_RANK must >= 0'
        args.local_rank = args.gpu_ids[args.local_rank]
    else:
        args.local_rank = args.gpu_ids[args.local_rank]

    return args


def train():
    # 一个进程一个train
    # TODO: 需要维护的参数：dist_url,world_size,rank,local_rank
    opt = ProjectOptions().parse(True)   # get training options
    print('option get ready')

    init_torch(gpu_id=opt.visible_gpu, deterministic=True)

    opt = repair_local_rank(opt)

    do_train(opt)


def do_train(opt):
    '''
    :param opt:
    local_rank: it is local_device
    :return:
    '''

    # 设置本程序默认的gpu,配合tensor.cuda()使用
    if opt.local_rank >= 0 and torch.cuda.is_available():
        torch.cuda.set_device(opt.local_rank)   # setup default cuda device

    if opt.DDP and torch.distributed.is_available():
        torch.distributed.init_process_group(backend=opt.dist_backend,
                                             init_method=opt.dist_url,
                                             world_size=opt.world_size,
                                             rank=opt.rank
                                             )
        print('local_rank:{}, rank:{}, world_size:{}'.format(opt.local_rank,
                                                             torch.distributed.get_rank(),
                                                             torch.distributed.get_world_size()))
        # print(opt.dist_backend, opt.dist_url)
        torch.cuda.empty_cache()

    on_master = (not opt.DDP) or (opt.DDP and torch.distributed.get_rank() == 0)

    # setting ddp_logger
    ddp_logger = get_logger(logname='ddp_logger', level=logging.INFO if on_master else logging.WARNING, is_save=False,
                            fmt="[%(process)d][%(filename)s][%(funcName)s]%(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    init_seed(opt.seed + (torch.distributed.get_rank() if opt.DDP else 0))

    dataloader = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataloader)    # get the number of images in the dataset.
    ddp_logger.info('The number of training images = %d' % dataset_size)

    model = create_model(opt)      # create a model given opt.model and other options

    if opt.DDP:
        torch.distributed.barrier()

    model.setup(opt)               # regular setup: load and print networks; create schedulers
    ddp_logger.warning('model get ready')

    if opt.DDP:
        torch.distributed.barrier()

    if on_master:
        visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
        ddp_logger.warning('visualizer get ready')
    else:
        visualizer = None

    if visualizer and opt.draw_model:
        nets = model.get_models()
        for net in nets:
            visualizer.draw_model_graph(net, shape=[4, 1, 128, 128, 128])

    if opt.DDP:
        torch.distributed.barrier()

    ddp_logger.warning('start training!')
    total_iters = 0                # the total number of training iterations
    for epoch in range(opt.epoch_start, opt.num_epochs + 1):
        # pbar = tqdm.tqdm(total=100)
        epoch_start_time = time.time()
        iter_data_time = time.time()

        model.update_learning_rate(epoch)   # update learning rates in the beginning/ending of every epoch.

        if not opt.serial_batches:
            dataloader.set_epoch(epoch)
        epoch_iter = 0
        for batch_idx, data in enumerate(dataloader):
            iter_start_time = time.time()

            total_iters += opt.batch_size
            epoch_iter += opt.batch_size

            model.set_input(data)
            model.optimize_parameters()

            if total_iters % opt.print_freq == 0:

                t_data = iter_start_time - iter_data_time
                t_comp = (time.time() - iter_start_time) / opt.batch_size

                model.compute_metrics()         # done reduce
                metrics = model.get_current_metrics()      # done reduce
                # print(str(metrics).replace('basic_metrics', 'tp fn tn fp'))

                losses = model.get_current_losses()

                if on_master:
                    lrs = model.get_current_lrs()
                    visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                    visualizer.print_current_metrics(metrics, epoch, epoch_iter)
                    visualizer.add_hparams({'lr': lrs[0]}, dict(metrics),
                                           name=f'result on epoch{epoch}', global_step=total_iters)

                    if total_iters % opt.plot_freq == 0:
                        for lr_i, lr in enumerate(lrs):
                            visualizer.plot_one_scalar(lr, total_iters, name=str(lr_i+1), tag='lrs')

                        visualizer.plot_current_losses(epoch, float(epoch_iter)/dataset_size, losses, total_iters)
                        # for key, value in losses.items():
                        #     visualizer.plot_one_scalar(value, total_iters, key)
                        for key, value in metrics.items():
                            if isinstance(value, tuple):
                                metrics.pop(key)
                        visualizer.plot_current_losses(epoch, float(epoch_iter)/dataset_size, metrics, total_iters,
                                                       tag='metrics over time')

            if on_master:
                # don't need to reduce
                if total_iters % opt.display_freq == 0:
                    model.compute_visuals()
                    visuals = model.get_current_visuals()
                    # ['predict', 'label']
                    if opt.display_histogram:
                        for name, image in visuals.items():
                            visualizer.add_histogram(name, image, total_iters)

                    # visuals_refine = {}
                    for name, image in visuals.items():
                        if image.ndim == 5:  # N C D H W
                            N, C, D, H, W = image.shape
                            for c in range(C):
                                if opt.play_video:
                                    visualizer.play_current_video(torch.unsqueeze(image[:, c], dim=2),
                                                                  total_iters, tag=name+'video')
                                for d in range(D):
                                    # visualizer.show_current_images_v2(name+f'N:{d} C{c}',image[:,c:c+1,d],total_iters)
                                    for n in range(N):
                                        visualizer.show_current_images({name+'N:{} C:{} D:{}'.format(n, c, d): image[n, c, d]}, total_iters)
                    #                     visuals_refine[name+'N:{} C:{} D:{}'.format(n, c, d)] = image[n, c, d]
                    #     else:
                    #         visuals_refine[name] = image
                    # visualizer.show_current_images(visuals_refine, total_iters)

                if total_iters > opt.save_iter_start and (total_iters-opt.save_iter_start) % opt.save_iter_freq == 0:
                    ddp_logger.warning('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                    save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                    model.save_networks(save_suffix)

            iter_data_time = time.time()
            # pbar.update(float(opt.batch_size*100)/dataset_size)

        if on_master:
            if epoch > opt.save_epoch_start and (epoch-opt.save_epoch_start) % opt.save_epoch_freq == 0:
                # cache our model every <save_epoch_freq> epochs
                ddp_logger.warning('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
                model.save_networks('latest')
                model.save_networks(epoch)
                visualizer.add_text(opt.name, f'saving checkpoint on {epoch}', total_iters)

        # TODO: do_test
        if opt.test_on_train and epoch % opt.val_epoch_freq == 0:
            pass

        # pbar.close()
        ddp_logger.info('End of epoch %d / %d \t Time Taken: %d sec' %
                        (epoch, opt.num_epochs, time.time() - epoch_start_time))
    ddp_logger.warning('end training!')
    if visualizer:
        visualizer.close()
    ddp_logger.warning('visualizer closed!')
    # TODO: bug,提前结束的话，由于各个进程运行进度不一致，会导致先前向完成的进程无法传数据给未完成的进程，导致最后一个epoch训练失败
    if opt.DDP:
        torch.distributed.barrier()
    ddp_logger.warning('barrier ending!')
    if opt.DDP:
        torch.distributed.destroy_process_group()

    # if not on_master:
    #     torch.distributed.barrier()     # 尝试修复此bug
    # if on_master:
    #     torch.distributed.barrier()
    sys.exit(0)


if __name__ == '__main__':
    train()


#
# # 给主要进程（rank=0）设置低输出等级，给其他进程设置高输出等级。
# logging.basicConfig(level=logging.INFO if rank in [-1, 0] else logging.WARN)
# logging.info("This is an ordinary log.")
# # 危险的warning、error，无论在哪个进程，都会被打印出来，从而方便debug。
# logging.error("This is a fatal log!")
# 设置device
# torch.cuda.set_device(opt.local_rank)
# with torch.cuda.device(args.local_rank):...


# 梯度累计，加速小技巧
# optimizer.zero_grad()
# for i, (data, label) in enumerate(dataloader):
#     my_context = model.no_sync if local_rank != -1 and i%k != 0 else nullcontext
#     with my_context():
#         prediction = model(data)
#         loss_fn(prediction, label).backward()
#     if i%k ==0:
#         optimizer.step()
#         optimizer.zero_grad()


# torch.cuda.synchronize()  # CPU和GPU的运行同步，
# torch.distributed.barrier()  # 多个GPU间的运行同步

