import os
import contextlib
import sys
import time
import tqdm
import logging
import numpy as np
import torch.distributed

from data import create_dataset
from models import create_model
from utils.forLogs import Visualizer, get_logger
from utils.others.utils import Timer
from utils.others.utils import init_seed, init_torch
from utils.others.distributed_utils import record_distribute_ddp, torch_distributed_zero_first
from configs.utils_config import pretty_print_opt, get_pretty_opt
from configs.simple_options import get_opt
# from configs.options.promise_3dunet import TrainOptions
from configs.options.dataset_network import ProjectOptions


# 不能用两层映射，在DDP的时候，rank=1，必须用device号也为1的gpu；否则会阻塞不动。。。。。。
def set_local_gpu(args):
    if not args.DDP:
        # args.local_rank = args.gpu_ids[0] if args.gpu_ids else -1
        args.local_gpu = args.gpu_ids[0] if args.gpu_ids else - 1
    elif args.dist_url == 'env://':
        args.local_rank = int(os.environ["LOCAL_RANK"])
        assert args.local_rank >= 0, 'LOCAL_RANK must >= 0'
        args.local_gpu = args.gpu_ids[args.local_rank]
    else:
        args.local_gpu = args.gpu_ids[args.local_rank]

    if args.local_gpu >= 0 and torch.cuda.is_available():
        torch.cuda.set_device(args.local_gpu)   # setup default cuda device

    return args


def train():
    # 一个进程一个train
    # opt = ProjectOptions().parse(True)   # get training options
    # opt = get_opt(args=None)
    # opt = get_opt(args=['--config_path=configs/defaults/trus_unet3d.yaml', '--use_config'])
    opt = get_opt(args=['--config_path=configs/defaults/trus_unet3d.yaml', '--use_config', '--use_current_local_rank'])

    init_torch(gpu_id=opt.visible_gpu, deterministic=opt.deterministic)
    assert torch.backends.cudnn.enabled, "Amp requires cudnn backend to be enabled."

    do_train(opt)


def do_train(opt):
    print('now is in do_train, if you are using DDP, please make sure that '
          'you had got (dist_backend, dist_url, world_size, rank, local_rank) ready')

    # 修补一个参数不能PicklingError的bug
    opt.random_state = np.random.RandomState(seed=opt.seed)

    if opt.DDP and torch.distributed.is_available():
        torch.distributed.init_process_group(backend=opt.dist_backend,
                                             init_method=opt.dist_url,
                                             world_size=opt.world_size,
                                             rank=opt.rank)
        print('backend:{}, dist_method:{}'.format(repr(torch.distributed.get_backend()), opt.dist_url))
        print('local_rank:{}, rank:{}, world_size:{}'.format(opt.local_rank,
                                                             torch.distributed.get_rank(),
                                                             torch.distributed.get_world_size()))
        # print(opt.dist_backend, opt.dist_url)
        torch.cuda.empty_cache()
        # 通过这一步把初始化后的rank等参数存入opt，统一不同框架的用法
        opt = record_distribute_ddp(opt)

    # setup default cuda device, 配合tensor.cuda()使用
    opt = set_local_gpu(opt)
    print('local_gpu:{}'.format(opt.local_gpu))
    on_master = (not opt.DDP) or (opt.DDP and opt.rank == 0)
    init_seed(opt.seed + (opt.rank if opt.DDP else 0))

    # setting ddp_logger
    ddp_logger = get_logger(logname='ddp_logger', level=logging.INFO if on_master else logging.WARNING, is_save=False,
                            fmt="[%(process)d][%(filename)s][%(funcName)s]%(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    # ddp_logger.warning(get_pretty_opt(opt))

    dataloader = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataloader)    # get the number of images in the dataset.
    ddp_logger.warning('The number of training images = %d' % dataset_size)

    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks
    ddp_logger.warning('model get ready')

    optimize_parameters = model.optimize_parameters_with_apex if opt.APEX else model.optimize_parameters
    save_networks = model.save_for_apex if opt.APEX else model.save_networks

    if on_master:
        visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
        ddp_logger.info('visualizer get ready')
        if opt.draw_model:
            [visualizer.draw_model_graph(net, shape=[4, 1, 128, 128, 128]) for net in model.get_models()]
    else:
        visualizer = None

    if opt.DDP:
        torch.distributed.barrier()
    # # new
    # visualizer = None
    # with torch_distributed_zero_first(opt.rank):
    #     visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
    #     ddp_logger.info('visualizer get ready')
    #     if opt.draw_model:
    #         [visualizer.draw_model_graph(net, shape=[4, 1, 128, 128, 128]) for net in model.get_models()]

    ddp_logger.warning('start training! on local_rank:{}'.format(opt.local_rank))

    total_iters = 0                # the total number of training iterations
    for epoch in range(opt.epoch_start, opt.num_epochs + 1):
        if epoch == 1 and opt.continue_train is False and opt.DDP is True:
            ddp_logger.info('saving networks and than load!')
            # 保证每个进程的网络初始权重相同
            load_networks = model.load_for_apex if opt.APEX else model.load_networks
            base_patten = '%s_net_apex_%s.pth' if opt.APEX else '%s_net_%s.pth'
            weitht_name = base_patten % (epoch, opt.name)
            weitht_path = os.path.join(opt.checkpoints_dir, opt.name, weitht_name)
            with torch_distributed_zero_first(opt.local_rank):
                save_networks(epoch)
            load_networks(weitht_path)
            torch.distributed.barrier()

        epoch_start_time = time.time()

        # 更新dataloader的seed和优化器的学习率
        if not opt.serial_batches:
            dataloader.set_epoch(epoch)
        model.update_learning_rate(epoch)   # update learning rates in the beginning/ending of every epoch.

        # 训练一个epoch
        epoch_iter = 0
        iter_data_time = time.time()
        for batch_idx, data in enumerate(dataloader):
            iter_start_time = time.time()

            total_iters += opt.batch_size
            epoch_iter += opt.batch_size

            model.set_input(data)
            optimize_parameters()

            # 计算当前训练数据的metrics、losses、lrs， 使用visualizer绘图并打印
            if total_iters % opt.print_freq == 0 or total_iters % opt.plot_freq == 0:
                # 因为要reduce结果，所以全部进程都要进行计算
                t_data = iter_start_time - iter_data_time
                t_comp = (time.time() - iter_start_time) / opt.batch_size

                model.compute_metrics()         # done reduce
                metrics = model.get_current_metrics()      # done reduce
                # print(str(metrics).replace('basic_metrics', 'tp fn tn fp'))

                losses = model.get_current_losses()   # done reduce

                if on_master:
                    lrs = model.get_current_lrs()   # 学习率不需要reduce

                    if total_iters % opt.print_freq == 0:
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

            # 获取当前训练数据的预测结果，使用visualizer展示图片；依据iter保存checkpoint
            if on_master:
                if total_iters % opt.display_freq == 0:
                    # don't need to reduce
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
                    save_networks(save_suffix)

            torch.cuda.synchronize()
            iter_data_time = time.time()
            # pbar.update(float(opt.batch_size*100)/dataset_size)

        # 用测试数据测试当前epoch训练完后的模型性能
        # TODO: do_test
        if opt.test_on_train and epoch % opt.val_epoch_freq == 0:
            # TODO:实现用测试数据测试结果，并返回一定的指标，以便后续通过指标来保存checkpoint
            test_epoch(dataloader, model, visualizer, opt)

        # pbar.close()
        ddp_logger.info('End of epoch %d / %d \t Time Taken: %d sec' %
                        (epoch, opt.num_epochs, time.time() - epoch_start_time))

        # 按照epoch保存checkpoint
        # TODO：根据需要保存optimizer和apex的state_dict
        if on_master and epoch > opt.save_epoch_start and (epoch-opt.save_epoch_start) % opt.save_epoch_freq == 0 and \
                not opt.DEBUG:
            # cache our model every <save_epoch_freq> epochs
            ddp_logger.warning('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            save_networks('latest')
            save_networks(epoch)
            visualizer.add_text(opt.name, f'saving checkpoint on {epoch}', total_iters)

    ddp_logger.info('end training!')

    if visualizer:
        visualizer.close()
    ddp_logger.info('visualizer closed!')

    # 尝试修复一个进程不完全关闭的bug
    if opt.DDP:
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()

    sys.exit(0)


def test_epoch(dataloader, model, visualizer, opt):
    pass


def train_epoch(dataloader, model, visualizer, opt):
    pass


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

