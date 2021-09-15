import sys
import time
import tqdm
import numpy as np
import torch.distributed

# from configs.options.promise_3dunet import TrainOptions
from configs.options.trus_unet3d import ProjectOptions
from data import create_dataset
from models import create_model
from utils.forLogs.visualizer import Visualizer
from utils.others.utils import Timer
#  没有用同步BN


# TODO: 在DDP的时候，需要给不同进程设置不同的seed
def global_configuration(seed, visible_gpu='0,1,2,3'):
    from utils.others.utils import init_seed, init_torch
    init_seed(seed)
    init_torch(gpu_id=visible_gpu, deterministic=True)


if __name__ == '__main__':
    opt = ProjectOptions().parse(True)   # get training options
    opt.random_state = np.random.RandomState(seed=opt.seed)
    assert not(opt.DP and opt.DDP)
    print('option get ready')

    not_on_master = opt.DDP and (opt.local_rank != 0)  # not opt.DDP or opt.local_rank == 0
    on_master = (not opt.DDP) or (opt.DDP and opt.local_rank == 0)      # dist.get_rank() == 0

    global_configuration(opt.seed + opt.local_rank, visible_gpu=opt.visible_gpu)

    if opt.DDP:
        # import os
        # os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2'   # 设置本程序可用的gpus
        # torch.cuda.set_device(gpu_id)  # 设置本程序默认的gpu,配合tensor.cuda()使用
        # torch.cuda.empty_cache()
        print('local_rank:', opt.local_rank, 'world_size:', opt.world_size)
        # print(opt.dist_backend, opt.dist_url)
        torch.distributed.init_process_group(backend=opt.dist_backend,
                                             init_method=opt.dist_url,
                                             world_size=opt.world_size,
                                             rank=opt.local_rank
                                             )
        print('rank:', torch.distributed.get_rank())

    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)    # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size)

    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    print('model get ready')
    if on_master:
        visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
        print('visualizer get ready')
    else:
        visualizer = None

    if visualizer and opt.draw_model and not opt.DP and not opt.DDP:
        nets = model.get_models()
        for net in nets:
            visualizer.draw_model_graph(net, shape=[4, 1, 128, 128, 128])

    total_iters = 0                # the total number of training iterations
    for epoch in range(opt.epoch_start, opt.num_epochs + 1):
        pbar = tqdm.tqdm(total=100)
        epoch_start_time = time.time()
        iter_data_time = time.time()
        dataset.set_epoch(epoch)
        epoch_iter = 0
        for i, data in enumerate(dataset):
            iter_start_time = time.time()

            total_iters += opt.batch_size
            epoch_iter += opt.batch_size

            model.set_input(data)
            model.optimize_parameters()

            if on_master:
                if total_iters % opt.plot_freq == 0:
                    lrs = model.get_current_lrs()
                    for lr_i, lr in enumerate(lrs):
                        visualizer.plot_one_scalar(lr, total_iters, name=str(lr_i+1), tag='lrs')

                    model.compute_metrics()
                    metrics = model.get_current_metrics()
                    print(str(metrics).replace('basic_metrics', 'tp fn tn fp'))
                    visualizer.add_hparams({'lr': lrs[0]}, dict(metrics),
                                           name=f'result on epoch{epoch}', global_step=total_iters)

                    for key, value in metrics.items():
                        if isinstance(value, tuple):
                            metrics.pop(key)
                    visualizer.plot_current_losses(epoch, float(epoch_iter)/dataset_size, metrics, total_iters,
                                                   tag='metrics over time')

                if total_iters % opt.print_freq == 0:
                    # if opt.DEBUG:
                    #     print(data.size())
                    losses = model.get_current_losses()
                    t_comp = (time.time() - iter_start_time) / opt.batch_size
                    t_data = iter_start_time - iter_data_time
                    visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                    visualizer.plot_current_losses(epoch, float(epoch_iter)/dataset_size, losses, total_iters)
                    # for key, value in losses.items():
                    #     visualizer.plot_one_scalar(value, total_iters, key)

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
                                # TODO: video似乎看不出来什么东西，暂且关闭，以便运行快一些
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
                    print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                    save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                    model.save_networks(save_suffix)
            iter_data_time = time.time()
            pbar.update(float(opt.batch_size*100)/dataset_size)
        pbar.close()
        model.update_learning_rate(epoch)
        # update learning rates in the beginning/ending of every epoch.
        if on_master:
            if epoch > opt.save_epoch_start and (epoch-opt.save_epoch_start) % opt.save_epoch_freq == 0:
                # cache our model every <save_epoch_freq> epochs
                print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
                model.save_networks('latest')
                model.save_networks(epoch)
                visualizer.add_text(opt.name, f'saving checkpoint on {epoch}', total_iters)

        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.num_epochs, time.time() - epoch_start_time))
    if visualizer:
        visualizer.close()
    sys.exit(0)


# import logging
#
# # 给主要进程（rank=0）设置低输出等级，给其他进程设置高输出等级。
# logging.basicConfig(level=logging.INFO if rank in [-1, 0] else logging.WARN)
# # 普通log，只会打印一次。
# logging.info("This is an ordinary log.")
# # 危险的warning、error，无论在哪个进程，都会被打印出来，从而方便debug。
# logging.error("This is a fatal log!")
