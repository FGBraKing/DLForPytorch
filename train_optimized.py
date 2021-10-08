'''
使用DDP时，使用gpu_ids作为运行的GPU号码和数量
优化版本，可以完整使用gpu_ids.也可以从script直接运行DDP
'''
import os
import torch.multiprocessing as mp

from configs.options.trus_unet3d import ProjectOptions
from utils.others.utils import init_seed, init_torch
from train import do_train


# 维护rank，local_rank
def correct_args(ind, args):
    # 当ind=-1，表示没有使用DDP训练，或者使用了DDP，但使用环境变量提供
    if ind == -1:
        # using environment variables to initialize or not in DDP
        args.rank = -1
        args.local_rank = args.gpu_ids[0] if args.gpu_ids else -1
    elif args.dist_url == 'env://' and args.rank == -1:
        args.rank = int(os.environ["RANK"]) + ind
        args.local_rank = args.gpu_ids[ind]
    else:
        args.rank = args.rank + ind
        args.local_rank = args.gpu_ids[ind]

    return args


def train(ind, *args):
    '''
    :param ind:process id, from 0 to len(opt.gpu_ids), when ind=-1, means not DDP
    :param args:
    :return:
    '''
    opt = args[0]
    init_torch(gpu_id=opt.visible_gpu, deterministic=True)

    opt = correct_args(ind, opt)

    do_train(opt)


def train_ddp(args):
    if args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])
    assert args.world_size > 0, 'world_size{} have to > 0'.format(args.world_size)
    assert len(args.gpu_ids) > 0, 'gpu_ids{} have to specified'.format(args.gpu_ids)
    nprocs = min(args.world_size, len(args.gpu_ids))
    if len(args.gpu_ids) > 0:
        mp.spawn(fn=train,
                 args=(args,),
                 nprocs=nprocs,
                 join=True,
                 daemon=False)
    else:
        raise ValueError('when use ddp, you must provide the correct gpu_ids,'
                         ' but got [] of {}'.format(repr(args.gpu_ids)))


def main():
    opt = ProjectOptions().parse(True)   # get training options
    print('option get ready')

    if opt.DDP:
        train_ddp(opt)
    else:
        train(-1, opt)


if __name__ == "__main__":
    main()

    # # using environment variables to initialize or not in DDP
    # warnings.warn('you are trying to use environment variables to initialize the DDP, please try to use the utils '
    #               'that torch.distributed.launch to run script. or simply run script on multi-shell')
