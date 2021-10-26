import torch
import horovod.torch as hvd


def reduce_mean(tensor, nprocs):
    rt = tensor.clone()
    hvd.allreduce(rt, name='barrier')
    # # horovod.allreduce calculates the average value by default
    # # https://github.com/tczhangzhi/pytorch-distributed/issues/14
    # rt /= nprocs
    return rt


def metric_average(val, name):
    tensor = torch.tensor(val)
    avg_tensor = hvd.allreduce(tensor, name=name)
    return avg_tensor.item()
