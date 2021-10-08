""" Distributed training/validation utils

Hacked together by / Copyright 2020 Ross Wightman
"""
import torch
from torch import distributed as dist

from .model import unwrap_model


def reduce_tensor(tensor, n):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= n
    return rt


def distribute_bn(model, world_size, reduce=False):
    # ensure every node has the same running bn stats
    for bn_name, bn_buf in unwrap_model(model).named_buffers(recurse=True):
        if ('running_mean' in bn_name) or ('running_var' in bn_name):
            if reduce:
                # average bn stats across whole group
                dist.all_reduce(bn_buf, op=dist.ReduceOp.SUM)
                bn_buf /= float(world_size)
            else:
                # broadcast bn stats from rank 0 to whole group
                dist.broadcast(bn_buf, 0)


def distribute_concat(tensor, num_total_examples):
    output_tensors = [tensor.clone() for _ in range(dist.get_world_size())]
    dist.all_gather(output_tensors, tensor)
    concat = torch.cat(output_tensors, dim=0)
    return concat[:num_total_examples]

