import torch
import logging
import contextlib
import torch.distributed
import torch.nn as nn
import torch.cuda.amp

from types import SimpleNamespace
from .base_model import BaseModel
from models.loss import losses, get_loss_criterion
from models.auxiliary_funs import get_init_func, get_activation
from models.optim import create_optimizer, create_optimizer_v2
from models.scheduler import create_scheduler
from utils.others.metrics import BinaryMetrics, SoftMetrics
from utils.others.distributed_utils import reduce_mean
from utils.others.utils import print_numpy
from models.modules.segmentation.three_d.unet3d_V0 import UNet3D as UNetV0
from models.modules.segmentation.three_d.unet3d_V1 import UNet3D as UNetV1
from models.modules.segmentation.three_d.unet3d_V2 import UNet3D as UNetV2
from models.modules.segmentation.three_d.unet3d_V3 import UNet3D as UNetV3

try:
    import apex.amp
    import apex.parallel
    # import apex.optimizers
    # from apex.fp16_utils import *
    has_apex = True
except ImportError:
    has_apex = False

try:
    import horovod.torch as hvd
    has_horovod = True
except ImportError:
    has_horovod = False


ddp_logger = logging.getLogger('ddp_logger')


def define_3dunet(opt, device):
    assert not(opt.DDP and opt.DP)
    # net = UNetV0(in_channels=opt.input_nc, out_channels=opt.output_nc,
    #              conv_layer_order=opt.conv_order, init_channel_number=opt.init_channel_number,
    #              final_sigmoid=True, use_activation=False, interpolate=True)
    net = UNetV1(in_channels=opt.input_nc, out_channels=opt.output_nc, init_features=opt.init_channel_number)  # cbr
    # net = UNetV2(in_channels=opt.input_nc, out_channels=opt.output_nc, f_maps=opt.init_channel_number,
    #              is_segmentation=False, num_levels=4, layer_order='cbr')
    # net = UNetV3(in_channels=opt.input_nc, n_classes=opt.output_nc,
    #              init_features=opt.init_channel_number, trilinear=True)

    init_func = get_init_func(init_type=opt.init_type, init_gain=opt.init_gain)
    net.apply(init_func)

    if opt.APEX and has_apex:
        if opt.SyncBatchNorm:
            net = apex.parallel.convert_syncbn_model(net).to(device)
        else:
            net = net.to(device)
        return net

    if opt.SyncBatchNorm and opt.DDP:
        ddp_logger.warning('using torch.nn.SyncBatchNorm.convert_sync_batchnorm')
        # only single gpu per process is currently supported
        net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net).to(device)
    else:
        net = net.to(device)

    if opt.DDP:
        ddp_logger.warning('using nn.parallel.DistributedDataParallel')
        # 使用DDP前，模型一定要进行初始化
        assert(torch.distributed.is_available())
        net = nn.parallel.DistributedDataParallel(module=net,
                                                  device_ids=[opt.local_gpu],  # 猜测填多个的时候每个进程都相当于DP
                                                  output_device=opt.local_gpu)
    elif opt.DP:
        ddp_logger.warning('using nn.parallel.DataParallel')
        # 必须先to(device)，再用DP封装
        assert(torch.cuda.is_available())
        net = nn.parallel.DataParallel(module=net,
                                       device_ids=opt.gpu_ids,
                                       output_device=opt.gpu_ids[0])    # 默认都用0号
        ddp_logger.warning('ending to use nn.parallel.DataParallel')
    else:
        ddp_logger.warning('It seems do not use the parallel mode')
    return net


class Unet3dModel(BaseModel):
    def __init__(self, opt):
        super(Unet3dModel, self).__init__(opt)

        self.model_names = ['segment']
        self.net_segment = define_3dunet(opt, self.device)
        self.finally_activate = get_activation('sigmoid').to(self.device)

        self.loss_names = ['seg']
        if self.isTrain:
            other_loss_kwargs = {}
            # (sample_weight)   (gamma_neg gamma_pos clip)  (num_splits)  (activate)  (bce_smooth)
            self.criterion = get_loss_criterion(name=opt.loss_name,
                                                ignore_index=opt.ignore_index, reduction=opt.reduction,
                                                eps=opt.loss_eps, smooth=opt.loss_smooth,
                                                alpha=opt.loss_alpha, beta=opt.loss_beta,
                                                gamma=opt.loss_gamma, weight=opt.loss_weight,
                                                **other_loss_kwargs).to(self.device)
            optimizer_kwargs = {'eps': 1e-8,
                                'betas': (opt.beta1, 0.999)
                                }
            if 'sgd' in opt.optimizer_name.lower():
                optimizer_kwargs.pop('betas', None)
            self.optimizer = create_optimizer_v2(self.net_segment.parameters(),
                                                 opt=opt.optimizer_name,
                                                 lr=opt.lr,
                                                 weight_decay=opt.weight_decay,
                                                 momentum=opt.momentum,
                                                 **optimizer_kwargs)

            self.optimizers.append(self.optimizer)
            self.schedulers = [create_scheduler(opt, optimizer)[0] for optimizer in self.optimizers]
            if opt.APEX and has_apex:
                self.net_segment, self.optimizers = apex.amp.initialize(self.net_segment,
                                                                        self.optimizers,
                                                                        opt_level=opt.APEX_opt_level)
                ddp_logger.error(f'apex init: {opt.local_rank}, {opt.DDP}')
                if opt.DDP:
                    self.net_segment = apex.parallel.DistributedDataParallel(self.net_segment, delay_allreduce=True)
                print(f'apex ddp: {opt.local_rank}')

        # specify the images you want to save/display.
        self.visual_names = ['predict', 'label', 'volume']
        self.metric_names = ['DC', 'recall', 'precision', 'specificity', 'accuracy']

        self.get_metrics = BinaryMetrics()
        self.get_metrics_soft = SoftMetrics(smooth=0., eps=1e-6)

        self.volume = None
        self.label = None
        self.predict = None
        self.loss_seg = None
        self.metrics = None
        # setattr(self, opt.loss_name, None)

        self.autocast_context = torch.cuda.amp.autocast if opt.use_mixed_precision else contextlib.nullcontext
        self.no_sync_context = self.net_segment.no_sync if opt.DDP else contextlib.nullcontext
        self.scaler = torch.cuda.amp.GradScaler()

        self.is_activated = False

    def warp_horovod_optimizer(self):
        if not has_horovod:
            raise RuntimeError('you do not have horovod, please install')
        # for ind in range(len(self.optimizers)):
        #     self.optimizers[ind] = hvd.DistributedOptimizer(optimizer=self.optimizers[ind],
        #                                                     named_parameters=None,
        #                                                     compression=hvd.Compression.none,
        #                                                     backward_passes_per_step=1,
        #                                                     op=hvd.Average,
        #                                                     gradient_predivide_factor=1.0,
        #                                                     num_groups=0,
        #                                                     groups=None,
        #                                                     sparse_as_dense=False)
        self.optimizers = []
        self.optimizer = hvd.DistributedOptimizer(optimizer=self.optimizer,
                                                  named_parameters=self.net_segment.named_parameters(),
                                                  compression=hvd.Compression.none,
                                                  backward_passes_per_step=1,
                                                  op=hvd.Average,
                                                  gradient_predivide_factor=1.0,
                                                  num_groups=0,
                                                  groups=None,
                                                  sparse_as_dense=False)
        # ['Compression', 'FP16Compressor', 'NoneCompressor']
        # hvd.compression.Compression
        self.optimizers.append(self.optimizer)
        self.schedulers = [create_scheduler(self.opt, optimizer)[0] for optimizer in self.optimizers]

    def broadcast_horovod_parameters(self):
        if not has_horovod:
            raise RuntimeError('you do not have horovod, please install')
        hvd.broadcast_parameters(self.net_segment.state_dict(), root_rank=0)
        hvd.broadcast_optimizer_state(self.optimizer, root_rank=0)

    def set_input(self, input):

        self.volume = input['volume'].to(self.device)   # bs C D H W, C=1
        self.label = input['label'].to(self.device)     # bs C D H W, C=1
        self.volume_path = input['volume_path']
        self.label_path = input['label_path']
        if self.opt.DEBUG:
            print('proportion: {:.2%}'.format(input['label'].sum()/input['label'].numpy().size))

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        with self.autocast_context():
            self.predict = self.net_segment(self.volume)
        self.is_activated = False

    def backward(self):
        with self.autocast_context():
            self.loss_seg = self.criterion(self.predict, self.label)
        self.loss_seg = self.loss_seg / self.opt.gradient_accumulation_k_step

        if self.opt.use_mixed_precision:
            self.scaler.scale(self.loss_seg).backward()
            self.scaler.step(self.optimizer)  # maybe apply to all optimizers
            self.scaler.update()
        else:
            self.loss_seg.backward()

    def optimize_parameters(self, update=True):
        if update:
            self.forward()
            self.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
        else:
            with self.no_sync_context():
                self.forward()
                self.backward()

    def optimize_parameters_with_apex(self):
        self.optimizer.zero_grad()
        self.predict = self.net_segment(self.volume)
        self.loss_seg = self.criterion(self.predict, self.label)
        with apex.amp.scale_loss(self.loss_seg, self.optimizers) as scaled_loss:
            scaled_loss.backward()
        self.optimizer.step()

    def compute_visuals(self):
        if not self.is_activated:
            self.predict = self.finally_activate(self.predict)
            self.is_activated = True

    def compute_metrics(self, *args, **kwargs):
        if not self.is_activated:
            self.predict = self.finally_activate(self.predict)
            self.is_activated = True

        keys = tuple(self.metric_names) + args

        predict = self.predict.clone().detach()
        label = self.label.clone().detach()
        predict = (predict > 0.5).float()
        label = (label > 0.5).float()
        self.metrics = self.get_metrics_soft(predict, label, *self.metric_names, *args, **kwargs)

        if self.opt.DDP:
            for i in range(len(self.metrics)):
                if isinstance(self.metrics[i], torch.Tensor):
                    self.metrics[i] = reduce_mean(self.metrics[i], torch.distributed.get_world_size())
        if self.opt.HOROVOD:
            pass
        self.metric_dict = dict(zip(keys, self.metrics))


def main():
    from configs.options.dataset_network import ProjectOptions
    opt = ProjectOptions().parse(True)   # get training options
    model = Unet3dModel(opt)
    opt.continue_train = True
    model.setup(opt)


if __name__ == '__main__':
    main()
