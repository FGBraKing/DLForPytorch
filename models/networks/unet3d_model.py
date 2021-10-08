import torch
import logging
import torch.distributed
import torch.nn as nn

from .base_model import BaseModel
from models.modules import UNet3D
from models.loss import losses, get_loss_criterion

from models.auxiliary_funs import get_init_func, get_activation
from models.optim import create_optimizer, create_optimizer_v2
from models.scheduler import create_scheduler
from utils.others.metrics import BinaryMetrics, SoftMetrics
from utils.others.distributed_utils import reduce_mean


ddp_logger = logging.getLogger('ddp_logger')


def define_3dunet(opt, device):
    assert not(opt.DDP and opt.DP)
    net = UNet3D(in_channels=opt.input_nc, out_channels=opt.output_nc, final_sigmoid=False,
                 conv_layer_order=opt.conv_order, init_channel_number=opt.init_channel_number)
    # init_net(net, opt.init_type, opt.init_gain, opt.gpu_ids)
    init_func = get_init_func(init_type=opt.init_type, init_gain=opt.init_gain)
    net.apply(init_func)

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
                                                  device_ids=[opt.local_rank],  # 猜测填多个的时候每个进程都相当于DP
                                                  output_device=opt.local_rank)
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

        self.loss_names = ['dice']
        if self.isTrain:
            self.criterion = get_loss_criterion(name='bdc', ignore_index=None, reducetion='mean',
                                                use_batch=True, use_sigmoid=True, smooth=0.).to(self.device)
            self.optimizer = create_optimizer_v2(self.net_segment.parameters(), opt='adam', lr=opt.lr,
                                                 betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer)
            self.schedulers = [create_scheduler(opt, optimizer)[0] for optimizer in self.optimizers]
        # specify the images you want to save/display.
        self.visual_names = ['predict', 'label']
        self.metric_names = ['DC', 'recall', 'precision', 'accuracy']

        self.get_metrics = BinaryMetrics()
        self.get_metrics_soft = SoftMetrics(smooth=0., eps=1e-9)

        self.volume = None
        self.label = None
        self.predict = None
        self.loss_dice = None
        self.metrics = None

    def set_input(self, input):
        self.volume = input['volume'].to(self.device)   # bs C D H W, C=1
        self.label = input['label'].to(self.device)     # bs C D H W, C=1
        self.volume_path = input['volume_path']
        self.label_path = input['label_path']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.predict = self.net_segment(self.volume)
        # self.predict = self.finally_activate(m)

    def backward(self):
        self.loss_dice = self.criterion(self.predict, self.label)
        self.loss_dice.backward()

    def optimize_parameters(self):
        self.forward()
        self.optimizer.zero_grad()
        self.backward()
        self.optimizer.step()

    def compute_visuals(self):
        self.predict = self.finally_activate(self.predict)

    def compute_metrics(self, *args, **kwargs):
        keys = tuple(self.metric_names) + args

        # old version by numpy
        # predict = self.predict.clone().detach().cpu().numpy()    # bs C D H W, C=1
        # label = self.label.clone().detach().cpu().numpy()
        # metrics = self.get_metrics(predict, label, *self.metric_names, *args, **kwargs)
        # print(dict(zip(keys, metrics)))

        predict = self.predict.clone().detach()
        label = self.label.clone().detach()
        predict = (predict > 0.5).float()
        label = (label > 0.5).float()
        self.metrics = self.get_metrics_soft(predict, label, *self.metric_names, *args, **kwargs)
        # print(type(self.metrics[0]), self.metrics[0].dtype, self.metrics[0].device,
        #       self.metrics[0].grad, self.metrics[0].grad_fn)

        # print(dict(zip(keys, self.metrics)))
        if self.opt.DDP:
            torch.distributed.barrier()
            for i in range(len(self.metrics)):
                if isinstance(self.metrics[i], torch.Tensor):
                    self.metrics[i] = reduce_mean(self.metrics[i], torch.distributed.get_world_size())
        self.metric_dict = dict(zip(keys, self.metrics))


def main():
    from configs.options.trus_unet3d import ProjectOptions
    opt = ProjectOptions().parse(True)   # get training options
    model = Unet3dModel(opt)
    opt.continue_train = True
    model.setup(opt)


if __name__ == '__main__':
    main()
