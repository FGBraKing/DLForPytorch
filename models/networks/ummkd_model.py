import torch
import torch.nn as nn
import torch.nn.functional as F

from models.networks.base_model import BaseModel
from models.modules.segmentation.two_d.ummkd2d import Ummkd2dMod, get_l2_Norm, KDLoss
from models.loss import losses
from models.auxiliary_funs import get_init_func, get_scheduler
from models.slovers import get_optimizer
from utils.others.metrics import MutiClassMetrics, expand_as_one_hot


def define_ummkd2d(opt, device):
    assert not(opt.DDP and opt.DP)
    net = Ummkd2dMod(names=['source', 'target'], in_channels=opt.input_nc, out_channels=opt.output_nc,
                     keep_prob=opt.keep_prob, feature_base=opt.feature_base,
                     down_times=opt.down_times, n_class=opt.n_class, batch_size=opt.batch_size)
    # init_net(net, opt.init_type, opt.init_gain, opt.gpu_ids)
    init_func = get_init_func(init_type=opt.init_type, init_gain=opt.init_gain)
    net.apply(init_func)
    if False:
        net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net).to(device)
    else:
        net = net.to(device)
    if opt.DDP:
        assert(torch.cuda.is_available())
        net = nn.parallel.DistributedDataParallel(module=net,
                                                  device_ids=[opt.local_rank],  # 猜测填多个的时候每个进程都相当于DP
                                                  output_device=opt.local_rank)
    elif opt.DP:
        assert(torch.cuda.is_available())
        net = nn.parallel.DataParallel(module=net,
                                       device_ids=opt.gpu_ids,
                                       output_device=opt.gpu_ids[0])    # 默认都用0号
    else:
        print('It seems do not use the parallel mode')
    return net


class UmmkdModel(BaseModel):
    def __init__(self, opt):
        super(UmmkdModel, self).__init__(opt)

        self.model_names = ['jointSeg']
        self.net_jointSeg = define_ummkd2d(opt, self.device)

        # self.finally_activate = nn.Softmax(dim=1).to(self.device)

        self.loss_names = ['diceA', 'diceB', 'wceA', 'wceB', 'kd', 'l2']

        if self.isTrain:
            self.criterionDice = losses.MutiClassDiceLoss(smooth=0., normalization=True)
            self.criterionSeg = getattr(self.net_jointSeg, '_get_segmentation_cost')
            # self.net_jointSeg._get_segmentation_cost
            self.criterionL2 = get_l2_Norm  # partial(F.mse_loss, reduction='sum')
            self.criterionKD = KDLoss(n_class=self.opt.n_class).to(self.device)

            self.optimizer = get_optimizer(self.net_jointSeg.parameters(), 'adam', lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer)
            self.schedulers = [get_scheduler(optimizer, opt) for optimizer in self.optimizers]
        # specify the images you want to save/display.
        self.visual_names = ['predictA', 'labelA', 'predictB', 'labelB']
        self.metric_names = ['dice', 'recall', 'precision', 'accuracy']

        self.get_metrics = MutiClassMetrics()

    def set_input(self, input):
        self.volumeA = input['volumeA'].to(self.device)   # bs C D H W, C=1
        self.labelA = input['labelA'].to(self.device)     # bs C D H W, C=1
        self.labelA_onehot = expand_as_one_hot(self.labelA, self.opt.n_class)
        self.volumeB = input['volumeB'].to(self.device)   # bs C D H W, C=1
        self.labelB = input['labelB'].to(self.device)     # bs C D H W, C=1
        self.labelB_onehot = expand_as_one_hot(self.labelB, self.opt.n_class)
        self.data_pathA = input['pathA']
        self.data_pathB = input['pathB']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.logitA = self.net_jointSeg(self.volumeA, name='source')
        self.logitB = self.net_jointSeg(self.volumeB, name='target')
        self.probA = F.softmax(self.logitA, dim=1, _stacklevel=5)
        self.probB = F.softmax(self.logitB, dim=1, _stacklevel=5)

    def backward(self):
        self.loss_diceA = self.criterionDice(self.probA, self.labelA, expand=True)
        _, self.loss_wceA = self.criterionSeg(self.logitA, self.labelA_onehot)
        self.loss_diceB = self.criterionDice(self.probB, self.labelB, expand=True)
        _, self.loss_wceB = self.criterionSeg(self.logitB, self.labelB_onehot)
        self.loss_kd = self.criterionKD(self.logitA, self.labelA_onehot, self.logitB, self.labelB_onehot)
        self.loss_l2 = self.criterionL2(self.net_jointSeg.parameters())

        self.loss_all = (self.loss_diceA + self.loss_diceB) * self.opt.miu_seg_dice + \
                        (self.loss_wceA + self.loss_wceB) * self.opt.miu_seg_ce + \
                        (self.loss_kd * self.opt.miu_kd + self.loss_l2 * self.opt.miu_seg_L2_norm)
        self.loss_all.backward()

    def optimize_parameters(self):
        self.forward()
        self.optimizer.zero_grad()
        self.backward()
        self.optimizer.step()

    def compute_visuals(self):
        pass

    def compute_metrics(self, *args, **kwargs):
        predictA = self.logitA.clone().detach().cpu().numpy()    # bs C D H W, C=1
        labelA = self.labelA_onehot.clone().detach().cpu().numpy()
        self.metricsA = self.get_metrics(predictA, labelA, *self.metric_names, *args, **kwargs)
        keys = tuple(self.metric_names) + args
        self.metric_dictA = dict(zip(keys, self.metricsA))

        predictB = self.logitB.clone().detach().cpu().numpy()    # bs C D H W, C=1
        labelB = self.labelB_onehot.clone().detach().cpu().numpy()
        self.metricsB = self.get_metrics(predictB, labelB, *self.metric_names, *args, **kwargs)
        keys = tuple(self.metric_names) + args
        self.metric_dictB = dict(zip(keys, self.metricsB))

    def get_models(self):
        nets = []
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net_' + name)
                nets.append(net)
        return nets

    def compute_training_dice(self):
        predictA_compact = torch.argmax(self.probA, dim=1)
        predictB_compact = torch.argmax(self.probB, dim=1)
        diceA = self.net_jointSeg._eval_dice_during_train(self.labelA_onehot, predictA_compact)
        diceB = self.net_jointSeg._eval_dice_during_train(self.labelB_onehot, predictB_compact)


def main():
    from configs.options.promise_3dunet import TrainOptions
    opt = TrainOptions().parse()   # get training options
    model = UmmkdModel(opt)
    opt.continue_train = True
    model.setup(opt)


if __name__ == '__main__':
    main()

