import os
import torch
import itertools
from .base_model import BaseModel
from models.modules.style_transfer.two_d.MUNIT_network import ContentEncoder, Decoder, MsImageDis
from models.auxiliary_funs import init_net
from ..loss import losses
from torch import nn as nn
from torch.optim import lr_scheduler
from torch.autograd import Variable
from models.modules.style_transfer.two_d.MUNIT_network import load_vgg16
from data.transforms.transformOnTensor import vgg_preprocess
from models.auxiliary_funs import get_model_list


def get_scheduler(optimizer, hyperparameters, iterations=-1):

    if hyperparameters['lr_policy'] == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + hyperparameters['epoch_count'] - hyperparameters['n_epochs'])\
                   / float(hyperparameters['n_epochs_decay'] + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif hyperparameters['lr_policy'] == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=hyperparameters['lr_decay_iters'],
                                        gamma=hyperparameters['gamma'], last_epoch=iterations)  # gamma=0.1
    elif hyperparameters['lr_policy'] == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif hyperparameters['lr_policy'] == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=hyperparameters['n_epochs'], eta_min=0)
    elif hyperparameters['lr_policy'] == 'constant':
        scheduler = None  # constant scheduler
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', hyperparameters['lr_policy'])
    return scheduler


class CriterionRecon(nn.Module):
    def __init__(self, cal_type='l1'):
        super(CriterionRecon, self).__init__()
        self.cal_type = cal_type

    def forward(self, in_data, target):
        if self.cal_type == 'l1':
            return torch.mean(losses.l1(in_data, target))
        elif self.cal_type == 'l2':
            return torch.mean(losses.l2(in_data, target))
        else:
            raise NotImplementedError('Generator loss name [%s] is not recognized' % self.cal_type)


def _compute_k1(mu):
    mu_2 = torch.pow(mu, 2)
    encoding_loss = torch.mean(mu_2)
    return encoding_loss


class UNIT_Model(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(no_dropout=True)
        if is_train:
            parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
            parser.add_argument('--lambda_identity', type=float, default=0.5, help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')
        return parser

    def __init__(self, opt):
        super(UNIT_Model, self).__init__(opt)

    def set_initialize(self, hyperparameters):
        self.hyperparameters = hyperparameters
        # define the models
        if 1:
            if self.isTrain:
                self.model_names = ['_enc_a', '_dec_a', '_enc_b', '_dec_b', '_dis_a', '_dis_b']
            else:  # during test time, only load Gs
                self.model_names = ['_enc_a', '_dec_a', '_enc_b', '_dec_b']
            # define networks (both Generators and discriminators)
            self.net_enc_a = ContentEncoder(
                n_downsample=hyperparameters['n_downsample'],
                n_res=hyperparameters['n_res'],
                input_dim=hyperparameters['input_dim_a'],
                dim=hyperparameters['dim'],
                norm='in',
                activ=hyperparameters['activ'],
                pad_type=hyperparameters['pad_type']
            )
            self.net_enc_b = ContentEncoder(
                n_downsample=hyperparameters['n_downsample'],
                n_res=hyperparameters['n_res'],
                input_dim=hyperparameters['input_dim_b'],
                dim=hyperparameters['dim'],
                norm='in',
                activ=hyperparameters['activ'],
                pad_type=hyperparameters['pad_type']
            )
            self.net_dec_a = Decoder(
                n_upsample=hyperparameters['n_downsample'],
                n_res=hyperparameters['n_res'],
                dim=hyperparameters['dim'],
                output_dim=hyperparameters['input_dim_a'],
                res_norm='in',
                activ=hyperparameters['activ'],
                pad_type=hyperparameters['pad_type']
            )
            self.net_dec_b = Decoder(
                n_upsample=hyperparameters['n_downsample'],
                n_res=hyperparameters['n_res'],
                dim=hyperparameters['dim'],
                output_dim=hyperparameters['input_dim_b'],
                res_norm='in',
                activ=hyperparameters['activ'],
                pad_type=hyperparameters['pad_type']
            )
            init_net(self.net_enc_a, init_type=hyperparameters['init'], init_gain=hyperparameters['init_gain'], gpu_ids=self.gpu_ids)
            init_net(self.net_enc_b, init_type=hyperparameters['init'], init_gain=hyperparameters['init_gain'], gpu_ids=self.gpu_ids)
            init_net(self.net_dec_a, init_type=hyperparameters['init'], init_gain=hyperparameters['init_gain'], gpu_ids=self.gpu_ids)
            init_net(self.net_dec_b, init_type=hyperparameters['init'], init_gain=hyperparameters['init_gain'], gpu_ids=self.gpu_ids)
            # load descrimeter
            if self.isTrain:  # define discriminators
                self.net_dis_a = MsImageDis(
                    input_dim=hyperparameters['input_dim_a'],
                    n_layer=hyperparameters['n_layer'],
                    gan_type=hyperparameters['gan_type'],
                    dim=hyperparameters['dim'],
                    norm=hyperparameters['norm'],
                    activ=hyperparameters['activ'],
                    num_scales=hyperparameters['num_scales'],
                    pad_type=hyperparameters['pad_type'])
                self.net_dis_b = MsImageDis(
                    input_dim=hyperparameters['input_dim_b'],
                    n_layer=hyperparameters['n_layer'],
                    gan_type=hyperparameters['gan_type'],
                    dim=hyperparameters['dim'],
                    norm=hyperparameters['norm'],
                    activ=hyperparameters['activ'],
                    num_scales=hyperparameters['num_scales'],
                    pad_type=hyperparameters['pad_type'])
                init_net(self.net_dis_a, init_type=hyperparameters['gaussian'], init_gain=0.02, gpu_ids=self.gpu_ids)
                init_net(self.net_dis_b, init_type=hyperparameters['gaussian'], init_gain=0.02, gpu_ids=self.gpu_ids)
            self.instancenorm = nn.InstanceNorm2d(512, affine=False)
            # Load VGG model if needed
            if 'vgg_w' in hyperparameters.keys() and hyperparameters['vgg_w'] > 0:
                self.vgg = load_vgg16(hyperparameters['vgg_model_path'] + '/models')
                self.vgg.eval()
                for param in self.vgg.parameters():
                    param.requires_grad = False
        # define the losses
        if 1:
            self.loss_names = ['gen_recon_a', 'gen_recon_b',
                               'gen_recon_k1_a', 'gen_recon_k1_b',
                               'cycle_a', 'cycle_b',
                               'cycle_gan_k1_a', 'cycle_gan_k1_b',
                               'dis_a', 'dis_b',
                               'vgg_a', 'vgg_b']
            # self.dis_criterion
            # self.vgg_criterion
            self.recon_criterion = CriterionRecon('l1')
            self.__compute_k1 = _compute_k1
        # define the optimizers
        if self.isTrain:
            lr = hyperparameters['lr']
            beta1 = hyperparameters['beta1']
            beta2 = hyperparameters['beta2']
            dis_params = itertools.chain(self.net_dis_a.parameters(), self.net_dis_b.parameters())
            # gen_params = list(self.gen_a.parameters()) + list(self.gen_b.parameters())
            gen_params = itertools.chain(self.net_enc_a.parameters(), self.net_dec_a.parameters(),
                                         self.net_enc_b.parameters(), self.net_dec_b.parameters())
            self.dis_opt = torch.optim.Adam([p for p in dis_params if p.requires_grad],
                                            lr=lr, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])
            self.gen_opt = torch.optim.Adam([p for p in gen_params if p.requires_grad],
                                            lr=lr, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])
            self.optimizers.append(self.dis_opt)
            self.optimizers.append(self.gen_opt)
            self.dis_scheduler = get_scheduler(self.dis_opt, hyperparameters)
            self.gen_scheduler = get_scheduler(self.gen_opt, hyperparameters)

    def set_input(self, input):
        self.real_A = input['A'].to(self.device)
        self.real_B = input['B'].to(self.device)
        self.A_paths = input['A_paths']
        self.B_paths = input['B_paths']

    def forward(self):
        self.eval()
        h_a = self.net_enc_a(self.real_A)
        h_b = self.net_enc_b(self.real_B)
        x_ba = self.net_dec_a(h_b)
        x_ab = self.net_dec_b(h_a)
        self.train()
        return x_ab, x_ba

    def gen_update(self, x_a, x_b, hyperparameters):
        self.gen_opt.zero_grad()
        # encode
        h_a = self.net_enc_a(x_a)
        n_a = Variable(torch.randn(h_a.size()).cuda(h_a.data.get_device()))
        h_b = self.net_enc_b(x_b)
        n_b = Variable(torch.randn(h_b.size()).cuda(h_b.data.get_device()))
        # decode (within domain)
        x_a_recon = self.net_dec_a(h_a + n_a)
        x_b_recon = self.net_dec_b(h_b + n_b)
        # decode (cross domain)
        x_ba = self.net_dec_a(h_b + n_b)
        x_ab = self.net_dec_b(h_a + n_a)
        # encode again
        h_b_recon = self.net_enc_a(x_ba)
        n_b_recon = Variable(torch.randn(h_b_recon.size()).cuda(h_b_recon.data.get_device()))
        h_a_recon = self.net_enc_b(x_ab)
        n_a_recon = Variable(torch.randn(h_a_recon.size()).cuda(h_a_recon.data.get_device()))
        # decode again (if needed)
        x_aba = self.net_dec_a(h_a_recon + n_a_recon) if hyperparameters['recon_x_cyc_w'] > 0 else None
        x_bab = self.net_dec_b(h_b_recon + n_b_recon) if hyperparameters['recon_x_cyc_w'] > 0 else None
        # reconstruction loss
        self.loss_gen_recon_x_a = self.recon_criterion(x_a_recon, x_a)
        self.loss_gen_recon_x_b = self.recon_criterion(x_b_recon, x_b)
        self.loss_gen_recon_kl_a = self.__compute_kl(h_a)
        self.loss_gen_recon_kl_b = self.__compute_kl(h_b)
        self.loss_gen_cyc_x_a = self.recon_criterion(x_aba, x_a)
        self.loss_gen_cyc_x_b = self.recon_criterion(x_bab, x_b)
        self.loss_gen_recon_kl_cyc_aba = self.__compute_kl(h_a_recon)
        self.loss_gen_recon_kl_cyc_bab = self.__compute_kl(h_b_recon)
        # GAN loss
        self.loss_gen_adv_a = self.net_dis_a.calc_gen_loss(x_ba)
        self.loss_gen_adv_b = self.net_dis_b.calc_gen_loss(x_ab)
        # domain-invariant perceptual loss
        self.loss_gen_vgg_a = self.compute_vgg_loss(self.vgg, x_ba, x_b) if hyperparameters['vgg_w'] > 0 else 0
        self.loss_gen_vgg_b = self.compute_vgg_loss(self.vgg, x_ab, x_a) if hyperparameters['vgg_w'] > 0 else 0
        # total loss
        self.loss_gen_total = hyperparameters['gan_w'] * self.loss_gen_adv_a + \
                              hyperparameters['gan_w'] * self.loss_gen_adv_b + \
                              hyperparameters['recon_x_w'] * self.loss_gen_recon_x_a + \
                              hyperparameters['recon_kl_w'] * self.loss_gen_recon_kl_a + \
                              hyperparameters['recon_x_w'] * self.loss_gen_recon_x_b + \
                              hyperparameters['recon_kl_w'] * self.loss_gen_recon_kl_b + \
                              hyperparameters['recon_x_cyc_w'] * self.loss_gen_cyc_x_a + \
                              hyperparameters['recon_kl_cyc_w'] * self.loss_gen_recon_kl_cyc_aba + \
                              hyperparameters['recon_x_cyc_w'] * self.loss_gen_cyc_x_b + \
                              hyperparameters['recon_kl_cyc_w'] * self.loss_gen_recon_kl_cyc_bab + \
                              hyperparameters['vgg_w'] * self.loss_gen_vgg_a + \
                              hyperparameters['vgg_w'] * self.loss_gen_vgg_b

        self.loss_gen_total.backward()
        self.gen_opt.step()

    def dis_update(self, x_a, x_b, hyperparameters):
        self.dis_opt.zero_grad()
        # encode
        h_a = self.net_enc_a(x_a)
        n_a = Variable(torch.randn(h_a.size()).cuda(h_a.data.get_device()))
        h_b = self.net_enc_b(x_b)
        n_b = Variable(torch.randn(h_b.size()).cuda(h_b.data.get_device()))
        # decode (cross domain)
        x_ba = self.net_dec_a(h_b + n_b)
        x_ab = self.net_dec_b(h_a + n_a)
        # D loss
        self.loss_dis_a = self.net_dis_a.calc_dis_loss(x_ba.detach(), x_a)
        self.loss_dis_b = self.net_dis_b.calc_dis_loss(x_ab.detach(), x_b)
        self.loss_dis_total = hyperparameters['gan_w'] * self.loss_dis_a + hyperparameters['gan_w'] * self.loss_dis_b
        self.loss_dis_total.backward()
        self.dis_opt.step()

    def compute_vgg_loss(self, vgg, img, target):
        img_vgg = vgg_preprocess(img)
        target_vgg = vgg_preprocess(target)
        img_fea = vgg(img_vgg)
        target_fea = vgg(target_vgg)
        return torch.mean((self.instancenorm(img_fea) - self.instancenorm(target_fea)) ** 2)

    def sample(self, x_a, x_b):
        self.eval()
        x_a_recon, x_b_recon, x_ba, x_ab = [], [], [], []
        for i in range(x_a.size(0)):
            h_a = self.net_enc_a(x_a[i].unsqueeze(0))
            h_b = self.net_enc_b(x_b[i].unsqueeze(0))
            x_a_recon.append(self.net_dec_a(h_a))
            x_b_recon.append(self.net_dec_b(h_b))
            x_ba.append(self.net_dec_a(h_b))
            x_ab.append(self.net_dec_b(h_a))
        x_a_recon, x_b_recon = torch.cat(x_a_recon), torch.cat(x_b_recon)
        x_ba = torch.cat(x_ba)
        x_ab = torch.cat(x_ab)
        self.train()
        return x_a, x_a_recon, x_ab, x_b, x_b_recon, x_ba

    def update_learning_rate(self):
        if self.dis_scheduler is not None:
            self.dis_scheduler.step()
        if self.gen_scheduler is not None:
            self.gen_scheduler.step()

    def resume(self, checkpoint_dir, hyperparameters):
        # Load generators
        last_model_name = get_model_list(checkpoint_dir, "gen")
        state_dict = torch.load(last_model_name)
        # self.gen_a.load_state_dict(state_dict['a'])
        # self.gen_b.load_state_dict(state_dict['b'])
        self.load_weight(self.net_enc_a, state_dict['encode_a'])
        self.load_weight(self.net_dec_a, state_dict['decode_a'])
        self.load_weight(self.net_enc_b, state_dict['encode_b'])
        self.load_weight(self.net_dec_b, state_dict['decode_b'])
        iterations = int(last_model_name[-11:-3])
        # Load discriminators
        last_model_name = get_model_list(checkpoint_dir, "dis")
        state_dict = torch.load(last_model_name)
        # self.dis_a.load_state_dict(state_dict['a'])
        # self.dis_b.load_state_dict(state_dict['b'])
        self.load_weight(self.net_dis_a, state_dict['a'])
        self.load_weight(self.net_dis_b, state_dict['b'])
        # Load optimizers
        state_dict = torch.load(os.path.join(checkpoint_dir, 'optimizer.pt'))
        # self.dis_opt.load_state_dict(state_dict['dis'])
        # self.gen_opt.load_state_dict(state_dict['gen'])
        self.load_weight(self.dis_opt, state_dict['dis'])
        self.load_weight(self.gen_opt, state_dict['gen'])
        # Reinitilize schedulers
        self.dis_scheduler = get_scheduler(self.dis_opt, hyperparameters, iterations)
        self.gen_scheduler = get_scheduler(self.gen_opt, hyperparameters, iterations)
        print('Resume from iteration %d' % iterations)
        return iterations

    def load_weight(self, net, state_dict):
        if isinstance(net, torch.nn.DataParallel):
            net = net.module
        net.load_state_dict(state_dict)

    def save(self, snapshot_dir, iterations):
        # Save generators, discriminators, and optimizers
        gen_name = os.path.join(snapshot_dir, 'gen_%08d.pt' % (iterations + 1))
        dis_name = os.path.join(snapshot_dir, 'dis_%08d.pt' % (iterations + 1))
        opt_name = os.path.join(snapshot_dir, 'optimizer.pt')
        torch.save({'encode_a': self.net_enc_a.state_dict(),
                    'decode_a': self.net_dec_a.state_dict(),
                    'encode_b': self.net_enc_b.state_dict(),
                    'decode_b': self.net_dec_b.state_dict()}, gen_name)
        torch.save({'a': self.net_dis_a.state_dict(), 'b': self.net_dis_b.state_dict()}, dis_name)
        torch.save({'gen': self.gen_opt.state_dict(), 'dis': self.dis_opt.state_dict()}, opt_name)

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        pass


