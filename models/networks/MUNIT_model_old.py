from models.modules.style_transfer.two_d.MUNIT_network import AdaINGen, MsImageDis, VAEGen

from models.auxiliary_funs import get_model_list
from models.modules.style_transfer.two_d.MUNIT_network import load_vgg16
from data.transforms.transformOnTensor import vgg_preprocess
from torch.autograd import Variable
import torch
import torch.nn as nn
import os
import math

from torch.optim import lr_scheduler
import itertools
from models.auxiliary_funs import init_net
from models.loss.losses import l1, l2


def define_G(net_type, input_dim, params, init_type, gpu_ids):
    # dim, style_dim, n_downsample, n_res, activ, pad_type, mlp_dim
    net = None
    if net_type == 'MUNIT':
        net = AdaINGen(input_dim,
                       dim=params['dim'],
                       style_dim=params['style_dim'],
                       n_downsample=params['n_downsample'],
                       n_res=params['n_res'],
                       activ=params['activ'],
                       pad_type=params['pad_type'],
                       mlp_dim=params['mlp_dim'])
    elif net_type == 'UNIT':
        net = VAEGen(input_dim,
                     dim=params['dim'],
                     n_downsample=params['n_downsample'],
                     n_res=params['n_res'],
                     activ=params['activ'],
                     pad_type=params['pad_type'])
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % net_type)
    return init_net(net, init_type=init_type, init_gain=math.sqrt(2), gpu_ids=gpu_ids)


def define_D(net_type, input_dim, params, init_type='gaussian', gpu_ids=[]):
    net = None
    if net_type == 'MUNIT':
        pass
    else:
        pass
    net = MsImageDis(input_dim,
                     n_layer=params['n_layer'],
                     gan_type=params['gan_type'],
                     dim=params['dim'],
                     norm=params['norm'],
                     activ=params['activ'],
                     num_scales=params['num_scales'],
                     pad_type=params['pad_type'])
    return init_net(net, init_type=init_type, init_gain=math.sqrt(2), gpu_ids=gpu_ids)


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
            return torch.mean(l1(in_data, target))
        elif self.cal_type == 'l2':
            return torch.mean(l2(in_data, target))
        else:
            raise NotImplementedError('Generator loss name [%s] is not recognized' % self.cal_type)


def _compute_k1(mu):
    mu_2 = torch.pow(mu, 2)
    encoding_loss = torch.mean(mu_2)
    return encoding_loss


class MUNITModel(nn.Module):
    def __init__(self, net_type, hyperparameters, gpu_ids=[]):
        super(MUNITModel, self).__init__()
        if len(gpu_ids) > 0:
            self.device = torch.device("cuda:{}".format(gpu_ids[0]))
        else:
            self.device = torch.device('cpu')
        self.net_type = net_type
        # fix the noise used in sampling
        display_size = int(hyperparameters['display_size'])
        self.style_dim = hyperparameters['gen']['style_dim']
        self.s_a = torch.randn(display_size, self.style_dim, 1, 1).to(self.device)
        self.s_b = torch.randn(display_size, self.style_dim, 1, 1).to(self.device)

        self.gen_a = define_G(self.net_type, hyperparameters['input_dim_a'], hyperparameters['gen'],
                              init_type=hyperparameters['init'], gpu_ids=gpu_ids)
        self.gen_b = define_G(self.net_type, hyperparameters['input_dim_b'], hyperparameters['gen'],
                              init_type=hyperparameters['init'], gpu_ids=gpu_ids)
        self.dis_a = define_D(self.net_type, hyperparameters['input_dim_a'], hyperparameters['dis'],
                              init_type='gaussian', gpu_ids=gpu_ids)
        self.dis_b = define_D(self.net_type, hyperparameters['input_dim_b'], hyperparameters['dis'],
                              init_type='gaussian', gpu_ids=gpu_ids)
        self.instancenorm = nn.InstanceNorm2d(512, affine=False)
        #
        # self.criterionCycle = CriterionRecon()
        self.recon_criterion = CriterionRecon()
        if self.net_type == 'UNIT':
            self.__compute_k1 = _compute_k1   # criterion_encoding
        # setup the optimizers
        lr = hyperparameters['lr']
        beta1 = hyperparameters['beta1']
        beta2 = hyperparameters['beta2']
        # dis_params = list(self.dis_a.parameters()) + list(self.dis_b.parameters())
        dis_params = itertools.chain(self.dis_a.parameters(), self.dis_b.parameters())
        # gen_params = list(self.gen_a.parameters()) + list(self.gen_b.parameters())
        gen_params = itertools.chain(self.gen_a.parameters(), self.gen_b.parameters())
        self.dis_opt = torch.optim.Adam([p for p in dis_params if p.requires_grad],
                                        lr=lr, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])
        self.gen_opt = torch.optim.Adam([p for p in gen_params if p.requires_grad],
                                        lr=lr, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])
        self.dis_scheduler = get_scheduler(self.dis_opt, hyperparameters)
        self.gen_scheduler = get_scheduler(self.gen_opt, hyperparameters)

        # Load VGG model if needed
        if 'vgg_w' in hyperparameters.keys() and hyperparameters['vgg_w'] > 0:
            self.vgg = load_vgg16(hyperparameters['vgg_model_path'] + '/models')
            self.vgg.eval()
            for param in self.vgg.parameters():
                param.requires_grad = False

    def forward(self, x_a, x_b):
        self.eval()
        if self.net_type == 'MUNIT':
            s_a = Variable(self.s_a)
            s_b = Variable(self.s_b)
            _, c_a, s_a_fake = self.gen_a(x_a)
            # c_a, s_a_fake = self.gen_a.encode(x_a)
            c_b, s_b_fake = self.gen_b.encode(x_b)
            c_b, s_b_fake = self.gen_b.encode(x_b)
            x_ba = self.gen_a.decode(c_b, s_a)
            x_ab = self.gen_b.decode(c_a, s_b)
        elif self.net_type == 'UNit':
            h_a, _ = self.gen_a.encode(x_a)
            h_b, _ = self.gen_b.encode(x_b)
            x_ba = self.gen_a.decode(h_b)
            x_ab = self.gen_b.decode(h_a)
        else:
            raise NotImplementedError('Generator model name [%s] is not recognized' % self.net_type)
        self.train()
        return x_ab, x_ba

    def gen_update(self, x_a, x_b, hyperparameters):
        self.gen_opt.zero_grad()
        if self.net_type == 'MUNIT':
            s_a = Variable(torch.randn(x_a.size(0), self.style_dim, 1, 1).to(self.device))
            s_b = Variable(torch.randn(x_b.size(0), self.style_dim, 1, 1).to(self.device))
            # encode
            c_a, s_a_prime = self.gen_a.encode(x_a)
            c_b, s_b_prime = self.gen_b.encode(x_b)
            # decode (within domain)
            x_a_recon = self.gen_a.decode(c_a, s_a_prime)
            x_b_recon = self.gen_b.decode(c_b, s_b_prime)
            # decode (cross domain)
            x_ba = self.gen_a.decode(c_b, s_a)
            x_ab = self.gen_b.decode(c_a, s_b)
            # encode again
            c_b_recon, s_a_recon = self.gen_a.encode(x_ba)
            c_a_recon, s_b_recon = self.gen_b.encode(x_ab)
            # decode again (if needed)
            x_aba = self.gen_a.decode(c_a_recon, s_a_prime) if hyperparameters['recon_x_cyc_w'] > 0 else None
            x_bab = self.gen_b.decode(c_b_recon, s_b_prime) if hyperparameters['recon_x_cyc_w'] > 0 else None

            # reconstruction loss
            self.loss_gen_recon_x_a = self.recon_criterion(x_a_recon, x_a)
            self.loss_gen_recon_x_b = self.recon_criterion(x_b_recon, x_b)
            self.loss_gen_recon_s_a = self.recon_criterion(s_a_recon, s_a)
            self.loss_gen_recon_s_b = self.recon_criterion(s_b_recon, s_b)
            self.loss_gen_recon_c_a = self.recon_criterion(c_a_recon, c_a)
            self.loss_gen_recon_c_b = self.recon_criterion(c_b_recon, c_b)
            self.loss_gen_cycrecon_x_a = self.recon_criterion(x_aba, x_a) if hyperparameters['recon_x_cyc_w'] > 0 else 0
            self.loss_gen_cycrecon_x_b = self.recon_criterion(x_bab, x_b) if hyperparameters['recon_x_cyc_w'] > 0 else 0
            # GAN loss
            self.loss_gen_adv_a = self.dis_a.calc_gen_loss(x_ba)
            self.loss_gen_adv_b = self.dis_b.calc_gen_loss(x_ab)
            # domain-invariant perceptual loss
            self.loss_gen_vgg_a = self.compute_vgg_loss(self.vgg, x_ba, x_b) if hyperparameters['vgg_w'] > 0 else 0
            self.loss_gen_vgg_b = self.compute_vgg_loss(self.vgg, x_ab, x_a) if hyperparameters['vgg_w'] > 0 else 0
            # total loss
            self.loss_gen_total = hyperparameters['gan_w'] * self.loss_gen_adv_a + \
                                  hyperparameters['gan_w'] * self.loss_gen_adv_b + \
                                  hyperparameters['recon_x_w'] * self.loss_gen_recon_x_a + \
                                  hyperparameters['recon_s_w'] * self.loss_gen_recon_s_a + \
                                  hyperparameters['recon_c_w'] * self.loss_gen_recon_c_a + \
                                  hyperparameters['recon_x_w'] * self.loss_gen_recon_x_b + \
                                  hyperparameters['recon_s_w'] * self.loss_gen_recon_s_b + \
                                  hyperparameters['recon_c_w'] * self.loss_gen_recon_c_b + \
                                  hyperparameters['recon_x_cyc_w'] * self.loss_gen_cycrecon_x_a + \
                                  hyperparameters['recon_x_cyc_w'] * self.loss_gen_cycrecon_x_b + \
                                  hyperparameters['vgg_w'] * self.loss_gen_vgg_a + \
                                  hyperparameters['vgg_w'] * self.loss_gen_vgg_b
        elif self.net_type == 'UNIT':
            # encode
            h_a, n_a = self.gen_a.encode(x_a)
            h_b, n_b = self.gen_b.encode(x_b)
            # decode (within domain)
            x_a_recon = self.gen_a.decode(h_a + n_a)
            x_b_recon = self.gen_b.decode(h_b + n_b)
            # decode (cross domain)
            x_ba = self.gen_a.decode(h_b + n_b)
            x_ab = self.gen_b.decode(h_a + n_a)
            # encode again
            h_b_recon, n_b_recon = self.gen_a.encode(x_ba)
            h_a_recon, n_a_recon = self.gen_b.encode(x_ab)
            # decode again (if needed)
            x_aba = self.gen_a.decode(h_a_recon + n_a_recon) if hyperparameters['recon_x_cyc_w'] > 0 else None
            x_bab = self.gen_b.decode(h_b_recon + n_b_recon) if hyperparameters['recon_x_cyc_w'] > 0 else None

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
            self.loss_gen_adv_a = self.dis_a.calc_gen_loss(x_ba)
            self.loss_gen_adv_b = self.dis_b.calc_gen_loss(x_ab)
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
        else:
            raise NotImplementedError('Generator model name [%s] is not recognized' % self.net_type)
        self.loss_gen_total.backward()
        self.gen_opt.step()

    def dis_update(self, x_a, x_b, hyperparameters):
        self.dis_opt.zero_grad()
        if self.net_type == 'MUNIT':
            s_a = Variable(torch.randn(x_a.size(0), self.style_dim, 1, 1).to(self.device))
            s_b = Variable(torch.randn(x_b.size(0), self.style_dim, 1, 1).to(self.device))
            # encode
            c_a, _ = self.gen_a.encode(x_a)
            c_b, _ = self.gen_b.encode(x_b)
            # decode (cross domain)
            x_ba = self.gen_a.decode(c_b, s_a)
            x_ab = self.gen_b.decode(c_a, s_b)
        elif self.net_type == 'UNIT':
            # encode
            h_a, n_a = self.gen_a.encode(x_a)
            h_b, n_b = self.gen_b.encode(x_b)
            # decode (cross domain)
            x_ba = self.gen_a.decode(h_b + n_b)
            x_ab = self.gen_b.decode(h_a + n_a)
        else:
            raise NotImplementedError('Generator model name [%s] is not recognized' % self.net_type)

        # D loss
        self.loss_dis_a = self.dis_a.calc_dis_loss(x_ba.detach(), x_a)
        self.loss_dis_b = self.dis_b.calc_dis_loss(x_ab.detach(), x_b)
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
        if self.net_type == 'MUNIT':
            self.eval()
            s_a1 = Variable(self.s_a)
            s_b1 = Variable(self.s_b)
            s_a2 = Variable(torch.randn(x_a.size(0), self.style_dim, 1, 1).to(self.device))
            s_b2 = Variable(torch.randn(x_b.size(0), self.style_dim, 1, 1).to(self.device))
            x_a_recon, x_b_recon, x_ba1, x_ba2, x_ab1, x_ab2 = [], [], [], [], [], []
            for i in range(x_a.size(0)):
                c_a, s_a_fake = self.gen_a.encode(x_a[i].unsqueeze(0))
                c_b, s_b_fake = self.gen_b.encode(x_b[i].unsqueeze(0))
                x_a_recon.append(self.gen_a.decode(c_a, s_a_fake))
                x_b_recon.append(self.gen_b.decode(c_b, s_b_fake))
                x_ba1.append(self.gen_a.decode(c_b, s_a1[i].unsqueeze(0)))
                x_ba2.append(self.gen_a.decode(c_b, s_a2[i].unsqueeze(0)))
                x_ab1.append(self.gen_b.decode(c_a, s_b1[i].unsqueeze(0)))
                x_ab2.append(self.gen_b.decode(c_a, s_b2[i].unsqueeze(0)))
            x_a_recon, x_b_recon = torch.cat(x_a_recon), torch.cat(x_b_recon)
            x_ba1, x_ba2 = torch.cat(x_ba1), torch.cat(x_ba2)
            x_ab1, x_ab2 = torch.cat(x_ab1), torch.cat(x_ab2)
            self.train()
            return x_a, x_a_recon, x_ab1, x_ab2, x_b, x_b_recon, x_ba1, x_ba2
        elif self.net_type == 'UNIT':
            self.eval()
            x_a_recon, x_b_recon, x_ba, x_ab = [], [], [], []
            for i in range(x_a.size(0)):
                h_a, _ = self.gen_a.encode(x_a[i].unsqueeze(0))
                h_b, _ = self.gen_b.encode(x_b[i].unsqueeze(0))
                x_a_recon.append(self.gen_a.decode(h_a))
                x_b_recon.append(self.gen_b.decode(h_b))
                x_ba.append(self.gen_a.decode(h_b))
                x_ab.append(self.gen_b.decode(h_a))
            x_a_recon, x_b_recon = torch.cat(x_a_recon), torch.cat(x_b_recon)
            x_ba = torch.cat(x_ba)
            x_ab = torch.cat(x_ab)
            self.train()
            return x_a, x_a_recon, x_ab, x_b, x_b_recon, x_ba
        else:
            raise NotImplementedError('Generator model name [%s] is not recognized' % self.net_type)

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
        self.load_weight(self.gen_a, state_dict['a'])
        self.load_weight(self.gen_b, state_dict['b'])
        iterations = int(last_model_name[-11:-3])
        # Load discriminators
        last_model_name = get_model_list(checkpoint_dir, "dis")
        state_dict = torch.load(last_model_name)
        # self.dis_a.load_state_dict(state_dict['a'])
        # self.dis_b.load_state_dict(state_dict['b'])
        self.load_weight(self.dis_a, state_dict['a'])
        self.load_weight(self.dis_b, state_dict['b'])
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
        torch.save({'a': self.gen_a.state_dict(), 'b': self.gen_b.state_dict()}, gen_name)
        torch.save({'a': self.dis_a.state_dict(), 'b': self.dis_b.state_dict()}, dis_name)
        torch.save({'gen': self.gen_opt.state_dict(), 'dis': self.dis_opt.state_dict()}, opt_name)


