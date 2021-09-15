import os
import math
import torch
import itertools
from .base_model import BaseModel
from models.modules.style_transfer.two_d.MUNIT_network import AdaINGen, MsImageDis
from models.auxiliary_funs import init_weights
from ..loss import losses
from torch import nn as nn
from torch.autograd import Variable
from models.modules.style_transfer.two_d.MUNIT_network import load_vgg16
from data.transforms.transformOnTensor import vgg_preprocess
from models.auxiliary_funs import get_model_list, get_scheduler


def define_G(input_dim, params, init_type, gpu_ids):
    # dim, style_dim, n_downsample, n_res, activ, pad_type, mlp_dim
    net = AdaINGen(input_dim,
                   dim=params['dim'],
                   style_dim=params['style_dim'],
                   n_downsample=params['n_downsample'],
                   n_res=params['n_res'],
                   activ=params['activ'],
                   pad_type=params['pad_type'],
                   mlp_dim=params['mlp_dim'])
    net.to_DataParallel(gpu_ids)
    init_weights(net, init_type, init_gain=math.sqrt(2))
    return net


def define_D(input_dim, params, init_type='gaussian', gpu_ids=[]):
    net = MsImageDis(input_dim,
                     n_layer=params['n_layer'],
                     gan_type=params['gan_type'],
                     dim=params['dim'],
                     norm=params['norm'],
                     activ=params['activ'],
                     num_scales=params['num_scales'],
                     pad_type=params['pad_type'])
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
    init_weights(net, init_type, init_gain=math.sqrt(2))
    return net


# def get_scheduler(optimizer, hyperparameters, iterations=-1):
#
#     if hyperparameters['lr_policy'] == 'linear':
#         def lambda_rule(epoch):
#             lr_l = 1.0 - max(0, epoch + hyperparameters['epoch_count'] - hyperparameters['n_epochs'])\
#                    / float(hyperparameters['n_epochs_decay'] + 1)
#             return lr_l
#         scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
#     elif hyperparameters['lr_policy'] == 'step':
#         scheduler = lr_scheduler.StepLR(optimizer, step_size=hyperparameters['lr_decay_iters'],
#                                         gamma=hyperparameters['gamma'], last_epoch=iterations)  # gamma=0.1
#     elif hyperparameters['lr_policy'] == 'plateau':
#         scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
#     elif hyperparameters['lr_policy'] == 'cosine':
#         scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=hyperparameters['n_epochs'], eta_min=0)
#     elif hyperparameters['lr_policy'] == 'constant':
#         scheduler = None  # constant scheduler
#     else:
#         return NotImplementedError('learning rate policy [%s] is not implemented', hyperparameters['lr_policy'])
#     return scheduler


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


class VGGLoss(nn.Module):
    def __init__(self):
        super(VGGLoss, self).__init__()
        self.instancenorm = nn.InstanceNorm2d(512, affine=False)

    def forward(self, vgg, img, target):
        img_vgg = vgg_preprocess(img)
        target_vgg = vgg_preprocess(target)
        img_fea = vgg(img_vgg)
        target_fea = vgg(target_vgg)
        return torch.mean((self.instancenorm(img_fea) - self.instancenorm(target_fea)) ** 2)


class MUNITModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
        if is_train:
            parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
            parser.add_argument('--lambda_identity', type=float, default=0.5,
                                help='use identity mapping. Setting lambda_identity other than 0 has an effect of '
                                     'scaling the weight of the identity mapping loss. For example, if the weight of'
                                     ' the identity loss should be 10 times smaller than the weight of the '
                                     'reconstruction loss, please set lambda_identity = 0.1')
        return parser

    def __init__(self, opt):
        super(MUNITModel, self).__init__(opt)

    def set_initialization(self, hyperparameters):
        self.hyperparameters = hyperparameters
        # define the network
        if 1:
            if self.isTrain:
                self.model_names = ['gen_a', 'gen_b', 'dis_a', 'dis_b']
            else:  # during test time, only load Gs
                self.model_names = ['gen_a', 'gen_b']
            self.net_gen_a = define_G(hyperparameters['input_dim_a'], hyperparameters['gen'],
                                      init_type=hyperparameters['init'], gpu_ids=self.gpu_ids)
            self.net_gen_b = define_G(hyperparameters['input_dim_b'], hyperparameters['gen'],
                                      init_type=hyperparameters['init'], gpu_ids=self.gpu_ids)
            if self.isTrain:  # define discriminators
                self.net_dis_a = define_D(hyperparameters['input_dim_a'], hyperparameters['dis'],
                                          init_type='gaussian', gpu_ids=self.gpu_ids)
                self.net_dis_b = define_D(hyperparameters['input_dim_b'], hyperparameters['dis'],
                                          init_type='gaussian', gpu_ids=self.gpu_ids)

        # define losses
        if 1:
            self.loss_names = ['gen_recon_x_a', 'gen_recon_x_b',
                               'gen_recon_s_a', 'gen_recon_s_b',
                               'gen_recon_c_a', 'gen_recon_c_b',
                               'gen_cycrecon_x_a', 'gen_cycrecon_x_b',
                               'gen_adv_a', 'gen_adv_b',
                               'gen_vgg_a', 'gen_vgg_b',
                               'dis_a', 'dis_b']
        if self.isTrain:
            # define criterion
            self.recon_criterion = CriterionRecon('l1').to(self.device)
            self.gen_criterion = None
            self.vgg_criterion = VGGLoss().to(self.device)  # wite to apply
            # define optimizer
            lr = hyperparameters['lr']
            beta1 = hyperparameters['beta1']
            beta2 = hyperparameters['beta2']
            # dis_params = list(self.dis_a.parameters()) + list(self.dis_b.parameters())
            dis_params = itertools.chain(self.net_dis_a.parameters(), self.net_dis_b.parameters())
            # gen_params = list(self.gen_a.parameters()) + list(self.gen_b.parameters())
            gen_params = itertools.chain(self.net_gen_a.parameters(), self.net_gen_b.parameters())
            self.dis_opt = torch.optim.Adam([p for p in dis_params if p.requires_grad],
                                            lr=lr, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])
            self.gen_opt = torch.optim.Adam([p for p in gen_params if p.requires_grad],
                                            lr=lr, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])
            self.optimizers.append(self.dis_opt)
            self.optimizers.append(self.gen_opt)
        # define visual_names
        if 1:
            # specify the images you want to save/display.
            visual_names_first = ['x_a', 'x_b']   # ['x_a', 'x_b', 's_a', 's_b']
            visual_names_second = []  # ['c_a', 'c_b', 's_a_prime', 's_b_prime']
            visual_names_third = ['x_a_recon', 'x_b_recon', 'x_ba', 'x_ab']
            self.visual_names = visual_names_first + visual_names_second + visual_names_third
            if self.isTrain and self.opt.lambda_identity > 0.0:
                visual_names_four = []  # ['c_a_recon', 'c_b_recon', 's_a_recon', 's_b_recon']
                visual_names_five = ['x_aba', 'x_bab']
                self.visual_names = self.visual_names + visual_names_four + visual_names_five
        # Load VGG model if needed
        if 'vgg_w' in hyperparameters.keys() and hyperparameters['vgg_w'] > 0:
            self.vgg = load_vgg16(hyperparameters['vgg_model_path'] + '/models')
            self.vgg.eval()
            for param in self.vgg.parameters():
                param.requires_grad = False
        # fix the noise used in sampling
        # display_size = int(hyperparameters['display_size'])
        self.style_dim = hyperparameters['gen']['style_dim']

    def setup(self, opt):
        """Load and print networks; create schedulers

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        if self.isTrain:
            self.dis_scheduler = get_scheduler(self.dis_opt, opt)
            self.gen_scheduler = get_scheduler(self.gen_opt, opt)
            self.schedulers = [self.dis_scheduler, self.gen_scheduler]
        if not self.isTrain or opt.continue_train:
            load_suffix = 'iter_%d' % opt.load_iter if opt.load_iter > 0 else opt.epoch
            self.load_networks(load_suffix)
        self.print_networks(opt.verbose)

    def set_input(self, input):
        self.x_a = input['A'].to(self.device)
        self.x_b = input['B'].to(self.device)
        self.image_paths = input['A_paths']
        self.image_paths.append(input['B_paths'])
        self.s_a = input['s_a'].to(self.device)
        self.s_b = input['s_b'].to(self.device)
        # self.s_a = torch.randn(display_size, self.style_dim, 1, 1).to(self.device)
        # self.s_b = torch.randn(display_size, self.style_dim, 1, 1).to(self.device)

    def forward(self):
        # self.eval()
        # s_a = Variable(self.s_a)
        # s_b = Variable(self.s_b)
        self.c_a, self.s_a_prime = self.net_gen_a.encode(self.x_a)
        self.c_b, self.s_b_prime = self.net_gen_b.encode(self.x_b)
        # print('s_a_prime shape:', self.s_a_prime.size())
        # print('s_a shape:', self.s_a.size())
        self.x_ba = self.net_gen_a.decode(self.c_b, self.s_a)
        self.x_ab = self.net_gen_b.decode(self.c_a, self.s_b)

        # self.train()
        return self.x_ab, self.x_ba

    def compute_visuals(self):
        """Calculate additional output images for visdom and HTML visualization"""
        # self.eval()
        self.x_a_recon = self.net_gen_a.decode(self.c_a, self.s_a_prime)
        self.x_b_recon = self.net_gen_b.decode(self.c_b, self.s_b_prime)
        # self.x_ba = self.net_gen_a.decode(self.c_b, self.s_a_prime)
        # self.x_ab = self.net_gen_b.decode(self.c_a, self.s_b_prime)
        self.c_a_recon, self.s_b_recon = self.net_gen_b.encode(self.x_ab)
        self.c_b_recon, self.s_a_recon = self.net_gen_a.encode(self.x_ba)
        self.x_aba = self.net_gen_a.decode(self.c_a_recon, self.s_a_prime)
        self.x_bab = self.net_gen_b.decode(self.c_b_recon, self.s_b_prime)
        # self.train()

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()
        self.compute_visuals()
        # G_A and G_B
        self.set_requires_grad([self.net_dis_a, self.net_dis_b], False)  # Ds require no gradients when optimizing Gs
        self.gen_update()
        # D_A and D_B
        self.set_requires_grad([self.net_dis_a, self.net_dis_b], True)
        self.dis_update()

    def get_image_paths(self):
        """ Return image paths that are used to load current data"""
        return self.image_paths

    def save_networks(self, epoch):
        super(MUNITModel, self).save_networks(epoch)
        opt_name = os.path.join(self.save_dir, '%s_optimizer.pt' % epoch)
        torch.save({'gen': self.gen_opt.state_dict(), 'dis': self.dis_opt.state_dict()}, opt_name)

    def load_networks(self, epoch):
        super(MUNITModel, self).load_networks(epoch)
        opt_name = os.path.join(self.save_dir, '%s_optimizer.pt' % epoch)
        state_dict = torch.load(opt_name)
        self.dis_opt.load_state_dict(state_dict['dis'])
        self.gen_opt.load_state_dict(state_dict['gen'])

    def gen_update(self):
        self.gen_opt.zero_grad()
        # reconstruction loss
        self.loss_gen_recon_x_a = self.recon_criterion(self.x_a_recon, self.x_a)
        self.loss_gen_recon_x_b = self.recon_criterion(self.x_b_recon, self.x_b)
        self.loss_gen_recon_s_a = self.recon_criterion(self.s_a_recon, self.s_a)
        self.loss_gen_recon_s_b = self.recon_criterion(self.s_b_recon, self.s_b)
        self.loss_gen_recon_c_a = self.recon_criterion(self.c_a_recon, self.c_a)
        self.loss_gen_recon_c_b = self.recon_criterion(self.c_b_recon, self.c_b)
        self.loss_gen_cycrecon_x_a = self.recon_criterion(self.x_aba, self.x_a) if self.hyperparameters['recon_x_cyc_w'] > 0 else 0
        self.loss_gen_cycrecon_x_b = self.recon_criterion(self.x_bab, self.x_b) if self.hyperparameters['recon_x_cyc_w'] > 0 else 0
        # GAN loss
        self.loss_gen_adv_a = self.net_dis_a.calc_gen_loss(self.x_ba)
        self.loss_gen_adv_b = self.net_dis_b.calc_gen_loss(self.x_ab)
        # domain-invariant perceptual loss
        self.loss_gen_vgg_a = self.vgg_criterion(self.vgg, self.x_ba, self.x_b) if self.hyperparameters['vgg_w'] > 0 else 0
        self.loss_gen_vgg_b = self.vgg_criterion(self.vgg, self.x_ab, self.x_a) if self.hyperparameters['vgg_w'] > 0 else 0
        # total loss
        self.loss_gen_total = self.hyperparameters['gan_w'] * self.loss_gen_adv_a + \
                              self.hyperparameters['gan_w'] * self.loss_gen_adv_b + \
                              self.hyperparameters['recon_x_w'] * self.loss_gen_recon_x_a + \
                              self.hyperparameters['recon_s_w'] * self.loss_gen_recon_s_a + \
                              self.hyperparameters['recon_c_w'] * self.loss_gen_recon_c_a + \
                              self.hyperparameters['recon_x_w'] * self.loss_gen_recon_x_b + \
                              self.hyperparameters['recon_s_w'] * self.loss_gen_recon_s_b + \
                              self.hyperparameters['recon_c_w'] * self.loss_gen_recon_c_b + \
                              self.hyperparameters['recon_x_cyc_w'] * self.loss_gen_cycrecon_x_a + \
                              self.hyperparameters['recon_x_cyc_w'] * self.loss_gen_cycrecon_x_b + \
                              self.hyperparameters['vgg_w'] * self.loss_gen_vgg_a + \
                              self.hyperparameters['vgg_w'] * self.loss_gen_vgg_b
        self.loss_gen_total.backward()
        self.gen_opt.step()

    def dis_update(self):
        self.dis_opt.zero_grad()
        # D loss
        self.loss_dis_a = self.net_dis_a.calc_dis_loss(self.x_ba.detach(), self.x_a)
        self.loss_dis_b = self.net_dis_b.calc_dis_loss(self.x_ab.detach(), self.x_b)
        self.loss_dis_total = self.hyperparameters['gan_w'] * self.loss_dis_a + self.hyperparameters['gan_w'] * self.loss_dis_b
        self.loss_dis_total.backward()
        self.dis_opt.step()

    def save(self, snapshot_dir, iterations):
        # Save generators, discriminators, and optimizers
        gen_name = os.path.join(snapshot_dir, 'gen_%08d.pt' % (iterations + 1))
        dis_name = os.path.join(snapshot_dir, 'dis_%08d.pt' % (iterations + 1))
        opt_name = os.path.join(snapshot_dir, 'optimizer.pt')
        torch.save({'a': self.net_gen_a.state_dict(), 'b': self.net_gen_b.state_dict()}, gen_name)
        torch.save({'a': self.net_dis_a.state_dict(), 'b': self.net_dis_b.state_dict()}, dis_name)
        torch.save({'gen': self.gen_opt.state_dict(), 'dis': self.dis_opt.state_dict()}, opt_name)

    def resume(self, checkpoint_dir, hyperparameters):
        # Load generators
        last_model_name = get_model_list(checkpoint_dir, "gen")
        state_dict = torch.load(last_model_name)
        self.net_gen_a.load_state_dict(state_dict['a'])
        self.net_gen_b.load_state_dict(state_dict['b'])
        iterations = int(last_model_name[-11:-3])
        # Load discriminators
        last_model_name = get_model_list(checkpoint_dir, "dis")
        state_dict = torch.load(last_model_name)
        self.net_dis_a.load_state_dict(state_dict['a'])
        self.net_dis_b.load_state_dict(state_dict['b'])

        # Load optimizers
        state_dict = torch.load(os.path.join(checkpoint_dir, 'optimizer.pt'))
        self.dis_opt.load_state_dict(state_dict['dis'])
        self.gen_opt.load_state_dict(state_dict['gen'])
        # Reinitilize schedulers
        self.dis_scheduler = get_scheduler(self.dis_opt, hyperparameters, iterations)
        self.gen_scheduler = get_scheduler(self.gen_opt, hyperparameters, iterations)
        self.schedulers = [self.dis_scheduler, self.gen_scheduler]
        print('Resume from iteration %d' % iterations)
        return iterations

    def sample(self, x_a, x_b):
        self.eval()
        s_a1 = Variable(torch.randn(x_a.size(0), self.style_dim, 1, 1).to(self.device))
        s_b1 = Variable(torch.randn(x_b.size(0), self.style_dim, 1, 1).to(self.device))
        s_a2 = Variable(torch.randn(x_a.size(0), self.style_dim, 1, 1).to(self.device))
        s_b2 = Variable(torch.randn(x_b.size(0), self.style_dim, 1, 1).to(self.device))
        x_a_recon, x_b_recon, x_ba1, x_ba2, x_ab1, x_ab2 = [], [], [], [], [], []
        for i in range(x_a.size(0)):
            c_a, s_a_fake = self.net_gen_a.encode(x_a[i].unsqueeze(0))
            c_b, s_b_fake = self.net_gen_b.encode(x_b[i].unsqueeze(0))
            x_a_recon.append(self.net_gen_a.decode(c_a, s_a_fake))
            x_b_recon.append(self.net_gen_b.decode(c_b, s_b_fake))
            x_ba1.append(self.net_gen_a.decode(c_b, s_a1[i].unsqueeze(0)))
            x_ba2.append(self.net_gen_a.decode(c_b, s_a2[i].unsqueeze(0)))
            x_ab1.append(self.net_gen_b.decode(c_a, s_b1[i].unsqueeze(0)))
            x_ab2.append(self.net_gen_b.decode(c_a, s_b2[i].unsqueeze(0)))
        x_a_recon, x_b_recon = torch.cat(x_a_recon), torch.cat(x_b_recon)
        x_ba1, x_ba2 = torch.cat(x_ba1), torch.cat(x_ba2)
        x_ab1, x_ab2 = torch.cat(x_ab1), torch.cat(x_ab2)
        self.train()
        return x_a, x_a_recon, x_ab1, x_ab2, x_b, x_b_recon, x_ba1, x_ba2