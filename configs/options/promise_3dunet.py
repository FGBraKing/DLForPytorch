import argparse
import os
import torch

from utils.others.utils import mkdirs, convert_str_to_list


class BaseOptions:
    def __init__(self):
        """Reset the class; indicates the class hasn't been initailized"""
        self.initialized = False
        self.isTrain = False
        self.parser = None
        self.opt = None

    def initialize(self, parser):
        """Define the common options that are used in both training and test."""
        # dataset parameters
        parser.add_argument('--dataroot', type=str,
                            default='/data/project_data_lf/PROJECT/DLForPytorch/datasets/promise12',
                            help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
        parser.add_argument('--phase', type=str, default='train')
        parser.add_argument('--preprocess', type=str, default='GaussianNoise_crop_rotate_centercrop_rot90_flip_bothscale',
                            help='scaling and cropping of images at load time ')  # 'crop_flip_rotate'
        parser.add_argument('--gaussian_sigma', type=str, default='0.0,0.1', help='gaussian sigma')
        parser.add_argument('--crop_size', type=str, default='128,128,32', help='the crop size of slide  windows')
        parser.add_argument('--angle_spectrum', type=int, default=30, help='random rotate, angle')
        parser.add_argument('--custom', action='store_true', help='whether to use custom configure')
        parser.add_argument('--serial_batches', action='store_true',
                            help='if true, takes images in order to make batches, otherwise takes them randomly')
        # dataloader parameters
        parser.add_argument('--num_threads', default=1, type=int,
                            help='# threads for loading data')
        parser.add_argument('--batch_size', type=int,
                            default=6, help='input batch size')
        parser.add_argument('--max_dataset_size', type=int, default=float("inf"),
                            help='Maximum number of samples allowed per dataset. '
                                 'If the dataset directory contains more than max_dataset_size, '
                                 'only a subset is loaded.')
        # model parameters
        parser.add_argument('--input_nc', type=int, default=1,
                            help='# of input volume channels')
        parser.add_argument('--output_nc', type=int, default=1,
                            help='# of output image channels: 3 for RGB and 1 for grayscale')
        parser.add_argument('--conv_order', type=str, default='crb',
                            help='# of the order of conv layer in the 3d-unet')
        parser.add_argument('--init_channel_number', type=int, default=32, help='the init channel number of unet')
        # initialization parameters
        parser.add_argument('--init_type', type=str, default='normal',
                            help='network initialization [normal | xavier | kaiming | orthogonal]')
        parser.add_argument('--init_gain', type=float, default=0.02,
                            help='scaling factor for normal, xavier and orthogonal.')
        # optimizer parameter
        parser.add_argument('--beta1', type=float, default=0.9,
                            help='momentum term of adam')
        parser.add_argument('--lr', type=float, default=1e-4,
                            help='initial learning rate for adam')
        parser.add_argument('--lr_policy', type=str, default='linear',
                            help='learning rate policy. [linear | step | plateau | cosine]')
        parser.add_argument('--epoch_start', type=int, default=1,
                            help='number of epochs with the initial learning rate')
        parser.add_argument('--epoch_retain', type=int, default=500,
                            help='number of epochs with the initial learning rate')
        parser.add_argument('--epochs_decay', type=int, default=500,
                            help='number of epochs to linearly decay learning rate to zero')

        # model control parameters
        parser.add_argument('--logs_dir', type=str,
                            default='/home/users/lf/CODE/PycharmProjects/DLForPytorch/traces/logs',
                            help='logs are saved here')
        parser.add_argument('--checkpoints_dir', type=str,
                            default='/home/users/lf/CODE/PycharmProjects/DLForPytorch/traces/checkpoints',
                            help='models are saved here')
        parser.add_argument('--continue_train', action='store_true',
                            help='continue training: load the latest model')
        parser.add_argument('--verbose', action='store_true',
                            help='if specified, print more debugging information')
        parser.add_argument('--weight_path', type=str, default='None')

        # basic parameters
        parser.add_argument('--name', type=str, default='promise_unet_default',
                            help='name of the experiment option. It decides where to store samples and models')
        parser.add_argument('--dataset_name', type=str, default='promise12',
                            help='chooses how datasets are loaded. [unaligned | aligned | single | colorization]')
        parser.add_argument('--model_name', type=str, default='unet3d',
                            help='chooses which model to use. [cycle_gan | pix2pix | test | colorization]')
        parser.add_argument('--seed', type=int, default=1008, help='random seed')
        parser.add_argument('--gpu_ids', type=str, default='1',
                            help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')

        # additional parameters
        parser.add_argument('--suffix', default='', type=str,
                            help='customized suffix: opt.name = opt.name + suffix:e.g., {model}_{netG}_size{load_size}')
        parser.add_argument('--DEBUG', action='store_true', help='in the debug mode, print moreover info')

        parser.add_argument('--DP', action='store_true', help='use torch.nn.DataParallel')
        parser.add_argument('--DDP', action='store_true', help='torch.nn.parallel.DistributedDataParallel')
        parser.add_argument('--world_size', default=3, type=int, help='number of distributed processes')
        parser.add_argument('--dist_url', default='tcp://172.21.141.4:30303', type=str,
                            help='url used to set up distributed training')
        parser.add_argument('--dist_backend', default='nccl', type=str, help='distributed backend')
        parser.add_argument('--local_rank', type=int, help='rank of distributed processes')

        self.initialized = True
        return parser

    def gather_options(self):
        if not self.initialized:  # check if it has been initialized
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)
        else:
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        # get the basic options
        opt, _ = parser.parse_known_args()
        # save and return the parser
        self.parser = parser
        return parser.parse_args()  # args=['--verbose']  args=['--verbose', '--custom', '--verbose']

    def print_options(self, opt):
        """Print and save options

        It will print both current options and default values(if different).
        It will save options into a text file / [checkpoints_dir] / opt.txt
        """
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        # save to the disk
        expr_dir = os.path.join(opt.checkpoints_dir, opt.name)  # opt.dataset_name + opt.model_name + opt.name
        mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, '{}_opt.txt'.format(opt.phase))
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')

    def parse(self):
        """Parse our options, create checkpoints directory suffix, and set up gpu device."""
        opt = self.gather_options()
        opt.isTrain = self.isTrain  # train or test
        assert not(opt.DP and opt.DDP)

        # process opt.suffix
        if opt.suffix:
            suffix = ('_' + opt.suffix.format(**vars(opt))) if opt.suffix != '' else ''
            opt.name = opt.name + suffix

        if (not opt.DDP) or (opt.DDP and opt.local_rank == 0):
            self.print_options(opt)

        opt.gpu_ids = convert_str_to_list(opt.gpu_ids, split=',', aim_type=int, condition=lambda x: x >= 0)
        # if len(opt.gpu_ids) > 0:
        #     torch.cuda.set_device(opt.gpu_ids[0])
        opt.crop_size = convert_str_to_list(opt.crop_size, split=',', aim_type=int, condition=lambda x: x > 0)
        opt.gaussian_sigma = convert_str_to_list(opt.gaussian_sigma, split=',',
                                                 aim_type=float, condition=lambda x: x >= 0)
        # print('opt.gaussian_sigma', opt.gaussian_sigma)
        # if isinstance(opt.init_gain, str):
        #     opt.init_gain = float(opt.init_gain)
        self.opt = opt
        return self.opt


class TrainOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        # visualizer parameters
        parser.add_argument('--no_html', action='store_true',
                            help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')
        parser.add_argument('--display_id', type=int, default=0, help='window id of the web display')
        parser.add_argument('--tensorboard', action='store_true', help='whether to use tensorboard')
        parser.add_argument('--save_log', action='store_true', help='whether to use logging file')

        # visdom and HTML visualization parameters
        parser.add_argument('--display_ncols', type=int, default=0,
                            help='if positive, display all images in a single visdom web panel '
                                 'with certain number of images per row.')
        parser.add_argument('--display_env', type=str, default='main',
                            help='visdom display environment name (default is "main")')
        parser.add_argument('--display_server', type=str, default="http://172.21.141.4",
                            help='visdom server of the web display')
        parser.add_argument('--display_port', type=int, default=30303,
                            help='visdom port of the web display')
        parser.add_argument('--display_winsize', type=int, default=256,
                            help='display windows size for both visdom and html')

        # network print and showing parameters
        parser.add_argument('--display_freq', type=int, default=64,
                            help='frequency of showing training results on screen')
        parser.add_argument('--print_freq', type=int, default=1,
                            help='frequency of showing training results on console')
        parser.add_argument('--save_epoch_freq', type=int, default=50,
                            help='frequency of saving checkpoints at the end of epochs')
        parser.add_argument('--save_iter_freq', type=int, default=500,
                            help='frequency of saving checkpoints at the end of epochs')

        # network saving and loading parameters
        parser.add_argument('--save_epoch_start', type=int, default=500,
                            help='we save the model by <save_epoch_start>, <save_epoch_start>+<save_latest_freq>, ...')

        parser.add_argument('--save_iter_start', type=int, default=5000,
                            help='we save the model by <save_epoch_start>, <save_epoch_start>+<save_latest_freq>, ...')
        parser.add_argument('--save_by_iter', action='store_true',
                            help='whether saves model by iteration')

        # useless parameter(Temporarily)
        parser.add_argument('--update_html_freq', type=int, default=1000,
                            help='frequency of saving training results to html')
        parser.add_argument('--lr_decay_iters', type=int, default=50,
                            help='multiply by a gamma every lr_decay_iters iterations')

        self.isTrain = True
        return parser


class TestOptions(BaseOptions):
    """This class includes test options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        # basic parameters
        parser.add_argument('--results_dir', type=str, default='./results/',
                            help='saves results here.')

        # Dropout and Batchnorm has different behavioir during training and test.
        parser.add_argument('--eval', action='store_true',
                            help='use eval mode during test time.')
        # rewrite devalue values
        parser.set_defaults(model='test')
        # To avoid cropping, the load_size should be the same as crop_size
        parser.set_defaults(load_size=parser.get_default('crop_size'))

        self.isTrain = False
        return parser


if __name__ == '__main__':
    # parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    #
    # parser.add_argument('--dataroot', default='/test', required=False,
    #                     help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
    # parser.add_argument('--name', type=str, default='experiment_name',
    #                     help='name of the experiment. It decides where to store samples and models')
    # parser.add_argument('--gpu_ids', type=str, default='0',
    #                     help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    # parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints',
    #                     help='models are saved here')
    # opt = parser.parse_args(args=[])
    opt = TrainOptions().parse()
    print('option get ready')
#  os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_ids[0]
#  torch.cuda.set_device(opt.gpu_ids[0])
#  torch.device('cuda:{}'.format(self.gpu_ids[0]))
#  data.to(device)
#  model.to(device)
#  net.to(gpu_ids[0])
#  net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs



