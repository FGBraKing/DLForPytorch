import os
import math
import argparse

from utils.others.utils import mkdirs, convert_str_to_list


class ProjectOptions:
    def __init__(self):
        """Reset the class; indicates the class hasn't been initailized"""
        self.initialized = False
        self.isTrain = False
        self.parser = None
        self.opt = None

    @staticmethod
    def base_initialize(parser):
        # basic parameters
        parser.add_argument('--name', type=str, default='promise_unet_default',
                            help='name of the experiment option. It decides where to store samples and models')
        parser.add_argument('--dataset_name', type=str, default='promise12', help='chooses how datasets are loaded')
        parser.add_argument('--model_name', type=str, default='unet3d',  help='chooses which model to use.')
        parser.add_argument('--seed', type=int, default=1008, help='random seed')
        parser.add_argument('--gpu_ids', type=str, default='1', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        parser.add_argument('--visible_gpu', type=str, default='0,1,2,3', help='visible gpu ids: e.g. 0  0,1,2, 0,2.')
        return parser

    @staticmethod
    def distribution_initialize(parser):
        # distribution parameters
        parser.add_argument('--DP', action='store_true', help='use torch.nn.DataParallel')
        parser.add_argument('--DDP', action='store_true', help='torch.nn.parallel.DistributedDataParallel')
        parser.add_argument('--world_size', type=int, default=3, help='number of distributed processes')
        parser.add_argument('--local_rank', type=int, default=0, help='rank of distributed processes')
        parser.add_argument('--dist_url', type=str, default='tcp://172.21.141.4:30303',
                            help='url used to set up distributed training')
        parser.add_argument('--dist_backend', default='nccl', type=str, help='distributed backend')
        return parser

    @staticmethod
    def dataset_initialize(parser):
        # dataset parameters
        parser.add_argument('--dataroot', type=str,
                            default='/data/project_data_lf/PROJECT/DLForPytorch/datasets/promise12',
                            help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
        parser.add_argument('--phase', type=str, default='train')
        parser.add_argument('--serial_batches', action='store_true',
                            help='if true, takes images in order to make batches, otherwise takes them randomly')

        parser.add_argument('--custom', action='store_true', help='whether to use custom configure')
        parser.add_argument('--preprocess', type=str,
                            default='randomscale_randomcrop_ranomrotate_centercrop_rot90_mirror_'
                                    'gaussianNoise_GaussianBlur_BrightnessMultiplicative_'
                                    'contrast_simulate_gammatransform',
                            help='scaling and cropping of images at load time ')
        parser.add_argument('--crop_size', type=str, default='128,128,32', help='the crop size of slide  windows')
        parser.add_argument('--target_size', type=str, default='128,128,128', help='the target size ')
        parser.add_argument('--scale', type=str, default='1.,1.,1.', help='the scale of target size')
        parser.add_argument('--bright_mu', type=float, default=0.0,  help='brightness')
        parser.add_argument('--bright_sigma', type=float, default=0.5,  help='brightness')
        parser.add_argument('--elastic_alpha', type=str, default=(0., 1000.),  help='ElasticDeformTransform ')
        parser.add_argument('--elastic_sigma', type=str, default=(10., 13.),  help='ElasticDeformTransform ')
        parser.add_argument('--shift_mu', type=str, default=(0., 1000.),  help='RandomShiftTransform ')
        parser.add_argument('--shift_sigma', type=str, default=(10., 13.),  help='RandomShiftTransform ')
        parser.add_argument('--order_data', type=int, default=3,  help='order_data ')
        parser.add_argument('--order_seg', type=int, default=0,  help='order_seg ')

        # dataloader parameters
        parser.add_argument('--num_threads', type=int, default=1, help='# threads for loading data')
        parser.add_argument('--batch_size', type=int, default=6, help='input batch size')
        parser.add_argument('--max_dataset_size', type=int, default=float("inf"),
                            help='Maximum number of samples allowed per dataset. If the dataset directory contains'
                                 ' more than max_dataset_size, only a subset is loaded.')
        return parser

    @staticmethod
    def module_initialize(parser):
        # model parameters
        parser.add_argument('--input_nc', type=int, default=1, help='# of input volume channels')
        parser.add_argument('--output_nc', type=int, default=1, help='# of output image channels:')
        parser.add_argument('--init_channel_number', type=int, default=32, help='the init channel number of unet')
        parser.add_argument('--up_interpolate', action='store_true', help='upsample_interpolate')
        parser.add_argument('--conv_order', type=str, default='crb', help='# of the order of conv layer in the 3d-unet')
        # loss parameters
        parser.add_argument('--reduction', type=str, default='mean', help='loss reduction')
        parser.add_argument('--ignore_index', type=str, default=None, help='which class should be ignore')
        parser.add_argument('--smooth', type=float, default=1., help='loss function smooth')
        # initialization parameters
        parser.add_argument('--init_type', type=str, default='kaiming',
                            help='network initialization [normal | xavier | kaiming | orthogonal]')
        parser.add_argument('--init_gain', type=float, default=math.sqrt(2),
                            help='scaling factor for normal, xavier and orthogonal.')
        parser.add_argument('--init_std', type=float, default=0.02,
                            help='scaling factor for normal, xavier and orthogonal.')
        return parser

    @staticmethod
    def optimizer_initialize(parser):
        # optimizer parameters
        parser.add_argument('--optimizer_name', type=str, default='adam', help='name of optimizer to create')
        parser.add_argument('--lr', type=float, default=1e-4, help='initial learning rate for adam')
        parser.add_argument('--weight_decay', type=float, default=0., help='weight decay to apply in optimizer')
        parser.add_argument('--momentum', type=float, default=0.9,
                            help='momentum for momentum based optimizers (others may use betas via kwargs)')
        parser.add_argument('--beta1', type=float, default=0.9, help='momentum term of adam')

        # scheduler parameters
        parser.add_argument('--lr_policy', type=str, default='step',
                            help='learning rate policy. [linear | step | plateau | cosine]')
        parser.add_argument('--decay_epochs', type=int, default=100,
                            help='multiply by a gamma every decay_epochs ')
        parser.add_argument('--decay_rate', type=float, default=0.1, help='the base to decay')
        parser.add_argument('--warmup_lr', type=float, default=1e-7, help='warmup_lr_init')
        parser.add_argument('--warmup_epochs', type=int, default=100, help='how many epoch to warmup')
        parser.add_argument('--lr_noise', type=float, default=None,
                            help='the range of epochs for applying noise to lr')
        return parser

    @staticmethod
    def model_initialize(parser):
        parser.add_argument('--logs_dir', type=str,
                            default='/home/users/lf/CODE/PycharmProjects/DLForPytorch/traces/logs',
                            help='logs are saved here')
        parser.add_argument('--checkpoints_dir', type=str,
                            default='/home/users/lf/CODE/PycharmProjects/DLForPytorch/traces/checkpoints',
                            help='models are saved here')
        parser.add_argument('--weight_path', type=str, default='None', help='')
        parser.add_argument('--continue_train', action='store_true',
                            help='continue training: load the latest model')

        parser.add_argument('--verbose', action='store_true',
                            help='if specified, print more debugging information')
        # additional parameters
        parser.add_argument('--suffix', default='', type=str,
                            help='customized suffix: opt.name = opt.name + suffix:e.g., {model}_{netG}_size{load_size}')
        parser.add_argument('--DEBUG', action='store_true', help='in the debug mode, print moreover info')
        parser.add_argument('--epoch_start', type=int, default=1, help='form which epoch to start')
        parser.add_argument('--num_epochs', type=int, default=1000, help='total epochs for training')
        return parser

    @staticmethod
    def train_initialize(parser):
        # network saving and loading parameters
        parser.add_argument('--save_epoch_start', type=int, default=500,
                            help='we save the model by <save_epoch_start>, <save_epoch_start>+<save_latest_freq>, ...')
        parser.add_argument('--save_epoch_freq', type=int, default=50,
                            help='frequency of saving checkpoints at the end of epochs')
        parser.add_argument('--save_iter_start', type=int, default=5000,
                            help='we save the model by <save_epoch_start>, <save_epoch_start>+<save_latest_freq>, ...')
        parser.add_argument('--save_iter_freq', type=int, default=500,
                            help='frequency of saving checkpoints at the end of epochs')
        parser.add_argument('--save_by_iter', action='store_true',
                            help='whether saves model by iteration')

        # network print and showing parameters
        parser.add_argument('--display_freq', type=int, default=64,
                            help='frequency of showing training results on screen')
        parser.add_argument('--print_freq', type=int, default=1,
                            help='frequency of print training loss on console')
        parser.add_argument('--plot_freq', type=int, default=1,
                            help='frequency of plot training metrics on console')

        # visualizer parameters
        parser.add_argument('--with_html', action='store_true',
                            help='whether save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')
        parser.add_argument('--with_tensorboard', action='store_true', help='whether to use tensorboard')
        parser.add_argument('--with_visdom', action='store_true', help='whether to use visdom')
        parser.add_argument('--save_log', action='store_true', help='whether to save logging file')

        # visdom  parameters
        parser.add_argument('--display_server', type=str, default="http://172.21.141.4",
                            help='visdom server of the web display')
        parser.add_argument('--display_port', type=int, default=30303,
                            help='visdom port of the web display')
        parser.add_argument('--display_env', type=str, default='main',
                            help='visdom display environment name (default is "main")')
        parser.add_argument('--display_id', type=int, default=0, help='window id of the web display')
        parser.add_argument('--display_ncols', type=int, default=0,
                            help='if positive, display all images in a single visdom web panel '
                                 'with certain number of images per row.')
        # html parameters
        parser.add_argument('--display_winsize', type=int, default=256, help='display windows size for html')

        # tensorboard and logging parameters
        parser.add_argument('--draw_model', action='store_true', help='whether to draw model on tensorboard')
        parser.add_argument('--display_histogram', action='store_true', help='whether display histogram ')
        parser.add_argument('--play_video', action='store_true', help='whether play volume as a video ')
        return parser

    @staticmethod
    def test_initialize(parser):
        # basic parameters
        parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')

        # Dropout and Batchnorm has different behavioir during training and test.
        parser.add_argument('--eval', action='store_true', help='use eval mode during test time.')
        # rewrite devalue values
        parser.set_defaults(model='test')
        # To avoid cropping, the load_size should be the same as crop_size
        parser.set_defaults(load_size=parser.get_default('crop_size'))
        return parser

    def initialize(self, parser, isTrain):
        """Define the common options that are used in both training and test."""
        parser = self.base_initialize(parser)
        parser = self.distribution_initialize(parser)
        parser = self.dataset_initialize(parser)
        parser = self.module_initialize(parser)
        parser = self.optimizer_initialize(parser)
        parser = self.model_initialize(parser)

        if isTrain:
            parser = self.train_initialize(parser)
        else:
            parser = self.test_initialize(parser)

        self.initialized = True
        return parser

    def gather_options(self, isTrain):
        if not self.initialized:  # check if it has been initialized
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser, isTrain)
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

    def parse(self, isTrain):
        """Parse our options, create checkpoints directory suffix, and set up gpu device."""
        opt = self.gather_options(isTrain)
        opt.isTrain = isTrain
        assert not(opt.DP and opt.DDP)

        # process opt.suffix
        if opt.suffix:
            suffix = ('_' + opt.suffix.format(**vars(opt))) if opt.suffix != '' else ''
            opt.name = opt.name + suffix

        if (not opt.DDP) or (opt.DDP and opt.local_rank == 0):
            self.print_options(opt)

        opt.gpu_ids = convert_str_to_list(opt.gpu_ids, split=',', aim_type=int, condition=lambda x: x >= 0)

        opt.crop_size = convert_str_to_list(opt.crop_size, split=',', aim_type=int, condition=lambda x: x > 0)
        opt.target_size = convert_str_to_list(opt.target_size, split=',', aim_type=int, condition=lambda x: x > 0)
        opt.scale = convert_str_to_list(opt.scale, split=',', aim_type=float, condition=lambda x: x > 0)
        opt.elastic_alpha = convert_str_to_list(opt.elastic_alpha, split=',', aim_type=float, condition=lambda x: x >= 0)
        opt.elastic_sigma = convert_str_to_list(opt.elastic_sigma, split=',', aim_type=float, condition=lambda x: x >= 0)
        opt.shift_mu = convert_str_to_list(opt.shift_mu, split=',', aim_type=float, condition=lambda x: x >= 0)
        opt.shift_sigma = convert_str_to_list(opt.shift_sigma, split=',', aim_type=float, condition=lambda x: x >= 0)
        if opt.ignore_index is not None:
            opt.ignore_index = convert_str_to_list(opt.ignore_index, split=',', aim_type=int, condition=lambda x: x >= 0)

        self.opt = opt
        return self.opt


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
    opt = ProjectOptions().parse()
    print('option get ready')
#  os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_ids[0]
#  torch.cuda.set_device(opt.gpu_ids[0])
#  torch.device('cuda:{}'.format(self.gpu_ids[0]))
#  data.to(device)
#  model.to(device)
#  net.to(gpu_ids[0])
#  net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs



