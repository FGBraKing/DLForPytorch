import os
import torch
import argparse

from utils.others.utils import mkdirs


class BaseOptions:
    def __init__(self):
        """Reset the class; indicates the class hasn't been initailized"""
        self.initialized = False
        self.isTrain = False
        self.parser = None
        self.opt = None

    def initialize(self, parser):
        """Define the common options that are used in both training and test."""
        # basic parameters
        parser.add_argument('--name', type=str, default='default',
                            help='name of the experiment option. It decides where to store samples and models')
        parser.add_argument('--dataroot',  # required=True,
                            default='/home/users/lf/CODE/PycharmProjects/DLForPytorch/traces/datasets/BraTs2018-IPML',
                            help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
        parser.add_argument('--logs_dir', type=str,
                            default='/home/users/lf/CODE/PycharmProjects/DLForPytorch/traces/checkpoints',
                            help='logs are saved here')
        parser.add_argument('--checkpoints_dir', type=str,
                            default='/home/users/lf/CODE/PycharmProjects/DLForPytorch/traces/checkpoints',
                            help='models are saved here')
        parser.add_argument('--seed', type=int, default=1008, help='random seed')

        # running parameters
        parser.add_argument('--gpu_ids', type=str, default='0,1,2',
                            help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        # model parameters
        parser.add_argument('--model_name', type=str, default='3dunet',
                            help='chooses which model to use. [cycle_gan | pix2pix | test | colorization]')
        # dataset parameters
        parser.add_argument('--dataset_name', type=str, default='promise',
                            help='chooses how datasets are loaded. [unaligned | aligned | single | colorization]')

        # additional parameters
        parser.add_argument('--suffix', default='', type=str,
                            help='customized suffix: opt.name = opt.name + suffix:e.g., {model}_{netG}_size{load_size}')

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
        return parser.parse_args()      # args=['--verbose']

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
        expr_dir = os.path.join(opt.checkpoints_dir, opt.dataset_name+opt.model_name+opt.name)
        mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, '{}_opt.txt'.format(opt.phase))
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')

    def parse(self):
        """Parse our options, create checkpoints directory suffix, and set up gpu device."""
        opt = self.gather_options()
        opt.isTrain = self.isTrain   # train or test

        # process opt.suffix
        if opt.suffix:
            suffix = ('_' + opt.suffix.format(**vars(opt))) if opt.suffix != '' else ''
            opt.name = opt.name + suffix

        self.print_options(opt)

        # set gpu ids
        assert isinstance(opt.gpu_ids, str), 'the gpu_ids have to be string!'
        print('gpu_ids:'+opt.gpu_ids)
        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)
        # if len(opt.gpu_ids) > 0:
        #     torch.cuda.set_device(opt.gpu_ids[0])

        self.opt = opt
        return self.opt


class TrainOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        # visdom and HTML visualization parameters

        # network saving and loading parameters
        parser.add_argument('--save_epoch_start', type=int, default=5000,
                            help='we save the model by <save_epoch_start>, <save_epoch_start>+<save_latest_freq>, ...')
        parser.add_argument('--save_epoch_freq', type=int, default=10,
                            help='frequency of saving checkpoints at the end of epochs')
        parser.add_argument('--save_iter_start', type=int, default=5000,
                            help='we save the model by <save_epoch_start>, <save_epoch_start>+<save_latest_freq>, ...')
        parser.add_argument('--save_iter_freq', type=int, default=10,
                            help='frequency of saving checkpoints at the end of epochs')
        parser.add_argument('--save_by_iter', action='store_true',
                            help='whether saves model by iteration')
        parser.add_argument('--continue_train', action='store_true',
                            help='continue training: load the latest model')

        # training parameters
        parser.add_argument('--epoch_start', type=int, default=100,
                            help='number of epochs with the initial learning rate')
        parser.add_argument('--epochs_decay', type=int, default=200,
                            help='number of epochs to linearly decay learning rate to zero')
        parser.add_argument('--epoch_end', type=int, default=1000,
                            help='number of epochs to end the training')

        parser.add_argument('--lr', type=float, default=0.0002,
                            help='initial learning rate for adam')
        parser.add_argument('--lr_policy', type=str, default='linear',
                            help='learning rate policy. [linear | step | plateau | cosine]')
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
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--dataroot',  default='/test', required=False,
                        help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
    parser.add_argument('--name', type=str, default='experiment_name',
                        help='name of the experiment. It decides where to store samples and models')
    parser.add_argument('--gpu_ids', type=str, default='0',
                        help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints',
                        help='models are saved here')
    opt = parser.parse_args(args=[])
    print(vars(opt))
#  os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_ids[0]
#  torch.cuda.set_device(opt.gpu_ids[0])
#  torch.device('cuda:{}'.format(self.gpu_ids[0]))
#  data.to(device)
#  model.to(device)
#  net.to(gpu_ids[0])
#  net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs



