import os
import torch
from collections import OrderedDict, defaultdict
from abc import ABC, abstractmethod
# from models.auxiliary_funs import get_scheduler


class BaseModel(ABC):
    """This class is an abstract base class (ABC) for models.
    To create a subclass, you need to implement the following five functions:
        -- <__init__>:                      initialize the class; first call BaseModel.__init__(self, opt).
        -- <set_input>:                     unpack data from dataset and apply preprocessing.
        -- <forward>:                       produce intermediate results.
        -- <optimize_parameters>:           calculate losses, gradients, and update network weights.
    """

    def __init__(self, opt):
        """Initialize the BaseModel class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions

        When creating your custom class, you need to implement your own initialization.
        In this function, you should first call <BaseModel.__init__(self, opt)>
        Then, you need to define four lists:
            -- self.loss_names (str list):          specify the training losses that you want to plot and save.
            -- self.model_names (str list):         define networks used in our training.
            -- self.visual_names (str list):        specify the images that you want to display and save.
            -- self.optimizers (optimizer list):    define and initialize optimizers. You can define one optimizer for each network. If two networks are updated at the same time, you can use itertools.chain to group them. See cycle_gan_model.py for an example.
        """
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        if opt.DDP:
            self.device = torch.device('cuda:{}'.format(opt.local_rank))  #
        else:
            self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)  # save all the checkpoints to save_dir
        self.logs_dir = os.path.join(opt.logs_dir, opt.name)
        # self.data_paths = []
        self.volume_path = []
        self.label_path = []
        self.model_names = []
        self.loss_names = []
        if self.isTrain:
            self.optimizers = []
            # define self.optimizers and self.loss_criterion
            # self.schedulers = [get_scheduler(optimizer, opt) for optimizer in self.optimizers]
            self.schedulers = []
        self.visual_names = []
        self.metric_names = []
        self.metric_dict = {}
        self.lr_metric = 0  # used for learning rate policy 'plateau'

    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new model-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        return parser

    @abstractmethod
    def set_input(self, inputs):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            inputs (dict): includes the data itself and its metadata information.
        """
        pass

    @abstractmethod
    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        pass

    @abstractmethod
    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        pass

    # TODO :修改命名得规则和策略
    def setup(self, opt):
        """Load and print networks; create schedulers
        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        if not self.isTrain or opt.continue_train:
            self.load_networks()
        self.print_networks(opt.verbose)

    def eval(self):
        """Make models eval mode during test time"""
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net_' + name)
                net.eval()

    def train(self):
        """Make models eval mode during test time"""
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net_' + name)
                net.train()

    def test(self):
        """Forward function used in test time.

        This function wraps <forward> function in no_grad() so we don't save intermediate steps for backprop
        It also calls <compute_visuals> to produce additional visualization results
        """
        with torch.no_grad():
            self.forward()
            self.compute_visuals()
            self.compute_metrics()

    def compute_visuals(self):
        """Calculate additional output images for visdom and HTML visualization"""
        pass

    def compute_metrics(self):
        pass

    def get_image_paths(self):
        """ Return image paths that are used to load current data"""
        return self.volume_path

    def update_learning_rate(self, epoch):
        """Update learning rates for all the networks; called at the end of every epoch"""
        old_lr = self.optimizers[0].param_groups[0]['lr']
        for scheduler in self.schedulers:
            scheduler.step(epoch)
            # if self.opt.lr_policy == 'plateau':
            #     scheduler.step(self.lr_metric)
            # else:
            #     scheduler.step(epoch)

        lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate %.7f -> %.7f' % (old_lr, lr))

    def get_current_lrs(self):
        lrs = []
        for optimizer in self.optimizers:
            for group in optimizer.param_groups:
                lrs.append(group['lr'])
        return lrs

    def get_current_visuals(self):
        """Return visualization images. train.py will display these images with visdom, and save the images to a HTML"""
        visual_ret = OrderedDict()
        for name in self.visual_names:
            if isinstance(name, str):
                visual_ret[name] = getattr(self, name)
        return visual_ret

    def get_current_metrics(self):
        metrics_ret = OrderedDict()
        for name in self.metric_names:
            if isinstance(name, str):
                metrics_ret[name] = self.metric_dict[name]
        return metrics_ret

    def get_current_losses(self):
        """Return traning losses / errors. train.py will print out these errors on console, and save them to a file"""
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                errors_ret[name] = float(getattr(self, 'loss_' + name))
                # float(...) works for both scalar tensor and float number
        return errors_ret

    def save_networks(self, epoch):
        """Save all the networks to the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        if self.opt.DDP and self.opt.local_rank != 0:
            return
        state_dict = defaultdict()
        # state_dict['lr'] = self.optimizers[0].param_groups[0]['lr']
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net_' + name)
                # if len(self.gpu_ids) > 0 and torch.cuda.is_available():
                if isinstance(net, torch.nn.parallel.DataParallel) \
                        or isinstance(net, torch.nn.parallel.DistributedDataParallel):
                    state_dict[name] = net.module.state_dict()
                else:
                    state_dict[name] = net.state_dict()
        save_filename = '%s_net_%s.pth' % (epoch, self.opt.name)
        save_path = os.path.join(self.save_dir, save_filename)
        torch.save(state_dict, save_path)

    def __patch_instance_norm_state_dict(self, state_dict, module, keys, i=0):
        """Fix InstanceNorm checkpoints incompatibility (prior to 0.4)"""
        # keys:[module parameter ]
        key = keys[i]   # 得到该parameter的module name
        if i + 1 == len(keys):  # at the end, pointing to a parameter/buffer
            if module.__class__.__name__.startswith('InstanceNorm') and (key == 'running_mean' or key == 'running_var'):
                if getattr(module, key) is None:
                    state_dict.pop('.'.join(keys))
            if module.__class__.__name__.startswith('InstanceNorm') and (key == 'num_batches_tracked'):
                state_dict.pop('.'.join(keys))
        else:
            self.__patch_instance_norm_state_dict(state_dict, getattr(module, key), keys, i + 1)

    def load_networks(self):
        """Load all the networks from the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        load_path = self.opt.weight_path
        if not os.path.exists(load_path):
            raise IOError(f"Checkpoint '{load_path}' does not exist")
        print('loading the model from %s' % load_path)
        state_dict = torch.load(load_path, map_location=str(self.device))
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net_' + name)
                if isinstance(net, torch.nn.parallel.DataParallel) \
                        or isinstance(net, torch.nn.parallel.DistributedDataParallel):
                    net = net.module
                if name in state_dict.keys():
                    net_state_dict = state_dict[name]
                else:
                    net_state_dict = state_dict
                if hasattr(net_state_dict, '_metadata'):
                    del net_state_dict._metadata
                for key in list(net_state_dict.keys()):
                    self.__patch_instance_norm_state_dict(net_state_dict, net, key.split('.'))
                net.load_state_dict(net_state_dict)

    def print_networks(self, verbose):
        """Print the total number of parameters in the network and (if verbose) network architecture

        Parameters:
            verbose (bool) -- if verbose: print the network architecture
        """
        print('---------- Networks initialized -------------')
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net_' + name)
                num_params = 0
                for param in net.parameters():
                    num_params += param.numel()
                if verbose:
                    print(net)
                print('[Network %s] Total number of parameters : %.3f M' % (name, num_params / 1e6))
        print('-----------------------------------------------')

    @staticmethod
    def set_requires_grad(nets, requires_grad=False):  # self,
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

