import os
import torch
import argparse
import numpy as np
from torch.utils import data
from collections import OrderedDict
from torchvision.transforms import transforms

from models.modules.segmentation.three_d.unet3d_V0 import UNet3D
from data.utils_data import h5_loader
from utils.others.utils import Timer, convert_str_to_list
from utils.others.metrics import BinaryMetrics, MutiClassMetrics
from data.transforms.transformOnArray import ToTensor, agent_resize


def define_net(opt, device):
    net = UNet3D(in_channels=opt.input_nc,
                 out_channels=opt.output_nc,
                 final_sigmoid=True,
                 conv_layer_order=opt.conv_order,
                 init_channel_number=opt.init_channel_number).to(device)
    return net


class PredictModel:
    def __init__(self, opt):
        self.opt = opt
        self.device = torch.device('cuda:{}'.format(opt.gpu_ids[0])) if opt.gpu_ids else torch.device('cpu')
        self.model_names = ['segment']
        self.visual_names = ['volume', 'segment']
        self.net_segment = define_net(opt, self.device)
        self.volume = None
        self.segment = None

    def set_input(self, data_input):
        self.volume = data_input.to(self.device)

    def set_up(self):
        self.load_networks()
        self.print_networks(opt.verbose)

    def forward(self):
        self.segment = self.net_segment(self.volume)

    def eval(self):
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

    def load_networks(self):
        load_path = self.opt.weight_path
        print('loading the model from %s' % load_path)
        state_dict = torch.load(load_path, map_location=str(self.device))
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net_' + name)
                if isinstance(net, torch.nn.DataParallel):
                    net = net.module
                try:
                    net.load_state_dict(state_dict[name])
                except KeyError:
                    net.load_state_dict(state_dict)

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

    def get_current_visuals(self):
        visual_ret = OrderedDict()
        for name in self.visual_names:
            if isinstance(name, str):
                visual_ret[name] = getattr(self, name)
        return visual_ret


class PromisePredictDataset(data.Dataset):
    def __init__(self, opt, loader):
        super(PromisePredictDataset, self).__init__()
        self.opt = opt
        self.root = opt.dataroot
        # get the image paths of your dataset;
        root = os.path.join(opt.dataroot, opt.phase)
        self.paths = [os.path.join(root, name) for name in os.listdir(root) if name.endswith('h5')]
        self.data_size = len(self.paths)
        self.loader = loader

        # transformer
        self.volume_transform = transforms.Compose([#NormalizeRange(dtype=np.float32),
                                                    # CenterCrop(crop_size=opt.crop_size),
                                                    ToTensor(expand_dims=True)])
        self.label_transform = transforms.Compose([#CenterCrop(crop_size=opt.crop_size),
                                                   ToTensor(expand_dims=True)])

    def __getitem__(self, index):
        index_used = index % self.data_size
        data_path = self.paths[index_used]
        volume, label = self.loader(data_path, 'volume', 'label')
        origin_shape = volume.shape
        if opt.resize:
            volume = agent_resize(volume, self.opt.crop_size[::-1], order=3,  mode='constant', cval=0.0)
            label = agent_resize(label, self.opt.crop_size[::-1], order=1,  mode='constant', cval=0.0)
        volume = self.volume_transform(volume)
        label = self.label_transform(label)
        return volume, label, origin_shape

    def __len__(self):
        """Return the total number of images."""
        return self.data_size

    def custom_debug(self, *args, **kwargs):
        pass


def get_option():
    parser = argparse.ArgumentParser()
    # model parameter
    parser.add_argument('--gpu_ids', type=str, default='2',
                        help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--weight_path', type=str,
                        default='/home/users/lf/CODE/PycharmProjects/DLForPytorch/traces/checkpoints/promise_unet_testDDP_45/latest_net_promise_unet_testDDP_45.pth')
    parser.add_argument('--input_nc', type=int, default=1, help='input volume channels')
    parser.add_argument('--output_nc', type=int, default=1, help=' output image channels')
    parser.add_argument('--conv_order', type=str, default='crb', help='# of the order of conv layer in the 3d-unet')
    parser.add_argument('--init_channel_number', type=int, default=32, help='the init channel number of unet')
    parser.add_argument('--verbose', action='store_true', help='if specified, print more debugging information')
    parser.add_argument('--crop_size', type=str, default='128, 128, 32', help='the crop size of slide windows')
    parser.add_argument('--resize', action='store_true', help='resize the input data')
    # dataset parameter
    parser.add_argument('--dataroot', type=str, default='/data/project_data_lf/PROJECT/DLForPytorch/datasets/promise12')
    parser.add_argument('--phase', type=str, default='test')

    opt = parser.parse_args(args=[])  # '--verbose' '--resize'
    opt.gpu_ids = convert_str_to_list(opt.gpu_ids, split=',', aim_type=int, condition=lambda x: x >= 0)
    opt.crop_size = convert_str_to_list(opt.crop_size, split=',', aim_type=int, condition=lambda x: x > 0)

    return opt


if __name__ == '__main__':
    metrics_keys = ['basic_metrics', 'dice', 'recall', 'precision', 'accuracy']
    opt = get_option()
    binary_metrics = BinaryMetrics()
    multi_metrics = MutiClassMetrics()
    dataset = PromisePredictDataset(opt, loader=h5_loader)  # partial(h5_loader, names=('volume', 'label'))
    print('dataset_len:', len(dataset))
    dataloader = torch.utils.data.DataLoader(dataset, 1, shuffle=False, num_workers=1)
    model = PredictModel(opt)
    model.set_up()
    model.eval()
    with Timer('no grad out loop: %f'):
        with torch.no_grad():
            for data_iter, data in enumerate(dataloader):
                volume, label, origin_shape = data
                model.set_input(volume)
                model.forward()
                visuals = model.get_current_visuals()

                volume = np.squeeze(visuals['volume'].detach().clone().cpu().numpy(), axis=(0, 1))
                label = np.squeeze(label.detach().clone().cpu().numpy(), axis=(0, 1))
                segment = np.squeeze(visuals['segment'].detach().clone().cpu().numpy(), axis=(0, 1))
                # print_numpy(visuals['segment'].detach().clone().cpu().numpy())
                # show_volume_label(volume, segment, 4, 4, title='volume_segment')
                # show_volume_label(segment, label, 4, 4, title='segment,label')
                # print(segment.shape, label.shape)
                if opt.resize:
                    segment = agent_resize(segment, origin_shape, order=1,  mode='constant', cval=0.0)
                    label = agent_resize(label, origin_shape, order=1,  mode='constant', cval=0.0)
                # print_numpy(segment)
                # print_numpy(label)
                # print(segment.shape, label.shape)
                print('all metrics:', str(dict(zip(metrics_keys, binary_metrics(segment, label, *metrics_keys, mode=0)))).replace('basic_metrics', 'tp fn tn fp'))
                # for c in range(segment.shape[0]):
                #     print('channel metrics:', str(dict(zip(metrics_keys, binary_metrics(segment[c, ...], label[c, ...], *metrics_keys, mode=0)))).replace('basic_metrics', 'tp fn tn fp'))
                # # print('channel metrics:', multi_metrics(np.expand_dims(segment, axis=1),
                # #                                         np.expand_dims(label, axis=1),
                # #                                         'basic_metrics', 'DC', 'recall', 'precision',
                # #                                         reduce='None'))

