import torch
import argparse
from torch.utils import data
from collections import OrderedDict

from models.modules.segmentation.three_d.unet3d_V0 import UNet3D
from utils.others.utils import convert_str_to_list
from utils.others.metrics import BinaryMetrics
from data.utils_data import combine_all_masks, get_unpad_image
from data.dataloads.promiseTest_dataset import PromiseTestDataset


# 训练和测试过程的数据处理过程差异极大，测试时采用滑窗+扩增方式产生数据，因此重新编写dataloader
# 模型部分也重新编写。模型的数据读取部分发生改变，最好重写。应该也可以继承覆盖，但是现在框架还不够成熟，多重写几遍可以更加精简干练。
# 如果需要的参数有所变化，也可以重写option部分


def define_net(opt, device):
    net = UNet3D(in_channels=opt.input_nc,
                 out_channels=opt.output_nc,
                 final_sigmoid=True,
                 conv_layer_order=opt.conv_order,
                 init_channel_number=opt.init_channel_number).to(device)
    return net


class TestModel:
    def __init__(self, opt):
        self.opt = opt
        self.device = torch.device('cuda:{}'.format(opt.gpu_ids[0])) if opt.gpu_ids else torch.device('cpu')
        self.model_names = ['segment']
        self.visual_names = ['segment_pad', 'label_pad']
        self.metric_names = ['basic_metrics', 'dice', 'recall', 'precision', 'accuracy']
        self.net_segment = define_net(opt, self.device)
        self.get_metrics = BinaryMetrics()
        if opt.no_augment:
            self.axis = ()
        else:
            self.axis = ((0,), (1,), (2,), (0, 1), (0, 2), (1, 2))

    def set_input(self, data_input):
        '''
        :param data_input:3 parts
         1:6D-tensor,which is bs,n,c,d,h,w. Generally bs have to be equal to 1
         2:4D-tensor,which is bs,d,h,w. Generally bs must be equal to 1
         3:2D-tensor, whose len is bs*3, meaning d,h,w of origin volume
        :return: None
        '''
        sub_volumes, label_pad, origin_shape = data_input
        bs, n, c, d, h, w = sub_volumes.size()
        self.origin_shape = origin_shape[0, :].type(torch.int16)
        self.label_pad = label_pad[0, :, :, :]
        self.sub_volumes_shape = sub_volumes.size()[1:]
        self.sub_volumes = sub_volumes.view(-1, 1, d, h, w).to(self.device)     # n*c,1,d,h,w

    def set_up(self):
        self.load_networks()
        self.print_networks(opt.verbose)

    def forward(self):
        try:
            self.segment = self.net_segment(self.sub_volumes)   # n*c,1,d,h,w
        except RuntimeError:
            self.segment = torch.zeros_like(self.sub_volumes, dtype=self.sub_volumes.dtype).to(self.device)
            for ind in range(self.sub_volumes.shape[0]):
                volume = self.sub_volumes[ind:ind+1, :, :, :, :]
                result = self.net_segment(volume)
                self.segment[ind:ind+1, :, :, :, :] = result
        # print(torch.mean(self.segment), torch.min(self.segment), torch.max(self.segment), torch.median(self.segment), torch.std(self.segment))
        # print(torch.max(self.segment))

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

    def test(self):
        with torch.no_grad():
            self.forward()
            self.compute_visuals()
            self.compute_metrics()

    def compute_visuals(self, recover=False, unpad=False):
        segment = self.segment.view(self.sub_volumes_shape).clone().cpu().numpy()  # n*c,1,d,h,w -> ncdhw
        label_pad = self.label_pad.clone().cpu().numpy()       # d h w
        # print_numpy(segment)
        # print_numpy(label_pad)
        segment[segment >= 0.5] = 1
        segment[segment < 0.5] = 0
        # print_numpy(segment)
        segment_pad = combine_all_masks(segment, aim_shape=label_pad.shape,
                                        stride=self.opt.stride, axises=self.axis)
        assert segment_pad.shape == label_pad.shape     # d,h,w
        if recover or unpad:
            segment_pad, label_pad = get_unpad_image(segment_pad.shape, self.origin_shape.tolist(), segment_pad, label_pad)
        self.segment_pad = segment_pad
        self.label_pad = label_pad   # dhw

    def compute_metrics(self, *args, **kwargs):
        self.metrics = self.get_metrics(self.segment_pad, self.label_pad, *self.metric_names, *args, **kwargs)
        keys = tuple(self.metric_names) + args
        self.metrics_dict = dict(zip(keys, self.metrics))

    def load_networks(self):
        load_path = self.opt.weight_path
        print('loading the model from %s' % load_path)
        state_dict = torch.load(load_path, map_location=self.device)
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net_' + name)
                if isinstance(net, torch.nn.DataParallel):
                    net = net.module
                if name in state_dict.keys():
                    net_state_dict = state_dict.get(name)
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

    def get_current_visuals(self):
        visual_ret = OrderedDict()
        for name in self.visual_names:
            if isinstance(name, str):
                visual_ret[name] = getattr(self, name)
        return visual_ret

    def get_current_metrics(self, need_name=True):
        if need_name:
            return self.metrics_dict
        else:
            return self.metrics


def get_option():
    parser = argparse.ArgumentParser()
    # common parameters
    parser.add_argument('--no_augment', action='store_true')
    parser.add_argument('--stride', type=str, default='24, 24, 8', help='the stride of slide windows')

    # model parameters
    parser.add_argument('--input_nc', type=int, default=1, help='# of input volume channels')
    parser.add_argument('--output_nc', type=int, default=1, help='# of output image channels')
    parser.add_argument('--conv_order', type=str, default='crb', help='# of the order of conv layer in the 3d-unet')
    parser.add_argument('--init_channel_number', type=int, default=32, help='the init channel number of unet')
    parser.add_argument('--gpu_ids', type=str, default='1', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--verbose', action='store_true', help='if specified, print more debugging information')
    parser.add_argument('--weight_path', type=str, default='/home/users/lf/CODE/PycharmProjects/DLForPytorch/traces/checkpoints/promise_unet_bs16_969632/latest_net_promise_unet_bs16_969632.pth')

    # dataset parameters
    parser.add_argument('--phase', type=str, default='test')
    parser.add_argument('--crop_size', type=str, default='96, 96, 32', help='the crop size of slide windows')
    parser.add_argument('--dataroot', type=str, default='/data/project_data_lf/PROJECT/DLForPytorch/datasets/promise12')
    parser.add_argument('--serial_batches', action='store_true')

    # others
    parser.add_argument('--DEBUG', action='store_true', help='if true, print the debug message')

    opt = parser.parse_args(args=['--DEBUG', '--serial_batches'])  # '--DEBUG', '--verbose'  , '--no_augment'
    opt.gpu_ids = convert_str_to_list(opt.gpu_ids, split=',', aim_type=int, condition=lambda x: x >= 0)
    opt.stride = convert_str_to_list(opt.stride, split=',', aim_type=int, condition=lambda x: x > 0)
    opt.crop_size = convert_str_to_list(opt.crop_size, split=',', aim_type=int, condition=lambda x: x > 0)

    if opt.DEBUG:
        print('stride:', opt.stride)
        print('gpu:', opt.gpu_ids)
        print('crop size:', opt.crop_size)
    return opt


if __name__ == '__main__':
    opt = get_option()
    dataset = PromiseTestDataset(opt)  # partial(h5_loader, names=('volume', 'label'))
    print('dataset_len:', len(dataset))
    dataloader = torch.utils.data.DataLoader(dataset, 1, shuffle=False, num_workers=1)

    model = TestModel(opt)
    model.set_up()
    model.eval()

    # tensor_dir = os.path.join(opt.logs_dir, opt.name, 'tensorboard_log_test')
    # mkdirs(tensor_dir)
    # # writer = SummaryWriter(logdir=tensor_dir, flush_secs=120,
    # #                        filename_suffix=opt.name, write_to_disk=True)

    for data_iter, data in enumerate(dataloader):
        model.set_input(data)
        model.test()
        visuals = model.get_current_visuals()
        metrics = model.get_current_metrics()
        print(str(metrics).replace('basic_metrics', 'tp fn tn fp'))
        # show_array_3d(visuals['segment_pad'])
        # show_array_3d(visuals['label_pad'])
        # show_volume_label(visuals['segment_pad'], visuals['label_pad'])
        # if writer:
        #     visuals_refine = {}
        #     for name, image in visuals.items():
        #         if image.ndim == 3:  # D H W
        #             for i_depth, d_image in enumerate(visuals):
        #                 visuals_refine[name+'D:{}'.format(i_depth)] = d_image
        #         else:
        #             visuals_refine[name] = image
        #     for name, image in visuals_refine.items():
        #         if image.ndim == 2:
        #             image = torch.unsqueeze(image, dim=0)
        #         writer.add_image(tag=name, img_tensor=image, global_step=data_iter)

