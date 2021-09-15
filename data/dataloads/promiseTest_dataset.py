import os
import re
import h5py
import torch
import random
import argparse
import numpy as np
import torch.utils.data
import torch.utils.data as data
import torchvision.transforms as transforms

from data.utils_data import nii_loader, h5_loader, get_flip_volumes, slide_crop, get_pad_image
from utils.others.utils import print_numpy, clip_array, slim_array, convert_str_to_list
from data.transforms.transformOnArray import random_scale, NormalizeRange, Normalize, normalize, ToTensor
from utils.others.img_io import show_array_3d, show_volume_label, show_array_histogram, show_pired_histogram


def get_promise_path(dataroot, data_phase):
    root = os.path.join(dataroot, data_phase)
    return [os.path.join(root, name) for name in os.listdir(root) if name.endswith('h5')]


class PromiseTestDataset(data.Dataset):
    def __init__(self, opt, loader=h5_loader):
        self.opt = opt
        self.dataroot = opt.dataroot

        self.paths = get_promise_path(opt.dataroot, opt.phase)
        self.data_size = len(self.paths)
        self.loader = loader

        if opt.no_augment:
            self.axis = ()
        else:
            self.axis = ((0,), (1,), (2,), (0, 1), (0, 2), (1, 2))

    def __getitem__(self, index):
        if self.opt.serial_batches:  # make sure index is within then range
            index_used = index % self.data_size
        else:
            index_used = random.randint(0, self.data_size - 1)

        data_path = self.paths[index_used]
        volume, label = self.loader(data_path, 'volume', 'label')

        # volume = NormalizeRange(dtype=np.float32)(volume)    # dhw
        assert isinstance(volume, np.ndarray)

        sub_volumes = slide_crop(volume, crop_size=self.opt.crop_size, stride=self.opt.stride, mode='minimum')  # NCDHW=n1dhw
        label_pad = get_pad_image(label, crop_size=self.opt.crop_size, stride=self.opt.stride, mode='minimum')  # dhw

        if self.opt.no_augment:
            volume_out = sub_volumes
        else:
            sub_volumes = sub_volumes.squeeze(axis=1)  # NDHW
            volume_list = []
            for i in range(sub_volumes.shape[0]):
                sub_volumes[i, ...] = sub_volumes[i, ...]  # normalize(sub_volumes[i, ...])
                volume_list.append(get_flip_volumes(sub_volumes[i, ...], self.axis))
            volume_out = np.stack(volume_list, axis=0)   # NCDHW

        volume_out = torch.from_numpy(volume_out.astype(dtype=np.float32))
        label_pad = torch.from_numpy(label_pad.astype(dtype=np.float32))
        shape = torch.tensor(volume.shape, requires_grad=False)
        return volume_out, label_pad, shape

    def __len__(self):
        """Return the total number of images."""
        return self.data_size

    def custom_debug(self, *args, **kwargs):
        pass
        # index = kwargs['index']
        # volume, label, volume_shape = self.__getitem__(index)  # volume:NCDHW
        # print(volume.size())
        # print(label.size())
        # n, c, d, h, w = volume.size()
        # volume = volume.view(-1, d, h, w)
        # predict = net(volume)
        # predict = predict.view(n, c, d, h, w)
        # predict = predict.cpu().numpy()
        # label = label.cpu().numpy()
        # pre_mask = combine_all_masks(predict, label.shape)
        # assert pre_mask.shape == label.shape
        # pad_shape = np.array(label.shape) - np.array(volume_shape)
        # d_pad_l, d_pad_r = int(np.floor(pad_shape[0]/2)), int(np.ceil(pad_shape[0]/2))
        # h_pad_l, h_pad_r = int(np.floor(pad_shape[1]/2)), int(np.ceil(pad_shape[1]/2))
        # w_pad_l, w_pad_r = int(np.floor(pad_shape[2]/2)), int(np.ceil(pad_shape[2]/2))
        # d_r = label.shape[0] - d_pad_r + 1
        # h_r = label.shape[1] - h_pad_r + 1
        # w_r = label.shape[2] - w_pad_r + 1
        # print(d_pad_l,d_pad_r,h_pad_l,h_pad_r,w_pad_l,w_pad_r)
        # pre_mask = pre_mask[d_pad_l:d_r, h_pad_l:h_r, w_pad_l:w_r]
        # label = label[d_pad_l:d_r, h_pad_l:h_r, w_pad_l:w_r]
        # print(pre_mask.shape)
        # print(volume_shape)


def net(m_in):
    return torch.randint(0, 2, m_in.size())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', type=str, default='/data/project_data_lf/PROJECT/DLForPytorch/datasets/promise12')
    parser.add_argument('--phase', type=str, default='test')
    parser.add_argument('--no_augment', action='store_true')
    parser.add_argument('--serial_batches', action='store_true')
    parser.add_argument('--stride', type=str, default='32, 32, 8', help='the stride of slide windows')
    parser.add_argument('--crop_size', type=str, default='128, 128, 32', help='the crop size of slide windows')
    opt = parser.parse_args(args=['--serial_batches'])

    opt.crop_size = convert_str_to_list(opt.crop_size, split=',', aim_type=int, condition=lambda x: x > 0)
    opt.stride = convert_str_to_list(opt.stride, split=',', aim_type=int, condition=lambda x: x > 0)

    dataset = PromiseTestDataset(opt, loader=h5_loader)
    print('dataset_len:', len(dataset))
    # dataset.custom_debug(index=45)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
    ind = 1
    for test_data in dataloader:
        ind += 1
        print(test_data[0].size())
        print(test_data[1].size())
        print(test_data[2].size())

        if ind > 1:
            break


if __name__ == '__main__':
    main()

