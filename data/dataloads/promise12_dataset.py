import os
import re
import h5py
import torch
import random
import argparse
import numpy as np
import torch.utils.data

from data.utils_data import nii_loader, h5_loader
from data.transforms import get_transform, get_pre_transform, get_post_transform
from data.dataloads.base_dataset import BaseDataset, CustomDataset
from data.transforms.transformOnArray import Normalize
from data.transforms.transformOnSample import random_scale, agent_resize
from utils.others.utils import print_numpy, clip_array, slim_array, convert_str_to_list
from utils.others.img_io import show_array_3d, show_volume_label, show_array_histogram, show_pired_histogram


def get_promise_path(dataroot, data_phase):
    # ------  Old version use nii
    # # if istrain:
    # #     A_root = os.path.join(dataroot, 'trainA')
    # #     B_root = os.path.join(dataroot, 'trainB')
    # # else:
    # #     A_root = os.path.join(dataroot, 'testA')
    # #     B_root = os.path.join(dataroot, 'testB')
    # root = os.path.join(dataroot, data_phase)
    # volume_paths = [os.path.join(root, name) for name in os.listdir(root) if 'itk_image' in name]
    # # label_paths = [name.replace('image', 'label') for name in volume_paths]
    # # return volume_paths, label_paths
    # return [{'volume': path, 'label': path.replace('image', 'label')} for path in volume_paths]
    # -----New version use h5
    root = os.path.join(dataroot, data_phase)
    return [os.path.join(root, name) for name in os.listdir(root) if name.endswith('h5')]


class Promise12Dataset(CustomDataset):
    def __init__(self, opt, loader=h5_loader):
        # save the option and dataset root
        super(Promise12Dataset, self).__init__(opt)
        # get the image paths of your dataset;
        self.paths = get_promise_path(opt.dataroot, opt.phase)
        self.data_size = len(self.paths)

        self.loader = loader
        self.pre_transform = get_pre_transform(opt)
        self.transform = get_transform(opt)
        self.post_transform = get_post_transform(opt)

    def __getitem__(self, index):
        index_used = self._get_used_index(index)
        ### Old version
        # volume_path = self.paths[index_used]['volume']
        # label_path = self.paths[index_used]['label']
        # volume = self.loader(volume_path)
        # label = self.loader(label_path)
        ### New version
        data_path = self.paths[index_used]
        volume, label = self.loader(data_path, 'volume', 'label')

        volume = self._apply_pre_transform(volume)
        volume = Normalize(volume.mean(), volume.std())(volume)
        # print_numpy(volume)

        if 'bothscale' in self.opt.preprocess.split('_'):
            volume, label = random_scale(volume, label, scale_in=0.2, execution_probability=0.2)
        if 'bothresize' in self.opt.preprocess.split('_'):
            volume = agent_resize(volume, self.opt.crop_size[::-1], order=3,  mode='constant', cval=0.0)
            label = agent_resize(label, self.opt.crop_size[::-1], order=1,  mode='constant', cval=0.0)

        volume, label = self._apply_transform(volume, label)
        volume = self._apply_post_transform(volume)

        return {'volume': volume, 'label': label, 'path': data_path}
        # return {'volume': volume, 'label': label, 'volume_path': volume_path, 'label_path': label_path}

    def __len__(self):
        """Return the total number of images."""
        return self.data_size

    def custom_debug(self, *args, **kwargs):
        pat = re.compile(r'.*(Case\d+).*')
        for index in range(self.data_size):
            if index < 100:
                tt = self.__getitem__(index)
                print(tt['path'])
                data = tt['volume'].cpu().numpy()
                label = tt['label'].cpu().numpy()
                print(f'data shape:{data.shape}')
                # print(f'label shape:{label.shape}')
                # print_numpy(data)
                # print_numpy(label)
                title = pat.match(tt['path']).groups()[0]
                # show_array_3d(data[0, ...], 4, 4, title='vol-'+title)
                show_volume_label(data[0, ...], label[0, ...], 4, 3, title=title, add_line=True, normalize_per=True)


class CustomDatasetDataLoader:
    def __init__(self, dataset):
        self.dataset = dataset
        self.batch_size = 8
        self.dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size, pin_memory=False,
                                                      shuffle=True, num_workers=1)
        self.max_dataset_size = float('inf')

    def load_data(self):
        return self

    def __len__(self):
        """Return the number of data in the dataset"""
        return min(len(self.dataset), self.max_dataset_size)

    def __iter__(self):
        """Return a batch of data"""
        for i, data in enumerate(self.dataloader):
            if i * self.batch_size >= self.max_dataset_size:
                print('max_dataset_size:{}'.format(self.max_dataset_size))
                break
            yield data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', type=str, default='/data/project_data_lf/PROJECT/DLForPytorch/datasets/promise12')
    parser.add_argument('--phase', type=str, default='test')
    parser.add_argument('--preprocess', type=str, default='GaussianNoise_bothresize_rot90_flip')  # 'GaussianNoise_crop_rotate_centercrop_rot90_flip'
    parser.add_argument('--gaussian_sigma', type=str, default='0.0,0.1')
    parser.add_argument('--crop_size', type=str, default='224,224,36', help='the crop size of slide windows')
    parser.add_argument('--angle_spectrum', type=int, default=45, help='random rotate, angle')
    parser.add_argument('--seed', type=int, default=1008)
    parser.add_argument('--custom', action='store_true')
    parser.add_argument('--serial_batches', action='store_true')
    opt = parser.parse_args(args=['--custom'])

    opt.crop_size = convert_str_to_list(opt.crop_size, split=',', aim_type=int, condition=lambda x: x > 0)
    opt.gaussian_sigma = convert_str_to_list(opt.gaussian_sigma, split=',', aim_type=float, condition=lambda x: x >= 0)
    dataset = Promise12Dataset(opt, loader=h5_loader)  # partial(h5_loader, names=('volume', 'label'))
    print('dataset_len:', len(dataset))
    dataset.custom_debug()

    # dataloader = CustomDatasetDataLoader(dataset)
    # ind = 0
    # for epoch in range(1000):
    #     for data in dataloader:
    #         ind += 1
    #         print('epoch', epoch)
    #         # print(ind)
    #         print(data['path'])
    #     # print(data.size())
    #     # print(data.keys())
    #     # print(type(data['volume']))
    #     # print(data['volume'].size())
    #     # print(type(data['path']))
    #     # print(data['path'])


def process_nii2h5(root):
    pat = re.compile(r'.*(Case\d+).*')
    volume_paths = [os.path.join(root, name) for name in os.listdir(root) if 'itk_image' in name]
    label_paths = [name.replace('image', 'label') for name in volume_paths]
    print('len:', (len(volume_paths)))
    i = 1
    for volume_path, label_path in zip(volume_paths, label_paths):
        volume = nii_loader(volume_path)
        label = nii_loader(label_path)
        print('loaded nii file')
        print('volume shape:', volume.shape)
        print_numpy(volume)
        print_numpy(label)
        data_label_slim = slim_array(np.stack([volume, label], axis=0), dims=(1,))
        data_slim = data_label_slim[0, ...]
        label_slim = data_label_slim[1, ...]
        print('slim end')
        bin_edge, data_post = clip_array(data_slim, rate=0.999, bins=1000, side_bin=True)
        print('clip end')
        data_nor = Normalize(np.mean(data_post), np.std(data_post), eps=1e-6)(data_post)
        print('normal end')
        print('data_nor shape:', data_nor.shape)
        print_numpy(data_nor)
        print_numpy(label_slim)

        name = pat.match(volume_path).groups()[0]
        show_volume_label(data_nor, label_slim, 4, 3, title=name, add_line=True, normalize_per=True)

        # save_name = os.path.join(os.path.dirname(volume_path), name+'.h5')
        # fw = h5py.File(save_name, mode='w')
        # fw.create_dataset(name='volume', data=data_nor)
        # fw.create_dataset(name='label', data=label_slim)
        # fw.close()
        # print(f'{i} end')
        # i += 1


if __name__ == '__main__':
    main()

