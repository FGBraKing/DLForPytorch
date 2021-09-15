import os

from data.utils_data import nii_loader
from data.dataloads.base_dataset import BaseDataset, CustomDataset
from data.transforms.transformOnArray import get_transform, get_pre_transform, get_post_transform, ToTensor


def get_trus_path(dataroot, data_phase):
    root = os.path.join(dataroot, data_phase)
    return [{'volume': os.path.join(root, name.replace('label', 'image')), 'label': os.path.join(root, name)}
            for name in os.listdir(root) if 'label' in name]


# teststr = 'elastic_resize_zoom_randomscale_randomcrop_ranomrotate_centercrop_transposeaxes_randomshift_rot90_mirror_'\
#           'gaussianNoise_GaussianBlur_brightness_BrightnessMultiplicative_contrast_simulate_gammatransform'


class TrusDataset(CustomDataset):
    def __init__(self, opt, loader=nii_loader):
        # save the option and dataset root
        super(TrusDataset, self).__init__(opt)

        self.paths = get_trus_path(opt.dataroot, opt.phase)  # should be [{'volume':volume,'label':label}, ...]
        self.data_size = len(self.paths)

        self.loader = loader
        self.pre_transform = get_pre_transform(opt)
        self.transform = get_transform(opt)
        self.post_transform = get_post_transform(opt)

        self.to_tensor = ToTensor(expand_dims=True)

    def __getitem__(self, index):

        index_used = self._get_used_index(index)

        volume_path = self.paths[index_used]['volume']
        label_path = self.paths[index_used]['label']
        volume = self.loader(volume_path)   # DHW, zyx
        label = self.loader(label_path)

        # 进行形状变换前的对volume进行的一些特殊处理,目前为空
        volume = self._apply_pre_transform(volume)
        # 同时对volume和label进行的一些处理，主要包括，旋转、放缩、剪切，镜像，通道变换等
        #
        volume, label = self._apply_transform(volume, label)
        # 单独对volume做的一些处理，主要包括亮度、对比度、噪声变换等
        volume = self._apply_post_transform(volume)

        volume = self.to_tensor(volume)
        label = self.to_tensor(label)

        return {'volume': volume, 'label': label, 'volume_path': volume_path, 'label_path': label_path}

    def custom_debug(self, *args, **kwargs):
        print(f'data_size:{self.data_size}')
        for index in range(self.data_size):
            if index < 5:
                tt = self.__getitem__(index)
                print(tt['volume_path'])
                print(tt['volume'].shape)
                print(type(tt['volume']))
                data = tt['volume'].cpu().numpy()
                label = tt['label'].cpu().numpy()
                print(type(label), label.shape)
                print(type(data), data.shape)
                # from utils.others.img_io import show_array_3d, show_volume_label
                # title = os.path.basename(tt['volume_path'])[:-4]
                # # show_array_3d(data[0, ...], 4, 4)
                # show_volume_label(data[0, ...], label[0, ...], 4, 4, title=title)
                # show_array_3d(label[0, ...])
            # print(torch.max(tt['volume']))
            # print(torch.max(tt['label']))


def main():
    import argparse
    parser = argparse.ArgumentParser(description='for the test of trus dataset')
    parser.add_argument('--dataroot', type=str,
                        default='/raid/lf/PROJECT/DLForPytorch/traces/datasets/prostate_daf3d_pre')
    parser.add_argument('--phase', type=str, default='train')
    parser.add_argument('--preprocess', type=str, default=None)
    parser.add_argument('--serial_batches', action='store_true')
    parser.add_argument('--seed', type=int, default=1008)
    parser.add_argument('--custom', action='store_true')
    parser.add_argument('--crop_size', type=list, default=[128, 128, 128])
    parser.add_argument('--order_data', type=int, default=3)
    parser.add_argument('--order_seg', type=int, default=0)
    opt = parser.parse_args(args=['--serial_batches', '--custom'])
    opt.preprocess = 'randomscale_randomcrop_ranomrotate_centercrop_rot90_mirror_' \
                     'gaussianNoise_GaussianBlur_BrightnessMultiplicative_contrast_simulate_gammatransform'

    dataset = TrusDataset(opt, loader=nii_loader)
    dataset.custom_debug(45)


if __name__ == '__main__':
    main()
