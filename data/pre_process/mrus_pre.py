# -*- coding:utf-8 -*-
import os
import re
import h5py
import random
import imageio
import pandas as pd
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from pprint import pprint
from data.utils_data import save_nii
from skimage.transform import resize
from scipy.ndimage.interpolation import map_coordinates
from utils.others.utils import get_foreground_shape, print_numpy, clip_array, slim_array, convert_str_to_list
from data.pre_process.dataset_pre import DatasetPre

from data.transforms.transforms import resize_image_itk, Compose
from data.transforms.transformOnArray import standardize
from scipy.ndimage.interpolation import zoom


class MrusPre(DatasetPre):

    @staticmethod
    def addition_process(img, img_info, *args, **kwargs):
        if 'kit' in kwargs.keys():
            kit = kwargs['kit']
        else:
            kit = 'itk'

        if 'do_separate_z' in kwargs.keys():
            do_separate_z = kwargs['do_separate_z']
        else:
            do_separate_z = False

        new_spacing = [2, 2, 2]
        is_label = kwargs['is_label']
        old_origin = img_info[0]
        old_direction = img_info[1]
        old_spacing = img_info[2]

        itk_img = sitk.GetImageFromArray(img)
        itk_img.SetSpacing(old_spacing)

        if is_label:
            resamplemethod = sitk.sitkNearestNeighbor
            order = 0
        else:
            resamplemethod = sitk.sitkBSplineResamplerOrder3        # sitkBSplineResamplerOrder3     sitk.sitkLinear
            order = 3

        # resampler.SetOutputSpacing([0.625, 0.625, 1.5])  # 设置输出图像间距
        # resampler.SetOutputOrigin([0, 0, 0])
        # resampler.SetOutputDirection([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
        if do_separate_z:
            old_shape = img.shape[::-1]
            new_shape = np.array(old_shape) * old_spacing / new_spacing
            new_shape = np.round(new_shape)
            # x, y
            reshaped_data = []
            for slice_id in range(img.shape[0]):
                reshaped_data.append(resize(img[slice_id, :, :], new_shape[:-1][::-1], order,
                                            cval=0, mode='edge', anti_aliasing=False))
            reshaped_data = np.stack(reshaped_data, axis=0)     # z y x
            # z
            resize_factor_z = old_spacing[0] / new_spacing[0]
            resize_factor = [1, 1, resize_factor_z]
            out_img = zoom(reshaped_data.transpose([2, 1, 0]), resize_factor, order=0, mode='nearest', cval=0.0)
            # other info
            new_spacing_refine = (np.array(old_shape) * old_spacing / out_img.shape).tolist()
            out_info = img_info[0], img_info[1], new_spacing_refine
            out_img = out_img.transpose([2, 1, 0])
            print('new spacing:{}'.format(new_spacing_refine))
            # if new_shape[-1] != img.shape[0]:
            #     # copied from nnunet
            #     rows, cols, dim = new_shape[0], new_shape[1], new_shape[2]
            #     orig_dim, orig_cols, orig_rows = reshaped_data.shape
            #
            #     row_scale = float(orig_rows) / rows
            #     col_scale = float(orig_cols) / cols
            #     dim_scale = float(orig_dim) / dim
            #
            #     map_rows, map_cols, map_dims = np.mgrid[:rows, :cols, :dim]
            #     map_rows = row_scale * (map_rows + 0.5) - 0.5
            #     map_cols = col_scale * (map_cols + 0.5) - 0.5
            #     map_dims = dim_scale * (map_dims + 0.5) - 0.5
            #
            #     coord_map = np.array([map_rows, map_cols, map_dims])
            #
            #     reshaped_final_data = map_coordinates(reshaped_data, coord_map,
            #                                           order=order, cval=0, mode='nearest')[None]
        elif kit == 'itk':
            print('origin spacing:{}'.format(old_spacing))
            itk_img_resized = resize_image_itk(itk_img,
                                               newSpacing=new_spacing,
                                               newOrigin=old_origin,
                                               newDirection=old_direction,
                                               resamplemethod=resamplemethod,
                                               N4BiasCorrect=False)
            out_img = sitk.GetArrayFromImage(itk_img_resized)  # z,y,x
            out_info = itk_img_resized.GetOrigin(), itk_img_resized.GetDirection(), itk_img_resized.GetSpacing()
            print('new spacing:{}'.format(old_spacing))
        else:
            print('origin spacing:{}'.format(old_spacing))
            resize_factor = np.array(old_spacing, float) / new_spacing
            out_img = zoom(img.transpose([2, 1, 0]), resize_factor, order=order, mode='constant', cval=0.0)
            new_spacing_refine = (np.array(img.shape[::-1]) * old_spacing / out_img.shape).tolist()
            out_info = img_info[0], img_info[1], new_spacing_refine
            out_img = out_img.transpose([2, 1, 0])
            print('new spacing:{}'.format(new_spacing_refine))
        if not is_label:
            out_img = standardize(out_img, out_img.mean(), out_img.std())

        return out_img, out_info

    def __init__(self, dataroot, seed=1008, mode='all', **kwargs):
        super(MrusPre).__init__(dataroot, seed, kwargs)
        # save the parameters
        self.mode = mode
        # get the image and label path among all modal
        assert os.path.isdir(self.dataroot)
        random.seed(seed)
        mr_case = []
        for root, dirs, files in os.walk(self.dataroot):
            for name in files:
                if name.endswith('.nii') and 'MR' in name and 'state' not in name:
                    # print(name)
                    mr_case.append(os.path.join(root, name))
        pat = re.compile(r'_MR')
        self.case_mr = [{'image': path,
                         'label': pat.sub('_MR_Prostate', path)} for path in mr_case]
        self.case_us = [{'image': pat.sub('_US', path),
                         'label': pat.sub('_US_Prostate', path)} for path in mr_case]

    def shuffle_list(self):
        if self.mode.lower() == 'mr':
            random.shuffle(self.case_mr)
        elif self.mode.lower() == 'us':
            random.shuffle(self.case_us)
        else:
            random.shuffle(self.case_mr)
            random.shuffle(self.case_us)

    def get_patient_list(self):
        if self.mode.lower() == 'mr':
            return self.case_mr
        elif self.mode.lower() == 'us':
            return self.case_us
        else:
            return {'mr': self.case_mr, 'us': self.case_us}

    def split_train_val_test(self, *ratio, shuffle=True):
        if np.sum(ratio[:3]) != 1:
            ratio = np.array(ratio) / np.sum(ratio[:3])

        if shuffle:
            self.shuffle_list()

        mr_num = len(self.case_mr)
        train_mr_num = int(mr_num * ratio[0])
        val_mr_num = int(mr_num * ratio[1])

        us_num = len(self.case_us)
        train_us_num = int(us_num * ratio[0])
        val_us_num = int(us_num * ratio[1])

        if self.mode.lower() == 'mr':
            return self.case_mr[:train_mr_num], \
                   self.case_mr[train_mr_num:train_mr_num+val_mr_num], \
                   self.case_mr[train_mr_num+val_mr_num:]

        elif self.mode.lower() == 'us':
            return self.case_us[:train_us_num], \
                   self.case_us[train_us_num:train_us_num+val_us_num], \
                   self.case_us[train_us_num+val_us_num:]
        else:
            return {'mr': [self.case_mr[:train_mr_num],
                           self.case_mr[train_mr_num:train_mr_num+val_mr_num],
                           self.case_mr[train_mr_num+val_mr_num:]],
                    'us': [self.case_us[:train_us_num],
                           self.case_us[train_us_num:train_us_num+val_us_num],
                           self.case_us[train_us_num+val_us_num:]]}

    def get_patient_num(self):
        if self.mode.lower() == 'mr':
            return len(self.case_mr)
        elif self.mode.lower() == 'us':
            return len(self.case_us)
        else:
            return {'mr': len(self.case_mr), 'us': len(self.case_us)}

    def process_and_save_data(self, save_root, split_ratio=(3, 1, 1), transform=None,
                              save_csv=False, split_name='split.csv', **kwargs):
        '''
        :param save_root: The save_root of preprocess data
        :param split_ratio: train:val:test
        :param transform:
        :param save_csv: whether to save csv file
        :param split_name: the csv file's name
        :param kwargs:
            1 process parameter:
                modal: folder's prefix
                save_type: file save type
                if_slim: whether to slim array
            2. addition_process's parameter:
                ...
        :return: None
        '''
        phase_list = ['train', 'val', 'test']
        data_list = self.split_train_val_test(*split_ratio)  # train_list, val_list, test_list
        if save_csv:
            if self.mode.lower() == 'mr':
                data_name_list = [map(lambda x:os.path.basename(x['image']).split('.')[0], data) for data in data_list]  # 三个map对象
                data_df = pd.DataFrame(data=data_name_list, index=phase_list)
                data_df.T.to_csv(os.path.join(save_root, 'mr_'+split_name), index=False)
            elif self.mode.lower() == 'us':
                data_name_list = [map(lambda x:os.path.basename(x['image']).split('.')[0], data) for data in data_list]  # 三个map对象
                data_df = pd.DataFrame(data=data_name_list, index=phase_list)
                data_df.T.to_csv(os.path.join(save_root, 'us_'+split_name), index=False)
            else:
                mr_name_list = [map(lambda x:os.path.basename(x['image']).split('.')[0], data) for data in data_list['mr']]
                us_name_list = [map(lambda x:os.path.basename(x['image']).split('.')[0], data) for data in data_list['us']]
                data_df = pd.DataFrame(data=mr_name_list+us_name_list, index=phase_list+phase_list)
                data_df.T.to_csv(os.path.join(save_root, 'mr_us_'+split_name), index=False)

        if self.mode.lower() == 'mr':
            for data, phase in zip(data_list, phase_list):
                self._process_and_save_data(data, phase, transform, save_root, modal='mr', save_type='nii', **kwargs)
        elif self.mode.lower() == 'us':
            for data, phase in zip(data_list, phase_list):
                self._process_and_save_data(data, phase, transform, save_root, modal='us', save_type='nii', **kwargs)
        else:
            for data, phase in zip(data_list['mr'], phase_list):
                self._process_and_save_data(data, phase, transform, save_root, modal='mr', save_type='nii', **kwargs)
            for data, phase in zip(data_list['us'], phase_list):
                self._process_and_save_data(data, phase, transform, save_root, modal='us', save_type='nii', **kwargs)

    def print_cunstom_info(self, *args, **kwargs):
        self.spec_info(*args, **kwargs)
        # shape_set = set()
        # spacing_set = set()
        # origin_set = set()
        # direction_set = set()
        # label_size_set = set()
        # label_shape_set = set()
        # split_ratio = (3, 1, 1)
        # phase_list = ['train', 'val', 'test']
        # data_list = self.split_train_val_test(*split_ratio)  # train_list, val_list, test_list
        # for patient_list, phase in zip(data_list, phase_list):
        #     print(phase)
        #     for patient in patient_list:
        #         print(patient['image'])
        #         img, img_info = self._read_img(patient['image'])    # z,y,x; D*H*W
        #         label, label_info = self._read_img(patient['label'])
        #
        #         label_size = tuple(map(lambda x: x[1]-x[0], get_foreground_shape(label)))
        #         scan_size = tuple(map(lambda x, y: x*y, img.shape[::-1], img_info[2]))
        #         print(f'scan_size:{scan_size}')
        #
        #         # print('shape:{}, origin:{}, direction:{}, spacing{}'.format(img.shape, *img_info))
        #         label_shape_set.add(tuple(get_foreground_shape(label)))
        #         label_size_set.add(label_size)
        #         shape_set.add(img.shape)
        #         spacing_set.add(img_info[2])
        #         origin_set.add(img_info[0])
        #         direction_set.add(img_info[1])
        # print('label_size: ')
        # for lb_size in label_size_set:
        #     print(lb_size)
        # print('label_shape: ')
        # for lb_shape in label_shape_set:
        #     print(lb_shape)
        # # print('shape: ')
        # # for shape in shape_set:
        # #     print(shape)
        # # print('spacing: ')
        # # for space in spacing_set:
        # #     print(space)
        # # print('origin:')
        # # for origin in origin_set:
        # #     print(origin)
        # # print('direction:')
        # # for direction in direction_set:
        # #     print(direction)
        # # print('origin', origin_set)
        # # print('direction', direction_set)
        # # print('shape:', shape_set)
        # # print('space:', spacing_set)

    def spec_info(self, *args, **kwargs):
        patient = self.get_patient_list()
        num = self.get_patient_num()
        print(num)
        print(patient)


def main():
    dataroot = r'/raid/lf/DATA/MR-USviaFenster20/'
    saveroot = r'/raid/lf/PROJECT/DLForPytorch/traces/datasets/MR-USviaFenster20/'
    if not os.path.exists(saveroot):
        os.mkdir(saveroot)

    dataset = MrusPre(dataroot=dataroot, mode='us')
    dataset.process_and_save_data(saveroot, save_csv=True, if_slim=True, kit='sci', do_separate_z=False)
    # dataset.print_cunstom_info()


if __name__ == "__main__":
    main()


