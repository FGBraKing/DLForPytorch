# -*- coding:utf-8 -*-
import os
import h5py
import random
import numpy as np
import pandas as pd
import SimpleITK as sitk
from glob import glob
from abc import ABC, abstractmethod
from data.utils_data import save_nii, nii_loader, npy_loader, h5_loader
from utils.others.utils import slim_array


class DatasetPre(ABC):
    ''' abstract class'''
    def __init__(self, dataroot, seed, **kwargs):
        self.dataroot = dataroot
        self.seed = seed
        random.seed(self.seed)
        self.random_state = np.random.RandomState(seed=seed)
        self.kwargs = kwargs

    @staticmethod
    @abstractmethod
    def addition_process(img, img_info, *args, **kwargs):
        return img, img_info

    @abstractmethod
    def shuffle_list(self):
        pass

    @abstractmethod
    def get_patient_list(self):
        pass

    @abstractmethod
    def get_patient_num(self):
        pass

    @abstractmethod
    def split_train_val_test(self, *ratio, shuffle=True):
        pass

    @abstractmethod
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
        pass

    def _process_and_save_data(self, patient_list, phase, transform=None, save_root=None,
                               modal='', save_type='nii', if_slim=True, **kwargs):
        save_dir = os.path.join(save_root, modal+phase)
        print('save_dir:{}'.format(save_dir))
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        for patient in patient_list:
            print(patient)
            img, img_info = self._read_img(patient['image'])    # z,y,x; D*H*W
            label, label_info = self._read_img(patient['label'])

            if if_slim:
                img_label_slim = slim_array(np.stack([img, label], axis=0), dims=(1, 2, 3))
                img = img_label_slim[0, ...]
                label = img_label_slim[1, ...]

            img, img_info = self.addition_process(img=img, img_info=img_info, is_label=False, **kwargs)
            label, label_info = self.addition_process(img=label, img_info=label_info, is_label=True, **kwargs)

            img_warp = self._process_img(img, transform)
            label = self._process_img(label, transform)

            # TODO need to be modified when type change
            patient_name = os.path.basename(patient['image']).split('.')[0]
            self._save_img(save_dir, patient_name, img_warp, label,
                           img_info=img_info, label_info=label_info, save_type=save_type)

    @staticmethod
    def _read_img(path):
        filename, filetype = os.path.splitext(path)
        if filetype.lower() == '.nii':
            itk_img = sitk.ReadImage(path)
            img_array = sitk.GetArrayFromImage(itk_img)     # indexes are z,y,x    DHW
            origin = itk_img.GetOrigin()
            direction = itk_img.GetDirection()
            spacing = itk_img.GetSpacing()
        elif filetype.lower() == '.npy':
            img_array = npy_loader(path)
            origin, direction, spacing = None, None, None
        elif filetype.lower() == '.h5':
            img_array = h5_loader(path, 'image', 'label')[-1]
            origin, direction, spacing = None, None, None
        else:
            raise TypeError('Filetype is unsupported')
        return img_array, (origin, direction, spacing)

    @staticmethod
    def _process_img(img, transform):
        if transform:
            img = transform(img)
        return img

    @staticmethod
    def _save_img(save_dir, case_name, image, label, **kwargs):
        '''
        :param save_dir:
        :param case_name:
        :param image:
        :param label:
        :param kwargs: the key may be ['img_info', 'label_info', 'mode']
        :return:
        '''
        if 'img_info' in kwargs.keys():
            img_info = kwargs['img_info']
        else:
            img_info = None

        if 'label_info' in kwargs.keys():
            label_info = kwargs['label_info']
        else:
            label_info = None

        if 'mode' in kwargs.keys():
            mode = kwargs['mode']
        else:
            mode = 'nii'

        if mode == 'nii':
            assert img_info
            assert label_info
            img_path = os.path.join(save_dir, case_name + '_image.nii')
            label_path = os.path.join(save_dir, case_name + '_label.nii')

            if img_info is None:
                img_info = img_info,
            if label_info is None:
                label_info = label_info,
            save_nii(img_path, image, *img_info)
            save_nii(label_path, label, *label_info)

        elif mode == 'h5':
            save_path = os.path.join(save_dir, case_name + '.h5')
            fwrite = h5py.File(save_path, mode='w')
            fwrite.create_dataset(name='image', data=image)
            fwrite.create_dataset(name='label', data=label)
            fwrite.close()

        elif mode == 'npy':
            save_path = os.path.join(save_dir, case_name + '.npy')
            np.save(save_path, {'image': image, 'label': label})
        else:
            raise TypeError('It is not supported type of \'{}\' until now'.format(mode))

    @abstractmethod
    def print_cunstom_info(self, *args, **kwargs):
        pass


class DatasetPreOne(DatasetPre):
    ''' just a example, subclass need to finish:__init__, addition_process,
     process_and_save_data and print_cunstom_info'''

    def __init__(self, dataroot, seed=1008, **kwargs):
        super(DatasetPreOne, self).__init__(dataroot, seed, **kwargs)
        assert os.path.isdir(dataroot)
        self.case_list = []
        self.case_num = len(self.case_list)

    @staticmethod
    def addition_process(img, img_info, *args, **kwargs):
        return img, img_info

    def shuffle_list(self):
        random.shuffle(self.case_list)
        # self.random_state.shuffle(self.case_list)

    def get_patient_list(self):
        return self.case_list

    def get_patient_num(self):
        return self.case_num

    def split_train_val_test(self, *ratio, shuffle=True):
        if np.sum(ratio[:3]) != 1:
            ratio = np.array(ratio) / np.sum(ratio[:3])

        if shuffle:
            self.shuffle_list()

        train_num = int(self.case_num * ratio[0])
        val_num = int(self.case_num * ratio[1])
        return self.case_list[:train_num], self.case_list[train_num:train_num + val_num], self.case_list[train_num + val_num:]

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
            data_name_list = [map(lambda x:os.path.basename(x['image']).split('.')[0], data) for data in data_list]
            data_df = pd.DataFrame(data=data_name_list, index=phase_list)
            data_df.T.to_csv(os.path.join(save_root, split_name), index=False)
        for data, phase in zip(data_list, phase_list):
            self._process_and_save_data(data, phase, transform, save_root, **kwargs)

    def print_cunstom_info(self, *args, **kwargs):
        pass


class DatasetPreMul(DatasetPre):
    ''' just a example, subclass need to finish all of rest method'''

    def __init__(self, dataroot, seed=1008, **kwargs):
        super(DatasetPreMul, self).__init__(dataroot, seed, kwargs)
        assert os.path.isdir(dataroot)

    @staticmethod
    def addition_process(img, img_info, *args, **kwargs):
        return img, img_info

    def shuffle_list(self):
        pass

    def get_patient_list(self):
        pass

    def get_patient_num(self):
        pass

    def split_train_val_test(self, *ratio, shuffle=True):
        pass

    def process_and_save_data(self, save_root, transform=None):
        pass

    def print_cunstom_info(self, *args, **kwargs):
        pass


def main():
    pass


if __name__ == "__main__":
    main()


