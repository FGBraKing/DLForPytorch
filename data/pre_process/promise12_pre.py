# -*- coding:utf-8 -*-
import os
import re
import h5py
import random
import numpy as np
from glob import glob
import SimpleITK as sitk
import pandas as pd
from data.transforms.transforms import resize_image_itk, Compose
from functools import partial
import matplotlib.pyplot as plt
from scipy.ndimage.interpolation import zoom
# from scipy.interpolate import interp1d, interp2d


# 该函数已废弃，实际使用时出现了问题
def get_resampler(reference_image=None,
                  new_size=None, new_spacing=None,
                  new_orgin=None, new_direction=None,
                  resamplemethod=sitk.sitkNearestNeighbor
                  ):
    # sitk.sitkNearestNeighbor
    # sitk.sitkLinear
    '''you have to set the size at lease'''
    resampler = sitk.ResampleImageFilter()
    # resampler.SetNumberOfThreads(8)
    if reference_image:
        resampler.SetReferenceImage(reference_image)
    if new_size:
        resampler.SetSize(new_size)
    # 0.625,0.625,1.5
    if new_spacing:
        resampler.SetOutputSpacing(new_spacing)
    # (0.0, 0.0, 0.0)
    if new_orgin:
        resampler.SetOutputOrigin(new_orgin)
    # (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    if new_direction:
        resampler.SetOutputDirection(new_direction)
    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    resampler.SetInterpolator(resamplemethod)
    return resampler
# /data/project_data_lf/BraTS2018
# /data/project_data_lf/prostate_daf3d
# /data/project_data_lf/promise12/origin_data
# /data/project_data_lf/DATA/promise12/origin_data


def addition_process(*args, **kwargs):
    is_label = kwargs['is_label']

    old_origin = kwargs['img_info'][0]
    old_direction = kwargs['img_info'][1]
    old_spacing = kwargs['img_info'][2]

    img_array = kwargs['img']
    itk_img = sitk.GetImageFromArray(img_array)
    itk_img.SetSpacing(old_spacing)

    # get interpolator
    if is_label:
        # resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        resamplemethod = sitk.sitkNearestNeighbor
        N4BiasCorrect = False
        order = 0
    else:
        # resampler.SetInterpolator(sitk.sitkLinear)
        resamplemethod = sitk.sitkLinear
        N4BiasCorrect = True
        order = 3

    # resampler.SetOutputSpacing([0.625, 0.625, 1.5])  # 设置输出图像间距
    # resampler.SetOutputOrigin([0, 0, 0])
    # resampler.SetOutputDirection([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])

    itk_img_resized = resize_image_itk(itk_img, newSpacing=[0.625, 0.625, 1.5],
                                       resamplemethod=resamplemethod, N4BiasCorrect=False)
    img = sitk.GetArrayFromImage(itk_img_resized)  # z,y,x
    img_info = itk_img_resized.GetOrigin(), itk_img_resized.GetDirection(), itk_img_resized.GetSpacing()

    resize_factor = np.array(old_spacing, float) / [0.625, 0.625, 1.5]
    sci_img = zoom(img_array.transpose([2, 1, 0]), resize_factor, order=order, mode='constant', cval=0.0)

    if args:
        plt.figure(1)
        plt.imshow(img[10, :, :], cmap='gray')
        plt.show()

    return img, img_info, sci_img.transpose([2, 1, 0])


class PromisePre:
    def __init__(self, dataroot, patient_base=True, seed=1008):
        random.seed(seed)
        self.dataroot = dataroot
        self.patient_base = patient_base
        assert os.path.isdir(dataroot)
        used_dir_list = glob(self.dataroot+'/Training*')
        self.case_list = []
        for used_dir in used_dir_list:
            filelist = [os.path.join(used_dir, name) for name in os.listdir(used_dir)
                        if name.endswith('segmentation.mhd')]
            self.case_list.extend(filelist)
        case_pat = re.compile(r'\_segment\w+')
        self.patientlist = [{'image': case_pat.sub('', name), 'label': name} for name in self.case_list]
        self.patient_num = len(self.patientlist)

    def get_patient_num(self):
        return self.patient_num

    def shuffle_list(self):
        random.shuffle(self.patientlist)

    def _split_train_val_test(self, *ratio, shuffle=True):
        if np.sum(ratio[:3]) != 1:
            ratio = np.array(ratio) / np.sum(ratio[:3])
        patient_num = self.get_patient_num()
        train_num = int(patient_num * ratio[0])
        val_num = int(patient_num * ratio[1])
        # test_num = patient_num - train_num - val_num
        if shuffle:
            self.shuffle_list()
        return self.patientlist[:train_num], self.patientlist[train_num:train_num+val_num], self.patientlist[train_num+val_num:]

    def get_patient_list(self):
        return self.patientlist

    def process_and_save_data(self, save_root, split_ratio=(7, 1, 1), transform=None, save_csv=False):
        phase_list = ['train', 'val', 'test']
        data_list = self._split_train_val_test(*split_ratio)  # train_list, val_list, test_list
        if save_csv:
            data_name_list = [map(lambda x:os.path.basename(x['image']).split('.')[0], data) for data in data_list]
            # print(type(data_name_list[0]))  # 'map'
            data_df = pd.DataFrame(data=data_name_list, index=phase_list)
            # print(data_df)
            # print(data_df.T)
            data_df.T.to_csv(os.path.join(save_root, 'split.csv'), index=False)
        for data, phase in zip(data_list, phase_list):
            self._process_and_save_data(data, phase, transform, save_root)

    def _process_and_save_data(self, patient_list, phase, transform=None, save_root=None):
        save_dir = os.path.join(save_root, phase)
        print('save_dir:{}'.format(save_dir))
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        for patient in patient_list:
            img, img_info = self._read_img(patient['image'])    # z,y,x; C*H*W
            label, label_info = self._read_img(patient['label'])
            print('pre-process:')
            print('img:', img.shape, 'img_info:', img_info)
            # print('label:', label.shape, 'label_info:', label_info)

            img, img_info, sci_img = addition_process(img=img, img_info=img_info, is_label=False)
            label, label_info, sci_label = addition_process(img=label, img_info=label_info, is_label=True)

            img_warp = self._process_img(img, transform)
            label = self._process_img(label, transform)

            print('post-process:')
            print('img:', img_warp.shape, 'img_info:', img_info)
            # print('img_sci:', sci_img.shape, 'img_info:', img_info)
            # print('label:', label.shape, 'label_info:', label_info)

            patient_name = os.path.basename(patient['image'])[:-4]
            self._save_img(save_dir, patient_name+'_itk', img_warp, label, img_info=img_info, label_info=label_info)

    def _read_img(self, path):
        itk_img = sitk.ReadImage(path)
        img_array = sitk.GetArrayFromImage(itk_img)     # indexes are z,y,x    DHW
        # img_array = img_array.transpose([2, 1, 0])
        origin = itk_img.GetOrigin()
        direction = itk_img.GetDirection()
        spacing = itk_img.GetSpacing()
        return img_array, (origin, direction, spacing)

    def _process_img(self, img, transform):
        if transform:
            img = transform(img)
        return img

    def _save_img(self, save_dir, case_name, img, label, *args, **kwargs):
        # print('img: ', img.shape, 'label: ', label.shape)
        # f_write = h5py.File(path, 'w')
        # f_write.create_dataset('img', data=img)
        # f_write.create_dataset('label', data=label)
        # f_write.close()
        print(case_name)
        img_info = kwargs['img_info']
        label_info = kwargs['label_info']
        img_path = os.path.join(save_dir, case_name + '_image.nii')
        label_path = os.path.join(save_dir, case_name + '_label.nii')
        saveimg = sitk.GetImageFromArray(img)
        saveimg.SetOrigin(img_info[0])
        saveimg.SetDirection(img_info[1])
        saveimg.SetSpacing(img_info[2])
        savelabel = sitk.GetImageFromArray(label)
        savelabel.SetOrigin(label_info[0])
        savelabel.SetDirection(label_info[1])
        savelabel.SetSpacing(label_info[2])
        sitk.WriteImage(saveimg, img_path)
        sitk.WriteImage(savelabel, label_path)

    def print_cunstom_info(self, *args, **kwargs):
        shape_set = set()
        spacing_set = set()
        origin_set = set()
        direction_set = set()
        for patient in self.patientlist:
            img, img_info = self._read_img(patient['image'])
            label, label_info = self._read_img(patient['label'])
            shape_set.add(img.shape)
            spacing_set.add(img_info[2])
            origin_set.add(img_info[0])
            direction_set.add(img_info[1])
            # print(img.shape, img_info[2])
            # from utils.others.utils import print_numpy
            # print(patient['image'])
            # print_numpy(img)
            # from utils.others.img_io import show_array_histogram, show_pired_histogram
            # from utils.others.utils import clip_array
            # # show_array_histogram(img, title='pre histogram')
            # bin_edge, data_post = clip_array(img, rate=0.999, bins=1000, side_bin=True)
            # print(f'win:{bin_edge}')
            # print_numpy(data_post)
            # show_pired_histogram(img, data_post, bins=1000, title=os.path.basename(patient['image'])[:-4])
            # # from utils.others.img_io import show_volume_label, show_array_3d
            # # show_array_3d(img, title='pre-'+os.path.basename(patient['image'])[:-4], normalize_per=True)
            # # show_array_3d(data_post, title='post-'+os.path.basename(patient['image'])[:-4], normalize_per=True)

        print('shape: ')
        for shape in shape_set:
            print(shape)
        print('spacing: ')
        for space in spacing_set:
            print(space)
        print('origin:')
        for origin in origin_set:
            print(origin)
        print('direction:')
        for direction in direction_set:
            print(direction)
        print('origin', origin_set)
        print('direction', direction_set)
        print('shape:', shape_set)
        print('space:', spacing_set)


def main():
    dataroot = '/data/project_data_lf/DATA/promise12/origin_data'
    seed = 1008
    dataset = PromisePre(dataroot, seed=seed)
    # '/data/project_data_lf/SegMRTS/datasets/prostate_daf3d'
    # '/data/project_data_lf/PROJECT/DLForPytorch/datasets/promise12'
    # '/data/project_data_lf/PROJECT/SegMRTS/datasets/promise12/origin_data/'
    save_root = '/data/project_data_lf/PROJECT/DLForPytorch/datasets/promise12'
    split_ratio = (3, 1, 1)
    transform = Compose([])  # lambda x: np.transpose(x, [2, 1, 0])
    # dataset.process_and_save_data(save_root, split_ratio, save_csv=False, transform=transform)
    print(dataset.get_patient_num())
    dataset.print_cunstom_info()
    print("end")

    # for root, dirs, files in os.walk(save_root):
    #     for file in files:
    #         if file.endswith('.nii') and 'itk' not in file and 'sci' not in file:
    #             print(os.path.join(root, file))
    #             os.remove(os.path.join(root, file))


if __name__ == "__main__":
    main()


