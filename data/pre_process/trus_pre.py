# -*- coding:utf-8 -*-
import os
import re
import h5py
import numpy as np
import pandas as pd
import SimpleITK as sitk
from collections import OrderedDict
from batchgenerators.utilities.file_and_folder_operations import save_json

from utils.others.utils import mkdir
from data.transforms.transformOnArray import normalize
from data.transforms.transforms import resize_image_itk, Compose
from data.transforms.transformOnArray import standardize
from data.pre_process.dataset_pre import DatasetPreOne
from skimage.transform import resize
from scipy.ndimage.interpolation import zoom

# 二维：cv2.resize()，np.resize()
# 三维
# 1. scipy.ndimage.interpolation.zoom()
# 2. torch.nn.functional.interpolate()


def h52nii(dataroot):
    phase_list = ['train', 'test', 'val']
    def h5_read(path, name):
        with h5py.File(path, mode='a') as f:
            data = f.get(name)[:]   # # W D H
            del f[name]
            f[name] = data.transpose([2, 0, 1])  # H W D
        return data.transpose([1, 2, 0])  # D H W
    def nii_save(path, data):
        itk_img = sitk.GetImageFromArray(data)
        sitk.WriteImage(itk_img, path)

    for phase in phase_list:
        root_dir = os.path.join(dataroot, phase)
        filelist = [file for file in os.listdir(root_dir) if file.endswith('h5')]
        for file in filelist:
            print(file)
            path = os.path.join(root_dir, file)
            image = h5_read(path, 'img')  # D H W (80,132,170)
            label = h5_read(path, 'label')
            print(image.shape)
            label_path = os.path.join(root_dir, file.replace('data', 'label').replace('h5', 'nii'))
            image_path = os.path.join(root_dir, file.replace('data', 'image').replace('h5', 'nii'))
            print(image_path)
            nii_save(label_path, label)
            nii_save(image_path, image)


def nii2gz(src_file, save_dir, is_label):
    name = os.path.basename(src_file)
    print(name)
    if is_label:
        save_file = os.path.join(save_dir, name[:-10]+'.nii.gz')
    else:
        save_file = os.path.join(save_dir, name[:-10]+'_0000.nii.gz')
    sitk.WriteImage(sitk.ReadImage(src_file), save_file)


# /data/project_data_lf/BraTS2018
# /data/project_data_lf/prostate_daf3d
class TrusPre(DatasetPreOne):
    @staticmethod
    def addition_process(img, img_info, *args, **kwargs):
        '''
        :param img: DHW
        :param img_info: origin, direction, spacing
        :param args:
        :param kwargs: 'kit','do_separate_z','is_label'
        :return:
        '''
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

    def __init__(self, dataroot, seed=1008, **kwargs):
        super(TrusPre, self).__init__(dataroot, seed, **kwargs)
        # self.img_list = []
        # self.label_list = []
        # for dirpath, dirnames, filenames in os.walk(dataroot):
        #     for filename in filenames:
        #         if filename.endswith('image.nii'):
        #             self.img_list.append(os.path.join(dirpath, filename))
        #         elif filename.endswith('label.nii'):
        #             self.label_list.append(os.path.join(dirpath, filename))
        self.img_root = os.path.join(self.dataroot, 'image')
        self.label_root = os.path.join(self.dataroot, 'label')
        pat_num = re.compile(r'P(\d+)\_')
        patient_numlist = [pat_num.findall(name)[0] for name in os.listdir(self.img_root)]
        self.case_list = [{'image': os.path.join(self.img_root, 'P'+name+r'_image.nii'),
                          'label': os.path.join(self.label_root, 'P' + name + r'_label.nii')}
                          for name in patient_numlist]
        self.case_num = len(self.case_list)

    # img_resized = resize_image_itk(img_itk, newSize=(170, 132, 80), resamplemethod=sitk.sitkLinear)

    def print_custom_info(self, *args, **kwargs):
        w_set = set()
        h_set = set()
        d_set = set()
        all_set = set()
        shape_set = set()
        for patient in self.case_list:
            label_path = patient['label']
            # print(patient)
            img_itk = sitk.ReadImage(label_path)
            img_shape = img_itk.GetSize()
            w_set.add(img_shape[0])
            h_set.add(img_shape[1])
            d_set.add(img_shape[-1])
            all_set.update(img_shape)
            shape_set.add(img_shape)
        print('width:', w_set)
        print('height:', h_set)
        print('depth:', d_set)
        print('all:', all_set)

    def process_for_nnunet(self, save_root, split_ratio=(3, 1)):
        phase_list = ['Tr', 'Ts']
        data_list_list = self.split_train_val_test(*split_ratio)  # train_list, val_list, test_list
        self._process_save_json_for_nnunet(save_root, data_list_list)
        for data_list, phase in zip(data_list_list, phase_list):
            img_save_dir = os.path.join(save_root, 'images'+phase)
            label_save_dir = os.path.join(save_root, 'labels'+phase)
            mkdir(img_save_dir)
            mkdir(label_save_dir)
            for data in data_list:
                img_path = data['image']
                label_path = data['label']
                nii2gz(img_path, img_save_dir, False)
                nii2gz(label_path, label_save_dir, True)

    def _process_save_json_for_nnunet(self, save_root, train_test_list):
        json_dict = OrderedDict()
        json_dict['name'] = "USProstate"
        json_dict['description'] = "prostate"
        json_dict['tensorImageSize'] = "4D"
        json_dict['reference'] = "see challenge website"
        json_dict['licence'] = "see challenge website"
        json_dict['release'] = "0.0"
        json_dict['modality'] = {
            "0": "US",
        }
        json_dict['labels'] = {
            "0": "background",
            "1": "prostate"
        }
        json_dict['numTraining'] = 30
        json_dict['numTest'] = 10
        train_list = train_test_list[0]
        test_list = train_test_list[1]

        json_dict['training'] = [{'image': "./imagesTr/%s.nii.gz" % train['image'].split("/")[-1][:-10],
                                  "label": "./labelsTr/%s.nii.gz" % train['label'].split("/")[-1][:-10]}
                                 for train in train_list]
        json_dict['test'] = ["./imagesTr/%s.nii.gz" % test['image'].split("/")[-1][:-10] for test in test_list]

        save_json(json_dict, os.path.join(save_root, "dataset.json"))


if __name__ == "__main__":
    dataroot = '/raid/lf/DATA/prostate_daf3d'
    save_root = '/raid/lf/PROJECT/DLForPytorch/traces/datasets/prostate_daf3d_pre'
    if not os.path.exists(save_root):
        os.mkdir(save_root)

    dataset = TrusPre(dataroot, seed=1008)

    dataset.process_and_save_data(save_root=save_root,
                                  split_ratio=(3, 1, 1),
                                  transform=None,
                                  save_csv=True,
                                  split_name='split.csv',
                                  save_type='nii',
                                  if_slim=True,
                                  do_separate_z=False,
                                  kit='sci')
    # dataset.print_custom_info()
    print(dataset.get_patient_num())
    # # dataset.process_for_nnunet(save_root, split_ratio)
    # h52nii(save_root)
    print("end")


# sitk.ReadImage(img_path)
# <class 'SimpleITK.SimpleITK.Image'>
# ['CopyInformation', 'EraseMetaData',
# 'GetDimension', 'GetDirection', 'GetDepth', 'GetHeight', 'GetWidth', 'GetSize', 'GetSpacing', 'GetOrigin',
# 'GetITKBase', 'GetMetaData', 'GetMetaDataKeys',
# 'GetNumberOfComponentsPerPixel', 'GetNumberOfPixels',
# 'GetPixel', 'GetPixelAsComplexFloat64', 'GetPixelID', 'GetPixelIDTypeAsString', 'GetPixelIDValue',
# 'HasMetaDataKey', 'MakeUnique',
# 'SetDirection', 'SetMetaData', 'SetOrigin', 'SetPixel', 'SetPixelAsComplexFloat64', 'SetSpacing',
# 'TransformContinuousIndexToPhysicalPoint', 'TransformIndexToPhysicalPoint',
# 'TransformPhysicalPointToContinuousIndex', 'TransformPhysicalPointToIndex', 'this']

# nib.load(img_path)
# <class 'nibabel.nifti1.Nifti1Image'>
# ['ImageArrayProxy', 'ImageSlicer', 'affine', 'as_reoriented', 'dataobj', 'extra',
# 'file_map', 'files_types', 'filespec_to_file_map', 'filespec_to_files', 'from_bytes',
# 'from_file_map', 'from_filename', 'from_files', 'from_image', 'get_affine', 'get_data',
# 'get_data_dtype', 'get_fdata', 'get_filename', 'get_header', 'get_qform', 'get_sform',
# 'get_shape', 'header', 'header_class', 'in_memory', 'instance_to_filename', 'load',
# 'make_file_map', 'makeable', 'ndim', 'orthoview', 'path_maybe_image', 'rw', 'set_data_dtype',
# 'set_filename', 'set_qform', 'set_sform', 'shape', 'slicer', 'to_bytes', 'to_file_map',
# 'to_filename', 'to_files', 'to_filespec', 'uncache', 'update_header', 'valid_exts']



