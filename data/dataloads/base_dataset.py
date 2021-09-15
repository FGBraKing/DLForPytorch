import torch.utils.data as data

from abc import ABC, abstractmethod


class BaseDataset(data.Dataset, ABC):
    def __init__(self, opt):
        self.opt = opt
        self.root = opt.dataroot

    @abstractmethod
    def __len__(self):
        """Return the total number of images in the dataset."""
        return 0

    @abstractmethod
    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns:
            a dictionary of data with their names. It ususally contains the data itself and its metadata information.
        """
        pass


class CustomDataset(BaseDataset):

    def __init__(self, opt):
        super(CustomDataset, self).__init__(opt)
        self.paths = []  # should be [{'volume':volume,'label':label}, ...]
        self.data_size = len(self.paths)

        self.loader = None
        self.pre_transform = None
        self.transform = None
        self.post_transform = None

    def __getitem__(self, index):
        pass

    def _get_volume_label_array(self, index_used):
        pass

    def __len__(self):
        """Return the total number of images."""
        return self.data_size

    def _get_used_index(self, index):
        if self.opt.serial_batches:  # make sure index is within then range
            index_used = index % self.data_size
        else:
            index_used = self.opt.random_state.randint(0, self.data_size - 1)
        return index_used

    # 进行形状变换前的对volume进行的一些特殊处理,目前为空
    def _apply_pre_transform(self, volume):
        if self.pre_transform:
            volume = self.pre_transform(volume)
        return volume

    # 同时对volume和label进行的一些处理，主要包括，旋转、放缩、剪切，镜像，通道变换等
    def _apply_transform(self, volume, label):
        if self.transform:
            # # print(volume.shape)
            # # print(volume_path)
            # # print(label.shape)
            # # print(label_path)
            # volume_label = np.stack([volume, label], axis=0)    # array
            # volume_label = self.transform(volume_label)         # tensor
            # volume, label = volume_label[:-1, ...], volume_label[-1:, ...]
            # # label = torch.unsqueeze(label, dim=0)
            volume, label = self.transform(volume, label)
        return volume, label

    # 单独对volume做的一些处理，主要包括亮度、对比度、噪声变换等
    def _apply_post_transform(self, volume):
        if self.post_transform:
            volume = self.post_transform(volume)
        return volume

