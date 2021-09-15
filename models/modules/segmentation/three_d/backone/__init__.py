from . import resnet_3d


def build_3dbackbone(backbone, in_channels=1):
    if backbone == 'resnet18_os8':
        return resnet_3d.ResNet18_OS8(in_channels=in_channels)
    elif backbone == 'resnet34_os8':
        return resnet_3d.ResNet34_OS8(in_channels=in_channels)
    elif backbone == 'resnet18_os16':
        return resnet_3d.ResNet18_OS16(in_channels=in_channels)
    elif backbone == 'resnet_os16':
        return resnet_3d.ResNet34_OS16(in_channels=in_channels)
    else:
        raise NotImplementedError
