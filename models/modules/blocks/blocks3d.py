import torch
import torch.nn as nn
from torch.nn import functional as F

#   Conv3d(self, in_channels, out_channels, kernel_size, stride=1,
#                  padding=0, dilation=1, groups=1,
#                  bias=True, padding_mode='zeros'):


def conv3x3(in_planes, out_planes, stride=1,
            padding=1, dilation=1, groups=1,
            bias=False, padding_mode='zeros'):
    """3x3 convolution with padding"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=padding, bias=bias,
                     dilation=dilation, groups=groups, padding_mode=padding_mode)


def conv1x1(in_planes, out_planes, stride=1,
            padding=1, dilation=1, groups=1,
            bias=False, padding_mode='zeros'):
    """1x1 convolution"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride, bias=bias,
                     padding=padding, dilation=dilation, groups=groups, padding_mode=padding_mode)


def conv1x1_bn_ac(in_planes, out_planes, stride=1, activation=nn.ReLU(inplace=True)):
    """1x1 convolution + 3d batchnorm + activation"""
    return nn.Sequential(
            conv1x1(in_planes, out_planes, stride),
            nn.BatchNorm3d(out_planes),
            activation)


def conv3x3_bn_ac(in_planes, out_planes, stride=1, activation=nn.ReLU(inplace=True)):
    """1x1 convolution + 3d batchnorm + activation"""
    return nn.Sequential(
            conv3x3(in_planes, out_planes, stride),
            nn.BatchNorm3d(out_planes),
            activation)


class Block3x3(nn.Module):

    def __init__(self, inplanes, midplanes, outplanes, activation=nn.ReLU(inplace=True)):
        super(Block3x3, self).__init__()
        self.ac = activation
        self.conv1 = conv3x3(inplanes, midplanes)
        self.bn1 = nn.BatchNorm3d(midplanes)
        self.conv2 = conv3x3(midplanes, outplanes)
        self.bn2 = nn.BatchNorm3d(outplanes)
        self.conv3 = conv3x3(outplanes, outplanes)
        self.bn3 = nn.BatchNorm3d(outplanes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.ac(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.ac(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.ac(out)

        return out


class ExpandConv(nn.Module):
    '''(conv => BN => ReLU) * 2'''

    def __init__(self, in_planes, mid_planes, out_planes, activation=nn.ReLU(inplace=True)):
        super(ExpandConv, self).__init__()
        self.conv = nn.Sequential(
            conv3x3(in_planes, mid_planes // 2),
            nn.BatchNorm3d(mid_planes // 2),
            activation,
            conv3x3(mid_planes // 2, mid_planes),
            nn.BatchNorm3d(mid_planes),
            activation,
            conv3x3(mid_planes, mid_planes),
            nn.BatchNorm3d(mid_planes),
            activation,
            conv1x1(mid_planes, out_planes)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class ExpandConvAddlow(nn.Module):
    '''(conv => BN => ReLU) * 2'''

    def __init__(self, in_planes, mid_planes, low_planes, out_planes):
        super(ExpandConvAddlow, self).__init__()
        self.conv = nn.Sequential(
            conv3x3(in_planes, mid_planes // 2),
            nn.BatchNorm3d(mid_planes // 2),
            # nn.ReLU(inplace=True),
            conv3x3(mid_planes // 2, mid_planes),
            nn.BatchNorm3d(mid_planes),
            # nn.ReLU(inplace=True)
        )
        self.merge = nn.Sequential(
            conv3x3(mid_planes + low_planes, out_planes * 2 // 3),
            nn.BatchNorm3d(out_planes * 2 // 3),
            # nn.ReLU(inplace=True),
            conv1x1(out_planes * 2 // 3, out_planes)
        )

    def forward(self, x, low_feat):
        x = self.conv(x)
        low_feat = F.interpolate(low_feat, size=x.size()[2:], mode='bilinear', align_corners=True)
        merge_input = torch.cat((x, low_feat), dim=1)
        out = self.merge(merge_input)
        return out


class MergeConv(nn.Module):

    def __init__(self, inplanes, midplanes, outplanes, activation=nn.ReLU(inplace=True)):
        super(MergeConv, self).__init__()
        self.conv = nn.Sequential(
            conv1x1(inplanes, midplanes),
            nn.BatchNorm3d(midplanes),
            activation,
            conv1x1(midplanes, outplanes)
        )

    def forward(self, x):
        x = self.conv(x)
        return x
