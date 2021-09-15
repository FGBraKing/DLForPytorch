# NOTE! OS: output stride, the ratio of input image resolution to final output resolution

import torch
import torch.nn as nn
import torch.nn.functional as F

from .resnet_3dbase import *


def make_layer(block, in_channels, channels, num_blocks, stride=1, dilation=1):
    strides = [stride] + [1] * (num_blocks - 1)

    blocks = []
    for stride in strides:
        blocks.append(block(in_channels=in_channels, channels=channels, stride=stride, dilation=dilation))
        in_channels = block.expansion * channels

    layer = nn.Sequential(*blocks)  # (*blocks: call with unpacked list entires as arguments)

    return layer


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, channels, stride=1, dilation=1):
        super(BasicBlock, self).__init__()

        out_channels = self.expansion * channels

        if type(dilation) != type(1):
            dilation = 1

        self.conv1 = nn.Conv3d(in_channels, channels, kernel_size=3, stride=(1, stride, stride), padding=(1, dilation, dilation), dilation=(1, dilation, dilation),
                               bias=False)
        # self.conv1 = nn.Conv3d(in_channels, channels, kernel_size=3, stride=stride, padding=dilation,
        #                        dilation=dilation, bias=False)
        self.bn1 = nn.BatchNorm3d(channels)

        self.conv2 = nn.Conv3d(channels, channels, kernel_size=3, stride=1, padding=(1, dilation, dilation), dilation=(1, dilation, dilation),
                               bias=False)
        self.bn2 = nn.BatchNorm3d(channels)

        if (stride != 1) or (in_channels != out_channels):
            conv = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
            conv = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=(1, stride, stride), bias=False)
            bn = nn.BatchNorm3d(out_channels)
            self.downsample = nn.Sequential(conv, bn)
        else:
            self.downsample = nn.Sequential()

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        out = out + self.downsample(x)

        out = F.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, channels, stride=1, dilation=1):
        super(Bottleneck, self).__init__()

        out_channels = self.expansion * channels

        self.conv1 = nn.Conv3d(in_channels, channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(channels)

        self.conv2 = nn.Conv3d(channels, channels, kernel_size=3, stride=stride, padding=dilation, dilation=dilation,
                               bias=False)
        self.bn2 = nn.BatchNorm3d(channels)

        self.conv3 = nn.Conv3d(channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(out_channels)

        if (stride != 1) or (in_channels != out_channels):
            conv = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
            bn = nn.BatchNorm3d(out_channels)
            self.downsample = nn.Sequential(conv, bn)
        else:
            self.downsample = nn.Sequential()

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        out = out + self.downsample(x)

        out = F.relu(out)

        return out


class ResNet_Bottleneck_OS16(nn.Module):
    def __init__(self, num_layers, in_channels):
        super(ResNet_Bottleneck_OS16, self).__init__()

        if num_layers == 50:
            resnet = resnet50(in_channels)
            resnet.load_state_dict(torch.load("./resources/resnet_50_23dataset.pth"))
            self.resnet = nn.Sequential(*list(resnet.children())[:-3])
        elif num_layers == 101:
            resnet = resnet101(in_channels)
            self.resnet = nn.Sequential(*list(resnet.children())[:-3])
        elif num_layers == 152:
            resnet = resnet152(in_channels)
            self.resnet = nn.Sequential(*list(resnet.children())[:-3])
        else:
            raise Exception("num_layers must be in {50, 101, 152}!")

        self.layer5 = make_layer(Bottleneck, in_channels=4 * 256, channels=512, num_blocks=3, stride=1, dilation=2)

    def forward(self, x):
        c4 = self.resnet(x)

        output = self.layer5(c4)

        return output


class ResNet_BasicBlock_OS16(nn.Module):
    def __init__(self, num_layers, in_channels):
        super(ResNet_BasicBlock_OS16, self).__init__()

        if num_layers == 18:
            resnet = resnet18(in_channels)
            net_dict = resnet.state_dict()
            pretrain = torch.load('./resources/resnet_18_23dataset.pth')
            pretrain_dict = {k[7:]: v for k, v in pretrain['state_dict'].items()}

            net_dict.update(pretrain_dict)
            resnet.load_state_dict(net_dict)
            self.resnet = nn.Sequential(*list(resnet.children())[:-3])

            num_blocks = 2

        elif num_layers == 34:
            resnet = resnet34(in_channels)
            net_dict = resnet.state_dict()
            pretrain = torch.load('./resources/resnet_34_23dataset.pth')
            pretrain_dict = {k[7:]: v for k, v in pretrain['state_dict'].items()}

            net_dict.update(pretrain_dict)
            resnet.load_state_dict(net_dict)
            print('Pretrain finished')
            self.resnet = nn.Sequential(*list(resnet.children())[:-3])

            num_blocks = 3
        else:
            raise Exception("num_layers must be in {18, 34}!")

        self.layer5 = make_layer(BasicBlock, in_channels=256, channels=512, num_blocks=num_blocks, stride=1, dilation=2)

    def forward(self, x):
        c4 = self.resnet(x)

        output = self.layer5(c4)

        return output


class ResNet_BasicBlock_OS8(nn.Module):
    def __init__(self, num_layers, in_channels):
        super(ResNet_BasicBlock_OS8, self).__init__()

        if num_layers == 18:
            resnet = resnet18(in_channels)

            # net_dict = resnet.state_dict()
            # pretrain = torch.load('./resources/resnet_18_23dataset.pth')
            # pretrain_dict = {k[7:]: v for k, v in pretrain['state_dict'].items()}
            #
            # net_dict.update(pretrain_dict)
            # resnet.load_state_dict(net_dict)
            # print('Pretrain finished')

            self.resnet = nn.Sequential(*list(resnet.children())[:-4])

            num_blocks_layer_4 = 2
            num_blocks_layer_5 = 2

        elif num_layers == 34:
            resnet = resnet34(in_channels)

            net_dict = resnet.state_dict()
            pretrain = torch.load('./resources/resnet_34_23dataset.pth')
            pretrain_dict = {k[7:]: v for k, v in pretrain['state_dict'].items()}

            net_dict.update(pretrain_dict)
            resnet.load_state_dict(net_dict)
            self.resnet = nn.Sequential(*list(resnet.children())[:-4])

            num_blocks_layer_4 = 6
            num_blocks_layer_5 = 3
        else:
            raise Exception("num_layers must be in {18, 34}!")

        self.layer4 = make_layer(BasicBlock, in_channels=128, channels=256, num_blocks=num_blocks_layer_4, stride=1,
                                 dilation=2)

        self.layer5 = make_layer(BasicBlock, in_channels=256, channels=512, num_blocks=num_blocks_layer_5, stride=1,
                                 dilation=4)

    def forward(self, x):
        c3 = self.resnet(x)

        output = self.layer4(c3)
        output = self.layer5(output)

        return output


def ResNet18_OS16(in_channels):
    return ResNet_BasicBlock_OS16(num_layers=18, in_channels=in_channels)


def ResNet50_OS16(in_channels):
    return ResNet_Bottleneck_OS16(num_layers=50, in_channels=in_channels)


def ResNet101_OS16(in_channels):
    return ResNet_Bottleneck_OS16(num_layers=101, in_channels=in_channels)


def ResNet152_OS16(in_channels):
    return ResNet_Bottleneck_OS16(num_layers=152, in_channels=in_channels)


def ResNet34_OS16(in_channels):
    return ResNet_BasicBlock_OS16(num_layers=34, in_channels=in_channels)


def ResNet18_OS8(in_channels):
    return ResNet_BasicBlock_OS8(num_layers=18, in_channels=in_channels)


def ResNet34_OS8(in_channels):
    return ResNet_BasicBlock_OS8(num_layers=34, in_channels=in_channels)