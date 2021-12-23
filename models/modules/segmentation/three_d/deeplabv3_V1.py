import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------Encode-------------------------
# 3D Resnet
def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, input_channels, block, layers, num_classes=1, zero_init_residual=False):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv3d(input_channels, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        # self.avgpool = nn.AdaptiveAvgPool3d((1, 1))
        self.avgpool = nn.AvgPool3d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm3d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet18(input_channels, **kwargs):
    model = ResNet(input_channels, BasicBlock, [2, 2, 2, 2], **kwargs)
    return model


def resnet34(input_channels, **kwargs):
    model = ResNet(input_channels, BasicBlock, [3, 4, 6, 3], **kwargs)
    return model


def resnet50(input_channels, **kwargs):
    model = ResNet(input_channels, Bottleneck, [3, 4, 6, 3], **kwargs)
    return model


def resnet101(input_channels, **kwargs):
    model = ResNet(input_channels, Bottleneck, [3, 4, 23, 3], **kwargs)
    return model


def resnet152(input_channels, **kwargs):
    model = ResNet(input_channels, Bottleneck, [3, 8, 36, 3], **kwargs)
    return model


# Resnet for Deeplab
def make_layer(block, in_channels, channels, num_blocks, stride=1, dilation=1):
    strides = [stride] + [1] * (num_blocks - 1)

    blocks = []
    for stride in strides:
        blocks.append(block(in_channels=in_channels, channels=channels, stride=stride, dilation=dilation))
        in_channels = block.expansion * channels

    layer = nn.Sequential(*blocks)  # (*blocks: call with unpacked list entires as arguments)

    return layer


class BasicBlockV1(nn.Module):
    expansion = 1

    def __init__(self, in_channels, channels, stride=1, dilation=1):
        super(BasicBlockV1, self).__init__()

        out_channels = self.expansion * channels

        if type(dilation) != type(1):
            dilation = 1

        self.conv1 = nn.Conv3d(in_channels, channels, kernel_size=3, stride=(1, stride, stride), padding=(1, dilation, dilation), dilation=(1, dilation, dilation),
                               bias=False)
        self.bn1 = nn.BatchNorm3d(channels)

        self.conv2 = nn.Conv3d(channels, channels, kernel_size=3, stride=1, padding=(1, dilation, dilation), dilation=(1, dilation, dilation),
                               bias=False)
        self.bn2 = nn.BatchNorm3d(channels)

        if (stride != 1) or (in_channels != out_channels):
            # conv = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
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


class BottleneckV1(nn.Module):
    expansion = 4

    def __init__(self, in_channels, channels, stride=1, dilation=1):
        super(BottleneckV1, self).__init__()

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

        self.layer5 = make_layer(BottleneckV1, in_channels=4 * 256, channels=512, num_blocks=3, stride=1, dilation=2)

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

        self.layer5 = make_layer(BasicBlockV1, in_channels=256, channels=512, num_blocks=num_blocks, stride=1, dilation=2)

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

        self.layer4 = make_layer(BasicBlockV1, in_channels=128, channels=256, num_blocks=num_blocks_layer_4, stride=1,
                                 dilation=2)

        self.layer5 = make_layer(BasicBlockV1, in_channels=256, channels=512, num_blocks=num_blocks_layer_5, stride=1,
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


# build
def build_3dbackbone(backbone, in_channels=1):
    if backbone == 'resnet18_os8':
        return ResNet18_OS8(in_channels=in_channels)
    elif backbone == 'resnet34_os8':
        return ResNet34_OS8(in_channels=in_channels)
    elif backbone == 'resnet18_os16':
        return ResNet18_OS16(in_channels=in_channels)
    elif backbone == 'resnet_os16':
        return ResNet34_OS16(in_channels=in_channels)
    else:
        raise NotImplementedError


# ----------------------------------------Decode-----------------------------------------------
class _ASPPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation, BatchNorm, activation=nn.ReLU(inplace=True)):
        super(_ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv3d(inplanes, planes, kernel_size=kernel_size,
                                     stride=1, padding=padding, dilation=dilation, bias=False)
        self.bn = BatchNorm(planes)
        self.ac = activation

        self._init_weight()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)
        x = self.ac(x)
        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class ASPP(nn.Module):
    def __init__(self, backbone, output_stride, BatchNorm, activation=nn.ReLU(inplace=True)):
        super(ASPP, self).__init__()
        assert backbone in ['resnet18_os8', 'resnet18_os16', 'resnet34_os8', 'resnet_os16', 'drn', 'mobilenet', 'xception'], \
            "Not support this backbone!"

        if backbone in ['drn', 'resnet18_os8', 'resnet18_os16', 'resnet34_os8']:
            inplanes = 512
        elif backbone == 'mobilenet':
            inplanes = 320
        else:
            inplanes = 2048
        if output_stride == 16:
            dilations = [1, 6, 12, 18]
        elif output_stride == 8:
            dilations = [1, 12, 24, 36]
        else:
            raise NotImplementedError

        self.aspp1 = _ASPPModule(inplanes, 256, 1, padding=0, dilation=dilations[0], BatchNorm=BatchNorm, activation=activation)
        self.aspp2 = _ASPPModule(inplanes, 256, 3, padding=dilations[1], dilation=dilations[1], BatchNorm=BatchNorm, activation=activation)
        self.aspp3 = _ASPPModule(inplanes, 256, 3, padding=dilations[2], dilation=dilations[2], BatchNorm=BatchNorm, activation=activation)
        self.aspp4 = _ASPPModule(inplanes, 256, 3, padding=dilations[3], dilation=dilations[3], BatchNorm=BatchNorm, activation=activation)
        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool3d((1, 1, 1)),
                                             nn.Conv3d(inplanes, 256, 1, stride=1, bias=False),
                                             # BatchNorm(256),
                                             activation)
        self.conv1 = nn.Conv3d(1280, 256, 1, bias=False)
        self.bn1 = BatchNorm(256)
        self.ac = activation

        self.dropout = nn.Dropout(0.5)
        # if backbone == 'resnet18_os8':
        #     self.dropout = nn.Sequential()
        # else:
        #     self.dropout = nn.Dropout(0.5)
        self._init_weight()

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='trilinear', align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.ac(x)

        return self.dropout(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


#  ---------------------------------------Model------------------------------------------
class DeepLabV3_3D(nn.Module):

    def __init__(self, backbone='resnet18_os8', in_channels=1, output_stride=16, n_classes=21, final_sigmoid=True):
        super(DeepLabV3_3D, self).__init__()
        assert backbone in ['resnet18_os8', 'resnet18_os16',
                            'resnet34_os8', 'resnet_os16'], "Not support this backbone!"

        if backbone in ['resnet18_os8', 'resnet34_os8']:
            output_stride = 8

        self.backbone = build_3dbackbone(backbone, in_channels=in_channels)
        self.aspp = ASPP(backbone, output_stride, nn.BatchNorm3d, activation=nn.ReLU(inplace=True))

        self.output_layer = nn.Conv3d(256, n_classes, kernel_size=1)

        if final_sigmoid:
            self.final_activation = nn.Sigmoid()
        else:
            self.final_activation = nn.Softmax(dim=1)

    def logits_resnet(self, x):
        # (x has shape (batch_size, 3, h, w))
        x = self.backbone(x)
        # (shape: (batch_size, 512, h/16, w/16)) (assuming self.resnet is ResNet18_OS16 or ResNet34_OS16.
        # If self.resnet is ResNet18_OS8 or ResNet34_OS8, it will be (batch_size, 512, h/8, w/8).
        # If self.resnet is ResNet50-152, it will be (batch_size, 4*512, h/16, w/16))
        return x

    def logits_aspp(self, x):
        x = self.logits_resnet(x)
        x = self.aspp(x)  # (shape: (batch_size, num_classes, h/16, w/16))
        return x

    def forward(self, x):

        features = self.logits_aspp(x)
        output = self.output_layer(features)
        output = F.upsample(output, size=(x.size()[2], x.size()[3], x.size()[4]), mode="trilinear", align_corners=True)
        # (shape: (batch_size, num_classes, d, h, w))

        if not self.training:
            output = self.final_activation(output)

        return output


if __name__ == "__main__":
    from torch.autograd import Variable
    from models.auxiliary_funs import print_model_parm_nums, print_model_parm_flops

    device = torch.device(f"cuda:{1}" if torch.cuda.is_available() else 'cpu')
    net = DeepLabV3_3D(backbone='resnet18_os8', in_channels=3, output_stride=8, n_classes=1, final_sigmoid=True).to(device)
    input = Variable(torch.rand((1, 3, 16, 256, 256)), requires_grad=True).to(device)

    print_model_parm_nums(net)  # 44.41M
    print_model_parm_flops(net, input, need_idx=False)  # 105.59G
