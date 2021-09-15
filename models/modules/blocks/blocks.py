import torch
import torch.nn as nn
import functools
from torch.nn import functional as F


class Identity(nn.Module):
    def forward(self, x):
        return x


class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.

        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not

        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out


class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet submodule with skip connections.

        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
        """
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:   # add skip connections
            return torch.cat([x, self.model(x)], 1)


# from CHAOS
# nn.Conv2d(self, in_channels, out_channels, kernel_size, stride=1,
#              padding=0, dilation=1, groups=1,
#              bias=True, padding_mode='zeros'):
def conv3x3(inplanes, outplanes, stride=1, padding=1, dilation=1, groups=1, bias=False, padding_mode='zeros'):
    """3x3 convolution with padding"""
    return nn.Conv2d(inplanes, outplanes, kernel_size=3, stride=stride,
                     padding=padding, bias=bias,
                     dilation=dilation, groups=groups, padding_mode=padding_mode)


def conv1x1(inplanes, outplanes, stride=1, padding=1, dilation=1, groups=1, bias=False, padding_mode='zeros'):
    """1x1 convolution"""
    return nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=stride, bias=bias,
                     padding=padding, dilation=dilation, groups=groups, padding_mode=padding_mode)


def conv1x1_bn_ac(inplanes, outplanes, stride=1, activation=nn.ReLU(inplace=True)):
    """1x1 convolution + 2d batchnorm + activation"""
    return nn.Sequential(
            conv1x1(inplanes, outplanes, stride),
            nn.BatchNorm2d(outplanes),
            activation)


def conv3x3_bn_ac(inplanes, outplanes, stride=1, activation=nn.ReLU(inplace=True)):
    """1x1 convolution + 2d batchnorm + activation"""
    return nn.Sequential(
            conv3x3(inplanes, outplanes, stride),
            nn.BatchNorm2d(outplanes),
            activation)


class Block3x3(nn.Module):

    def __init__(self, inplanes, midplanes, outplanes, activation=nn.ReLU(inplace=True)):
        super(Block3x3, self).__init__()
        self.conv_bn_ac1 = conv3x3_bn_ac(inplanes, midplanes, 1, activation)

        self.conv_bn_ac2 = conv3x3_bn_ac(midplanes, outplanes, 1, activation)

        self.conv_bn_ac3 = conv3x3_bn_ac(outplanes, outplanes, 1, activation)

    def forward(self, x):
        out = self.conv_bn_ac1(x)
        out = self.conv_bn_ac2(out)
        out = self.conv_bn_ac3(out)

        return out


class DoubleConv(nn.Module):
    '''(conv => BN => ReLU) * 2'''

    def __init__(self, inplanes, outplanes, activation=nn.ReLU(inplace=True)):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            conv3x3(inplanes, outplanes),
            nn.BatchNorm2d(outplanes),
            activation,
            conv3x3(outplanes, outplanes),
            nn.BatchNorm2d(outplanes),
            activation
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class ExpandConv(nn.Module):
    '''(conv => BN => ReLU) * 2'''

    def __init__(self, inplanes, mid_planes, outplanes, activation=nn.ReLU(inplace=True)):
        super(ExpandConv, self).__init__()
        self.conv = nn.Sequential(
            conv3x3(inplanes, mid_planes // 2),
            nn.BatchNorm2d(mid_planes // 2),
            activation,
            conv3x3(mid_planes // 2, mid_planes),
            nn.BatchNorm2d(mid_planes),
            activation,
            conv3x3(mid_planes, mid_planes),
            nn.BatchNorm2d(mid_planes),
            activation,
            conv1x1(mid_planes, outplanes)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class ExpandConvAddlow(nn.Module):
    '''(conv => BN => ReLU) * 2'''

    def __init__(self, inplanes, mid_planes, low_planes, outplanes):
        super(ExpandConvAddlow, self).__init__()
        self.conv = nn.Sequential(
            conv3x3(inplanes, mid_planes // 2),
            nn.BatchNorm2d(mid_planes // 2),
            # nn.ReLU(inplace=True),
            conv3x3(mid_planes // 2, mid_planes),
            nn.BatchNorm2d(mid_planes),
            # nn.ReLU(inplace=True)
        )
        self.merge = nn.Sequential(
            conv3x3(mid_planes + low_planes, outplanes * 2 // 3),
            nn.BatchNorm2d(outplanes * 2 // 3),
            # nn.ReLU(inplace=True),
            conv1x1(outplanes * 2 // 3, outplanes)
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
            nn.BatchNorm2d(midplanes),
            activation,
            conv1x1(midplanes, outplanes)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class MergeConv1(nn.Module):

    def __init__(self, inplanes, midplanes, outplanes, activation=nn.ReLU(inplace=True)):
        super(MergeConv1, self).__init__()
        self.conv = nn.Sequential(
            DoubleConv(inplanes, midplanes, activation),
            nn.Dropout(0.1),
            conv1x1(midplanes, outplanes)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class tofp16(nn.Module):
    def __init__(self):
        super(tofp16, self).__init__()

    def forward(self, input):
        return input.half()


class tofp32(nn.Module):
    def __init__(self):
        super(tofp32, self).__init__()

    def forward(self, input):
        return input.float()
