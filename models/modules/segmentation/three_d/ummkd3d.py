import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from models.auxiliary_funs import print_model_parm_nums, print_model_parm_flops


# ----------------------------------function--------------------------
def conv3d(in_channels, out_channels, kernel_size,
           stride=(1,), padding=(0,), padding_mode='zeros',
           dilation=(1,), groups=1, bias=True):
    return nn.Conv3d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        bias=bias,
        padding_mode=padding_mode
    )


def batch_norm3d(num_features, momentum=0.1, affine=True,
                 track_running_stats=True, eps=1e-5):
    return nn.BatchNorm3d(num_features=num_features,
                          momentum=momentum,
                          affine=affine,
                          track_running_stats=track_running_stats,
                          eps=eps)


def instance_norm3d(num_features, momentum=0.1, affine=True,
                    track_running_stats=True, eps=1e-5):
    return nn.InstanceNorm3d(num_features=num_features,
                             momentum=momentum,
                             affine=affine,
                             track_running_stats=track_running_stats,
                             eps=eps)


def conv_bn_lrelu_3d(in_channels, out_channels, kernel_size, stride, padding=0, use_bias=False):
    return nn.Sequential(
        conv3d(in_channels, out_channels, kernel_size, stride, padding, bias=use_bias),
        batch_norm3d(out_channels, affine=True),
        nn.LeakyReLU(negative_slope=0.2, inplace=False)
    )


def conv_in_lrelu_3d(in_channels, out_channels, kernel_size, stride, padding=0, use_bias=False):
    return nn.Sequential(
        conv3d(in_channels, out_channels, kernel_size, stride, padding, bias=use_bias),
        instance_norm3d(out_channels, affine=True),
        nn.LeakyReLU(negative_slope=0.2, inplace=False)
    )


def deconv3d(in_channels, out_channels,
             kernel_size, stride=1, padding=0,
             output_padding=0, padding_mode='zeros',
             dilation=1, groups=1, bias=True):
    # D_out=(D_in−1)×stride[0]−2×padding[0]+kernel_size[0]+output_padding[0]
    return nn.ConvTranspose3d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        dilation=dilation,
        groups=groups,
        bias=bias,
        padding_mode=padding_mode
    )


def deconv_bn_relu3d(in_channels, out_channels, kernel_size, stride, padding=0, output_padding=0,
                     use_bias=False, padding_mode='zeros'):
    return nn.Sequential(
        deconv3d(in_channels, out_channels, kernel_size, stride, padding,
                 output_padding=output_padding, bias=use_bias, padding_mode=padding_mode),
        batch_norm3d(out_channels, affine=True),
        nn.LeakyReLU(negative_slope=0.2, inplace=False)
    )


def deconv_in_relu3d(in_channels, out_channels, kernel_size, stride, padding=0, output_padding=0,
                     use_bias=False, padding_mode='zeros'):
    return nn.Sequential(
        deconv3d(in_channels, out_channels, kernel_size, stride, padding,
                 output_padding=output_padding, bias=use_bias, padding_mode=padding_mode),
        instance_norm3d(out_channels, affine=True),
        nn.LeakyReLU(negative_slope=0.2, inplace=False)
    )


# nn.Softmax(dim=1)
def pixel_wise_softmax_2(output_map):
    # n c d h w
    exponential_map = torch.exp(output_map)
    sum_exp = torch.sum(exponential_map, dim=1, keepdims=True)
    tensor_sum_exp = torch.tile(sum_exp, [1, output_map.shape[1], 1, 1, 1])
    return torch.clip(torch.div(exponential_map, tensor_sum_exp), -1.0 * 1e15, 1.0 * 1e15)


class MutiNorm3d(nn.Module):
    def __init__(self, names, norm_type='instance', **kwargs) -> None:
        super(MutiNorm3d, self).__init__()
        names = [n for n in names if isinstance(n, str)]
        self.num = len(names)
        # like nn.ModuleDict
        self._norm_dict = OrderedDict()
        for n in names:
            if norm_type.lower() == 'instance':
                self._norm_dict[n] = nn.InstanceNorm3d(**kwargs)
            elif norm_type.lower() == 'batch':
                self._norm_dict[n] = nn.BatchNorm3d(**kwargs)
            elif norm_type.lower() == 'layer':
                self._norm_dict[n] = nn.LayerNorm(**kwargs)
            elif norm_type.lower() == 'group':
                self._norm_dict[n] = nn.GroupNorm(**kwargs)
            else:
                self._norm_dict[n] = nn.BatchNorm3d(**kwargs)
        self.norm = nn.ModuleDict(self._norm_dict)
        # self.norm_dict.__setitem__(name, nn.ReLU)

    def forward(self, x, key):
        return self.norm[key](x)


class ConvMutiNormLrelu3d(nn.Module):
    def __init__(self, names, norm_type, in_channels, out_channels, kernel_size, stride, padding=0, use_bias=False):
        super(ConvMutiNormLrelu3d, self).__init__()
        self.conv = conv3d(in_channels, out_channels, kernel_size, stride, padding, bias=use_bias)
        self.norm = MutiNorm3d(names, norm_type, num_features=out_channels, affine=True)  # , track_running_stats=True
        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=False)

    def forward(self, x, name):
        x = self.conv(x)
        x = self.norm(x, name)
        x = self.act(x)
        return x


class DeconvMutiNormLrelu3d(nn.Module):
    def __init__(self, names, norm_type, in_channels, out_channels, kernel_size, stride, padding=0, output_padding=1,
                 use_bias=False):
        super(DeconvMutiNormLrelu3d, self).__init__()
        self.deconv = deconv3d(in_channels, out_channels, kernel_size, stride, padding,
                               output_padding=output_padding, bias=use_bias)
        self.norm = MutiNorm3d(names, norm_type, num_features=out_channels, affine=True)  # , track_running_stats=True
        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=False)

    def forward(self, x, name):
        x = self.deconv(x)
        x = self.norm(x, name)
        x = self.act(x)
        return x


# （conv+norm+relu）*2
class DoubleConvMutiNorm(nn.Module):
    def __init__(self, names, norm_type, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DoubleConvMutiNorm, self).__init__()
        if in_channels < out_channels:
            # if in_channels < out_channels we're in the encoder path
            conv1_in_channels, conv1_out_channels = in_channels, out_channels // 2
            conv2_in_channels, conv2_out_channels = conv1_out_channels, out_channels
        else:
            # otherwise we're in the decoder path
            conv1_in_channels, conv1_out_channels = in_channels, out_channels
            conv2_in_channels, conv2_out_channels = out_channels, out_channels

        self.conv1 = ConvMutiNormLrelu3d(names, norm_type, conv1_in_channels, conv1_out_channels,
                                         kernel_size, stride, padding, use_bias=True)
        self.conv2 = ConvMutiNormLrelu3d(names, norm_type, conv2_in_channels, conv2_out_channels,
                                         kernel_size, stride, padding, use_bias=True)

    def forward(self, x, name):
        x = self.conv1(x, name)
        x = self.conv2(x, name)
        return x


# pooling+doubleconv
class DownBlockMutiNorm(nn.Module):
    def __init__(self, names, norm_type, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                 is_max_pool=True, max_pool_kernel_size=(2, 2, 2)):
        super(DownBlockMutiNorm, self).__init__()
        self.max_pool = nn.MaxPool3d(kernel_size=max_pool_kernel_size, padding=1) if is_max_pool else None
        self.double_conv = DoubleConvMutiNorm(names, norm_type,
                                              in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x, name):
        if self.max_pool is not None:
            x = self.max_pool(x)
        x = self.double_conv(x, name)
        return x


# upsample+doubleconv
class UpBlockMutiNorm(nn.Module):
    def __init__(self, names, norm_type, in_channels, out_channels, kernel_size=3, scale_factor=(2, 2, 2)):
        super(UpBlockMutiNorm, self).__init__()
        # make sure that the output size reverses the MaxPool3d
        # D_out=(D_in−1)×stride[0]−2×padding[0]+kernel_size[0]+output_padding[0]
        self.up = DeconvMutiNormLrelu3d(names, norm_type, in_channels, out_channels,   # in_channels=2*out_channels
                                        kernel_size, stride=scale_factor, padding=1, output_padding=1)
        self.double_conv = DoubleConvMutiNorm(names, norm_type, in_channels, out_channels,
                                              kernel_size, scale_factor, padding=1)

    def forward(self, x1, x2, name):

        x = self.up(x1, name)

        # for padding issues, see
        diffZ = x2.size()[2] - x.size()[2]
        diffY = x2.size()[3] - x.size()[3]
        diffX = x2.size()[4] - x.size()[4]

        x = F.pad(x, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2, diffZ // 2, diffZ - diffZ // 2])

        # concatenate encoder_features (encoder path) with the upsampled input across channel dimension
        x = torch.cat((x, x2), dim=1)
        x = self.double_conv(x, name)
        return x


class Unet3dWithSpecBN(nn.Module):
    def __init__(self, names, norm_type, in_channels, out_channels, final_sigmoid, init_channel_number=64):
        super(Unet3dWithSpecBN, self).__init__()
        self.encoders = nn.ModuleList([
            DownBlockMutiNorm(names, norm_type, in_channels, init_channel_number, is_max_pool=False),
            DownBlockMutiNorm(names, norm_type, init_channel_number, 2*init_channel_number, is_max_pool=True),
            DownBlockMutiNorm(names, norm_type, 2*init_channel_number, 4*init_channel_number, is_max_pool=True),
            DownBlockMutiNorm(names, norm_type, 4*init_channel_number, 8*init_channel_number, is_max_pool=True)
            # DownBlockMutiNorm(names, norm_type, 8*init_channel_number, 16*init_channel_number, is_max_pool=True),
        ])

        self.decoders = nn.ModuleList([
            UpBlockMutiNorm(names, norm_type, 8 * init_channel_number, 4 * init_channel_number),
            UpBlockMutiNorm(names, norm_type, 4 * init_channel_number, 2 * init_channel_number),
            UpBlockMutiNorm(names, norm_type, 2 * init_channel_number, init_channel_number),
        ])

        self.final_conv = nn.Conv3d(init_channel_number, out_channels, 1, 1, 0)
        if final_sigmoid:
            self.final_activation = nn.Sigmoid()
        else:
            self.final_activation = nn.Softmax(dim=1)

    def forward(self, x, name):
        encoders_features = []
        for encoder in self.encoders:
            x = encoder(x, name)
            # reverse the encoder outputs to be aligned with the decoder
            encoders_features.insert(0, x)
        encoders_features = encoders_features[1:]

        for decoder, encoder_features in zip(self.decoders, encoders_features):
            x = decoder(x, encoder_features, name)

        x = self.final_conv(x)
        if not self.training:
            x = self.final_activation(x)
        return x


if __name__ == "__main__":
    device = torch.device(f"cuda:{0}" if torch.cuda.is_available() else 'cpu')
    net = Unet3dWithSpecBN(names=['target', 'source'], norm_type='batch',
                           in_channels=1, out_channels=1, init_channel_number=32, final_sigmoid=True).to(device)

    print('---------------------------------------------------------')
    for name, layer in net.named_modules():
        print(name, type(layer))
    print('---------------------------------------------------------')
    for name, layer in net.named_children():
        print(name, type(layer))
