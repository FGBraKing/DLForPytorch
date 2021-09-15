import torch
import torch.nn as nn
import torch.nn.functional as F

from models.auxiliary_funs import print_model_parm_nums, print_model_parm_flops


def conv3x3(in_planes, out_planes, stride=1,
            padding=1, dilation=1, groups=1,
            bias=False, padding_mode='zeros'):
    """3x3 convolution with padding"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride, padding=padding,
                     bias=bias, dilation=dilation, groups=groups, padding_mode=padding_mode)


class DoubleConv(nn.Module):
    '''(conv => BN => ReLU) * 2'''

    def __init__(self, in_planes, out_planes, activation=nn.ReLU(inplace=True)):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            conv3x3(in_planes, out_planes),
            nn.BatchNorm3d(out_planes),
            nn.ReLU(inplace=True),
            conv3x3(out_planes, out_planes),
            nn.BatchNorm3d(out_planes),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class Down(nn.Module):
    def __init__(self, in_ch, out_ch, slice_down=True):
        super(Down, self).__init__()
        if slice_down:
            self.mpconv = nn.Sequential(
                nn.MaxPool3d(2),
                DoubleConv(in_ch, out_ch)
            )
        else:
            self.mpconv = nn.Sequential(
                nn.MaxPool3d((1, 2, 2)),
                DoubleConv(in_ch, out_ch)
            )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class Up(nn.Module):
    def __init__(self, in_ch, out_ch, trilinear=True, slice_up=True):
        super(Up, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if slice_up:
            if trilinear:
                self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
            else:
                self.up = nn.ConvTranspose3d(in_ch // 2, in_ch // 2, kernel_size=2, stride=2)
        else:
            self.up = nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear', align_corners=True)

        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # input is CHW
        diffZ = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]
        diffX = x2.size()[4] - x1.size()[4]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2, diffZ // 2, diffZ - diffZ // 2])

        # for padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

        x = torch.cat((x2, x1), dim=1)
        x = self.conv(x)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x


class UNet3D(nn.Module):
    def __init__(self, in_channels, n_classes, final_sigmoid=True):
        super(UNet3D, self).__init__()
        self.inc = inconv(in_ch=in_channels, out_ch=64)

        self.down1 = Down(64, 128, slice_down=False)
        self.down2 = Down(128, 256, slice_down=True)
        self.down3 = Down(256, 512, slice_down=False)
        self.down4 = Down(512, 512, slice_down=True)                    # 512 1024
        self.up1 = Up(1024, 256, trilinear=True, slice_up=True)         # 1024+512, 512
        self.up2 = Up(512, 128, trilinear=True, slice_up=False)
        self.up3 = Up(256, 64, trilinear=True, slice_up=True)
        self.up4 = Up(128, 64, trilinear=True, slice_up=False)

        self.outc = outconv(64, n_classes)

        if final_sigmoid:
            self.final_activation = nn.Sigmoid()
        else:
            self.final_activation = nn.Softmax(dim=1)

    # def forward(self, x, idx):
    def forward(self, x):
        x1 = self.inc(x)

        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        x = self.outc(x)

        if not self.training:
            x = self.final_activation(x)

        return x


if __name__ == '__main__':

    device = torch.device(f"cuda:{0}" if torch.cuda.is_available() else 'cpu')
    net = UNet3D(in_channels=1, n_classes=1, final_sigmoid=True).to(device)
    inputs = torch.rand((16, 1, 32, 64, 64), requires_grad=True).to(device)
    print_model_parm_nums(net)  # 40.15M
    print_model_parm_flops(net, inputs, need_idx=False)  # 751.84G

    from torchsummary import summary
    summary(net, input_size=(1, 32, 128, 128), batch_size=1, device='cuda')

    for name, module in net.named_modules():  # named_children():
        print(name, type(module))

    for name, layer in net.named_children():
        print(name, type(layer))

    for k, v in net.named_parameters():
        print(k, v.size())
        print(v.nelement())

