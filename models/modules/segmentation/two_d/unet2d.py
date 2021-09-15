import torch
import torch.nn as nn
import torch.nn.functional as F
from models.modules.blocks.blocks import DoubleConv

from models.auxiliary_funs import print_model_parm_nums, print_model_parm_flops

# nn.Conv2d(self, in_channels, out_channels, kernel_size, stride=1,
#              padding=0, dilation=1, groups=1,
#              bias=True, padding_mode='zeros'):


# from CHAOs and github
class UNet(nn.Module):
    def __init__(self, in_channels, n_classes, final_sigmoid=True):
        super(UNet, self).__init__()
        self.inc = inconv(in_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)
        self.outc = outconv(64, n_classes)
        # lower version
        # self.inc = inconv(n_channels, 32)
        # self.down1 = down(32, 64)
        # self.down2 = down(64, 128)
        # self.down3 = down(128, 128)
        # # self.down3 = down(256, 512)
        # # self.down4 = down(512, 512)
        # # self.up1 = up(1024, 256)
        # self.up2 = up(256, 64)
        # self.up3 = up(128, 32)
        # self.up4 = up(64, 32)
        # self.outc = outconv(32, n_classes)

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

        if not self.training:  # evaluation mode的时候self.training才为False，才执行Sigmoid
            x = self.final_activation(x)  # 二值图？

        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)

        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])

        # for padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x


if __name__ == '__main__':
    device = torch.device(f"cuda:{0}" if torch.cuda.is_available() else 'cpu')
    net = UNet(1, 1, final_sigmoid=True)
    input = torch.rand((32, 1, 256, 256), requires_grad=True)
    #
    print_model_parm_nums(net)  # 13.39M

    print_model_parm_flops(net, input, need_idx=False)  # 498.13G

    # from torchstat import stat
    # stat(net, (3,256,256))
