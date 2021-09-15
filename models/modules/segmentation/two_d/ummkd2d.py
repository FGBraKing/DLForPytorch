import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import numpy as np
from functools import partial
from models.auxiliary_funs import print_model_parm_nums, print_model_parm_flops


def conv3x3(in_planes, out_planes, stride=1, dilation=1, padding=1):
    # same
    return nn.Conv2d(in_planes, out_planes, stride=stride, dilation=dilation,
                     padding=padding, kernel_size=3, bias=False, padding_mode='zeros')


def conv3x3_sym(in_planes, out_planes, stride=1, dilation=1, padding=1):
    return nn.Conv2d(in_planes, out_planes,  stride=stride, dilation=dilation,
                     padding=padding, kernel_size=3, bias=False, padding_mode='reflect')


def conv3x3_sym_block(in_planes, out_planes, stride=1, dilation=1, padding=1):
    block_list = [
        nn.Conv2d(in_planes, out_planes,  stride=stride, dilation=dilation,
                  padding=padding, kernel_size=3, bias=False, padding_mode='reflect'),
        nn.BatchNorm2d(out_planes),
        nn.LeakyReLU()
    ]
    return nn.Sequential(*block_list)


class Conv3x3SymBlockMultiBN(nn.Module):
    def __init__(self, names, in_planes, out_planes, stride=1, dilation=1, padding=1):
        super(Conv3x3SymBlockMultiBN, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,  stride=stride, dilation=dilation,
                              padding=padding, kernel_size=3, bias=False, padding_mode='reflect')
        self.norm = MultiNorm2d(names, 'bn', num_features=out_planes)
        self.act = nn.LeakyReLU()

    def forward(self, x, name):
        x = self.conv(x)
        x = self.norm(x, name)
        x = self.act(x)
        return x


def max_pool2d(kernel_size):
    return nn.MaxPool2d(kernel_size, padding=0)   # torch.floor  int(kernel_size / 2)


class MultiNorm2d(nn.Module):
    def __init__(self, names, norm_type='instance', **kwargs) -> None:
        super(MultiNorm2d, self).__init__()
        names = [name for name in names if isinstance(name, str)]
        self.num = len(names)
        # like nn.ModuleDict
        self._norm_dict = OrderedDict()
        for name in names:
            if norm_type.lower() == 'instance':
                self._norm_dict[name] = nn.InstanceNorm2d(**kwargs)
            elif norm_type.lower() == 'batch':
                self._norm_dict[name] = nn.BatchNorm2d(**kwargs)
            elif norm_type.lower() == 'layer':
                self._norm_dict[name] = nn.LayerNorm(**kwargs)
            elif norm_type.lower() == 'group':
                self._norm_dict[name] = nn.GroupNorm(**kwargs)
            else:
                self._norm_dict[name] = nn.BatchNorm2d(**kwargs)
        self.norm = nn.ModuleDict(self._norm_dict)
        # self.norm_dict.__setitem__(name, nn.ReLU)

    def forward(self, x, name):
        return self.norm[name](x)


# bn_leaky_relu_conv2d_layer
class ConvBlock2d(nn.Module):
    def __init__(self, inplanes, planes, stride=1, keep_prob=0.8):
        super(ConvBlock2d, self).__init__()
        self.bn = nn.BatchNorm2d(inplanes)
        self.relu = nn.LeakyReLU(inplace=True)
        self.conv = conv3x3(inplanes, planes, stride, dilation=1)
        self.dropout = nn.Dropout(keep_prob)

    def forward(self, x):
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv(x)
        x = self.dropout(x)
        return x


class ConvBlock2dMultiBN(nn.Module):
    def __init__(self, names, inplanes, planes, stride=1, keep_prob=0.8):
        super(ConvBlock2dMultiBN, self).__init__()
        self.bn = MultiNorm2d(names, 'bn', num_features=inplanes)
        # self.bn = nn.BatchNorm2d(inplanes)
        self.relu = nn.LeakyReLU(inplace=True)
        self.conv = conv3x3(inplanes, planes, stride, dilation=1)
        self.dropout = nn.Dropout(keep_prob)

    def forward(self, x, name):
        x = self.bn(x, name)
        x = self.relu(x)
        x = self.conv(x)
        x = self.dropout(x)
        return x


class DilateBlock2d(nn.Module):
    def __init__(self, inplanes, planes, stride=1, keep_prob=0.8, padding=2, dilation_rate=2):
        super(DilateBlock2d, self).__init__()
        self.bn = nn.BatchNorm2d(inplanes)
        self.relu = nn.LeakyReLU(inplace=True)
        self.conv = conv3x3(inplanes, planes, stride, padding=padding, dilation=dilation_rate)
        self.dropout = nn.Dropout(keep_prob)

    def forward(self, x):
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv(x)
        x = self.dropout(x)
        return x


class DilateBlock2dMulti(nn.Module):
    def __init__(self, names, inplanes, planes, stride=1, keep_prob=0.8, padding=2, dilation_rate=2):
        super(DilateBlock2dMulti, self).__init__()
        self.bn = MultiNorm2d(names, 'bn', num_features=inplanes)
        self.relu = nn.LeakyReLU(inplace=True)
        self.conv = conv3x3(inplanes, planes, stride, padding=padding, dilation=dilation_rate)
        self.dropout = nn.Dropout(keep_prob)

    def forward(self, x, name):
        x = self.bn(x, name)
        x = self.relu(x)
        x = self.conv(x)
        x = self.dropout(x)
        return x


class ResidualBlock(nn.Module):

    def __init__(self, in_planes, mid_planes, out_planes, keep_prob, stride=1, inc_dim=False):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvBlock2d(in_planes, mid_planes, stride, keep_prob)
        self.conv2 = ConvBlock2d(mid_planes, out_planes, stride, keep_prob)

        self.inc_dim = inc_dim

    def forward(self, x):
        x_channel = x.size()[1]
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.inc_dim:
            residual = F.pad(residual, [0, 0, 0, 0, x_channel//2, x_channel//2, 0, 0])
        return residual + out


class ResidualBlockMulti(nn.Module):

    def __init__(self, names, in_planes, mid_planes, out_planes, keep_prob, stride=1, inc_dim=False):
        super(ResidualBlockMulti, self).__init__()
        self.conv1 = ConvBlock2dMultiBN(names, in_planes, mid_planes, stride, keep_prob)
        self.conv2 = ConvBlock2dMultiBN(names, mid_planes, out_planes, stride, keep_prob)

        self.inc_dim = inc_dim

    def forward(self, x, name):
        x_channel = x.size()[1]
        residual = x
        out = self.conv1(x, name)
        out = self.conv2(out, name)
        if self.inc_dim:
            residual = F.pad(residual, [0, 0, 0, 0, x_channel//2, x_channel//2, 0, 0])
        return residual + out


class DRBlock(nn.Module):
    def __init__(self, in_planes, mid_planes, out_planes, stride=1,
                 keep_prob=0.8, padding=2, dilation_rate=2, inc_dim=False):
        super(DRBlock, self).__init__()
        self.dilate_conv1 = DilateBlock2d(in_planes, mid_planes, keep_prob=keep_prob, stride=stride,
                                          dilation_rate=dilation_rate, padding=padding)
        self.dilate_conv2 = DilateBlock2d(mid_planes, out_planes, keep_prob=keep_prob, stride=stride,
                                          dilation_rate=dilation_rate, padding=padding)
        self.inc_dim = inc_dim

    def forward(self, x):
        x_channel = x.size()[1]
        residual = x
        out = self.dilate_conv1(x)
        out = self.dilate_conv2(out)
        if self.inc_dim:
            residual = F.pad(residual, [0, 0, 0, 0, x_channel//2, x_channel//2, 0, 0])
        return residual + out


class DRBlockMulti(nn.Module):
    def __init__(self, names, in_planes, mid_planes, out_planes, stride=1,
                 keep_prob=0.8, padding=2, dilation_rate=2, inc_dim=False):
        super(DRBlockMulti, self).__init__()
        self.dilate_conv1 = DilateBlock2dMulti(names, in_planes, mid_planes, keep_prob=keep_prob, stride=stride,
                                               dilation_rate=dilation_rate, padding=padding)
        self.dilate_conv2 = DilateBlock2dMulti(names, mid_planes, out_planes, keep_prob=keep_prob, stride=stride,
                                               dilation_rate=dilation_rate, padding=padding)
        self.inc_dim = inc_dim

    def forward(self, x, name):
        x_channel = x.size()[1]
        residual = x
        out = self.dilate_conv1(x, name)
        out = self.dilate_conv2(out, name)
        if self.inc_dim:
            residual = F.pad(residual, [0, 0, 0, 0, x_channel//2, x_channel//2, 0, 0])
        return residual + out


def make_one_hot(label_vol, n_cls=None):
    if n_cls is None:
        n_cls = label_vol.max() + 1
    shape = list(label_vol.shape)
    shape.insert(1, n_cls)
    shape = tuple(shape)
    result = torch.zeros(shape)
    result = result.scatter_(1, label_vol.cpu().long(), 1)
    return result


def _phase_shift(img, r, batch_size) -> torch.Tensor:
    _, c, a, b = list(img.shape)            # c*_ ==bsize*r*r
    X = torch.reshape(img, (batch_size, a, b, r, r))
    X = torch.transpose(X, 3, 4)
    X = torch.split(X, 1, dim=1)   # a, [bsize, 1, b, r, r]
    X = torch.cat([x for x in X], dim=3)  # [bsize, 1, b, a*r, r]
    X = torch.split(X, 1, dim=2)
    X = torch.cat([x for x in X], dim=4)  # [bsize, 1, 1, a*r, b*r]
    X = torch.squeeze(X, dim=1)     # [bsize, 1, a*r, b*r]
    return X


def PS(X, r, batch_size, n_channel=8):
    t_c = X.shape[1]
    Xc = torch.split(X, t_c//n_channel, dim=1)  # n_channel,[n, t_c//n_channel, h, w]
    X = torch.cat([_phase_shift(x, r, batch_size) for x in Xc], dim=1)  # [bsize, (n*c)/(r*r*bsize), h*r, w*r]
    return X


def _eval_dice(gt_y, pred_y, detail=False):

    class_map = {  # a map used for mapping label value to its name, used for output
        "0": "bg",
        "1": "lv_myo",
        "2": "la_blood",
        "3": "lv_blood",
        "4": "aa"
    }

    dice = []

    for cls in range(1, 5):
        gt = np.zeros(gt_y.shape)
        pred = np.zeros(pred_y.shape)

        gt[gt_y == cls] = 1
        pred[pred_y == cls] = 1
        dice_this = 2*np.sum(gt*pred)/(np.sum(gt)+np.sum(pred))
        dice.append(dice_this)

        if detail is True:
            print("class {}, dice is {:2f}".format(class_map[str(cls)], dice_this))

    return dice


class Ummkd2d(nn.Module):

    @staticmethod
    def _get_residual_block(name, times, in_channels, keep_prob, inc_dim=False, downsample=False):
        if inc_dim:
            out_channels = in_channels*2
        else:
            out_channels = in_channels
        model = nn.Sequential()
        for i in range(times):
            if i == 0:
                model.add_module(name+f'ResidualBlock{i+1}',
                                 ResidualBlock(in_channels, out_channels, out_channels, keep_prob, inc_dim=inc_dim))
            else:
                model.add_module(name+f'ResidualBlock{i+1}',
                                 ResidualBlock(out_channels,out_channels,out_channels, keep_prob))
        if downsample:
            model.add_module(name+'max_pool2d', max_pool2d(2))
        return model

    def __init__(self, in_channels, keep_prob, batch_size, feature_base=16, n_class=4):
        super(Ummkd2d, self).__init__()
        self.n_class = n_class
        self.in_conv = conv3x3(in_channels, feature_base)

        self.block1 = self._get_residual_block('block1', 1, feature_base, keep_prob, inc_dim=False, downsample=True)
        self.block2 = self._get_residual_block('block2', 1, feature_base, keep_prob, inc_dim=True, downsample=True)
        self.block3 = self._get_residual_block('block3', 2, feature_base*2, keep_prob, inc_dim=True, downsample=True)
        self.block4 = self._get_residual_block('block4', 2, feature_base*4, keep_prob, inc_dim=True, downsample=False)
        self.block5 = self._get_residual_block('block5', 2, feature_base*8, keep_prob, inc_dim=True, downsample=False)
        self.block6 = self._get_residual_block('block6', 2, feature_base*16, keep_prob, inc_dim=False, downsample=False)
        self.block7 = self._get_residual_block('block7', 2, feature_base*16, keep_prob, inc_dim=True, downsample=False)


        self.dr_block_8_1 = DRBlock(feature_base*32, feature_base*32, feature_base*32,
                                    dilation_rate=2, padding=2,
                                    keep_prob=keep_prob, inc_dim=False)
        self.dr_block_8_2 = DRBlock(feature_base*32, feature_base*32, feature_base*32,
                                    dilation_rate=2, padding=2,
                                    keep_prob=keep_prob, inc_dim=False)

        self.conv_block9_1 = ConvBlock2d(feature_base*32, feature_base*32, keep_prob=keep_prob)
        self.conv_block9_2 = ConvBlock2d(feature_base*32, feature_base*32, keep_prob=keep_prob)

        local_size = 8*8
        # since the input feature is 8* downsampled, therefore, we need to recover corresponding size.
        # In this case 1 pixel in feature space encodes the label of a 8*8 region of the original image
        self.conv10_1 = conv3x3_sym_block(feature_base*32, local_size*self.n_class*8)   # [? 5*8*(8*8) 32 32]
        self.flat = partial(PS, r=8, batch_size=batch_size, n_channel=self.n_class*8)   # [? 256 256 5*8]

        self.out_conv = nn.Conv2d(self.n_class * 8, self.n_class, kernel_size=5,
                                  padding=2, bias=False, padding_mode='reflect')

    def _residual_forward(self, x):
        out1 = self.block1(x)
        out2 = self.block2(out1)
        out3 = self.block3(out2)

        block4_2 = self.block4(out3)
        block5_2 = self.block5(block4_2)
        block6_2 = self.block6(block5_2)
        block7_2 = self.block7(block6_2)
        return block7_2

    def forward(self, x):
        conv_in = self.in_conv(x)

        block7_2 = self._residual_forward(conv_in)

        block8_1 = self.dr_block_8_1(block7_2)
        block8_2 = self.dr_block_8_2(block8_1)

        conv9_1 = self.conv_block9_1(block8_2)
        conv9_2 = self.conv_block9_2(conv9_1)

        conv10_1 = self.conv10_1(conv9_2)
        flat_conv10_1 = self.flat(conv10_1)

        logits = self.out_conv(flat_conv10_1)
        return logits

    def _get_segmentation_cost(self, seg_logits, seg_gt):
        softmaxpred = F.softmax(seg_logits, dim=1)

        dice = 0
        for i in range(self.n_class):
            inse = torch.sum(softmaxpred[:, i, :, :]*seg_gt[:, i, :, :])
            l = torch.sum(softmaxpred[:, i, :, :]*softmaxpred[:, i, :, :])
            r = torch.sum(seg_gt[:, i, :, :])
            dice += 2.0 * inse/(l+r+1e-7) # here 1e-7 is relaxation eps
        dice_loss = -1.0 * dice / self.n_class

        # calculate cross-entropy weighted loss
        ce_weighted = 0
        for i in range(self.n_class):
            gti = seg_gt[:, i, :, :]
            predi = softmaxpred[:, i, :, :]
            weighted = 1-(torch.sum(gti) / torch.sum(seg_gt))
            ce_weighted += -1.0 * weighted * gti * torch.log(torch.clamp(predi, 0.005, 1))
        ce_weighted_loss = torch.mean(torch.Tensor(ce_weighted))
        return dice_loss, ce_weighted_loss

    def _eval_dice_during_train(self, labels, compact_pred):
        """
        calculate standard dice for evaluation, here uses the class prediction, not the probability
        """
        dice_arr = []
        # dice = 0
        eps = 1e-7
        pred = make_one_hot(compact_pred, self.n_class)
        for i in range(self.n_class):
            inse = torch.sum(pred[:, i, :, :] * labels[:, i, :, :])
            union = torch.sum(pred[:, i, :, :]) + torch.sum(labels[:, i, :, :])
            dice_arr.append(2.0 * inse / (union + eps))
        # return 1.0 * dice  / self.n_class, dice_arr
        return dice_arr


class KDLoss(nn.Module):
    def __init__(self, n_class):
        super(KDLoss, self).__init__()
        self.n_class = n_class

    def _cal_soft_prob(self, logits, mask, temperature = 2.0,  eps=1e-6):
        if mask.ndim == logits.ndim:
            p_mask = mask.repeat([1, self.n_class, 1, 1])
        elif mask.ndim == logits.ndim - 1:
            p_mask = mask.unsqueeze(1).repeat([1, self.n_class, 1, 1])
        else:
            raise ValueError
        logits_mask_out = logits * p_mask
        logits_avg = torch.sum(logits_mask_out, [0, 2, 3]) / (torch.sum(mask) + eps)  # C*1 (A 交 B/A)
        soft_prob = F.softmax(logits_avg/temperature,  dim=1, _stacklevel=5)
        return soft_prob

    def forward(self, source_logits, source_gt, target_logits, target_gt):
        # source_logits source_gt target_logits target_gt : n,C,h,w
        kd_loss = 0.0
        # source_prob = []
        # target_prob = []

        for i in range(self.n_class):

            s_soft_prob = self._cal_soft_prob(source_logits, source_gt[:, i:i+1, :, :])
            # source_prob.append(s_soft_prob)

            t_soft_prob = self._cal_soft_prob(target_logits, target_gt[:, i:i+1, :, :])
            # target_prob.append(s_soft_prob)

            # ## KL divergence loss
            loss = (torch.sum(s_soft_prob * torch.log(s_soft_prob/t_soft_prob)) +
                    torch.sum(t_soft_prob * torch.log(t_soft_prob/s_soft_prob))) / 2.0

            kd_loss += loss

        kd_loss = kd_loss / self.n_class

        return kd_loss


def get_l2_Norm(parameters):
    L2_norm = torch.sum(torch.Tensor([torch.sum(torch.pow(parameter, 2))/2
                                      for parameter in parameters if parameter.requires_grad]))
    return L2_norm


class AuxModuleList(nn.Module):
    def __init__(self, module_list):
        super(AuxModuleList, self).__init__()
        self.layers = nn.ModuleList(module_list)

    def forward(self, x, name):
        for layer in self.layers:
            if layer.__class__.__name__.find('Multi') != -1:
                x = layer(x, name)
            else:
                x = layer(x)
        return x


class Ummkd2dModMain(nn.Module):

    @staticmethod
    def _get_residual_block(bn_names, name, times, in_channels, keep_prob, inc_dim=False, downsample=False):
        if inc_dim:
            out_channels = in_channels*2
        else:
            out_channels = in_channels
        # model = nn.Sequential()
        module_list = []
        for i in range(times):
            if i == 0:
                module_list.append(ResidualBlockMulti(bn_names, in_channels, out_channels, out_channels,
                                                      keep_prob, inc_dim=inc_dim))
                # model.add_module(name+f'ResidualBlock{i+1}',
                #                  ResidualBlockMulti(bn_names, in_channels, out_channels, out_channels,
                #                                     keep_prob, inc_dim=inc_dim))
            else:
                module_list.append(ResidualBlockMulti(bn_names, out_channels, out_channels, out_channels, keep_prob))
                # model.add_module(name+f'ResidualBlock{i+1}',
                #                  ResidualBlockMulti(bn_names, out_channels, out_channels, out_channels, keep_prob))
        if downsample:
            module_list.append(max_pool2d(2))
            # model.add_module(name+'max_pool2d', max_pool2d(2))

        return AuxModuleList(module_list)

    def __init__(self, names, in_channels, keep_prob, feature_base=16):
        super(Ummkd2dModMain, self).__init__()
        self.in_conv = conv3x3(in_channels, feature_base)

        self.block1 = self._get_residual_block(names, 'block1', 1, feature_base, keep_prob, inc_dim=False, downsample=True)
        self.block2 = self._get_residual_block(names, 'block2', 1, feature_base, keep_prob, inc_dim=True, downsample=True)
        self.block3 = self._get_residual_block(names, 'block3', 2, feature_base*2, keep_prob, inc_dim=True, downsample=True)
        self.block4 = self._get_residual_block(names, 'block4', 2, feature_base*4, keep_prob, inc_dim=True, downsample=False)
        self.block5 = self._get_residual_block(names, 'block5', 2, feature_base*8, keep_prob, inc_dim=True, downsample=False)
        self.block6 = self._get_residual_block(names, 'block6', 2, feature_base*16, keep_prob, inc_dim=False, downsample=False)
        self.block7 = self._get_residual_block(names, 'block7', 2, feature_base*16, keep_prob, inc_dim=True, downsample=False)

        self.dr_block_8_1 = DRBlockMulti(names, feature_base*32, feature_base*32, feature_base*32,
                                         dilation_rate=2, padding=2,
                                         keep_prob=keep_prob, inc_dim=False)

        self.dr_block_8_2 = DRBlockMulti(names, feature_base*32, feature_base*32, feature_base*32,
                                         dilation_rate=2, padding=2,
                                         keep_prob=keep_prob, inc_dim=False)

        self.conv_block9_1 = ConvBlock2dMultiBN(names, feature_base*32, feature_base*32, keep_prob=keep_prob)
        self.conv_block9_2 = ConvBlock2dMultiBN(names, feature_base*32, feature_base*32, keep_prob=keep_prob)

        # local_size = 8*8
        # # since the input feature is 8* downsampled, therefore, we need to recover corresponding size.
        # # In this case 1 pixel in feature space encodes the label of a 8*8 region of the original image
        # self.conv10_1 = Conv3x3SymBlockMultiBN(names, feature_base*32, local_size*self.n_class*8)  # [? 5*8*(8*8) 32 32]
        # self.flat = partial(PS, r=8, batch_size=batch_size, n_channel=self.n_class*8)   # [? 256 256 5*8]
        #
        # self.out_conv = nn.Conv2d(self.n_class * 8, self.n_class, kernel_size=5,
        #                           padding=2, bias=False, padding_mode='reflect')

    def _residual_forward(self, x, name):
        out1 = self.block1(x, name)
        out2 = self.block2(out1, name)
        out3 = self.block3(out2, name)

        block4_2 = self.block4(out3, name)
        block5_2 = self.block5(block4_2, name)
        block6_2 = self.block6(block5_2, name)
        block7_2 = self.block7(block6_2, name)
        return block7_2

    def forward(self, x, name):
        conv_in = self.in_conv(x)

        block7_2 = self._residual_forward(conv_in, name)

        block8_1 = self.dr_block_8_1(block7_2, name)
        block8_2 = self.dr_block_8_2(block8_1, name)

        conv9_1 = self.conv_block9_1(block8_2, name)
        conv9_2 = self.conv_block9_2(conv9_1, name)

        # conv10_1 = self.conv10_1(conv9_2)
        # flat_conv10_1 = self.flat(conv10_1)
        #
        # logits = self.out_conv(flat_conv10_1)
        return conv9_2


class Ummkd2dModOutBlock(nn.Module):
    def __init__(self, names, in_channels, scale, n_class, batch_size, out_channels=None):
        super(Ummkd2dModOutBlock, self).__init__()
        if out_channels is None:
            out_channels = n_class

        self.n_class = n_class
        self.conv10_1 = Conv3x3SymBlockMultiBN(names, in_channels, scale*scale*n_class*8)  # [? 5*8*(8*8) 32 32]
        self.flat = partial(PS, r=scale, batch_size=batch_size, n_channel=n_class*8)   # [? 256 256 5*8]

        self.out_conv = nn.Conv2d(self.n_class * 8, out_channels, kernel_size=5,
                                  padding=2, bias=False, padding_mode='reflect')

    def forward(self, x, name):
        conv10_1 = self.conv10_1(x, name)
        flat_conv10_1 = self.flat(conv10_1)
        logits = self.out_conv(flat_conv10_1)
        return logits


class Ummkd2dMod(nn.Module):
    def __init__(self, names, in_channels, keep_prob, feature_base=16, down_times=3, n_class=5, batch_size=8,
                 out_channels=None):
        super(Ummkd2dMod, self).__init__()
        scale = pow(2, down_times)
        feature_scale = int(scale*scale/2)
        self.main = Ummkd2dModMain(names, in_channels, keep_prob, feature_base)
        self.out = Ummkd2dModOutBlock(names, feature_base*feature_scale, scale, n_class, batch_size, out_channels)

    def forward(self, x, name='target'):
        x = self.main(x, name)
        x = self.out(x, name)
        return x

    def _get_segmentation_cost(self, seg_logits, seg_gt):
        softmaxpred = F.softmax(seg_logits, dim=1, _stacklevel=5)

        dice = 0
        for i in range(self.n_class):
            inse = torch.sum(softmaxpred[:, i, :, :]*seg_gt[:, i, :, :])
            l = torch.sum(softmaxpred[:, i, :, :]*softmaxpred[:, i, :, :])
            r = torch.sum(seg_gt[:, i, :, :])
            dice += 2.0 * inse/(l+r+1e-7)  # here 1e-7 is relaxation eps
        dice_loss = -1.0 * dice / self.n_class

        # calculate cross-entropy weighted loss
        ce_weighted = 0
        for i in range(self.n_class):
            gti = seg_gt[:, i, :, :]
            predi = softmaxpred[:, i, :, :]
            weighted = 1-(torch.sum(gti) / torch.sum(seg_gt))
            ce_weighted += -1.0 * weighted * gti * torch.log(torch.clamp(predi, 0.005, 1))
        ce_weighted_loss = torch.mean(torch.Tensor(ce_weighted))
        return dice_loss, ce_weighted_loss

    def _eval_dice_during_train(self, labels, compact_pred):
        """
        calculate standard dice for evaluation, here uses the class prediction, not the probability
        """
        dice_arr = []
        # dice = 0
        eps = 1e-7
        pred = make_one_hot(compact_pred, self.n_class)
        for i in range(self.n_class):
            inse = torch.sum(pred[:, i, :, :] * labels[:, i, :, :])
            union = torch.sum(pred[:, i, :, :]) + torch.sum(labels[:, i, :, :])
            dice_arr.append(2.0 * inse / (union + eps))
        # return 1.0 * dice  / self.n_class, dice_arr
        return dice_arr


def main():
    net = Ummkd2d(in_channels=1, keep_prob=0.75, feature_base=16, n_class=5, batch_size=2)
    for name, mudule in net.named_children():
        print(name, mudule)

    data = torch.rand(2, 1, 256, 256)
    print_model_parm_nums(net)
    print_model_parm_flops(net, data, False)

    from torchsummary import summary
    net = net.to('cuda')
    data = torch.rand(2, 1, 256, 256).to('cuda')
    out = net(data)
    print(out.shape, out.device)
    summary(net, input_size=(1, 256, 256), batch_size=2, device='cuda')


def main_mod():
    net = Ummkd2dMod(names=['target', 'source'], in_channels=1, keep_prob=0.75, feature_base=16,
                     down_times=3, n_class=5, batch_size=2)

    for name, mudule in net.named_children():
        print(name, mudule)

    data = torch.rand(2, 1, 256, 256)
    print_model_parm_nums(net)
    print_model_parm_flops(net, data, False)

    from torchsummary import summary
    net = net.to('cuda')
    data = torch.rand(2, 1, 256, 256).to('cuda')
    out = net(data)
    print(out.shape, out.device)
    summary(net, input_size=(1, 256, 256), batch_size=2, device='cuda')





if __name__ == '__main__':
    main_mod()















