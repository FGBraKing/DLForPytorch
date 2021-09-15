import torch
import torch.nn as nn
import torch.nn.functional as F

from .backone.aspp_3d import build_aspp
from .backone import build_3dbackbone


class DeepLabV3_3D(nn.Module):

    def __init__(self, backbone='resnet18_os8', in_channels=1, output_stride=16, n_classes=21, final_sigmoid=True):
        super(DeepLabV3_3D, self).__init__()
        assert backbone in ['resnet18_os8', 'resnet18_os16',
                            'resnet34_os8', 'resnet_os16'], "Not support this backbone!"

        if backbone in ['resnet18_os8', 'resnet34_os8']:
            output_stride = 8

        self.backbone = build_3dbackbone(backbone, in_channels=in_channels)
        self.aspp = build_aspp(backbone, output_stride, nn.BatchNorm3d)

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
