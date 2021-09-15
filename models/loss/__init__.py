from torch import nn

from .distribution_based.asymmetric_loss import AsymmetricLossOptimized, AsymmetricLossMultiLabel
from .distribution_based.cross_entropy import *
from .distribution_based.focal_loss import BinaryFocalLoss, FocalLoss
from .distribution_based.jsd import JsdCrossEntropy
from .distribution_based.others import *

from .region_based.dice_loss import BinaryDiceLoss, MutiClassDiceLoss
from .region_based.iou_loss import IOULoss
from .region_based.lovasz_loss import LovaszSoftmax
from .region_based.tverskyloss import BinaryTverskyLoss, MultiTverskyLoss, FocalTverskyLoss, TverskyLoss

from .generic_loss import *
from .combo_loss import *

SUPPORTED_LOSSES = ['bdc', 'dc', 'bce', 'ce', 'wce', 'pce', 'asymmetric', 'b_focal', 'focal', 'jsd', 'l1', 'l2', 'mse',
                    'lovasz', 'BinaryTversky', 'MultiTversky', 'tversky', 'combo', 'others']


# --------------------------------------------------------CUSTOM------------------------------------------------

def get_loss_criterion(name, ignore_index=None, reducetion='mean', **kwargs):
    assert name in SUPPORTED_LOSSES, f'Invalid loss: {name}'
    if name == 'bce':
        if ignore_index is None:
            return nn.BCEWithLogitsLoss(weight=None, reduction=reducetion, pos_weight=None)
        else:
            return WBCEWithLogitLoss(weight=None, ignore_index=ignore_index, reduction='mean', smooth=0.01)
            # return IgnoreIndexLossWrapper(nn.BCEWithLogitsLoss(), ignore_index=ignore_index)
    elif name == 'ce':
        if ignore_index is None:
            ignore_index = -100  # use the default 'ignore_index' as defined in the CrossEntropyLoss
        return nn.CrossEntropyLoss(weight=None, ignore_index=ignore_index, reduction='mean')
    elif name == 'wce':
        if ignore_index is None:
            ignore_index = -100  # use the default 'ignore_index' as defined in the CrossEntropyLoss
        return WeightedCrossEntropyLoss(weight=None, ignore_index=ignore_index)
    elif name == 'pce':
        return PixelWiseCrossEntropyLoss(class_weights=None, ignore_index=ignore_index)
    elif name == 'l1':
        return nn.L1Loss(reduction='mean')
    elif name == 'l2' or name == 'mse':
        return nn.MSELoss(reduction='mean')
    elif name == 'asymmetric':
        return AsymmetricLossMultiLabel(gamma_pos=1, gamma_neg=4, reduction='mean')
    elif name == 'focal':
        return FocalLoss(gamma=2, alpha=0.5, reduction='mean')
    elif name == 'b_focal':
        return BinaryFocalLoss(alpha=3, gamma=2, ignore_index=ignore_index, reduction='mean')
    elif name == 'jsd':
        return JsdCrossEntropy(num_splits=4, alpha=12, smoothing=0.1)
    elif name == 'iou':
        return IOULoss(weight=None, ignore_index=ignore_index, is_logit=True, reduction='mean', smooth=1)
    elif name == 'lovasz':
        return LovaszSoftmax(reduction='mean')
    elif name == 'tversky':
        return TverskyLoss(alpha=0.3, beta=0.7, ignore_index=ignore_index, reduction='mean',
                           smooth=1., normalization='sigmoid')
    elif name == 'BinaryTversky':
        return BinaryTverskyLoss(alpha=0.3, beta=0.7, ignore_index=ignore_index, reduction='mean',
                                 use_sigmoid=True, smooth=10)
    elif name == 'MultiTversky':
        return MultiTverskyLoss(alpha=0.5, beta=0.5, weights=None,
                                reduction='mean', is_logit=True, ignore_index=ignore_index)
    elif name == 'combo':
        return WBCE_DiceLoss(alpha=1.0, weight=1.0, ignore_index=ignore_index, reduction='mean')
    elif name == 'bdc':
        return BinaryDiceLoss(ignore_index=ignore_index, reducetion='mean', **kwargs)
    else:
        return MutiClassDiceLoss(weight=None, ignore_index=ignore_index, is_logit=True, reduction='mean', smooth=1.)
