import torch
import torch.nn as nn
import numpy as np
from torch.nn.functional import l1_loss, mse_loss

from .region_based.dice_loss import dice_loss


# -------------------------------------------------------Function--------------------------------------------------
# I don't know from where
def _fast_hist(label_pred, label_true, num_classes):
    mask = (label_true >= 0) & (label_true < num_classes)
    hist = np.bincount(
        num_classes * label_true[mask].astype(int) +
        label_pred[mask], minlength=num_classes ** 2).reshape(num_classes, num_classes)
    return hist


def adopt_weight(weight, global_step, threshold=0, value=0.):
    if global_step < threshold:
        weight = value
    return weight


# maybe from MUNIT
def kl_loss(mu, logvar, N):
    return (1 / N) * torch.sum(mu.pow(2) + logvar.exp() - logvar - 1)


def vae_loss(predicted_x, mu, logvar, x, recon_loss=mse_loss, divergence_loss=kl_loss, recon_weight=1, kl_weight=1):
    loss_recon = recon_loss(predicted_x, x)
    loss_kl = divergence_loss(mu, logvar, x.numel()/x.shape[0])
    return recon_weight * loss_recon + kl_weight * loss_kl


def vae_dice_loss(predicted, mu, logvar, target, loss=dice_loss, divergence_loss=kl_loss, weight=1, kl_weight=1):
    return vae_loss(predicted_x=predicted, mu=mu, logvar=logvar, x=target, recon_loss=loss,
                    divergence_loss=divergence_loss, recon_weight=weight, kl_weight=kl_weight)


def vae_l1_loss(predicted, mu, logvar, target, loss=l1_loss, divergence_loss=kl_loss, weight=1, kl_weight=1):
    return vae_loss(predicted_x=predicted, mu=mu, logvar=logvar, x=target, recon_loss=loss,
                    divergence_loss=divergence_loss, recon_weight=weight, kl_weight=kl_weight)


def regularized_loss(predicted_y, predicted_x, x, y, pred_loss=l1_loss, decoder_loss=mse_loss, decoder_weight=0.1):
    return pred_loss(predicted_y, y) + decoder_weight * decoder_loss(predicted_x, x)


def variational_regularized_loss(predicted, vae_x, mu, logvar, x, y, pred_loss=l1_loss, decoder_loss=mse_loss,
                                 vae_weight=0.1, kl_weight=0.1):
    loss_pred = pred_loss(predicted, y)
    loss_vae = decoder_loss(vae_x, x)
    N = x.numel()/x.shape[0]
    loss_kl = (1 / N) * torch.sum(mu.pow(2) + logvar.exp() - logvar - 1)
    return loss_pred + (vae_weight * loss_vae) + (kl_weight * loss_kl)


# -------------------------------------------------------Class-------------------------------------------------
# copy from CycleGAN
class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss




