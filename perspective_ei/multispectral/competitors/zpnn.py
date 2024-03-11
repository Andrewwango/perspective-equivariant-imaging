from math import ceil
import torch.nn as nn
import torch
from deepinv.physics.blur import Blur, gaussian_blur
from .. import MultispectralUtils, Pansharpen

class ZPNN_CorrelationLoss(nn.Module, MultispectralUtils):
    """Z-PNN structural loss from Ciotola et al. 
    "Pansharpening by Convolutional Neural Networks in the Full Resolution Framework",
    IEEE TGRS 2022. https://ieeexplore.ieee.org/document/9745494 

    Code adapted from https://github.com/matciotola/Lambda-PNN/blob/main/loss.py

    :param int lowpass_kernel_size: Gaussian MTF (as assumed in our experiments)
        kernel size, defaults to 41 (as per Lambda-PNN paper https://ieeexplore.ieee.org/document/10198408)
    :param str device: torch device, defaults to "cpu"
    """
    def __init__(self, *args, lowpass_kernel_size: int = 41, device="cpu", **kwargs):

        super().__init__(*args, **kwargs)
        self.blur = Blur(
            filter=gaussian_blur(lowpass_kernel_size), #41 
            padding="reflect",
            device=device
        ).A
        self.name = "ZPNNStructural"

    def forward(self, y, x_net, physics: Pansharpen, *args, **kwargs):
        window_size = physics.factor #window_size set to ratio in Lambda-PNN paper

        hrms = self.hrms_from_volume(x_net)
        pan = self.pan_from_volume(y)

        one_minus_rho = 1. - torch.clamp(xcorr_torch(
            hrms, 
            pan, 
            half_width=ceil(window_size / 2)
        ), min=-1)
        
        pan_lp = self.blur(pan)
        lrms_up = self.hrms_from_volume(physics.A_adjoint(y))

        one_minus_rho_max = 1. - xcorr_torch(
            pan_lp, 
            lrms_up, 
            half_width=ceil((window_size ** 2) / 2)
        )

        return torch.mean(one_minus_rho * one_minus_rho.gt(one_minus_rho_max))


def xcorr_torch(img_1: torch.Tensor, img_2: torch.Tensor, half_width: int) -> torch.Tensor:
    """A PyTorch implementation of Cross-Correlation Field computation.
    Taken from https://github.com/matciotola/Lambda-PNN/blob/main/tools/cross_correlation.py

    :param torch.Tensor img_1: First image on which calculate the cross-correlation. Dimensions: 1, 1, H, W
    :param torch.Tensor img_2: Second image on which calculate the cross-correlation. Dimensions: 1, 1, H, W
    :param int half_width: The semi-size of the window on which calculate the cross-correlation
    :return torch.Tensor: The cross-correlation map between img_1 and img_2
    """

    w = ceil(half_width)
    ep = 1e-20
    img_1 = img_1.double()
    img_2 = img_2.double()

    img_1 = nn.functional.pad(img_1, (w, w, w, w))
    img_2 = nn.functional.pad(img_2, (w, w, w, w))

    img_1_cum = torch.cumsum(torch.cumsum(img_1, dim=-1), dim=-2)
    img_2_cum = torch.cumsum(torch.cumsum(img_2, dim=-1), dim=-2)

    img_1_mu = (img_1_cum[:, :, 2*w:, 2*w:] - img_1_cum[:, :, :-2*w, 2*w:] - img_1_cum[:, :, 2*w:, :-2*w] + img_1_cum[:, :, :-2*w, :-2*w]) / (4*w**2)
    img_2_mu = (img_2_cum[:, :, 2*w:, 2*w:] - img_2_cum[:, :, :-2*w, 2*w:] - img_2_cum[:, :, 2*w:, :-2*w] + img_2_cum[:, :, :-2*w, :-2*w]) / (4*w**2)

    img_1 = img_1[:, :, w:-w, w:-w] - img_1_mu
    img_2 = img_2[:, :, w:-w, w:-w] - img_2_mu

    img_1 = nn.functional.pad(img_1, (w, w, w, w))
    img_2 = nn.functional.pad(img_2, (w, w, w, w))

    i2_cum = torch.cumsum(torch.cumsum(img_1**2, dim=-1), dim=-2)
    j2_cum = torch.cumsum(torch.cumsum(img_2**2, dim=-1), dim=-2)
    ij_cum = torch.cumsum(torch.cumsum(img_1*img_2, dim=-1), dim=-2)

    sig2_ij_tot = (ij_cum[:, :, 2*w:, 2*w:] - ij_cum[:, :, :-2*w, 2*w:] - ij_cum[:, :, 2*w:, :-2*w] + ij_cum[:, :, :-2*w, :-2*w])
    sig2_ii_tot = (i2_cum[:, :, 2*w:, 2*w:] - i2_cum[:, :, :-2*w, 2*w:] - i2_cum[:, :, 2*w:, :-2*w] + i2_cum[:, :, :-2*w, :-2*w])
    sig2_jj_tot = (j2_cum[:, :, 2*w:, 2*w:] - j2_cum[:, :, :-2*w, 2*w:] - j2_cum[:, :, 2*w:, :-2*w] + j2_cum[:, :, :-2*w, :-2*w])

    sig2_ii_tot = torch.clip(sig2_ii_tot, ep, sig2_ii_tot.max().item())
    sig2_jj_tot = torch.clip(sig2_jj_tot, ep, sig2_jj_tot.max().item())

    L = sig2_ij_tot / ((sig2_ii_tot * sig2_jj_tot) ** 0.5 + ep)

    return L