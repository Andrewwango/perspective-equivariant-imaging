import torch
from torch import Tensor
import deepinv as dinv

from ..utils import Pansharpen, MultispectralUtils
from ..loss import BaseStructuralLoss
from ..qnr import cal_ssim, QNR


class SSQBase:
    """Base class for reimplementation of SSQ loss calculations, from Luo et al.
    "Pansharpening via Unsupervised Convolutional Neural Networks", IEEE J-STARS 2020
    https://ieeexplore.ieee.org/document/9136909
    """

    # Weight reg term handled by Adam optimizer
    def loss_fn(self, pred: Tensor, target: Tensor) -> Tensor:
        """Calculate MSE + 1 - SSIM"""
        return (
            torch.nn.functional.mse_loss(pred, target)
            + 1
            - cal_ssim(pred, target, gaussian_kernel=False, kernel_size=11)
        )


class SSQSpectralLoss(torch.nn.Module, SSQBase, MultispectralUtils):
    """Reimplementation of spectral component of SSQ measurement consistency loss."""

    def __init__(self, device, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "SSQSpectral"
        self.device = device

    def rescale_tensor(self, b: Tensor, a: Tensor) -> Tensor:
        """Rescale b according to a's min and max."""
        min_a, max_a = a.min(), a.max()
        return ((b - b.min()) / (b.max() - b.min())) * (max_a - min_a) + min_a

    def forward(
        self, y: Tensor, x_net: Tensor, physics: Pansharpen, *args, **kwargs
    ) -> Tensor:
        lrms = self.lrms_from_volume(y)
        lrms_up = dinv.physics.Downsampling(
            physics.imsize, factor=4, filter="bilinear", device=self.device
        ).A_adjoint(lrms)
        lrms_up = self.rescale_tensor(lrms_up, lrms)
        hrms = self.hrms_from_volume(x_net)
        blur = dinv.physics.Blur(physics.filter, device=self.device)
        hrms_lp = blur(hrms)
        return self.loss_fn(hrms_lp, lrms_up)


class SSQStructuralLoss(BaseStructuralLoss, SSQBase):
    """Reimplementation of structural component of SSQ measurement consistency loss."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "SSQStructural"

    def forward(self, y: Tensor, x_net: Tensor, *args, **kwargs) -> Tensor:
        pan, pan_recon = self.pans_from_inputs(y, x_net)
        return self.loss_fn(pan_recon, pan)


class QNRLoss(torch.nn.Module):
    """Reimplementation of QNR component of SSQ measurement consistency loss."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "QNR"

    def forward(
        self, y: Tensor, x_net: Tensor, physics: Pansharpen, *args, **kwargs
    ) -> Tensor:
        loss = 1 - QNR(physics)(x_net, y)
        return torch.tensor(loss, device=y.device)
