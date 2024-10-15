import torch.nn as nn
from torch import Tensor
import deepinv as dinv

from kornia.color import rgb_to_ycbcr
from torchmetrics.functional import structural_similarity_index_measure

from .utils import MultispectralUtils
from .pansharpen import Pansharpen


def cal_ssim(x_hat: Tensor, x: Tensor, y_channel: bool = False, **kwargs):
    """
    Compute the SSIM between two images

    :param torch.Tensor x_hat: reconstructed image
    :param torch.Tensor x: ground truth image
    :param bool y_channel: compute SSIM on the Y channel in CbCr space
    """
    if y_channel:
        x_hat = rgb_to_ycbcr(x_hat)[:, 0:1, :, :]
        x = rgb_to_ycbcr(x)[:, 0:1, :, :]
    return (
        structural_similarity_index_measure(x_hat, x, data_range=1.0, **kwargs)
        .detach()
        .cpu()
        .item()
    )


class QNR(nn.Module, MultispectralUtils):
    """Quality with No Reference metric from Alparone et al. "Multispectral and Panchromatic
        Data Fusion Assessment Without Reference", Photogrammetric Engineering & Remote Sensing 2008.

    :param Pansharpen physics: pansharpening physics
    :param float alpha: weight for spectral quality, defaults to 1
    :param float beta: weight for structural quality, defaults to 1
    :param float p: power exponent for spectral D, defaults to 1
    :param float q: power exponent for structural D, defaults to 1
    :param bool return_D: return D_\lambda, D_s along with QNR, defaults to False
    """

    def __init__(
        self,
        physics: Pansharpen,
        alpha: float = 1,
        beta: float = 1,
        p: float = 1,
        q: float = 1,
        return_D=False,
        **kwargs
    ):
        super().__init__()
        self.physics = physics
        self.alpha = alpha
        self.beta = beta
        self.p = p
        self.q = q
        self.return_D = return_D
        self.Q = cal_ssim  # original paper uses Wang-Bovik which is predecessor to SSIM

    def D_lambda(self, hrms: Tensor, lrms: Tensor) -> float:
        _, n_bands, _, _ = hrms.shape
        out = 0
        for b in range(n_bands):
            for c in range(n_bands):
                if b == c:
                    continue  # this shouldn't be necessary as 1-1=0 anyway
                out += (
                    abs(
                        self.Q(hrms[:, [b], :, :], hrms[:, [c], :, :])
                        - self.Q(lrms[:, [b], :, :], lrms[:, [c], :, :])
                    )
                    ** self.p
                )
        return (out / (n_bands * (n_bands - 1))) ** (1 / self.p)

    def D_s(self, hrms: Tensor, lrms: Tensor, pan: Tensor, pan_lr: Tensor) -> float:
        _, n_bands, _, _ = hrms.shape
        out = 0
        for b in range(n_bands):
            out += (
                abs(
                    self.Q(hrms[:, [b], :, :], pan) - self.Q(lrms[:, [b], :, :], pan_lr)
                )
                ** self.q
            )
        return (out / n_bands) ** (1 / self.q)

    def forward(self, x_hat: Tensor, y: Tensor) -> tuple | float:
        """Calculate QNR. In noiseless scenario, use inputs_from="y" as these are the real inputs.
        *QNR is NOT defined for noisy scenario*. Nevertheless, we calculate it when we have clean GT x.
        In noisy scenario, inputs_from="x" takes lrms+pan by downsampling clean GT. DO NOT use inputs_from="x_net"
        as even a bad network will have good QNR as it will be comparing bad to bad.

        Note in noiseless scenario, inputs_from="x" is same as inputs_from="y".

        :param Tensor x_hat: estimated HRMS+PAN volume
        :param Tensor y: input LRMS+PAN volume
        :return tuple | float: QNR, or if return_D, QNR, D_\lambda, D_s
        """
        hrms = self.hrms_from_volume(x_hat)
        lrms = self.lrms_from_volume(y, scale_factor=self.physics.factor)
        pan = self.pan_from_volume(y)

        pan_lr = self.physics.A_pan(pan)

        d_lambda = self.D_lambda(hrms, lrms)
        d_s = self.D_s(hrms, lrms, pan, pan_lr)
        qnr = (1 - d_lambda) ** self.alpha * (1 - d_s) ** self.beta

        return (qnr, d_lambda, d_s) if self.return_D else qnr
