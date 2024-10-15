import torch.nn as nn
import torch
from .utils import MultispectralUtils


class LRMS_MSELoss(nn.MSELoss, MultispectralUtils):
    """MSELoss on the LRMS image in the volume y"""

    def __init__(self, factor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.factor = factor

    def forward(self, y_hat, y):
        return super().forward(
            self.lrms_from_volume(y_hat, scale_factor=self.factor),
            self.lrms_from_volume(y, scale_factor=self.factor),
        )


class HRMS_MSELoss(nn.MSELoss, MultispectralUtils):
    """MSELoss on the HRMS image in the volume x"""

    def forward(self, x_hat, x):
        return super().forward(self.hrms_from_volume(x_hat), self.hrms_from_volume(x))


class LRMS_L1Loss(nn.L1Loss, MultispectralUtils):
    def __init__(self, factor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.factor = factor

    def forward(self, y_hat, y):
        return super().forward(
            self.lrms_from_volume(y_hat, scale_factor=self.factor),
            self.lrms_from_volume(y, scale_factor=self.factor),
        )


class BaseStructuralLoss(nn.Module, MultispectralUtils):
    """Abstract class for structural losses in pansharpening. Supports spectral response function (SRF)
    coming from different sources depending on model.

    The "constant" SRF coefficients are from reduced-resolution pre-training using stochastic linear
    regression on our SpaceNet-4 training set.

    :param str srf_from: from "estimate" (jointly train with model), "average" (simply average bands),
        "constant" (perform weighted average), defaults to "estimate"
    """

    def __init__(self, *args, srf_from: str = "estimate", **kwargs):
        super().__init__(*args, **kwargs)
        self.srf_from = srf_from
        assert self.srf_from in ("estimate", "average", "constant")
        self.name = "BaseStructural"
        self.constant_srf = [-0.40107638, 0.95721835, 0.21377654, 0.16151386]

    def pans_from_inputs(self, y, x_net):
        pan = self.pan_from_volume(y)
        hrms = self.hrms_from_volume(x_net)

        match self.srf_from:
            case "average":
                pan_recon = torch.mean(hrms, dim=1, keepdim=True)
            case "constant":
                weighted_hrms = hrms * torch.tensor(
                    self.constant_srf, device=hrms.device
                ).view(1, 4, 1, 1)
                pan_recon = torch.sum(weighted_hrms, dim=1, keepdim=True)
            case "estimate":
                pan_recon = self.pan_from_volume(x_net)
        return pan, pan_recon

    def forward(self, y, x_net, *args, **kwargs):
        raise NotImplementedError()


class TVStructuralLoss(BaseStructuralLoss):
    """
    Structural loss squared taken from Uezato et al. Guided Deep Decoder, ECCV 2020
    https://arxiv.org/abs/2007.11766

    We use their implementation from here:
    https://github.com/tuezato/guided-deep-decoder/blob/master/GDD_code/GDD_demo_PAN.py#L16

    Note this takes the full form rather than the anisotropic approximation in torchmetrics:
    https://github.com/Lightning-AI/torchmetrics/blob/v1.3.0.post0/src/torchmetrics/functional/image/tv.py#L20
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "TVStructural"

    def tv(self, img):
        w_variance = torch.mean(torch.pow(img[:, :, :, :-1] - img[:, :, :, 1:], 2))
        h_variance = torch.mean(torch.pow(img[:, :, :-1, :] - img[:, :, 1:, :], 2))
        return h_variance + w_variance

    def forward(self, y, x_net, *args, **kwargs):
        pan, pan_recon = self.pans_from_inputs(y, x_net)
        return self.tv(pan - pan_recon)


class SurePoissonSpectralLoss(nn.Module, MultispectralUtils):
    """SURE loss for Poisson noise on the multispectral bands of the input volume y.
    See deepinv.loss.SurePoissonLoss for details.
    """

    def __init__(self, gain, tau=1e-3):
        super().__init__()
        self.name = "SurePoissonSpectral"
        self.gain = gain
        self.tau = tau

    def forward(self, y, x_net, physics, model, **kwargs):
        # generate a random vector b
        b = torch.rand_like(y) > 0.5
        b = (2 * b - 1) * 1.0  # binary [-1, 1]

        y1 = physics.A(x_net)
        y2 = physics.A(model(y + self.tau * b, physics=physics))

        y1 = self.lrms_from_volume(y1)
        y2 = self.lrms_from_volume(y2)
        y = self.lrms_from_volume(y)
        b = self.lrms_from_volume(b)

        loss_sure = (
            (y1 - y).pow(2).mean()
            - self.gain * y.mean()
            + 2.0 / self.tau * (b * y * self.gain * (y2 - y1)).mean()
        )

        return loss_sure


class SurePoissonStructuralLoss(BaseStructuralLoss):
    """SURE loss for Poisson noise on the panchromatic bands.
    See deepinv.loss.SurePoissonLoss for details.
    """

    def __init__(self, gain, tau=1e-3, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "SurePoissonStructural"
        self.gain = gain
        self.tau = tau

    def forward(self, y, x_net, physics, model, *args, **kwargs):
        b = torch.rand_like(y) > 0.5
        b = (2 * b - 1) * 1.0  # binary [-1, 1]

        pan, pan1 = self.pans_from_inputs(y, x_net)
        _, pan2 = self.pans_from_inputs(y, model(y + self.tau * b, physics=physics))
        b = self.pan_from_volume(b)

        loss_sure = (
            (pan1 - pan).pow(2).mean()
            - self.gain * pan.mean()
            + 2.0 / self.tau * (b * pan * self.gain * (pan2 - pan1)).mean()
        )

        return loss_sure
