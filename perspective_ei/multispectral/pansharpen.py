import deepinv as dinv
from torch import Tensor
from .utils import MultispectralUtils

class Pansharpen(dinv.physics.Downsampling, MultispectralUtils):
    """Multispectral pansharpening operator. The forward physics decimates a 
    high-resolution multispectral image (HRMS) into a low-resolution multispectral image (LRMS)
    and high-resolution panchromatic (single-band) image (PAN).

    The LRMS is modelled by downsampling with an anti-aliasing kernel and the PAN by the
    satellite's spectral response function (SRF).

    In our implementation, we assume knowledge of the kernel but *not* of the SRF.

    Input x is a volume of shape (B,C+1,H,W) consisting of HRMS + PAN concatenated, 
    output y is a volume of same shape (B,C+1,H,W) consisting of LRMS + PAN concatenated, 
    where the LRMS is upsampled with zeros. Therefore when using y remember to 
    downsample it again before using with y[:, :, ::factor, ::factor] or with
    MultispectralUtils().lrms_from_volume(y).

    :param tuple[int] img_size: size of the input image
    :param int factor: pansharpening ratio of HR to LR
    """
    def __init__(self, img_size: tuple, factor: int = 2, device="cpu", **kwargs):
        super().__init__(img_size=img_size, factor=factor, device=device, **kwargs)
        self.device = device

    def A(self, x: Tensor):
        hrms = self.hrms_from_volume(x)
        pan = self.pan_from_volume(x)
        lrms = super().A(hrms)
        return self.lrms_pan_to_volume(lrms, pan, scale_factor=self.factor)

    def A_adjoint(self, y: Tensor):
        pan = self.pan_from_volume(y)
        lrms = self.lrms_from_volume(y, self.factor)
        hrms = super().A_adjoint(lrms)
        return self.hrms_pan_to_volume(hrms, pan)
        
    def A_pan(self, pan: Tensor):
        # Takes pure pan (B,1,H,W) and downsamples it
        return super().A(pan)

