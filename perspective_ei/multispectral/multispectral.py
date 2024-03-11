import torch
from torch import Tensor
import torch.nn as nn

class MultispectralUtils:
    """Utility class to obtain lrms, hrms and pan images from concatenated volumes.
    We assume that all volumes are passed around as (B,C+1,H,W) where the extra channel
    is the pan band, and H,W are the HR dimensions. LRMS is stored by upsampling and 
    zero-filling. The RGB and NIR utilities assume that the RGB are the first 3 channels 
    followed by NIR.
    """
    def lrms_from_volume(self, volume: Tensor, scale_factor: int = 4) -> Tensor:
        return volume[:, :-1, ::scale_factor, ::scale_factor]
    def pan_from_volume(self, volume: Tensor) -> Tensor:
        return volume[:, [-1], :, :]
    def hrms_from_volume(self, volume: Tensor) -> Tensor:
        return volume[:, :-1, :, :]
    def rgb_from_ms(self, ms: Tensor) -> Tensor:
        return ms[:, :3, :, :]
    def nir_from_ms(self, ms: Tensor) -> Tensor:
        assert ms.shape[1] == 4
        return ms[:, 3, :, :]
    def rgb_nir_from_ms(self, ms: Tensor) -> tuple:
        return self.rgb_from_ms(ms), self.nir_from_ms(ms)
    def hrms_pan_to_volume(self, hrms: Tensor, pan: Tensor) -> Tensor:
        return torch.cat([hrms, pan], dim=1)
    def lrms_pan_to_volume(self, lrms: Tensor, pan: Tensor, scale_factor: int = 4) -> Tensor:
        lrms_up = torch.zeros((
            lrms.shape[0], 
            lrms.shape[1], 
            lrms.shape[2] * scale_factor, 
            lrms.shape[3] * scale_factor
            ), device=lrms.device)
        lrms_up[:, :, :: scale_factor, :: scale_factor] = lrms
        return self.hrms_pan_to_volume(lrms_up, pan)
    
class LRMS_MSELoss(nn.MSELoss, MultispectralUtils):
    """MSELoss on the LRMS image in the volume y"""
    def __init__(self, factor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.factor = factor

    def forward(self, y_hat, y):
        return super().forward(
            self.lrms_from_volume(y_hat, scale_factor=self.factor),
            self.lrms_from_volume(y, scale_factor=self.factor)
        )
    
class HRMS_MSELoss(nn.MSELoss, MultispectralUtils):
    """MSELoss on the HRMS image in the volume x"""
    def forward(self, x_hat, x):
        return super().forward(
            self.hrms_from_volume(x_hat),
            self.hrms_from_volume(x)
        )