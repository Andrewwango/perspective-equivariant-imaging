import torch
from torch import Tensor
import deepinv as dinv


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

    def lrms_pan_to_volume(
        self, lrms: Tensor, pan: Tensor, scale_factor: int = 4
    ) -> Tensor:
        lrms_up = torch.zeros(
            (
                lrms.shape[0],
                lrms.shape[1],
                lrms.shape[2] * scale_factor,
                lrms.shape[3] * scale_factor,
            ),
            device=lrms.device,
        )
        lrms_up[:, :, ::scale_factor, ::scale_factor] = lrms
        return self.hrms_pan_to_volume(lrms_up, pan)


def plot_multispectral(x: torch.Tensor | list, y: torch.Tensor = None):
    """Plot HRMS, LRMS and PAN images from x and y volumes.

    :param torch.Tensor x: _description_
    :param torch.Tensor y: _description_, defaults to None
    :return _type_: _description_
    """
    msu = MultispectralUtils()

    if isinstance(x, (tuple, list)):
        return dinv.utils.plot([msu.rgb_from_ms(msu.hrms_from_volume(_x)) for _x in x])
    elif y is None:
        return dinv.utils.plot(msu.rgb_from_ms(msu.hrms_from_volume(x)))
    else:
        return dinv.utils.plot(
            [
                msu.rgb_from_ms(msu.hrms_from_volume(x)),
                msu.rgb_from_ms(msu.lrms_from_volume(y)),
                msu.rgb_from_ms(msu.pan_from_volume(y)),
            ],
            titles=["HRMS", "LRMS", "PAN"],
        )
