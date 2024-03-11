from .multispectral import MultispectralUtils, HRMS_MSELoss, LRMS_MSELoss
from .pansharpen import Pansharpen
from .pannet import ResNet, PanNet
from .qnr import QNR
from .loss import TVStructuralLoss, SurePoissonSpectralLoss, SurePoissonStructuralLoss

import torch
import deepinv as dinv

def plot_multispectral(x: torch.Tensor | list, y: torch.Tensor=None):
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
        return dinv.utils.plot([
            msu.rgb_from_ms(msu.hrms_from_volume(x)), 
            msu.rgb_from_ms(msu.lrms_from_volume(y)), 
            msu.rgb_from_ms(msu.pan_from_volume(y))
        ], titles=["HRMS", "LRMS", "PAN"])