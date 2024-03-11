from typing import Callable
import numpy as np
import torch
from skimage.io import imread

from torchvision.datasets.folder import has_file_allowed_extension
from torchvision import transforms

def spacenet_loader(filename: str) -> list:
    """Load HRMS + PAN images from SpaceNet-4. The images were downloaded as .tif,
    processed with read_tif_spacenet() and saved as npz volumes. 
    For a new dataset, reimplement this function.

    :param str filename: filename of image being loaded
    :return list: [hrms image, pan image] as ndarrays
    """
    f = np.load(filename)
    return [f["hrms"], f["pan"]]

def spacenet_is_valid_file() -> Callable:
    """Return callable for flagging which files are to be read into dataset.
    For a new dataset, reimplement this function."""
    return lambda f: has_file_allowed_extension(f, (".npz"))

def spacenet_transform() -> Callable:
    """Return torchvision transform for preprocessing HRMS and PAN images
    and concatenating into a volume of shape (B,C+1,H,W) for loading into 
    training. For a new dataset, reimplement this function."""
    transform_hrms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(1024, interpolation=transforms.InterpolationMode.BICUBIC),
    ])
    transform_pan = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(1024, interpolation=transforms.InterpolationMode.BICUBIC),
    ])
    return lambda f: torch.cat([
        transform_hrms(f[0]),
        transform_pan(f[1])
    ], dim=0)

def read_tif_spacenet(filename: str, channel: int = 0, thresh: float = 3000.) -> np.ndarray:
    """Read tif from SpaceNet-4 downloaded data, process according to official code
    https://github.com/CosmiQ/CosmiQ_SN4_Baseline and output as ndarray

    :param str filename: filename of tif
    :param int channel: axis where channel dimension is, defaults to 0
    :param float thresh: clipping threshold, defaults to 3000.
    :return np.ndarray: output processed image as array
    """
    I = imread(filename).clip(min=0, max=thresh)
    I = np.floor_divide(I, thresh/255).astype('uint8')

    if I.ndim == 2:
        pass
    else:
        match channel:
            case 0:
                I = np.moveaxis(I, (0, 1, 2), (2, 0, 1))
            case 1:
                raise ValueError("Channel dim can't be between img dims")

    return I