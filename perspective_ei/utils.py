import torch
from deepinv.utils import get_freer_gpu
from PIL import Image
from torchvision.transforms import PILToTensor


def get_device():
    """Get CUDA device is available else cpu"""
    return get_freer_gpu() if torch.cuda.is_available() else "cpu"


def make_optimizer_scheduler(model: torch.nn.Module, lr_init: float = 1e-3) -> tuple:
    """Make Adam optimizer and learning rate scheduler

    :param torch.nn.Module model: neural network
    :param float lr_init: initial learning rate, defaults to 1e-3
    :return tuple: optimizer, scheduler
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_init, weight_decay=1e-8)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.9)
    return optimizer, scheduler


def PIL2tensor(filename: str) -> torch.Tensor:
    """Open PIL Image and convert to tensor

    :param str filename: image filename
    :return torch.Tensor: image tensor of shape (B,C,H,W)
    """
    return PILToTensor()(Image.open(filename))[None].double()
