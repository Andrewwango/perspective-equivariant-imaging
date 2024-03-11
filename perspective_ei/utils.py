import torch
from deepinv.utils import get_freer_gpu

def get_device():
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