from dataclasses import dataclass
from contextlib import nullcontext
import torch
from torch import Tensor
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from .. import Pansharpen, MultispectralUtils
from ..loss import BaseStructuralLoss

class PanganDiscriminator(nn.Module):
    """Reimplementation of PanGan discriminator from Ma et al. 
    "Pan-GAN: An unsupervised pan-sharpening method for remote sensing image fusion", Information Fusion 2020
    https://www.sciencedirect.com/science/article/abs/pii/S1566253520302591

    Modified to accept 1024x1024 input

    :param int input_channels: number of channels in input image, defaults to 1
    """
    def __init__(self, input_channels: int = 1):
        super().__init__()
        
        self.layer1 = self.conv_block(input_channels, 16, kernel_size=6, stride=4, padding=1)
        self.layer2 = self.conv_block(16, 32, kernel_size=6, stride=4, padding=1)
        self.layer3 = self.conv_block(32, 64, kernel_size=6, stride=4, padding=1)
        self.layer4 = self.conv_block(64, 128, kernel_size=6, stride=4, padding=1)
        self.layer5 = self.conv_block(128, 1, kernel_size=3, stride=2, padding=0)

    def conv_block(self, in_channels, out_channels, kernel_size, stride, padding, activation=True):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=True),
            nn.LeakyReLU(0.2, inplace=True) if activation else nn.Identity()
        )

    def forward(self, img):
        out = self.layer1(img)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out).view(-1, 1)
        return out


class PanganStructuralLoss(BaseStructuralLoss):
    """
    Reimplementation of PanGan structural loss
    Following https://github.com/yuwei998/PanGAN/blob/master/PanGan.py
    """
    def __init__(self, *args, device="cpu", **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "PanganStructural"
        kernel = torch.tensor(
            [[1., 1., 1.], [1., -8., 1.], [1., 1., 1.]], dtype=torch.float32, device=device
        ).unsqueeze(0).unsqueeze(0)
        self.kernel = nn.Parameter(kernel, requires_grad=False)

    def highpass(self, img):
        return nn.functional.conv2d(img, self.kernel, stride=1, padding=1)
    
    def forward(self, y: Tensor, x_net: Tensor, *args, **kwargs) -> Tensor:
        pan, pan_recon = self.pans_from_inputs(y, x_net)
        return nn.functional.mse_loss(
            self.highpass(pan_recon), 
            self.highpass(pan)
            )

class DiscriminatorLoss:
    """Generic GAN discriminator loss building block. Compares discriminator output
    with labels depending on if the image should be real or not.

    :param nn.Module metric: loss with which to compare outputs, defaults to nn.MSELoss()
    :param float real_label: value for ideal real image, defaults to 1.
    :param float fake_label: value for ideal fake image, defaults to 0.
    :param bool no_grad: whether to no_grad the metric computation, defaults to False
    :param str device: torch device, defaults to "cpu"
    """
    def __init__(
            self, 
            metric: nn.Module = nn.MSELoss(), 
            real_label: float = 1., 
            fake_label: float = 0., 
            no_grad: bool = False,
            device="cpu"):
        self.real = torch.Tensor([real_label]).to(device)
        self.fake = torch.Tensor([fake_label]).to(device)
        self.no_grad = no_grad
        self.metric = metric

    def __call__(self, pred: Tensor, real: bool = None) -> Tensor:
        """Call discriminator loss.

        :param torch.Tensor pred: discriminator classification output
        :param bool real: whether image should be real or not, defaults to None
        :return torch.Tensor: loss value
        """
        target = (self.real if real else self.fake).expand_as(pred)
        with torch.no_grad() if self.no_grad else nullcontext():
            return self.metric(pred, target)

class PanganAdvGenLoss(BaseStructuralLoss):
    """Reimplementation of PanGan generator's adversarial loss

    :param float weight_adv_spectral: spectral weight for adversarial loss, defaults to 0.002
    :param float weight_adv_structural: spectral weight for adversarial loss, defaults to 0.001
    :param str device: torch device, defaults to "cpu"
    """
    def __init__(
            self, 
            weight_adv_spectral: float = 0.002,
            weight_adv_structural: float = 0.001,
            device="cpu",
            **kwargs):
        super().__init__(**kwargs)
        self.name = "PanganAdvGen"
        self.metric_gan = DiscriminatorLoss(device=device)
        self.weight_adv_spectral = weight_adv_spectral
        self.weight_adv_structural = weight_adv_structural
    
    def forward(self, y: Tensor, x_net: Tensor, D_spectral: nn.Module, D_structural: nn.Module, **kwargs) -> Tensor:
        """D_spectral is the spectral discriminator (see PanGan paper),
        D_structural is the structural discriminator.
        """
        _, pan_recon = self.pans_from_inputs(y, x_net)

        pred_fake_structural = D_structural(pan_recon)
        pred_fake_spectral = D_spectral(self.hrms_from_volume(x_net))

        loss_structural = self.metric_gan(pred_fake_structural, real=True)
        loss_spectral   = self.metric_gan(pred_fake_spectral, real=True)

        return loss_spectral * self.weight_adv_spectral + loss_structural * self.weight_adv_structural

class PanganAdvDiscrimSpectralLoss(nn.Module, MultispectralUtils):
    """Reimplementation of PanGan spectral discriminator's adversarial loss

    :param float weight_adv: weight for adversarial loss, defaults to 1
    :param str device: torch device, defaults to "cpu"
    """
    def __init__(self, weight_adv=1, device="cpu"):
        super().__init__()
        self.metric_gan = DiscriminatorLoss(device=device)
        self.weight_adv = weight_adv
        self.name = "PanganAdvDiscrimSpectral"
    
    def forward(self, y: Tensor, x_net: Tensor, physics: Pansharpen, D_spectral: nn.Module, **kwargs) -> Tensor:
        """D_spectral is the spectral discriminator (see PanGan paper).
        """
        hrms_adjoint = self.hrms_from_volume(physics.A_adjoint(y))
        hrms_pred = self.hrms_from_volume(x_net)

        pred_real = D_spectral(hrms_adjoint)
        pred_fake = D_spectral(hrms_pred.detach())

        adv_loss_real = self.metric_gan(pred_real, real=True)
        adv_loss_fake = self.metric_gan(pred_fake, real=False)

        return (adv_loss_real + adv_loss_fake) * self.weight_adv 
    
class PanganAdvDiscrimStructuralLoss(BaseStructuralLoss):
    """Reimplementation of PanGan structural discriminator's adversarial loss

    :param float weight_adv: weight for adversarial loss, defaults to 1
    :param str device: torch device, defaults to "cpu"
    """
    def __init__(self, weight_adv=1, device="cpu", **kwargs):
        super().__init__(**kwargs)
        self.metric_gan = DiscriminatorLoss(device=device)
        self.weight_adv = weight_adv
        self.name = "PanganAdvDiscrimStructural"
    
    def forward(self, y: Tensor, x_net: Tensor, physics: Pansharpen, D_structural: nn.Module, **kwargs) -> Tensor:
        """D_structural is the structural discriminator (see PanGan paper).
        """
        pan, pan_recon = self.pans_from_inputs(y, x_net)

        pred_real = D_structural(pan)
        pred_fake = D_structural(pan_recon.detach())

        adv_loss_real = self.metric_gan(pred_real, real=True)
        adv_loss_fake = self.metric_gan(pred_fake, real=False)

        return (adv_loss_real + adv_loss_fake) * self.weight_adv 

class PanganOptimizer:
    """Torch optimizer that wraps individual PanGan models' optimizers

    :param Optimizer optimizer_g: generator's optimizer
    :param Optimizer optimizer_d_spectral: spectral discriminator's optimizer
    :param Optimizer optimizer_d_structural: structural discriminator's optimizer
    :param bool zero_grad_g_only: on zero_grad, only affect generator, defaults to False
    :param bool zero_grad_d_only: on zero_grad, only affect discriminators, defaults to False
    """
    def __init__(self, optimizer_g: Optimizer, optimizer_d_spectral: Optimizer, optimizer_d_structural: Optimizer, zero_grad_g_only=False, zero_grad_d_only=False):
        self.G = optimizer_g
        self.D_spectral   = optimizer_d_spectral
        self.D_structural = optimizer_d_structural
        if zero_grad_d_only and zero_grad_g_only:
            raise ValueError("zero_grad_d_only or zero_grad_d_only must be False")
        self.zero_grad_d_only = zero_grad_d_only
        self.zero_grad_g_only = zero_grad_g_only
    
    def load_state_dict(self, state_dict):
        return NotImplementedError()
    
    def state_dict(self):
        return self.G.state_dict()
    
    def zero_grad(self, set_to_none: bool = True):
        if not self.zero_grad_d_only:
            self.G.zero_grad(set_to_none=set_to_none)
        if not self.zero_grad_g_only:
            self.D_spectral.zero_grad(set_to_none=set_to_none)
            self.D_structural.zero_grad(set_to_none=set_to_none)

class PanganScheduler:
    """Torch scheduler that wraps individual PanGan models' schedulers

    :param LRScheduler scheduler_g: generator's scheduler
    :param LRScheduler scheduler_d_spectral: spectral discriminator's optimizer
    :param LRScheduler scheduler_d_structural: structural discriminator's optimizer
    """
    def __init__(self, scheduler_g: LRScheduler, scheduler_d_spectral: LRScheduler, scheduler_d_structural: LRScheduler):
        self.G = scheduler_g
        self.D_spectral = scheduler_d_spectral
        self.D_structural = scheduler_d_structural
    
    def get_last_lr(self):
        return self.G.get_last_lr()
    
    def step(self):
        self.G.step()
        self.D_spectral.step()
        self.D_structural.step()


@dataclass
class PanganTrainer(Trainer):
    """Trainer class for training PanGan. Inherits from deepinv
    modular Trainer code, see https://github.com/deepinv/deepinv/pull/147
    
    Introduces D_spectral and D_structural, the two discriminator networks,
    as additional parameters.

    Usage (ensure that losses passed in this exact order, see demo.ipynb):

    >>> losses = [
    >>>    MCLoss(),
    >>>    PanganStructuralLoss(),
    >>>    PanganAdvGenLoss(),
    >>>    PanganAdvDiscrimSpectralLoss(),
    >>>    PanganAdvDiscrimStructuralLoss(),
    >>> ]
    >>> 
    >>> D_spectral = PanganDiscriminator(input_channels=4).to(device)
    >>> D_structural = PanganDiscriminator(input_channels=1).to(device)
    >>> optimizer_d_spectral,   scheduler_d_spectral   = make_optimizer_scheduler(config, D_spectral,   lr_init=config.lr_init)
    >>> optimizer_d_structural, scheduler_d_structural = make_optimizer_scheduler(config, D_structural, lr_init=config.lr_init)
    >>> 
    >>> trainer = PanganTrainer(
    >>>    D_spectral=D_spectral,
    >>>    D_structural=D_structural,
    >>>    optimizer=PanganOptimizer(optimizer_g=optimizer, optimizer_d_spectral=optimizer_d_spectral, optimizer_d_structural=optimizer_d_structural, zero_grad_g_only=True, zero_grad_d_only=False),
    >>>    scheduler=PanganScheduler(scheduler_g=scheduler, scheduler_d_spectral=scheduler_d_spectral, scheduler_d_structural=scheduler_d_structural),
    >>> )
    >>> trainer.train()
    """
    optimizer: PanganOptimizer = None
    D_spectral: nn.Module = None
    D_structural: nn.Module = None

    def backward_pass(self, g, x, y, x_net):
        # Train G
        loss_mc     = self.losses[0](x=x, x_net=x_net, y=y, physics=self.physics[g], model=self.model)
        loss_struct = self.losses[1](x=x, x_net=x_net, y=y, physics=self.physics[g], model=self.model)
        loss_g_adv  = self.losses[2](x=x, x_net=x_net, y=y, physics=self.physics[g], model=self.model, D_spectral=self.D_spectral, D_structural=self.D_structural)
        
        loss_g = loss_mc + loss_struct + loss_g_adv
        loss_g.backward(retain_graph=True)
        
        self.check_clip_grad()

        self.optimizer.G.step()

        # Train D_spectral
        self.optimizer.D_spectral.zero_grad()
        loss_d_spectral = self.losses[3](x=x, x_net=x_net, y=y, physics=self.physics[g], model=self.model, D_spectral=self.D_spectral)

        loss_d_spectral.backward(retain_graph=True)

        self.optimizer.D_spectral.step()

        # Train D_structural
        self.optimizer.D_structural.zero_grad()
        loss_d_structural = self.losses[4](x=x, x_net=x_net, y=y, physics=self.physics[g], model=self.model, D_structural=self.D_structural)

        loss_d_structural.backward()

        self.optimizer.D_structural.step()

        for k, (l, loss) in enumerate(zip(self.losses, [loss_mc, loss_struct, loss_g_adv, loss_d_spectral, loss_d_structural])):
            self.losses_verbose[k].update(loss.item())
            if len(self.losses) > 1:
                self.log_dict["loss_" + l.name] = self.losses_verbose[k].avg
                if self.wandb_vis:
                    self.wandb_log_dict_iter["loss_" + l.name] = loss.item()
        if self.wandb_vis:
            self.wandb_log_dict_iter["training loss"] = loss_g.item()
        self.total_loss.update(loss_g.item())
        self.log_dict["total_loss"] = self.total_loss.avg
