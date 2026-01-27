"""Loss composition helpers for CycleGAN training."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from config import TrainingConfig


class CycleGANLosses:
    """Container that encapsulates CycleGAN loss computations."""

    def __init__(
        self,
        config: TrainingConfig,
        lpips_module: Optional[nn.Module] = None,
        cycle_criterion: Optional[nn.Module] = None,
        identity_criterion: Optional[nn.Module] = None,
    ) -> None:
        self.config = config
        self.lpips = lpips_module
        self.crit_cycle = cycle_criterion or nn.L1Loss()
        self.crit_idt = identity_criterion or nn.L1Loss()

    def cycle_loss(self, reconstructed: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Cycle-consistency loss (L1 + LPIPS) with weighting."""
        l1 = self.crit_cycle(reconstructed, target) * self.config.lambda_cycle
        lpips_loss = self._lpips_loss(reconstructed, target, self.config.lambda_cycle_lpips)
        return l1 + lpips_loss

    def identity_loss(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Identity loss (L1 + LPIPS) with weighting."""
        l1 = self.crit_idt(output, target) * self.config.lambda_idt
        lpips_loss = self._lpips_loss(output, target, self.config.lambda_idt_lpips)
        return l1 + lpips_loss

    def gan_loss(self, discriminator_output: torch.Tensor) -> torch.Tensor:
        """Adversarial loss scaled by lambda_gan."""
        return discriminator_output.mean() * self.config.lambda_gan

    def _lpips_loss(self, output: torch.Tensor, target: torch.Tensor, weight: float) -> torch.Tensor:
        if weight == 0 or self.lpips is None:
            return output.new_zeros(())
        return self.lpips(output, target).mean() * weight
