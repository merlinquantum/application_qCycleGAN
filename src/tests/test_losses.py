"""Tests for CycleGAN loss helpers."""

import torch
import torch.nn as nn

from config import TrainingConfig
from training.losses import CycleGANLosses


class DummyLPIPS(nn.Module):
    def forward(self, x, y):
        _ = (x, y)
        # Return constant tensor to keep gradients simple
        return torch.ones(x.size(0), 1, 1, 1, device=x.device, dtype=x.dtype)


def test_cycle_loss_zero_when_identical():
    """Cycle loss should be zero if both tensors match and LPIPS weight is zero."""
    cfg = TrainingConfig(lambda_cycle=1.0, lambda_cycle_lpips=0.0)
    losses = CycleGANLosses(cfg)
    x = torch.randn(2, 3, 4, 4, requires_grad=True)
    loss = losses.cycle_loss(x, x)
    assert torch.allclose(loss, torch.tensor(0.0, dtype=loss.dtype), atol=1e-6)


def test_identity_loss_includes_lpips():
    """Identity loss should include LPIPS contribution when provided."""
    cfg = TrainingConfig(lambda_idt=1.0, lambda_idt_lpips=2.0)
    losses = CycleGANLosses(cfg, lpips_module=DummyLPIPS())
    output = torch.zeros(1, 3, 4, 4, requires_grad=True)
    target = torch.ones_like(output)
    loss = losses.identity_loss(output, target)
    # L1 = mean absolute difference = 1, LPIPS = 1 * lambda = 2
    expected = torch.tensor(3.0, dtype=loss.dtype)
    assert torch.allclose(loss, expected, atol=1e-6)


def test_gan_loss_scales_mean():
    """GAN loss should multiply mean logits by lambda."""
    cfg = TrainingConfig(lambda_gan=0.5)
    losses = CycleGANLosses(cfg)
    logits = torch.tensor([[2.0, 4.0], [1.0, 3.0]])
    loss = losses.gan_loss(logits)
    assert torch.allclose(loss, torch.tensor(0.5 * logits.mean()))
