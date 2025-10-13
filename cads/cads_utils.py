# -*- coding: utf-8 -*-
import torch
Tensor = torch.Tensor


@torch.no_grad()
def _stats_lastdim(x: Tensor) -> tuple[Tensor, Tensor]:
    """Return (mean, std) along the last dimension (numerically stable)."""
    mu = x.mean(dim=-1, keepdim=True)
    sig = x.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-6)
    return mu, sig


__all__ = ["_stats_lastdim"]
