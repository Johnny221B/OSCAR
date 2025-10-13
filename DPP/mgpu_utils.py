# -*- coding: utf-8 -*-
from typing import Tuple
import torch
import torch.nn.functional as F


def _to01(x: torch.Tensor) -> torch.Tensor:
    """Map tensor values from [-1, 1] to [0, 1]."""
    return (x.clamp(-1, 1) + 1) * 0.5


def _resize(x: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
    """Bilinear resize without corner alignment."""
    return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


def _pairwise_sqdist(x: torch.Tensor) -> torch.Tensor:
    """Pairwise squared Euclidean distance for row vectors in x (shape [N, D])."""
    x2 = (x * x).sum(dim=1, keepdim=True)
    return (x2 + x2.t() - 2.0 * (x @ x.t())).clamp_min(0.0)


def _logdet_stable(mat: torch.Tensor) -> torch.Tensor:
    """Stable log-determinant via slogdet with a small diagonal jitter."""
    eps = 1e-6
    I = torch.eye(mat.shape[-1], device=mat.device, dtype=mat.dtype)
    return torch.linalg.slogdet(mat + eps * I)[1]


__all__ = ["_to01", "_resize", "_pairwise_sqdist", "_logdet_stable"]
