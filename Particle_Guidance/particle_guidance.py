# ======================================
# FILE: Particle_Guidance/particle_guidance.py
# ======================================
# -*- coding: utf-8 -*-
"""
Particle Guidance (fixed potential) in latent space for Flow Matching sampling.

Training-free: adds a pairwise repulsive term via an RBF kernel on latents
at each step. This version includes stability guards (per-sample normalization,
1/N scaling) and a safer default alpha(t) schedule to avoid latent blow-ups.
"""
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Optional, Tuple
import torch


@dataclass
class PGConfig:
    group_size: int
    mode: str = "ode"  # "ode" or "sde"
    # Safer defaults
    alpha_scale: float = 1.0              # step strength scale
    sigma_a: float = 0.5                  # for alpha_mode="sigma2": sigma(t)=a*sqrt(t/(1-t))
    alpha_mode: str = "cos2"              # "cos2" (safe), or "sigma2" (paper-style)
    bandwidth: Optional[float] = None     # None -> median trick
    min_bandwidth: float = 1e-6
    eps: float = 1e-8
    center_latents: bool = False
    normalize_force: bool = True          # per-sample norm normalization
    dtype: Optional[torch.dtype] = torch.float32  # compute PG in fp32 for stability


class ParticleGuidance:
    def __init__(self, cfg: PGConfig):
        self.cfg = cfg
        if self.cfg.mode not in {"ode", "sde"}:
            raise ValueError("PGConfig.mode must be 'ode' or 'sde'")

    # ----- schedules -----
    def sigma(self, t: torch.Tensor) -> torch.Tensor:
        a = self.cfg.sigma_a
        eps = self.cfg.eps
        return a * torch.sqrt(torch.clamp(t, eps, 1 - eps) / torch.clamp(1 - t, eps, 1.0))

    def alpha(self, t: torch.Tensor) -> torch.Tensor:
        if self.cfg.alpha_mode == "sigma2":
            return self.cfg.alpha_scale * self.sigma(t) ** 2
        # cos^2 schedule: high early (t≈1), small late (t≈0)
        # alpha(t) = alpha_scale * (sin(pi * t / 2))^2
        return self.cfg.alpha_scale * torch.sin(0.5 * math.pi * t) ** 2

    # ----- internals -----
    @torch.no_grad()
    def _prep(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, ...]]:
        orig = x.shape
        if x.dim() > 2:
            x_flat = x.view(x.size(0), -1)
        else:
            x_flat = x
        if self.cfg.center_latents:
            x_flat = x_flat - x_flat.mean(dim=1, keepdim=True)
        if self.cfg.dtype is not None and x_flat.dtype != self.cfg.dtype:
            x_flat = x_flat.to(self.cfg.dtype)
        return x_flat, orig

    @torch.no_grad()
    def _pairwise_sq(self, x_flat: torch.Tensor) -> torch.Tensor:
        x2 = (x_flat ** 2).sum(1, keepdim=True)
        d2 = x2 + x2.T - 2.0 * (x_flat @ x_flat.T)
        d2 = d2.clamp_min(0.0)
        d2.fill_diagonal_(0.0)
        return d2

    @torch.no_grad()
    def _median_bandwidth(self, d2: torch.Tensor) -> float:
        n = d2.size(0)
        if n < 2:
            return 1.0
        idx = torch.triu_indices(n, n, 1, device=d2.device)
        vals = torch.sqrt(torch.clamp(d2[idx[0], idx[1]], min=0.0))
        if vals.numel() == 0:
            m = torch.tensor(1.0, device=d2.device, dtype=d2.dtype)
        else:
            m = vals.median()
            if not torch.isfinite(m):
                m = torch.tensor(1.0, device=d2.device, dtype=d2.dtype)
        h = float((m * m) / max(math.log(n + 1.0), 1.0))
        return max(h, self.cfg.min_bandwidth)

    @torch.no_grad()
    def kernel_grad(self, x: torch.Tensor) -> torch.Tensor:
        x_flat, orig = self._prep(x)
        n = x_flat.size(0)
        if n <= 1:
            return torch.zeros_like(x)
        d2 = self._pairwise_sq(x_flat)
        h = self._median_bandwidth(d2) if self.cfg.bandwidth is None else max(float(self.cfg.bandwidth), self.cfg.min_bandwidth)
        K = torch.exp(-d2 / h)
        K1 = K.sum(1, keepdim=True)
        Kx = K @ x_flat
        grad_flat = (2.0 / h) * (K1 * x_flat - Kx)
        # Stability: scale by 1/N (matches average-pair potential) and normalize per sample
        grad_flat = grad_flat / max(1, n)
        if self.cfg.normalize_force:
            gn = grad_flat.norm(dim=1, keepdim=True).clamp_min(self.cfg.eps)
            grad_flat = grad_flat / gn
        return grad_flat.view(orig) if len(orig) > 2 else grad_flat

    @torch.no_grad()
    def apply(self, x: torch.Tensor, t: float | torch.Tensor, dt: float, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        dev = x.device
        dtype = self.cfg.dtype or x.dtype
        if not torch.is_tensor(t):
            t = torch.tensor(t, device=dev, dtype=dtype)
        t = t.reshape(())
        t = t.clamp(0.0, 1.0)

        grad = self.kernel_grad(x).to(dtype)
        alpha_t = self.alpha(t).to(dtype)
        x_next = x.to(dtype) - alpha_t * grad * dt

        if self.cfg.mode == "sde":
            std = float(torch.sqrt(torch.clamp(self.sigma(t) ** 2 * dt, min=0.0)))
            if std > 0.0:
                noise = torch.randn_like(x_next, generator=generator)
                x_next = x_next + std * noise
        return x_next.type_as(x)
