# -*- coding: utf-8 -*-
# File: flow_base/DPP/dpp_coupler_mgpu.py
import math
from typing import List, Tuple

import torch

from .mgpu_config import MGPUConfig
from .mgpu_utils import _to01, _resize, _pairwise_sqdist, _logdet_stable

# ---- helpers local to this module ----


def _gamma_t(t: float, grad_mean_norm: torch.Tensor, cfg: MGPUConfig) -> float:
    """Time-dependent step scale γ(t), normalized by mean grad-norm."""
    if cfg.gamma_sched == "sqrt":
        sched = (1.0 - t) ** 0.5
    elif cfg.gamma_sched == "sin2":
        sched = math.sin(math.pi * t) ** 2
    else:
        sched = t * (1.0 - t)
    return float(cfg.gamma_max * sched / (grad_mean_norm.item() + 1e-8))

# ---- core ----


class DPPCouplerMGPU:
    """
    Multi-GPU DPP coupler.

    Device placement:
      - transformer/text encoders on `dev_tr`
      - VAE on `dev_vae`
      - CLIP (possibly JIT) on `dev_clip`

    Per-chunk callback flow:
      z (on dev_vae, requires_grad) -> decode(imgs) -> to(dev_clip) -> DPP loss -> dL/dimgs
      dimgs back to dev_vae -> autograd.grad(imgs, z, dimgs) -> dL/dz
      dL/dz back to latents.device (dev_tr) to update the corresponding chunk of latents.
    """

    def __init__(
        self,
        pipe,
        # CLIP feature tower (already on dev_clip)
        feat_module,
        dev_tr: torch.device,
        dev_vae: torch.device,
        dev_clip: torch.device,
        cfg: MGPUConfig = MGPUConfig(),
    ):
        self.pipe = pipe
        self.feat = feat_module
        self.dev_tr = dev_tr
        self.dev_vae = dev_vae
        self.dev_clip = dev_clip
        self.cfg = cfg
        self.vae = pipe.vae
        self.vae_scale = getattr(self.vae.config, "scaling_factor", 1.0)
        self.vae_dtype = next(self.vae.parameters()).dtype

    # ---- helpers ----
    def _vae_decode(self, z_vae: torch.Tensor) -> torch.Tensor:
        """Decode VAE latents on dev_vae to [0, 1] images and optionally resize for CLIP."""
        img = self.vae.decode(z_vae / self.vae_scale,
                              return_dict=False)[0]  # [-1, 1]
        img = _to01(img).float()
        if img.shape[-2] > self.cfg.decode_size or img.shape[-1] > self.cfg.decode_size:
            img = _resize(img, (self.cfg.decode_size, self.cfg.decode_size))
        return img

    def _dpp_ll(self, feats_full: torch.Tensor) -> torch.Tensor:
        """
        DPP log-likelihood for L2-normalized features on dev_clip.
        Uses RBF kernel with median heuristic and returns log det(K) - log det(K + I).
        """
        D2 = _pairwise_sqdist(feats_full.float())
        triu = torch.triu(D2, diagonal=1)
        med = torch.median(triu[triu > 0]).clamp_min(1e-6)
        K = torch.exp(-self.cfg.kernel_spread * D2 / med)
        I = torch.eye(K.shape[0], device=K.device, dtype=K.dtype)
        return _to_loglik(K, I)

    def __call__(self, step_index: int, t_norm: float, latents: torch.Tensor) -> torch.Tensor:
        """
        Args:
            step_index: Current step (unused here but kept for interface parity).
            t_norm:     Normalized time in [0, 1] for γ(t) scheduling.
            latents:    Tensor [B, C, H, W] on dev_tr (transformer device).

        Returns:
            Updated latents tensor on dev_tr with the same shape.
        """
        B = latents.size(0)
        if B < 2:
            return latents

        # Pass 1: compute constant features for "other" samples (no graph)
        feats_const_chunks = self._precompute_feats_const(
            latents)  # each [bs, D] on dev_clip
        # [B, D] on dev_clip
        feats_const = torch.cat(feats_const_chunks, dim=0)

        # Chunked backprop
        lat_new = latents.clone()
        grad_norm_acc = []

        for s in range(0, B, self.cfg.chunk_size):
            e = min(B, s + self.cfg.chunk_size)

            # 1) Leaf latent on VAE device
            z = (
                latents[s:e]
                .to(self.dev_vae, dtype=self.vae_dtype, non_blocking=True)
                .detach()
                .clone()
                .requires_grad_(True)
            )

            with torch.enable_grad():
                # 2) Decode on dev_vae -> move images to dev_clip
                # dev_vae
                imgs = self._vae_decode(z)
                imgs_clip = imgs.to(
                    self.dev_clip, dtype=torch.float32, non_blocking=True)

                # 3) Features for this chunk with grad
                # dev_clip, requires grad
                feats_chunk = self.feat(imgs_clip)

                # 4) Assemble full features (others constant, this chunk variable)
                feats_full = feats_const.clone()
                feats_full[s:e] = feats_chunk

                # 5) DPP objective on dev_clip
                ll = self._dpp_ll(feats_full)

                # 6) ∂L/∂imgs_clip on dev_clip
                grad_img_clip = torch.autograd.grad(
                    ll, imgs_clip, retain_graph=False, create_graph=False)[0]
                grad_img_clip = grad_img_clip.to(dtype=imgs_clip.dtype)

            # 7) Move image grads back to dev_vae and compute VJP ∂L/∂z
            grad_img_vae = grad_img_clip.to(
                self.dev_vae, non_blocking=True).to(dtype=imgs.dtype)
            grad_z = torch.autograd.grad(outputs=imgs, inputs=z, grad_outputs=grad_img_vae,
                                         retain_graph=False, create_graph=False, allow_unused=False)[0]  # dev_vae

            # 8) Normalize/clip and scale by γ(t)
            gn = grad_z.flatten(1).norm(dim=1).mean().clamp_min(1e-8)
            grad_norm_acc.append(gn)
            if self.cfg.clip_grad_norm and self.cfg.clip_grad_norm > 0:
                per = grad_z.flatten(1).norm(dim=1, p=2)
                scale = (self.cfg.clip_grad_norm / (per + 1e-6)).clamp_max(1.0)
                grad_z = grad_z * scale.view(-1, 1, 1, 1)
            gamma = _gamma_t(t_norm, gn, self.cfg)

            # 9) Write back to transformer device slice
            delta = (gamma * grad_z).to(lat_new.device,
                                        non_blocking=True).to(dtype=lat_new.dtype)
            lat_new[s:e] = lat_new[s:e] - delta

            # 10) Cleanup and sync
            del z, imgs, imgs_clip, feats_chunk, feats_full, ll, grad_img_clip, grad_img_vae, grad_z, delta

            if self.dev_clip.type == 'cuda':
                torch.cuda.synchronize(self.dev_clip)
            if self.dev_vae.type == 'cuda':
                torch.cuda.synchronize(self.dev_vae)

        return lat_new

    @torch.no_grad()
    def _precompute_feats_const(self, latents: torch.Tensor) -> List[torch.Tensor]:
        """
        Precompute chunked feature tensors without building a graph (const w.r.t. z).
        Returns a list of tensors [Tensor(B_i, D)] in batch order, all on dev_clip.
        """
        B = latents.size(0)
        feats_const: List[torch.Tensor] = []
        for s in range(0, B, self.cfg.chunk_size):
            e = min(B, s + self.cfg.chunk_size)
            z = latents[s:e].to(
                self.dev_vae, dtype=self.vae_dtype, non_blocking=True)
            # dev_vae
            imgs = self._vae_decode(z)
            imgs_c = imgs.to(self.dev_clip, dtype=torch.float32,
                             non_blocking=True)     # CLIP side uses fp32
            # dev_clip, no grad
            feats = self.feat(imgs_c).detach()
            feats_const += [feats]
            del z, imgs, imgs_c, feats
        return feats_const


def _to_loglik(K: torch.Tensor, I: torch.Tensor) -> torch.Tensor:
    """Helper: log det(K) - log det(K + I) with stability."""
    return _logdet_stable(K) - _logdet_stable(K + I)
