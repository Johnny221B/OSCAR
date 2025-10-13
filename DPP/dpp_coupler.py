# =============================
# File: diverse_flow/DPP/dpp_coupler.py
# =============================

import math
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple

import torch
import torch.nn.functional as F


# ---- small utils ----

def _to_range_0_1(img: torch.Tensor) -> torch.Tensor:
    return (img.clamp(-1, 1) + 1) * 0.5


def _resize_bilinear(x: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
    return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


def _pairwise_sqdist(x: torch.Tensor) -> torch.Tensor:
    x2 = (x * x).sum(dim=1, keepdim=True)
    dist = x2 + x2.t() - 2.0 * (x @ x.t())
    return dist.clamp_min(0.0)


def _safe_slogdet(mat: torch.Tensor) -> torch.Tensor:
    eps = 1e-6
    mat = mat + eps * torch.eye(mat.shape[-1], device=mat.device, dtype=mat.dtype)
    sign, logabsdet = torch.linalg.slogdet(mat)
    mask_bad = (sign <= 0)
    if mask_bad.any():
        logabsdet = torch.where(mask_bad, torch.full_like(logabsdet, -50.0), logabsdet)
    return logabsdet


# ---- config ----

@dataclass
class DPPConfig:
    kernel_spread: float = 3.0
    use_quality_term: bool = False
    quality_rho: float = 2.5
    quality_eps: float = 1e-2
    gamma_max: float = 0.12
    gamma_sched: str = "sqrt"  # ["sqrt", "sin2", "poly"]
    clip_grad_norm: float = 5.0
    decode_size: int = 256
    use_predicted_image_if_available: bool = False  # must be False to keep graph


class DPPCoupler:
    def __init__(self, pipe, feature_extractor, device: torch.device = torch.device("cuda:0"), cfg: DPPConfig = DPPConfig()):
        self.pipe = pipe
        self.device = device
        self.cfg = cfg
        self.feat = feature_extractor  # imgs->[B,D], normalized; must stay on same device for autograd
        self.prev_latents: Optional[torch.Tensor] = None
        self.prev_t: Optional[float] = None
        self.vae = pipe.vae
        self.vae_scale = getattr(self.vae.config, "scaling_factor", 0.18215)

        # Ensure same device
        feat_dev = getattr(self.feat, "device", None)
        if feat_dev is not None and torch.device(feat_dev) != self.device:
            raise RuntimeError(
                f"feature_extractor.device={feat_dev} 与 SD/latents device={self.device} 不一致。"
                "请将 --vision_device 与 --sd_device 设为相同（可借助 CUDA_VISIBLE_DEVICES 选择物理卡）。"
            )

    # --- helpers ---
    def _decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        z = latents / self.vae_scale
        img = self.vae.decode(z, return_dict=False)[0]  # [-1,1]
        img = _to_range_0_1(img)
        if (img.shape[-1] > self.cfg.decode_size) or (img.shape[-2] > self.cfg.decode_size):
            img = _resize_bilinear(img, (self.cfg.decode_size, self.cfg.decode_size))
        return img

    def _estimate_v_from_diff(self, x_t: torch.Tensor, t: float) -> Optional[torch.Tensor]:
        if (self.prev_latents is None) or (self.prev_t is None):
            return None
        dt = float(t) - float(self.prev_t)
        if abs(dt) < 1e-9:
            return None
        return (x_t - self.prev_latents) / dt

    def _estimate_x1(self, x_t: torch.Tensor, v_t_detached: Optional[torch.Tensor], t: float) -> torch.Tensor:
        if v_t_detached is None:
            return x_t
        return x_t + v_t_detached * (1.0 - float(t))

    def _quality_vector(self, x_t: torch.Tensor, v_t_detached: torch.Tensor, t: float) -> torch.Tensor:
        if not self.cfg.use_quality_term:
            return torch.ones(x_t.shape[0], device=x_t.device, dtype=x_t.dtype)
        x0_hat = x_t - v_t_detached * float(t)
        norm2 = x0_hat.flatten(1).pow(2).sum(dim=1)
        rho2 = self.cfg.quality_rho ** 2
        q = torch.where(norm2 <= rho2, torch.ones_like(norm2), torch.exp(-(norm2 - rho2)))
        return torch.clamp(q, min=self.cfg.quality_eps)

    def _gamma_t(self, t: float, grad_norm: torch.Tensor) -> float:
        if self.cfg.gamma_sched == "sqrt":
            sched = (1.0 - t) ** 0.5
        elif self.cfg.gamma_sched == "sin2":
            sched = math.sin(math.pi * t) ** 2
        else:
            sched = t * (1.0 - t)
        return float(self.cfg.gamma_max * sched / (grad_norm.item() + 1e-8))

    def _compute_dpp_ll(self, feats: torch.Tensor, quality: Optional[torch.Tensor] = None) -> torch.Tensor:
        feats = feats.float()
        D2 = _pairwise_sqdist(feats)
        triu = torch.triu(D2, diagonal=1)
        med = torch.median(triu[triu > 0]).clamp_min(1e-6)
        K = torch.exp(-self.cfg.kernel_spread * D2 / med)
        if quality is not None:
            K = K * (quality.view(-1, 1) * quality.view(1, -1))
        I = torch.eye(K.shape[0], device=K.device, dtype=K.dtype)
        log_det_L = torch.linalg.slogdet(K + 1e-6 * I)[1]
        log_det_LI = torch.linalg.slogdet(K + I + 1e-6 * I)[1]
        return log_det_L - log_det_LI

    def __call__(self, i: int, t: float, latents: torch.Tensor, callback_kwargs: Dict[str, Any]) -> torch.Tensor:
        B = latents.shape[0]
        if B < 2:
            self.prev_latents = latents.detach()
            self.prev_t = t
            return latents

        # try to get model velocity; otherwise finite-diff estimate
        v_t_detached = None
        for key in ("model_pred", "velocity", "predicted_velocity"):
            if key in callback_kwargs and isinstance(callback_kwargs[key], torch.Tensor):
                v_t_detached = callback_kwargs[key].detach()
                break
        if v_t_detached is None:
            v_t_detached = self._estimate_v_from_diff(latents.detach(), t)

        x1_hat_lat = self._estimate_x1(latents, v_t_detached, t)
        latents_req = x1_hat_lat.detach().to(self.device)
        latents_req.requires_grad_(True)

        # ---- Critical: re-enable grad inside diffusers' no_grad context ----
        with torch.enable_grad():
            with torch.amp.autocast('cuda', enabled=False):
                imgs = self._decode_latents(latents_req)  # MUST depend on latents_req
                # optional type cast to float32 for stability
                imgs = imgs.float()
                feats = self.feat(imgs)                  # keep autograd (no no_grad inside)
                q_vec = None
                if self.cfg.use_quality_term and (v_t_detached is not None):
                    q_vec = self._quality_vector(latents.detach(), v_t_detached, t)
                ll = self._compute_dpp_ll(feats, q_vec)

        # safety checks
        if not imgs.requires_grad:
            raise RuntimeError("[DPP] imgs 不带梯度。请确认未在前向中使用 no_grad/inference_mode，并关闭 predicted_image 路径。")

        grad = torch.autograd.grad(ll, latents_req, retain_graph=False, create_graph=False)[0]
        grad_norm = grad.flatten(1).norm(dim=1).mean().clamp_min(1e-8)

        if self.cfg.clip_grad_norm and self.cfg.clip_grad_norm > 0:
            total_norm = grad.flatten(1).norm(dim=1, p=2)
            scale = (self.cfg.clip_grad_norm / (total_norm + 1e-6)).clamp_max(1.0)
            grad = grad * scale.view(-1, 1, 1, 1)

        gamma = self._gamma_t(t, grad_norm)
        latents_new = latents - gamma * grad

        self.prev_latents = latents.detach()
        self.prev_t = t
        return latents_new

# ---- diffusers callback wrapper ----

def make_diffusers_callback(coupler: DPPCoupler):
    def _callback(pipe, step_index: int, timestep: int, callback_kwargs: Dict[str, Any]):
        num_inference_steps = getattr(pipe, "num_timesteps", None) or getattr(pipe.scheduler, "num_inference_steps", None)
        if num_inference_steps is None:
            t = (step_index + 1) / max(1, getattr(pipe, "_num_timesteps", 100))
        else:
            t = (step_index + 1) / float(num_inference_steps)
        latents = callback_kwargs.get("latents", None)
        if latents is None:
            return callback_kwargs
        new_latents = coupler(i=step_index, t=float(t), latents=latents.to(coupler.device), callback_kwargs=callback_kwargs)
        callback_kwargs["latents"] = new_latents
        return callback_kwargs
    return _callback
