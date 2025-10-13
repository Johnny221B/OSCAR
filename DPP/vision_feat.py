# =============================
# File: diverse_flow/DPP/vision_feat.py
# =============================

from dataclasses import dataclass
from typing import Optional, Tuple
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class VisionCfg:
    backend: str = "openai_clip_jit"  # only JIT is supported to stay fully offline
    jit_path: Optional[str] = None     # path to local OpenAI CLIP JIT .pt (e.g., ~/.cache/clip/ViT-B-32.pt)
    device: str = "cuda"
    image_size: Optional[int] = None   # optional manual override; otherwise read from model


def _resize_bilinear(x: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
    return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


class OpenAIJITVision(nn.Module):
    """OpenAI CLIP JIT (.pt) image tower — strictly local, differentiable.
    This module must keep autograd enabled so gradients flow back to latents via VAE.
    """
    def __init__(self, jit_path: str, device: str = "cuda"):
        super().__init__()
        p = Path(jit_path).expanduser()
        if not p.exists():
            raise FileNotFoundError(f"[DPP][OpenAI-JIT] 本地 JIT 权重不存在: {p}")
        core = torch.jit.load(str(p), map_location="cpu").eval()
        self.core = core.to(device)
        self.device = torch.device(device)

        # Resolution: read from model; fallback to 224 or user override
        inp_res = getattr(getattr(core, "visual", None), "input_resolution", 224)
        self.res = int(inp_res) if isinstance(inp_res, (int, float)) else 224

        # Standard OpenAI CLIP normalization
        mean = [0.48145466, 0.4578275, 0.40821073]
        std  = [0.26862954, 0.26130258, 0.27577711]
        self.register_buffer("mean", torch.tensor(mean).view(1,3,1,1))
        self.register_buffer("std",  torch.tensor(std).view(1,3,1,1))

    def forward(self, imgs_0_1: torch.Tensor) -> torch.Tensor:
        # Keep grads! No inference_mode / no_grad here.
        x = _resize_bilinear(imgs_0_1, (self.res, self.res))
        # align dtype/device for normalization tensors
        mean = self.mean.to(device=x.device, dtype=x.dtype)
        std  = self.std.to(device=x.device, dtype=x.dtype)
        x = (x - mean) / std
        feats = self.core.encode_image(x)
        return F.normalize(feats.float(), dim=-1)


def build_vision_feature(cfg: VisionCfg) -> nn.Module:
    if cfg.backend != "openai_clip_jit":
        raise ValueError("仅支持 backend='openai_clip_jit'（本地 OpenAI CLIP JIT .pt）。")
    if not cfg.jit_path:
        raise ValueError("backend='openai_clip_jit' 需要提供 jit_path (本地 .pt)。")
    return OpenAIJITVision(cfg.jit_path, device=cfg.device)
