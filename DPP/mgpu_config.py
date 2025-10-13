# -*- coding: utf-8 -*-
from dataclasses import dataclass


@dataclass
class MGPUConfig:
    decode_size: int = 256           # Decode up to this size before sending to CLIP
    kernel_spread: float = 3.0       # DPP kernel width (RBF scale)
    # Global step scale (normalized by mean grad-norm)
    gamma_max: float = 0.12
    gamma_sched: str = "sqrt"        # "sqrt" | "sin2" | "poly"
    clip_grad_norm: float = 5.0
    chunk_size: int = 2              # Backprop per chunk
    use_quality_term: bool = False   # Keep pure DPP baseline (no quality term)


__all__ = ["MGPUConfig"]
