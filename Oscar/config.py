from dataclasses import dataclass
from typing import Optional, Tuple
import torch

# source flowqd/bin/activate

@dataclass
class DiversityConfig:
    num_steps: int = 30 # flow steps
    t_start: float = 1.0 # time goes from 1 -> 0
    t_end: float = 0.0
    tau: float = 1.0 # logdet(I + τK)
    eps_logdet: float = 1e-3 # numerical stability
    feature_center: bool = True # center features (translation invariance)
    feature_l2norm: bool = True # L2 normalize features (angle-focused)
    whiten: bool = True # optional feature whitening
    whiten_min_B: int = 16
    gamma0: float = 0.15 # base diversity strength
    gamma_max_ratio: float = 0.3 # trust region cap: ||γ g|| ≤ ratio * ||v||
    partial_ortho: float = 0.5 # partial orthogonal coefficient λ∈[0,1]
    t_gate: Tuple[float,float] = (0.2, 0.9) # only apply diversity in this t range
    sched_shape: str = "sin2" # 'sin2' or 't1mt'
    update_every: int = 1 # recompute K/grad every m steps
    clip_image_size: int = 224
    angle_gate_deg: Optional[float] = None # None disables angle gating
    leverage_alpha: float = 0.5 # >0 enables leverage weights
    device: Optional[torch.device] = None
    noise_beta0: float = 0.0                     
    noise_use_same_gate: bool = True             
    noise_t_gate: Tuple[float,float] = (0.0, 0.7)
