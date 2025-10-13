from typing import Tuple
import math
import torch
import torch.nn.functional as F
from typing import Tuple, Union, Optional

def make_t_steps(n:int, t0:float=1.0, t1:float=0.0) -> torch.Tensor:
    return torch.linspace(t0, t1, n+1)


def _parse_gate(t_gate: Optional[Union[str, Tuple[float, float]]]) -> Tuple[float, float]:
    """Robustly parse t_gate into an ordered (t1, t2) float tuple."""
    if t_gate is None:
        return 0.0, 1.0
    if isinstance(t_gate, str):
        parts = t_gate.split(',')
        if len(parts) != 2:
            raise ValueError(f"t_gate string must be 'a,b', got: {t_gate!r}")
        t1, t2 = float(parts[0]), float(parts[1])
    else:
        # tuple/list/np array -> 2 floats
        try:
            t1, t2 = t_gate  # type: ignore
            t1, t2 = float(t1), float(t2)
        except Exception as e:
            raise ValueError(f"t_gate must be (float,float) or 'a,b'; got {t_gate!r}") from e
    if t2 < t1:
        t1, t2 = t2, t1
    return t1, t2

def sched_factor(t: Union[float, int],
                 t_gate: Optional[Union[str, Tuple[float, float]]],
                 mode: str = "sin2") -> float:
    """Return a scalar schedule factor in [0,1] given time t∈[0,1]."""
    t = float(t)
    t1, t2 = _parse_gate(t_gate)
    if not (t1 <= t <= t2):
        return 0.0
    denom = (t2 - t1)
    u = (t - t1) / (denom if abs(denom) > 1e-8 else 1e-8)
    if mode == "sin2":
        return float(math.sin(math.pi * u) ** 2)
    elif mode == "t1mt":
        return float(u * (1.0 - u))
    else:
        return 1.0


def project_partial_orth(g: torch.Tensor, v: torch.Tensor, lam: float, eps: float=1e-12) -> torch.Tensor:
    v_flat = v.view(v.size(0), -1)
    g_flat = g.view(g.size(0), -1)
    v2 = (v_flat*v_flat).sum(dim=1, keepdim=True) + eps
    coeff = ((g_flat*v_flat).sum(dim=1, keepdim=True) / v2)
    proj = coeff * v_flat
    g_orth = g_flat - lam * proj
    return g_orth.view_as(g)


def batched_norm(x: torch.Tensor, eps=1e-12) -> torch.Tensor:
    return x.flatten(1).norm(dim=1, keepdim=True).clamp_min(eps)


def pairwise_cosine_angles(Z: torch.Tensor) -> torch.Tensor:
    S = Z @ Z.t()
    S = S - torch.eye(S.size(0), device=S.device)
    S = S.clamp(-1, 1)
    angles = torch.acos(S) * 180.0 / math.pi
    return angles