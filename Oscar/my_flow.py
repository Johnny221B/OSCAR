# ============================
# FILE: diverse_flow/my_flow.py
# ============================
from typing import Optional, Dict, Any
import importlib
import os
import torch
from safetensors.torch import load_file as load_safetensors
from .base_flow import BaseFlow


def _load_state_dict_any(path: str) -> Dict[str, torch.Tensor]:
    ext = os.path.splitext(path)[1].lower()
    if ext in ['.safetensors', '.sft']:
        return load_safetensors(path)
    state = torch.load(path, map_location='cpu')
    # common wrappers
    for key in ['state_dict', 'model', 'net', 'ema_state_dict']:
        if isinstance(state, dict) and key in state and isinstance(state[key], dict):
            return state[key]
    if isinstance(state, dict):
        return state
    raise ValueError(f"Unsupported checkpoint structure: {type(state)} from {path}")


def _instantiate(class_path: str, **kwargs):
    """Instantiate a class from 'pkg.module:ClassName' string."""
    if ':' not in class_path:
        raise ValueError("class_path must be 'pkg.mod:ClassName'")
    mod_name, cls_name = class_path.split(':', 1)
    mod = importlib.import_module(mod_name)
    cls = getattr(mod, cls_name)
    return cls(**kwargs)


class MyFlowFromModule(BaseFlow):
    """
    Generic wrapper to load your local velocity network and expose a .velocity(x,t,cond)->v API.

    Args:
        class_path: 'pkg.mod:ClassName' of your model (must be importable locally)
        ckpt_path:  path to local weights (.pt/.ckpt/.safetensors)
        ctor_kwargs: dict of constructor kwargs for your ClassName
        device: torch.device

    Your model is expected to implement forward(x, t, **cond)->v OR forward(x, t)->v.
    If your model expects scaled timesteps (e.g., 0..999), override self.t_map.
    """
    def __init__(self, class_path: str, ckpt_path: str, ctor_kwargs: Optional[Dict[str, Any]] = None, device: Optional[torch.device] = None):
        super().__init__()
        self.net = _instantiate(class_path, **(ctor_kwargs or {}))
        sd = _load_state_dict_any(ckpt_path)
        missing, unexpected = self.net.load_state_dict(sd, strict=False)
        if len(missing)+len(unexpected) > 0:
            print(f"[MyFlow] load_state_dict: missing={len(missing)}, unexpected={len(unexpected)}")
        self.net.eval()
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net.to(self.device)

        # default maps t in [1,0] to itself (continuous); override if your model expects int steps.
        self.t_map = lambda t: t

    @torch.no_grad()
    def velocity(self, x: torch.Tensor, t_scalar: float, cond: Optional[Dict]=None) -> torch.Tensor:
        t_in = self.t_map(t_scalar)
        if cond is None:
            try:
                return self.net(x, t_in)
            except TypeError:
                return self.net(x, t_in, {})
        else:
            return self.net(x, t_in, **cond) if isinstance(cond, dict) else self.net(x, t_in, cond)

    # Helper to use discrete timesteps; call e.g. myflow.use_linear_timesteps(1000) after init.
    def use_linear_timesteps(self, n_steps:int=1000):
        self.t_map = lambda t: int(round((1.0 - max(0.0, min(1.0, t))) * (n_steps-1)))
        return self

