from typing import Tuple
import math
import torch
import torch.nn.functional as F
import torch.nn as nn
import os
import re
import time
from typing import Tuple, Union, Optional
from collections import OrderedDict

# ---- logging / mem ----

def _gb(x): 
    return f"{x/1024**3:.2f} GB"

def log(s: str, debug: bool = True):
    ts = time.strftime("%H:%M:%S")
    if debug:
        print(f"[{ts}] {s}", flush=True)

def print_mem_all(tag: str, devices: list):
    lines = [f"[{tag}]"]
    for d in devices:
        if d.type == 'cuda':
            free, total = torch.cuda.mem_get_info(d)
            used = total - free
            lines.append(f"  {d}: used={_gb(used)}, free={_gb(free)}")
        else:
            lines.append(f"  {d}: CPU")
    print("\n".join(lines), flush=True)

# ---- model/device introspection ----

def _first_device_of_module(m: nn.Module):
    if not isinstance(m, nn.Module):
        return None
    for p in m.parameters(recurse=False):
        return p.device
    for b in m.buffers(recurse=False):
        return b.device
    for sm in m.children():
        for p in sm.parameters(recurse=False):
            return p.device
        for b in sm.buffers(recurse=False):
            return b.device
    return None

def inspect_pipe_devices(pipe):
    names = [
        "transformer",
        "text_encoder", "text_encoder_2", "text_encoder_3",
        "vae",
        "tokenizer", "tokenizer_2", "tokenizer_3", "scheduler",
    ]
    report = {}
    for name in names:
        if not hasattr(pipe, name):
            continue
        obj = getattr(pipe, name)
        if obj is None:
            report[name] = "None"
            continue
        if isinstance(obj, nn.Module):
            dev = _first_device_of_module(obj)
            report[name] = str(dev) if dev is not None else "module(no params)"
        else:
            report[name] = "non-module"
    print("[pipe-devices]", report, flush=True)

def assert_on(m, want):
    if not isinstance(m, nn.Module):
        return
    for p in m.parameters():
        if str(p.device) != str(want):
            raise RuntimeError(f"Param on {p.device}, expected {want}")
    for b in m.buffers():
        if str(b.device) != str(want):
            raise RuntimeError(f"Buffer on {b.device}, expected {want}")

# ---- path & text utils ----

def slugify(text: str, maxlen: int = 120) -> str:
    s = re.sub(r'\s+', '_', text.strip())
    s = re.sub(r'[^A-Za-z0-9._-]+', '', s)
    s = re.sub(r'_{2,}', '_', s).strip('._-')
    return s[:maxlen] if maxlen and len(s) > maxlen else s

def resolve_model_dir(path: str) -> str:
    p = os.path.abspath(path)
    if os.path.isfile(os.path.join(p, 'model_index.json')):
        return p
    for root, _, files in os.walk(p):
        if 'model_index.json' in files:
            return root
    raise FileNotFoundError(f'Could not find model_index.json under {path}')

def parse_concepts_spec(obj: dict) -> "OrderedDict[str, list]":
    if not isinstance(obj, dict):
        raise ValueError("Spec must be a JSON object: {concept: [prompts...]}")
    out = OrderedDict()
    for concept, plist in obj.items():
        if not isinstance(concept, str) or not isinstance(plist, (list, tuple)):
            continue
        seen, cleaned = set(), []
        for p in plist:
            if isinstance(p, str):
                s = p.strip()
                if s and s not in seen:
                    seen.add(s); cleaned.append(s)
        if cleaned:
            out[concept] = cleaned
    if not out:
        raise ValueError("No valid {concept: [prompts...]} found in spec.")
    return out

def build_root_out(project_root: str, method: str, concept: str):
    base = os.path.join(project_root, "outputs", f"{method}_{slugify(concept)}")
    imgs = os.path.join(base, "imgs")
    eval_dir = os.path.join(base, "eval")
    os.makedirs(imgs, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)
    return base, imgs, eval_dir

def prompt_run_dir(imgs_root: str, prompt: str, seed: int, guidance: float, steps: int) -> str:
    pslug = slugify(prompt)
    return os.path.join(imgs_root, f"{pslug}_seed{seed}_g{guidance}_s{steps}")


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