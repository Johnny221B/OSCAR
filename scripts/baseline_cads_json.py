# -*- coding: utf-8 -*-
"""
Grid evaluation for CADS on SD-3.5 (Flow-Matching).

- Inputs:
  * A JSON file with multiple concepts:
      {
        "dog": ["a photo of a dog", "a dog", ...],
        "truck": ["a photo of a truck", "a truck", ...],
        ...
      }
    (top-level keys are concepts; values are lists of prompts)
  * Or a single --prompt string (back-compat).

- Outputs:
  outputs/{method}_{concept}/
    ├── eval/             # reserved for metrics
    └── imgs/
        └── {prompt_slug}_seed{SEED}_g{GUIDANCE}_s{STEPS}/
            00.png, 01.png, ...

Example:
  python -u scripts/cads_eval_grid.py \
    --model "./models/stable-diffusion-3.5-medium" \
    --spec ./specs/prompt.json \
    --negative "" \
    --G 16 --steps 10 \
    --guidances 3.0 7.5 12.0 \
    --seeds 1111 2222 3333 4444 \
    --height 512 --width 512 \
    --dtype fp16 \
    --device-transformer cuda:0 --device-vae cuda:1 --device-clip cuda:0 \
    --method cads
"""

from __future__ import annotations

import argparse
import json
import sys
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple
from collections import OrderedDict

import torch
from PIL import Image
from diffusers import StableDiffusion3Pipeline, DiffusionPipeline

# Repo root on sys.path
_THIS = Path(__file__).resolve()
_REPO = _THIS.parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

# Try CADS imports (lower/upper/fallback)
try:
    from cads import CADS
except Exception:
    try:
        from CADS import CADS
    except Exception:
        from cads.cads import CADSConditionAnnealer as CADS


# ---------------------------
# Utilities
# ---------------------------
def _log(msg: str):
    print(msg, flush=True)

def slugify(text: str, maxlen: int = 120) -> str:
    s = re.sub(r"\s+", "_", text.strip())
    s = re.sub(r"[^A-Za-z0-9._-]+", "", s)
    s = re.sub(r"_{2,}", "_", s).strip("._-")
    return s[:maxlen] if maxlen and len(s) > maxlen else s

def _parse_concepts_spec(spec_obj: Dict[str, Any]) -> "OrderedDict[str, List[str]]":
    """
    Strictly parse the multi-concept JSON:
      { "dog": ["a dog", ...], "truck": ["a truck", ...], ... }
    Returns an OrderedDict in file order; each prompt list is de-duped (stable).
    """
    if not isinstance(spec_obj, dict):
        raise ValueError("[CADS] Spec must be a JSON object mapping concept -> list[str].")

    concept_to_prompts: "OrderedDict[str, List[str]]" = OrderedDict()
    for concept, plist in spec_obj.items():
        if not isinstance(concept, str) or not isinstance(plist, (list, tuple)):
            continue
        seen, cleaned = set(), []
        for p in plist:
            if isinstance(p, str):
                s = p.strip()
                if s and s not in seen:
                    seen.add(s)
                    cleaned.append(s)
        if cleaned:
            concept_to_prompts[concept] = cleaned

    if not concept_to_prompts:
        raise ValueError("[CADS] No valid {concept: [prompts...]} found in spec.")
    return concept_to_prompts

def _outputs_root(method: str, concept: str) -> Tuple[Path, Path, Path]:
    """
    Create:
      outputs/{method}_{concept}/eval
      outputs/{method}_{concept}/imgs
    """
    base = _REPO / "outputs" / f"{method}_{slugify(concept)}"
    eval_dir = base / "eval"
    imgs_dir = base / "imgs"
    eval_dir.mkdir(parents=True, exist_ok=True)
    imgs_dir.mkdir(parents=True, exist_ok=True)
    return base, eval_dir, imgs_dir

def _prompt_run_dir(imgs_root: Path, prompt: str, seed: int, guidance: float, steps: int) -> Path:
    pslug = slugify(prompt)
    return imgs_root / f"{pslug}_seed{seed}_g{guidance}_s{steps}"

def _save_images(imgs: List[Image.Image], out_dir: Path, wh: Tuple[int, int]):
    W, H = int(wh[0]), int(wh[1])
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, img in enumerate(imgs):
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)
        if img.size != (W, H):
            img = img.resize((W, H), resample=Image.BICUBIC)
        img.save(out_dir / f"{i:02d}.png")

def _select_dtype(name: str):
    if name == "fp16": return torch.float16
    if name == "bf16": return torch.bfloat16
    return torch.float32

def _move_if_exists(mod, device: str | None):
    if mod is not None and device is not None:
        mod.to(torch.device(device))

def _generators_for_K(exec_device: str, base_seed: int, K: int) -> List[torch.Generator]:
    return [torch.Generator(device=torch.device(exec_device)).manual_seed(int(base_seed) + i) for i in range(K)]

# ---- Device-mismatch fix (ensure VAE decode gets latents on the same device/dtype) ----
def _final_latents_to_vae(ppl, latents: torch.Tensor) -> torch.Tensor:
    vae = getattr(ppl, "vae", None)
    if vae is None or latents is None:
        return latents
    try:
        vae_dtype = next(vae.parameters()).dtype
    except StopIteration:
        vae_dtype = latents.dtype
    return latents.to(device=vae.device, dtype=vae_dtype, non_blocking=True)

def _wrap_cads_callback(cads_obj, total_steps: int):
    """
    Wrap CADS __call__ so that at the *last* step we move latents to VAE's device/dtype.
    Matches diffusers' callback_on_step_end: (pipeline, step_index, timestep, kwargs) -> dict(kwargs)
    """
    total_steps = int(total_steps)
    def _cb(ppl, i, t, kw: Dict[str, Any]):
        out = cads_obj(ppl, i, t, kw)
        if not isinstance(out, dict):
            out = kw
        if (i + 1) == total_steps and out.get("latents", None) is not None:
            out["latents"] = _final_latents_to_vae(ppl, out["latents"])
        return out
    return _cb


# ---------------------------
# Robust SD3.5 loader
# ---------------------------
def _resolve_model_dir(root: Path) -> Path:
    # exact dir
    if (root / "model_index.json").exists():
        return root
    if root.exists() and root.is_dir():
        # HF cache snapshots
        snaps = root / "snapshots"
        if snaps.is_dir():
            for d1 in snaps.iterdir():
                if (d1 / "model_index.json").exists():
                    return d1
                if d1.is_dir():
                    for d2 in d1.iterdir():
                        if (d2 / "model_index.json").exists():
                            return d2
        # generic search up to two levels deep
        for d1 in root.iterdir():
            if (d1 / "model_index.json").exists():
                return d1
            if d1.is_dir():
                for d2 in d1.iterdir():
                    if (d2 / "model_index.json").exists():
                        return d2
    raise FileNotFoundError(
        f"[CADS] model_index.json not found under '{root}'. "
        f"Point --model to a diffusers dir (or HF snapshot subdir). "
        f"For single-file .safetensors/.ckpt, pass that file directly."
    )

def load_sd35(model_path: str, torch_dtype):
    p = Path(model_path)
    if not p.exists():
        raise FileNotFoundError(f"[CADS] model path not found: {p}")
    # single file
    if p.is_file() and p.suffix.lower() in {".safetensors", ".ckpt"}:
        _log(f"[CADS] Loading single-file checkpoint: {p}")
        pipe = DiffusionPipeline.from_single_file(p.as_posix(), torch_dtype=torch_dtype)
        if not isinstance(pipe, StableDiffusion3Pipeline):
            pipe = StableDiffusion3Pipeline(**pipe.components)
        return pipe
    # directory
    idx_dir = _resolve_model_dir(p)
    _log(f"[CADS] Using diffusers folder: {idx_dir}")
    return StableDiffusion3Pipeline.from_pretrained(
        idx_dir.as_posix(), torch_dtype=torch_dtype, local_files_only=True
    )


# ---------------------------
# Args
# ---------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="CADS grid evaluation on SD-3.5 (Flow-Matching)")

    # Model / output
    p.add_argument("--model", type=str, default="models/stable-diffusion-3.5-medium")
    p.add_argument("--method", type=str, default="cads", help="outputs/{method}_{concept}")
    p.add_argument("--out", type=str, default="outputs", help="(unused for structure; kept for compatibility)")

    # Prompts (multi-concept spec or single prompt)
    p.add_argument("--spec", type=str, default=None,
                   help="Path to JSON: {concept: [prompts...]} (overrides --prompt)")
    p.add_argument("--prompt", type=str, default=None, help="Fallback if --spec not provided")
    p.add_argument("--negative", type=str, default="low quality, blurry")

    # Grid
    p.add_argument("--G", type=int, default=4, help="images per prompt (group size)")
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--guidances", type=float, nargs="+", default=None, help="e.g., 3.0 7.5 12.0")
    p.add_argument("--guidance", type=float, default=7.5, help="used if --guidances omitted")
    p.add_argument("--seeds", type=int, nargs="+", default=[1111, 2222, 3333, 4444])

    # Resolution
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--width", type=int, default=1024)

    # Precision
    p.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "fp32", "bf16"])

    # Devices (optional fine placement)
    p.add_argument("--device-transformer", type=str, default=None)
    p.add_argument("--device-vae", type=str, default=None)
    p.add_argument("--device-clip", type=str, default=None, help="alias for text encoders")
    p.add_argument("--device-text1", type=str, default=None)
    p.add_argument("--device-text2", type=str, default=None)
    p.add_argument("--device-text3", type=str, default=None)

    # CADS hyperparams
    g = p.add_argument_group("CADS")
    g.add_argument("--tau1", type=float, default=0.4)
    g.add_argument("--tau2", type=float, default=0.9)
    g.add_argument("--cads-s", type=float, default=0.10, dest="cads_s")
    g.add_argument("--psi", type=float, default=1.0)
    g.add_argument("--no-rescale", action="store_true")
    g.add_argument("--dynamic-cfg", action="store_true")

    # Debug
    p.add_argument("--debug", action="store_true")

    return p


# ---------------------------
# Main
# ---------------------------
def main():
    args = build_parser().parse_args()
    torch_dtype = _select_dtype(args.dtype)

    # Parse prompts source (preserve file order)
    if args.spec:
        with open(args.spec, "r", encoding="utf-8") as fp:
            spec = json.load(fp, object_pairs_hook=OrderedDict)
        concept_to_prompts = _parse_concepts_spec(spec)  # OrderedDict[str, List[str]]
    elif args.prompt:
        concept_to_prompts = OrderedDict([("single", [args.prompt])])
    else:
        raise ValueError("Provide either --spec (JSON file) or --prompt")

    guidances = args.guidances if args.guidances is not None else [args.guidance]
    guidances = [float(g) for g in guidances]

    # ---- Build pipeline once (then move modules as requested) ----
    _log(f"[CADS] Resolving model from: {args.model}")
    pipe = load_sd35(args.model, torch_dtype=torch_dtype)

    # Optional device placement
    _move_if_exists(getattr(pipe, "transformer", None), args.device_transformer)
    _move_if_exists(getattr(pipe, "vae", None), args.device_vae)

    te_dev1 = args.device_text1 or args.device_clip
    te_dev2 = args.device_text2 or args.device_clip
    te_dev3 = args.device_text3 or args.device_clip
    _move_if_exists(getattr(pipe, "text_encoder", None), te_dev1)
    _move_if_exists(getattr(pipe, "text_encoder_2", None), te_dev2)
    _move_if_exists(getattr(pipe, "text_encoder_3", None), te_dev3)

    # If none specified, send entire pipe to a default device
    if all(x is None for x in [args.device_transformer, args.device_vae, te_dev1, te_dev2, te_dev3]):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pipe = pipe.to(device)

    if args.debug:
        # quick device print
        def dev_of(m):
            try:
                for prm in m.parameters():
                    return prm.device
            except Exception:
                return None
        parts = {
            "transformer": getattr(pipe, "transformer", None),
            "vae": getattr(pipe, "vae", None),
            "text_encoder": getattr(pipe, "text_encoder", None),
            "text_encoder_2": getattr(pipe, "text_encoder_2", None),
            "text_encoder_3": getattr(pipe, "text_encoder_3", None),
        }
        for k, m in parts.items():
            if m is not None:
                _log(f"- {k}: {dev_of(m)}")

    # Try get execution device string for Generators
    try:
        exec_dev = str(pipe._execution_device)
    except Exception:
        exec_dev = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- Iterate by concept to write into outputs/{method}_{concept}/... ----
    for concept, prompts in concept_to_prompts.items():
        base_dir, eval_dir, imgs_root = _outputs_root(args.method, concept)
        _log(f"[CADS] outputs base: {base_dir}")
        _log(f"[CADS] eval dir:     {eval_dir}")
        _log(f"[CADS] imgs root:    {imgs_root}")

        # Grid: prompt × guidance × seed
        for ptxt in prompts:
            for g in guidances:
                for sd in args.seeds:
                    # CADS instance per (guidance, seed) for reproducibility
                    cads = CADS(
                        num_inference_steps=int(args.steps),
                        tau1=float(args.tau1), tau2=float(args.tau2),
                        s=float(args.cads_s), psi=float(args.psi),
                        rescale=(not args.no_rescale),
                        dynamic_cfg=bool(args.dynamic_cfg),
                        seed=int(sd),
                    )

                    gens = _generators_for_K(exec_dev, int(sd), int(args.G))
                    cb = _wrap_cads_callback(cads, int(args.steps))

                    run_dir = _prompt_run_dir(
                        imgs_root=imgs_root,
                        prompt=ptxt, seed=int(sd),
                        guidance=float(g), steps=int(args.steps)
                    )
                    _log(f"[CADS] sampling: concept='{concept}' | prompt='{ptxt}' | seed={sd} | guidance={g} | steps={args.steps} -> {run_dir}")

                    # Try passing height/width; if unsupported, call without and resize on save
                    try:
                        result = pipe(
                            prompt=ptxt,
                            negative_prompt=(args.negative if args.negative else None),
                            height=int(args.height), width=int(args.width),
                            num_images_per_prompt=int(args.G),
                            num_inference_steps=int(args.steps),
                            guidance_scale=float(g),
                            generator=gens,
                            callback_on_step_end=cb,
                            callback_on_step_end_tensor_inputs=["latents"],
                            output_type="pil",
                        )
                    except TypeError:
                        result = pipe(
                            prompt=ptxt,
                            negative_prompt=(args.negative if args.negative else None),
                            num_images_per_prompt=int(args.G),
                            num_inference_steps=int(args.steps),
                            guidance_scale=float(g),
                            generator=gens,
                            callback_on_step_end=cb,
                            callback_on_step_end_tensor_inputs=["latents"],
                            output_type="pil",
                        )

                    images = result.images if hasattr(result, "images") else result
                    _save_images(images, run_dir, (args.width, args.height))

    _log("[CADS] Done.")


if __name__ == "__main__":
    main()