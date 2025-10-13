# -*- coding: utf-8 -*-
"""
Grid evaluation for Particle Guidance (PG) on SD3.5 (Flow Matching).

- Inputs:
  * New JSON format: { "dog": [...], "truck": [...], ... }
    (top-level keys are concepts; each value is a list of prompts)
  * Or a single --prompt string (back-compat).

- Outputs:
  outputs/{method}_{concept}/
    ├── eval/             # reserved for metrics
    └── imgs/
        └── {prompt_slug}_seed{SEED}_g{GUIDANCE}_s{STEPS}/
            00.png, 01.png, ...
"""

from __future__ import annotations
import argparse
import gc
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple
from collections import OrderedDict

import torch
from PIL import Image

# Repo root for imports
REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

# PG backend (your code)
from Particle_Guidance.flow_backend import FlowSamplerConfig, FlowSampler
from Particle_Guidance.particle_guidance import PGConfig
from Particle_Guidance.utils import seed_everything, slugify


# ---------------------------
# Utilities
# ---------------------------
def _parse_concepts_spec(spec: Dict[str, Any]) -> "OrderedDict[str, List[str]]":
    if not isinstance(spec, dict):
        raise ValueError("[PG] Spec must be a JSON object: {concept: [prompts...]}")
    concept_to_prompts: "OrderedDict[str, List[str]]" = OrderedDict()
    for concept, plist in spec.items():
        if not isinstance(concept, str) or not isinstance(plist, (list, tuple)):
            continue
        seen, cleaned = set(), []
        for p in plist:
            if isinstance(p, str):
                s = p.strip()
                if s and s not in seen:
                    seen.add(s); cleaned.append(s)
        if cleaned:
            concept_to_prompts[concept] = cleaned
    if not concept_to_prompts:
        raise ValueError("[PG] No valid {concept: [prompts...]} found in spec.")
    return concept_to_prompts


def _outputs_root(method: str, concept: str) -> Tuple[Path, Path, Path]:
    base = Path(REPO_DIR) / "outputs" / f"{method}_{slugify(concept)}"
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


def _cuda_clear():
    """Best-effort to release cached CUDA memory between iterations."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.synchronize()


# ---------------------------
# Args
# ---------------------------
def build_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SD3.5 + Particle Guidance (grid evaluation)")

    # Input spec OR single prompt
    p.add_argument("--spec", type=str, default=None,
                   help="Path to JSON: {concept: [prompts...]} (overrides --prompt)")
    p.add_argument("--prompt", type=str, default=None,
                   help="Single prompt if --spec not provided")
    p.add_argument("--negative", type=str, default=None)

    # Grid
    p.add_argument("--G", type=int, default=4, help="Group size (images per prompt)")
    p.add_argument("--steps", type=int, default=30, help="#flow steps (NFE)")
    p.add_argument("--guidances", type=float, nargs="+", default=None,
                   help="List of cfg scales; e.g., 3.0 7.5 12.0")
    p.add_argument("--cfg", type=float, default=5.0,
                   help="Single cfg if --guidances is omitted")
    p.add_argument("--seeds", type=int, nargs="+", default=[1111, 2222, 3333, 4444])

    # Resolution
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--width", type=int, default=512)

    # Model/device
    p.add_argument("--model", type=str, required=True,
                   help="Path to SD3.5 (Diffusers layout)")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--attention-slicing", action="store_true")

    # PG hyper-params
    p.add_argument("--pg-mode", type=str, choices=["ode", "sde"], default="ode")
    p.add_argument("--pg-alpha-scale", type=float, default=30.0,
                   help="alpha_t = alpha_scale * sigma(t)^2")
    p.add_argument("--pg-sigma-a", type=float, default=0.5,
                   help="sigma(t)=a*sqrt(t/(1-t))")
    p.add_argument("--pg-bandwidth", type=float, default=None,
                   help="Fixed RBF bandwidth; None -> median trick")

    # For directory naming
    p.add_argument("--method", type=str, default="pg",
                   help="Method name for outputs/{method}_{concept}")

    return p.parse_args()


# ---------------------------
# Main
# ---------------------------
def main():
    args = build_args()

    # Parse source of prompts (preserve order in file)
    if args.spec:
        with open(args.spec, "r", encoding="utf-8") as fp:
            spec = json.load(fp, object_pairs_hook=OrderedDict)
        concept_to_prompts = _parse_concepts_spec(spec)
    elif args.prompt:
        concept_to_prompts = OrderedDict([("single", [args.prompt])])
    else:
        raise ValueError("Provide either --spec (JSON file) or --prompt")

    # Helpful echo
    total_prompts = sum(len(v) for v in concept_to_prompts.values())
    print(f"[PG] concepts: {len(concept_to_prompts)} | total prompts: {total_prompts}")
    for k, v in concept_to_prompts.items():
        print(f"[PG]   - {k}: {len(v)} prompts")

    guidances = args.guidances if args.guidances is not None else [args.cfg]
    guidances = [float(g) for g in guidances]

    dtype = torch.float16 if args.fp16 else torch.float32
    device = torch.device(args.device)

    # Static PG config
    pg_cfg = PGConfig(
        group_size=args.G,
        mode=args.pg_mode,
        alpha_scale=args.pg_alpha_scale,
        sigma_a=args.pg_sigma_a,
        bandwidth=args.pg_bandwidth,
    )

    # Iterate by concept to write into outputs/{method}_{concept}/...
    for concept, prompts in concept_to_prompts.items():
        base_dir, eval_dir, imgs_root = _outputs_root(args.method, concept)
        print(f"[PG] outputs base: {base_dir}")
        print(f"[PG] eval dir:     {eval_dir}")
        print(f"[PG] imgs root:    {imgs_root}")

        for g in guidances:
            # Build one sampler per (concept, guidance) and reuse for all prompts & seeds
            fs_cfg = FlowSamplerConfig(
                model_path=args.model,
                device=str(device),
                dtype=dtype,
                steps=args.steps,
                cfg_scale=float(g),
                height=args.height,
                width=args.width,
                enable_attention_slicing=args.attention_slicing,
            )
            sampler = FlowSampler(fs_cfg, pg_cfg)

            try:
                for ptxt in prompts:
                    for sd in args.seeds:
                        seed_everything(int(sd))
                        out_dir = _prompt_run_dir(
                            imgs_root=imgs_root,
                            prompt=ptxt,
                            seed=int(sd),
                            guidance=float(g),
                            steps=int(args.steps),
                        )
                        print(f"[PG] sampling: concept='{concept}' | prompt='{ptxt}' | seed={sd} | cfg={g} | steps={args.steps} -> {out_dir}")

                        with torch.inference_mode():
                            images = sampler.generate(
                                prompt=ptxt,
                                negative_prompt=args.negative,
                                num_images=args.G,
                                seed=int(sd),
                            )
                        _save_images(images, out_dir, (args.width, args.height))

                        # Explicitly release references & cached memory between seeds
                        del images
                        _cuda_clear()

            finally:
                # Release model/sampler for this guidance before moving to next
                del sampler
                _cuda_clear()

    print("[PG] Done.")


if __name__ == "__main__":
    main()