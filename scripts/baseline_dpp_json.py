# -*- coding: utf-8 -*-

from DPP.dpp_coupler_mgpu import DPPCouplerMGPU, MGPUConfig
from DPP.vision_feat import VisionCfg, build_vision_feature
import os
import sys
import argparse
import random
import re
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple
from collections import OrderedDict

import torch
from PIL import Image

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ------------------------
# Utils
# ------------------------
def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _resolve_model_dir(p: str) -> Path:
    base = Path(p)
    if base.is_file():
        raise RuntimeError(
            f"[DPP] Expected a Diffusers directory, not a single file: {base}")
    if (base / "model_index.json").exists():
        return base
    for cand in base.rglob("model_index.json"):
        return cand.parent
    raise FileNotFoundError(
        f"[DPP] Could not find Diffusers `model_index.json` under: {base}. "
        "Point --model-dir to the directory that contains `model_index.json` (for ModelScope packages, point to the subdirectory that holds the Diffusers weights)."
    )


_slug_pat_space = re.compile(r"\s+")


def _slugify(text: str, maxlen: int = 120) -> str:
    s = text.strip().lower()
    s = _slug_pat_space.sub("_", s)
    s = re.sub(r"[^a-z0-9._-]+", "", s)
    s = re.sub(r"_{2,}", "_", s).strip("._-")
    return s[:maxlen] if maxlen and len(s) > maxlen else s


def _build_root_out(method: str, concept: str) -> Tuple[Path, Path, Path]:
    base = REPO_ROOT / "outputs" / f"{method}_{_slugify(concept)}"
    imgs = base / "imgs"
    eval_dir = base / "eval"
    imgs.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)
    return base, imgs, eval_dir


def _flatten_prompts_from_spec(spec: Dict[str, Any]) -> "OrderedDict[str, List[str]]":
    """
    Parse the new JSON spec format (as provided), e.g.:
      {
        "dog": ["a photo of a dog", "a dog", ...],
        "truck": ["a photo of a truck", "a truck", ...],
        ...
      }
    Returns: OrderedDict{ concept -> [prompts...] } preserving file order.
    """
    if not isinstance(spec, dict):
        raise ValueError(
            "[DPP] Top-level JSON must be an object: {concept: [prompts...]}")

    concept_to_prompts: "OrderedDict[str, List[str]]" = OrderedDict()
    for concept, plist in spec.items():
        if not isinstance(concept, str):
            continue
        if not isinstance(plist, (list, tuple)):
            continue
        # Clean and de-duplicate while preserving order
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
        raise ValueError(
            "[DPP] No valid prompts parsed; ensure the format is {concept: [\"a dog\", ...]}")

    return concept_to_prompts


def _generators_for_K(device: torch.device, base_seed: int, K: int) -> List[torch.Generator]:
    return [torch.Generator(device=device).manual_seed(int(base_seed) + i) for i in range(K)]


# ------------------------
# Argparse
# ------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="DPP grid sampler for DIM/CIM evaluation")
    # Input: choose one
    p.add_argument("--spec", type=str, default=None,
                   help="Path to a JSON file formatted as {concept: [prompts...]}. If provided, overrides --prompt.")
    p.add_argument("--prompt", type=str, default=None,
                   help="Single prompt (used when --spec is not provided).")

    # Grid
    p.add_argument("--K", type=int, default=4,
                   help="Number of images per prompt (K) for DPP-coupled sampling.")
    p.add_argument("--steps", type=int, default=28)
    p.add_argument("--guidances", type=float, nargs="+", default=None,
                   help="List of guidance scales to run.")
    p.add_argument("--guidance", type=float, default=7.0,
                   help="Single guidance scale if --guidances is not set.")
    p.add_argument("--seeds", type=int, nargs="+", default=[1111])

    # Unified resolution
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--height", type=int, default=512)

    # Model & multi-device
    p.add_argument("--device_transformer", type=str, default="cuda:1")
    p.add_argument("--device_vae", type=str, default="cuda:0")
    p.add_argument("--device_clip", type=str, default="cuda:0")
    p.add_argument("--model-dir", type=str,
                   default=os.path.abspath(os.path.join(os.path.dirname(
                       __file__), "..", "models", "stable-diffusion-3.5-medium")),
                   help="Local path to SD3.5 Diffusers model directory.")

    # Precision & memory saving
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--enable_vae_tiling", action="store_true")
    p.add_argument("--enable_xformers", action="store_true")

    # CLIP JIT
    p.add_argument("--openai_clip_jit_path", type=str, required=True,
                   help="Local path to OpenAI CLIP JIT .pt (e.g., ~/.cache/clip/ViT-B-32.pt)")

    # DPP hyperparameters
    p.add_argument("--gamma_max", type=float, default=0.12)
    p.add_argument("--kernel_spread", type=float, default=3.0)
    p.add_argument("--gamma_sched", type=str, default="sqrt",
                   choices=["sqrt", "sin2", "poly"])
    p.add_argument("--clip_grad_norm", type=float, default=5.0)
    p.add_argument("--decode_size", type=int, default=256)
    p.add_argument("--chunk_size", type=int, default=2)

    # Method name (used for output directory naming)
    p.add_argument("--method", type=str, default="dpp")

    return p.parse_args()


# ------------------------
# Build pipeline & DPP coupler
# ------------------------
def build_pipe_cpu(model_dir: str):
    from diffusers import StableDiffusion3Pipeline
    sd_dir = _resolve_model_dir(model_dir)
    pipe = StableDiffusion3Pipeline.from_pretrained(
        sd_dir, torch_dtype=torch.float32, local_files_only=True)
    pipe.set_progress_bar_config(disable=False)
    return pipe.to("cpu")


def build_all(args):
    dev_tr = torch.device(args.device_transformer)
    dev_vae = torch.device(args.device_vae)
    dev_clip = torch.device(args.device_clip)
    dtype = torch.float16 if args.fp16 else torch.float32

    pipe = build_pipe_cpu(args.model_dir)
    # Place modules on requested devices
    if hasattr(pipe, "transformer"):
        pipe.transformer.to(dev_tr,  dtype=dtype)
    if hasattr(pipe, "text_encoder"):
        pipe.text_encoder.to(dev_tr, dtype=dtype)
    if hasattr(pipe, "text_encoder_2"):
        pipe.text_encoder_2.to(dev_tr, dtype=dtype)
    if hasattr(pipe, "text_encoder_3"):
        pipe.text_encoder_3.to(dev_tr, dtype=dtype)
    if hasattr(pipe, "vae"):
        pipe.vae.to(dev_vae,        dtype=dtype)

    if args.enable_vae_tiling and hasattr(pipe, "vae"):
        if hasattr(pipe.vae, "enable_slicing"):
            pipe.vae.enable_slicing()
        if hasattr(pipe.vae, "enable_tiling"):
            pipe.vae.enable_tiling()

    if args.enable_xformers:
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception as e:
            print(f"[DPP] enable_xformers failed: {e}")

    # JIT CLIP
    vcfg = VisionCfg(backend="openai_clip_jit",
                     jit_path=args.openai_clip_jit_path, device=str(dev_clip))
    feat = build_vision_feature(vcfg)

    # DPP coupler
    mcfg = MGPUConfig(
        decode_size=args.decode_size,
        kernel_spread=args.kernel_spread,
        gamma_max=args.gamma_max,
        gamma_sched=args.gamma_sched,
        clip_grad_norm=args.clip_grad_norm,
        chunk_size=args.chunk_size,
        use_quality_term=False,
    )
    coupler = DPPCouplerMGPU(pipe, feat, dev_tr=dev_tr,
                             dev_vae=dev_vae, dev_clip=dev_clip, cfg=mcfg)
    return pipe, coupler, dev_tr


# ------------------------
# Sampling (one prompt × one guidance × one seed)
# ------------------------
def run_one(pipe, coupler, device_tr: torch.device, prompt: str, K: int, steps: int,
            guidance: float, seed: int, target_wh: Tuple[int, int], out_dir: Path):
    """Run one DPP-coupled sampling for a single (prompt, guidance, seed) producing K images and save to out_dir."""
    W, H = int(target_wh[0]), int(target_wh[1])

    # DPP update callback: on the final step, move latents to VAE device and dtype
    def dpp_callback(ppl, i, t, kw: Dict[str, Any]):
        num_steps = getattr(
            ppl.scheduler, "num_inference_steps", None) or steps
        t_norm = (i + 1) / float(num_steps)
        lat = kw.get("latents")
        if lat is None:
            return kw
        lat_new = coupler(step_index=i, t_norm=float(t_norm), latents=lat)
        if (i + 1) == num_steps:
            vae = ppl.vae
            vae_dtype = next(vae.parameters()).dtype
            lat_new = lat_new.to(
                device=vae.device, dtype=vae_dtype, non_blocking=True)
        kw["latents"] = lat_new
        return kw

    # K independent generators (offset from a shared base seed)
    gens = _generators_for_K(device_tr, seed, K)

    prompts = [prompt] * K
    kwargs = dict(
        prompt=prompts,
        num_inference_steps=steps,
        guidance_scale=guidance,
        callback_on_step_end=dpp_callback,
        callback_on_step_end_tensor_inputs=["latents"],
        generator=gens,
        output_type="pil",
    )

    # Prefer pipeline-native resizing; if unsupported, fall back to resize-on-save
    try:
        result = pipe(height=H, width=W, **kwargs)
    except TypeError:
        result = pipe(**kwargs)

    imgs = result.images if hasattr(result, "images") else result
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, img in enumerate(imgs):
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)
        # Ensure target size
        if img.size != (W, H):
            img = img.resize((W, H), resample=Image.BICUBIC)
        img.save(out_dir / f"{i:02d}.png")


# ------------------------
# Main
# ------------------------
def main():
    args = parse_args()

    # Parse concept -> prompts mapping (new format); if --spec is not used, --prompt can be provided
    if args.spec:
        with open(args.spec, "r", encoding="utf-8") as fp:
            spec = json.load(fp)
        concept_to_prompts = _flatten_prompts_from_spec(spec)  # OrderedDict
    elif args.prompt:
        concept_to_prompts = OrderedDict([("single", [args.prompt])])
    else:
        raise ValueError(
            "Provide either --spec (path to JSON) or --prompt (single prompt).")

    # Guidance list
    guidances = args.guidances if args.guidances is not None else [
        args.guidance]
    guidances = [float(g) for g in guidances]

    # Build pipeline/DPP once and reuse
    pipe, coupler, dev_tr = build_all(args)
    dtype = torch.float16 if args.fp16 else torch.float32
    print(
        f"[DPP] Ready. dtype={dtype}, K={args.K}, steps={args.steps}, W×H={args.width}×{args.height}")

    # Save outputs under outputs/{method}_{concept}
    for concept, prompts in concept_to_prompts.items():
        _, imgs_root, eval_dir = _build_root_out(args.method, concept)
        print(f"[DPP] Outputs base: {imgs_root.parent}")
        print(f"[DPP] Eval dir:     {eval_dir}")

        for ptext in prompts:
            p_slug = _slugify(ptext)  # folder name base derived from prompt
            for g in guidances:
                for s in args.seeds:
                    subdir = imgs_root / f"{p_slug}_seed{s}_g{g}_s{args.steps}"
                    print(
                        f"[DPP] Sampling: concept='{concept}' | prompt='{ptext}' | seed={s} | guidance={g} | steps={args.steps} -> {subdir}")
                    run_one(pipe, coupler, dev_tr,
                            prompt=ptext, K=args.K, steps=args.steps,
                            guidance=float(g), seed=int(s),
                            target_wh=(args.width, args.height),
                            out_dir=subdir)

    print("[DPP] Done.")


if __name__ == "__main__":
    main()
