#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Image Evaluation Tool (method_concept, CSV-only)

Expected layout:
  /.../outputs/<method>_<concept>/imgs

The tool auto-infers method and concept, and writes a CSV to:
  /.../outputs/<method>_<concept>/eval/<method>_<concept>.csv

It scans one level of subfolders under `imgs` (each subfolder is a generated K-set)
whose names should look like:
  <prompt>_seed<seed>_g<guidance>_s<steps>

Computed metrics (written to CSV):
  * Vendi Score (pixel & inception embedding)
  * FID: PIQ feature-based and optional clean-FID
  * CLIP cosine (text-image cosine; not the CLIPScore from papers)
  * 1 - MS-SSIM (multi-scale; diversity)
  * BRISQUE (quality; lower is better)

Dependencies:
  torch, torchvision, PIL, numpy, tqdm
  vendi_score (image_utils, vendi)
  piq (FID, BRISQUELoss, multi_scale_ssim); cleanfid (
      optional); open_clip_torch (optional)
  pytorch_msssim (optional: fallback MS-SSIM if piq is missing)
"""

import argparse
import csv
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from PIL import Image
from tqdm import tqdm
import torchvision.transforms as T

# -------- Optional deps --------
try:
    import piq
    from piq.feature_extractors import InceptionV3
except ImportError:
    piq = None
    InceptionV3 = None

try:
    import open_clip  # fallback for CLIP
except ImportError:
    open_clip = None

try:
    import cleanfid
except ImportError:
    cleanfid = None

try:
    from pytorch_msssim import ms_ssim as ms_ssim_torch
except ImportError:
    ms_ssim_torch = None

# Vendi score
from vendi_score import image_utils, vendi

# -------------------------------
# IO helpers
# -------------------------------
IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def list_images(dir_path: Path) -> List[Path]:
    return sorted([p for p in dir_path.rglob("*") if p.suffix.lower() in IMG_EXTS])


def list_child_dirs(root: Path) -> List[Path]:
    return sorted([p for p in root.iterdir() if p.is_dir()])


def derive_method_concept_and_eval_paths(gen_root: Path) -> Tuple[str, str, Path, Path]:
    """
    Infer method, concept, eval_dir and csv_path from:
      /.../outputs/<method>_<concept>/imgs

    Rules:
      - method = first token before the first underscore in "<method>_<concept>"
      - concept = the rest (may contain underscores)
    """
    parts = gen_root.parts
    if "imgs" not in parts:
        # Fallback: treat the parent directory as "<method>_<concept>"
        method_concept = gen_root.name
        base = gen_root.parent
    else:
        idx = parts.index("imgs")
        method_concept = parts[idx - 1] if idx - 1 >= 0 else gen_root.name
        base = Path(*parts[:idx]) if idx > 0 else Path(".")

    # Parse method and concept
    if "_" in method_concept:
        # split at the first underscore only
        tokens = method_concept.split("_", 1)
        method = tokens[0]
        concept = tokens[1]
    else:
        method = method_concept
        concept = ""

    eval_dir = base / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    csv_path = eval_dir / \
        f"{method}_{concept}.csv" if concept else eval_dir / f"{method}.csv"
    return method, concept, eval_dir, csv_path


# -------------------------------
# Parse folder meta
# -------------------------------
META_RE = re.compile(
    r"^(?P<prompt>.+)_seed(?P<seed>-?\d+)_g(?P<guidance>[0-9.]+)_s(?P<steps>\d+)$")


def parse_meta_from_name(name: str) -> Dict[str, Any]:
    m = META_RE.match(name)
    if not m:
        return {"prompt": name.replace("_", " "), "seed": None, "guidance": None, "steps": None}
    d = m.groupdict()
    d["prompt"] = d["prompt"].replace("_", " ")
    d["seed"] = int(d["seed"])
    d["guidance"] = float(d["guidance"])
    d["steps"] = int(d["steps"])
    return d

# -------------------------------
# Vendi: embedding fallback
# -------------------------------


def _embedding_vendi_score_fallback(imgs: List[Image.Image], device: str = "cuda") -> float:
    import torchvision
    from torchvision.models import Inception_V3_Weights
    weights = Inception_V3_Weights.DEFAULT
    tfm = weights.transforms()
    model = torchvision.models.inception_v3(
        weights=weights, aux_logits=True, transform_input=True)
    model.fc = nn.Identity()
    model.eval().to(device)

    feats: List[torch.Tensor] = []
    bs = 32
    with torch.inference_mode():
        for i in range(0, len(imgs), bs):
            batch = [tfm(img) for img in imgs[i: i + bs]]
            x = torch.stack(batch, dim=0).to(device)
            f = model(x)
            if isinstance(f, (tuple, list)):
                f = f[0]
            f = F.normalize(f, dim=1)
            feats.append(f.cpu())
    X = torch.cat(feats, dim=0).numpy()
    return float(vendi.score_dual(X, normalize=False))


def compute_vendi_for_images(imgs: List[Image.Image], device: str = "cuda") -> Dict[str, float]:
    pix_vs = float(image_utils.pixel_vendi_score(imgs))
    try:
        emb_vs = float(image_utils.embedding_vendi_score(imgs, device=device))
    except Exception as e:
        if "init_weights" in str(e) or "weights" in str(e):
            print("Vendi default failed, using torchvision compatibility fallback...")
            emb_vs = _embedding_vendi_score_fallback(imgs, device=device)
        else:
            raise
    return {"vendi_pixel": pix_vs, "vendi_inception": emb_vs}

# -------------------------------
# FID utilities
# -------------------------------


def _build_inception_extractor(device: torch.device):
    if piq is None or InceptionV3 is None:
        print("Warning: piq not installed; FID(piq) will be skipped.")
        return None
    return InceptionV3().to(device)


def _make_fid_transform():
    return T.Compose([
        T.Resize(299, interpolation=T.InterpolationMode.BICUBIC, antialias=True),
        T.CenterCrop(299),
        T.ToTensor(),
    ])


def _extract_inception_features(
    image_paths: List[Path],
    extractor: nn.Module,
    device: torch.device,
    num_workers: int = 4,
    batch_size: int = 32,
) -> torch.Tensor:
    transform = _make_fid_transform()

    class ImagePathDataset(torch.utils.data.Dataset):
        def __init__(self, files):
            self.files = files

        def __len__(self):
            return len(self.files)

        def __getitem__(self, i):
            path = self.files[i]
            img = Image.open(path).convert('RGB')
            return transform(img)

    if not image_paths:
        return torch.empty(0, 2048)

    dataset = ImagePathDataset(image_paths)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=(
            "cuda" in device.type)
    )

    all_features = []
    with torch.inference_mode():
        for images in tqdm(loader, desc=f"Extracting features: {Path(image_paths[0]).parent.name}"):
            feats = extractor(images.to(device))
            if isinstance(feats, list):
                feats = feats[0]
            feats = torch.nn.functional.adaptive_avg_pool2d(
                feats, (1, 1)).squeeze(-1).squeeze(-1)
            all_features.append(feats.cpu())

    return torch.cat(all_features, dim=0) if all_features else torch.empty(0, 2048)


def compute_fid_from_features(real_features: torch.Tensor, fake_features: torch.Tensor) -> Optional[float]:
    if piq is None:
        return None
    if real_features.numel() == 0 or fake_features.numel() == 0:
        return None
    fid_metric = piq.FID()
    value = fid_metric(real_features, fake_features)
    return float(value)


def compute_fid_cleanfid(real_dir: Path, gen_dir: Path, mode: str = "clean") -> Optional[float]:
    if cleanfid is None:
        print("cleanfid not installed; skipping clean-FID.")
        return None
    try:
        return float(cleanfid.compute_fid(str(real_dir), str(gen_dir), mode=mode, model_name="inception_v3"))
    except Exception as e:
        print(f"clean-fid failed on {gen_dir}: {e}")
        return None

# -------------------------------
# CLIP cosine
# -------------------------------


def load_clip_from_jit(jit_path: Path):
    model = torch.jit.load(str(jit_path), map_location="cpu").eval()
    preprocess = T.Compose([
        T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize((0.48145466, 0.4578275, 0.40821073),
                    (0.26862954, 0.26130258, 0.27577711)),
    ])
    return model, preprocess


def load_clip_fallback_openclip(device: torch.device):
    if open_clip is None:
        return None, None, None
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k", device=device
    )
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    model.eval()
    return model, preprocess, tokenizer


def calculate_clip_cosine(
    image_folder: Path,
    model: Any,
    preprocess: T.Compose,
    device: torch.device,
    batch_size: int,
    text: Optional[str] = None,
    tokenizer=None,
    is_openclip: bool = False
) -> Optional[float]:
    image_paths = list(image_folder.rglob("*.jpg")) + \
        list(image_folder.rglob("*.png"))
    if not image_paths:
        return None

    prompt = text
    if prompt is None:
        meta = parse_meta_from_name(image_folder.name)
        prompt = meta.get("prompt") or image_folder.name.replace(
            "_", " ").replace("-", " ")

    all_scores = []
    with torch.inference_mode():
        if is_openclip:
            text_tokens = tokenizer([prompt]).to(device)
            text_features = model.encode_text(text_tokens)
        else:
            if open_clip is None:
                print(
                    "open_clip not installed; cannot tokenize text for JIT CLIP. Skipping CLIP cosine.")
                return None
            text_tokens = open_clip.tokenize([prompt]).to(device)
            text_features = model.encode_text(text_tokens)
        text_features = text_features / \
            text_features.norm(dim=-1, keepdim=True)

        for i in tqdm(range(0, len(image_paths), batch_size), desc=f"CLIP cosine: {image_folder.name}"):
            batch_paths = image_paths[i:i + batch_size]
            images = [preprocess(Image.open(p).convert("RGB"))
                      for p in batch_paths]
            image_batch = torch.stack(images).to(device)
            image_features = model.encode_image(image_batch)
            image_features = image_features / \
                image_features.norm(dim=-1, keepdim=True)
            sim = (100.0 * image_features @
                   text_features.T).squeeze(-1).detach().cpu().numpy()
            all_scores.extend(sim.tolist())

    return float(np.mean(all_scores)) if all_scores else None

# -------------------------------
# 1 - MS-SSIM (true multi-scale)
# -------------------------------


def calculate_ms_ssim_diversity(gen_dir: Path, max_pairs: int, device: torch.device) -> Optional[float]:
    paths = list(gen_dir.rglob("*.jpg")) + list(gen_dir.rglob("*.png"))
    if len(paths) < 2:
        return None

    rng = np.random.default_rng(0)
    target_n = max(2, int(np.sqrt(max_pairs * 2)) +
                   1) if max_pairs > 0 else len(paths)
    if len(paths) > target_n:
        paths = rng.choice(paths, size=target_n, replace=False)

    pairs = [(paths[i], paths[j]) for i in range(len(paths))
             for j in range(i+1, len(paths))]
    if max_pairs > 0 and len(pairs) > max_pairs:
        idx = rng.choice(len(pairs), size=max_pairs, replace=False)
        pairs = [pairs[i] for i in idx]

    tfm = T.Compose([T.ToTensor()])  # [0,1]
    scores = []

    for p1, p2 in tqdm(pairs, desc=f"MS-SSIM: {gen_dir.name}"):
        try:
            im1 = Image.open(p1).convert("RGB")
            im2 = Image.open(p2).convert("RGB")
        except Exception:
            continue
        im2 = im2.resize(im1.size, Image.BICUBIC)
        x1 = tfm(im1).unsqueeze(0).to(device)
        x2 = tfm(im2).unsqueeze(0).to(device)

        with torch.inference_mode():
            if piq is not None:
                s = piq.multi_scale_ssim(
                    x1, x2, data_range=1.0, reduction="none")
                score = float(s.item())
            elif ms_ssim_torch is not None:
                score = float(ms_ssim_torch(x1, x2, data_range=1.0).item())
            else:
                print("Neither piq nor pytorch_msssim installed; skipping MS-SSIM.")
                return None
        scores.append(score)

    if not scores:
        return None
    return float(1.0 - np.mean(scores))

# -------------------------------
# BRISQUE
# -------------------------------


def calculate_brisque_quality(image_folder: Path, device: torch.device) -> Optional[float]:
    if piq is None:
        print("piq not installed; BRISQUE will be skipped.")
        return None

    image_paths = list(image_folder.rglob("*.jpg")) + \
        list(image_folder.rglob("*.png"))
    if not image_paths:
        return None

    transform = T.Compose([T.ToTensor()])
    metric = piq.BRISQUELoss(data_range=1.0, reduction='none')

    scores = []
    for p in tqdm(image_paths, desc=f"BRISQUE: {image_folder.name}"):
        try:
            img = Image.open(p).convert("RGB")
        except Exception:
            continue
        x = transform(img).unsqueeze(0).to(device)
        with torch.inference_mode():
            s = metric(x)
        scores.append(float(s.item()))
    if not scores:
        return None
    return float(np.mean(scores))

# -------------------------------
# Main
# -------------------------------


def main():
    ap = argparse.ArgumentParser(
        description="Unified evaluator (method_concept, CSV-only).")
    ap.add_argument("--gen-root", type=str, required=True,
                    help="Path to '<method>_<concept>/imgs' directory.")
    ap.add_argument("--real", required=True,
                    help="Reference (real) image folder for FID.")
    ap.add_argument("--device", type=str, default="cuda:0",
                    help='Device, e.g., "cpu", "cuda", "cuda:0".')
    ap.add_argument("--batch-size", type=int, default=32,
                    help="Batch size for feature extraction.")
    ap.add_argument("--num-workers", type=int, default=4,
                    help="Number of DataLoader workers.")
    ap.add_argument("--clip-jit", type=str, default=os.path.expanduser(\"~/.cache/clip/ViT-B-32.pt\"), help="Local OpenAI CLIP JIT path.")
                    ap.add_argument("--max-pairs", type=int, default=100,
                    help="Max pairs to evaluate for MS-SSIM diversity (cost control).")
                    ap.add_argument("--fid-mode", type=str, choices=["piq", "clean", "both", "none"], default="both",
                    help="Which FID variant(s) to compute.")
                    ap.add_argument("--match-real-count", action="store_true",
                    help="For FID(piq), subsample real features to match the number of generated images per folder.")
                    ap.add_argument("--text", type=str, default=None,
                    help="Global text prompt for CLIP cosine; defaults to the parsed prompt from folder name.")

                    args=ap.parse_args()

                    device=torch.device(args.device if (
                        "cuda" in args.device and torch.cuda.is_available()) else "cpu")
                    print(f"Using device: {device}")

                    gen_root=Path(args.gen_root)
                    if not gen_root.exists() or not gen_root.is_dir():
                    raise SystemExit(
        f"--gen-root not found or not a directory: {gen_root}")

        method, concept, eval_dir, csv_path=derive_method_concept_and_eval_paths(
            gen_root)
        print(f"[io] method={method}, concept={concept}")
        print(f"[io] CSV -> {csv_path}")

        real_dir=Path(args.real)
        if not real_dir.exists() or not real_dir.is_dir():
        raise SystemExit(
            f"Real folder not found or not a directory: {real_dir}")

        # --- Build shared tools/models ---
        # Inception for FID (piq)
        inception=None
        if args.fid_mode in ("piq", "both"):
        inception=_build_inception_extractor(device)

        # Precompute real features once (for FID-piq)
        real_features=None
        if inception is not None:
        real_paths=list_images(real_dir)
        if real_paths:
            real_features=_extract_inception_features(
                real_paths, inception, device, num_workers=args.num_workers, batch_size=args.batch_size
    )
        else:
            print("No real images found; FID(piq) will be skipped.")
            inception=None

        # CLIP: prefer JIT, fallback to open_clip
        clip_model=None
        clip_preproc=None
        openclip_tokenizer=None
        clip_is_openclip=False
        jit_path=Path(args.clip_jit) if args.clip_jit else None
        if jit_path and jit_path.exists():
        try:
            clip_model, clip_preproc=load_clip_from_jit(jit_path)
            clip_model=clip_model.to(device)
            print(f"Loaded CLIP JIT from {jit_path}")
        except Exception as e:
            print(
                f"Failed to load CLIP JIT: {e}. Trying open_clip fallback...")
        if clip_model is None:
        model, preproc, tok=load_clip_fallback_openclip(device)
        if model is not None:
            clip_model, clip_preproc, openclip_tokenizer=model, preproc, tok
            clip_is_openclip=True
            print("Loaded CLIP via open_clip fallback.")
        else:
            print("No CLIP available; CLIP cosine will be skipped.")

        # --- Discover generation folders under gen_root ---
        gen_dirs=[d for d in list_child_dirs(gen_root) if list_images(d)]
        if not gen_dirs:
        raise SystemExit(f"No image subfolders found under: {gen_root}")

        # --- Prepare CSV ---
        csv_columns=[
        "method", "concept", "folder", "prompt", "seed", "guidance", "steps", "num_images",
        "vendi_pixel", "vendi_inception",
        "fid", "fid_clean",
        "clip_score", "one_minus_ms_ssim", "brisque"
    ]
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, "w", newline="", encoding="utf-8") as csv_f:
        csv_writer=csv.DictWriter(csv_f, fieldnames=csv_columns)
        csv_writer.writeheader()

        # --- Evaluate each subfolder ---
        for gen_dir in gen_dirs:
            print(f"\n=== Evaluating: {gen_dir.name} ===")
            paths=list_images(gen_dir)
            if not paths:
                print(f"No images in {gen_dir}, skipping.")
                continue

            meta=parse_meta_from_name(gen_dir.name)

            # Vendi
            try:
                imgs=[Image.open(p).convert("RGB") for p in paths]
            except Exception:
                imgs=[]
            if not imgs:
                print(f"Could not load any images from {gen_dir}, skipping.")
                continue
            vendi_scores=compute_vendi_for_images(imgs, device=str(device))

            # FID (piq)
            fid_value=None
            if (args.fid_mode in ("piq", "both")) and (inception is not None) and (real_features is not None) and (real_features.numel() > 0):
                fake_features=_extract_inception_features(
                    paths, inception, device, num_workers=args.num_workers, batch_size=args.batch_size
    )
                real_feats_used=real_features
                if args.match_real_count and real_features.shape[0] >= fake_features.shape[0] and fake_features.shape[0] > 0:
                    rng=np.random.default_rng(
                        abs(hash(gen_dir.name)) % (2**32))
                    idx=rng.choice(
                        real_features.shape[0], size=fake_features.shape[0], replace=False)
                    real_feats_used=real_features[idx]
                fid_value=compute_fid_from_features(
                    real_feats_used, fake_features)

            # FID (clean-fid)
            fid_clean=None
            if args.fid_mode in ("clean", "both"):
                fid_clean=compute_fid_cleanfid(real_dir, gen_dir, mode="clean")

            # CLIP cosine
            clip_cos=None
            if clip_model is not None and clip_preproc is not None:
                clip_text=args.text if args.text else meta.get("prompt")
                clip_cos=calculate_clip_cosine(
                    gen_dir, clip_model, clip_preproc, device, args.batch_size,
                    text=clip_text,
                    tokenizer=openclip_tokenizer if clip_is_openclip else None,
                    is_openclip=clip_is_openclip
    )

            # 1 - MS-SSIM (multi-scale)
            msssim_div=calculate_ms_ssim_diversity(
                gen_dir, max_pairs=args.max_pairs, device=device)

            # BRISQUE
            brisque_quality=calculate_brisque_quality(gen_dir, device=device)

            # Write CSV row (includes prompt / seed / guidance / steps)
            row={
                "method": method,
                "concept": concept,
                "folder": gen_dir.name,
                "prompt": meta.get("prompt"),
                "seed": meta.get("seed"),
                "guidance": meta.get("guidance"),
                "steps": meta.get("steps"),
                "num_images": len(imgs),
                "vendi_pixel": vendi_scores.get("vendi_pixel"),
                "vendi_inception": vendi_scores.get("vendi_inception"),
                "fid": fid_value,
                "fid_clean": fid_clean,
                "clip_score": clip_cos,
                "one_minus_ms_ssim": msssim_div,
                "brisque": brisque_quality,
    }
            csv_writer.writerow(row)

        print(f"\nSaved CSV to: {csv_path}")

        if __name__ == "__main__":
        main()
