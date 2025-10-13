# -*- coding: utf-8 -*-
"""
Batch CSV evaluator by concept for methods: dpp, pg, cads (exclude ours).
Now with KID (Kernel Inception Distance).
Usage example:
  python eval_by_concept.py \
    --outputs-root /mnt/data6t/yyz/flow_grpo/flow_base/outputs \
    --concept truck \
    --real-root /mnt/data6t/yyz/flow_grpo/flow_base/real_cls_crops \
    --fid-mode both --match-real-count --max-pairs 200 \
    --clip-jit ~/.cache/clip/ViT-B-32.pt --clip-image-size 224 \
    --kid-subset-size 1000 --kid-subsets 10
"""

import argparse, os, re
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import csv
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

# Optional deps
try:
    import piq
    from piq.feature_extractors import InceptionV3
except ImportError:
    piq = None
    InceptionV3 = None

# only for tokenization fallback (model not used)
try:
    import open_clip
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

from vendi_score import image_utils, vendi

IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
META_RE = re.compile(r"^(?P<prompt>.+)_seed(?P<seed>-?\d+)_g(?P<guidance>[0-9.]+)_s(?P<steps>\d+)$")

def list_images(dir_path: Path) -> List[Path]:
    return sorted([p for p in dir_path.rglob("*") if p.suffix.lower() in IMG_EXTS])

def list_child_dirs(root: Path) -> List[Path]:
    return sorted([p for p in root.iterdir() if p.is_dir()])

def parse_meta_from_name(name: str) -> Dict[str, Any]:
    m = META_RE.match(name)
    if not m:
        return {"prompt": name.replace("_", " "), "seed": None, "guidance": None, "steps": None}
    d = m.groupdict()
    d["prompt"]   = d["prompt"].replace("_", " ")
    d["seed"]     = int(d["seed"])
    d["guidance"] = float(d["guidance"])
    d["steps"]    = int(d["steps"])
    return d

# --------- Vendi ----------
def _embedding_vendi_score_fallback(imgs: List[Image.Image], device: str = "cuda") -> float:
    import torchvision
    from torchvision.models import Inception_V3_Weights
    weights = Inception_V3_Weights.DEFAULT
    tfm = weights.transforms()
    model = torchvision.models.inception_v3(weights=weights, aux_logits=True, transform_input=True)
    model.fc = nn.Identity()
    model.eval().to(device)
    feats = []
    bs = 32
    with torch.inference_mode():
        for i in range(0, len(imgs), bs):
            x = torch.stack([tfm(img) for img in imgs[i:i+bs]], dim=0).to(device)
            f = model(x)
            if isinstance(f, (tuple, list)): f = f[0]
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
            print("Vendi default failed, use torchvision fallback...")
            emb_vs = _embedding_vendi_score_fallback(imgs, device=device)
        else:
            raise
    return {"vendi_pixel": pix_vs, "vendi_inception": emb_vs}

# --------- Inception features (for FID & KID) ----------
def _build_inception_extractor(device: torch.device):
    if piq is None or InceptionV3 is None:
        print("Warning: piq not installed; FID(piq)/KID using piq features will be skipped.")
        return None
    return InceptionV3().to(device)

def _make_fid_transform():
    return T.Compose([
        T.Resize(299, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(299),
        T.ToTensor(),
    ])

def _extract_inception_features(paths: List[Path], extractor: nn.Module, device: torch.device,
                                num_workers=4, batch_size=32) -> torch.Tensor:
    transform = _make_fid_transform()
    class D(torch.utils.data.Dataset):
        def __init__(self, files): self.files = files
        def __len__(self): return len(self.files)
        def __getitem__(self, i):
            img = Image.open(self.files[i]).convert("RGB")
            return transform(img)
    if not paths: return torch.empty(0, 2048)
    loader = torch.utils.data.DataLoader(D(paths), batch_size=batch_size, num_workers=num_workers,
                                         pin_memory=("cuda" in device.type))
    feats_all = []
    with torch.inference_mode():
        for imgs in tqdm(loader, desc=f"Extracting features: {Path(paths[0]).parent.name}"):
            feats = extractor(imgs.to(device))
            if isinstance(feats, list): feats = feats[0]
            feats = torch.nn.functional.adaptive_avg_pool2d(feats, (1,1)).squeeze(-1).squeeze(-1)
            feats_all.append(feats.cpu())
    return torch.cat(feats_all, dim=0) if feats_all else torch.empty(0, 2048)

# --------- FID ----------
def compute_fid_from_features(real_features: torch.Tensor, fake_features: torch.Tensor) -> Optional[float]:
    if piq is None or real_features.numel()==0 or fake_features.numel()==0: return None
    return float(piq.FID()(real_features, fake_features))

def compute_fid_cleanfid(real_dir: Path, gen_dir: Path, mode="clean") -> Optional[float]:
    if cleanfid is None:
        print("cleanfid not installed; skip clean-FID.")
        return None
    try:
        return float(cleanfid.compute_fid(str(real_dir), str(gen_dir), mode=mode, model_name="inception_v3"))
    except Exception as e:
        print(f"clean-fid failed on {gen_dir}: {e}")
        return None

# --------- KID (Kernel Inception Distance) ----------
# === KID ===
def _poly_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Polynomial kernel k(x,y) = (x^T y / d + 1)^3  where d=feature_dim.
    x: [m, d], y: [n, d]
    returns: [m, n]
    """
    d = x.shape[1]
    return (x @ y.T / float(d) + 1.0) ** 3

# === KID ===
def compute_kid_from_features(
    real_features: torch.Tensor,
    fake_features: torch.Tensor,
    subset_size: int = 1000,
    n_subsets: int = 10,
    rng: Optional[np.random.Generator] = None,
) -> Optional[Tuple[float, float]]:
    """
    Unbiased KID estimate (mean ± std over subsets) using polynomial kernel of degree 3.
    Returns (kid_mean, kid_std) or None if inputs are invalid.
    """
    if real_features is None or fake_features is None: return None
    if real_features.numel()==0 or fake_features.numel()==0: return None
    if real_features.shape[1] != fake_features.shape[1]:
        print("KID: feature dim mismatch.")
        return None

    m_full, n_full = real_features.shape[0], fake_features.shape[0]
    if m_full < 2 or n_full < 2:
        return None

    subset = min(subset_size, m_full, n_full)
    if subset < 2:
        return None

    if rng is None:
        rng = np.random.default_rng(0)

    # Work on float64 for numerics (optional)
    X = real_features.to(torch.float64)
    Y = fake_features.to(torch.float64)
    stats = []

    for _ in range(max(1, n_subsets)):
        ridx = rng.choice(m_full, size=subset, replace=False)
        fidx = rng.choice(n_full, size=subset, replace=False)
        xs = X[ridx]  # [s,d]
        ys = Y[fidx]  # [s,d]

        k_xx = _poly_kernel(xs, xs)
        k_yy = _poly_kernel(ys, ys)
        k_xy = _poly_kernel(xs, ys)

        m = k_xx.shape[0]
        n = k_yy.shape[0]

        # Unbiased MMD^2 estimator
        sum_xx = (k_xx.sum() - k_xx.diag().sum()) / (m * (m - 1))
        sum_yy = (k_yy.sum() - k_yy.diag().sum()) / (n * (n - 1))
        sum_xy = k_xy.mean()

        mmd2 = sum_xx + sum_yy - 2.0 * sum_xy
        stats.append(float(mmd2.item()))

    stats = np.array(stats, dtype=np.float64)
    return float(stats.mean()), float(stats.std(ddof=1)) if stats.size > 1 else 0.0

# --------- CLIP cosine (JIT only, preprocess aligned to pipeline) ----------
def load_clip_from_jit(jit_path: Path, image_size: int = 224):
    """
    Load OpenAI CLIP JIT and return (model, preprocess) where preprocess matches pipeline:
      - Resize((S,S), bicubic, antialias=True), no aspect ratio keep, no center crop
      - ToTensor + OpenAI mean/std
    """
    model = torch.jit.load(str(jit_path), map_location="cpu").eval()
    preprocess = T.Compose([
        T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BICUBIC, antialias=True),
        T.ToTensor(),
        T.Normalize((0.48145466, 0.4578275, 0.40821073),
                    (0.26862954, 0.26130258, 0.27577711)),
    ])
    return model, preprocess

def calculate_clip_cosine(folder: Path, model, preprocess, device, batch_size,
                          text: Optional[str] = None, tokenizer=None):
    """
    Compute CLIP cosine using the SAME preprocessing as pipeline (passed in via preprocess).
    Model is the OpenAI CLIP JIT. No model fallback.
    """
    image_paths = list(folder.rglob("*.jpg")) + list(folder.rglob("*.png"))
    if not image_paths:
        return None

    # prompt
    prompt = text
    if prompt is None:
        meta = parse_meta_from_name(folder.name)
        prompt = meta.get("prompt") or folder.name.replace("_"," ").replace("-"," ")

    # tokenizer: prefer open_clip.tokenize, fallback to openai-clip's clip.tokenize
    if tokenizer is None:
        tok = None
        try:
            import open_clip as _oc
            tok = _oc.tokenize
        except Exception:
            try:
                import clip as openai_clip
                tok = openai_clip.tokenize
            except Exception:
                tok = None
        tokenizer = tok

    if tokenizer is None:
        print("No tokenizer available (open_clip or clip). Skip CLIP cosine.")
        return None

    with torch.inference_mode():
        t = tokenizer([prompt]).to(device)
        tfeat = model.encode_text(t)
        tfeat = tfeat / (tfeat.norm(dim=-1, keepdim=True) + 1e-12)

        sims = []
        for i in tqdm(range(0, len(image_paths), batch_size), desc=f"CLIP cosine: {folder.name}"):
            ims = []
            for p in image_paths[i:i+batch_size]:
                try:
                    ims.append(preprocess(Image.open(p).convert("RGB")))
                except Exception:
                    continue
            if not ims:
                continue
            x = torch.stack(ims).to(device=device, dtype=torch.float32)  # [B,3,S,S]
            # JIT encode_image
            if not hasattr(model, "encode_image"):
                print("JIT model has no encode_image; skip.")
                return None
            feat = model.encode_image(x)
            feat = feat / (feat.norm(dim=-1, keepdim=True) + 1e-12)
            sim = (100.0 * feat @ tfeat.T).squeeze(-1).detach().cpu().numpy()
            sims.extend(sim.tolist())

    return float(np.mean(sims)) if sims else None

# --------- MS-SSIM (multi-scale) ----------
def calculate_ms_ssim_diversity(folder: Path, max_pairs: int, device: torch.device) -> Optional[float]:
    paths = list(folder.rglob("*.jpg")) + list(folder.rglob("*.png"))
    if len(paths) < 2: return None
    rng = np.random.default_rng(0)
    target_n = max(2, int(np.sqrt(max_pairs * 2)) + 1) if max_pairs > 0 else len(paths)
    if len(paths) > target_n:
        paths = rng.choice(paths, size=target_n, replace=False)
    pairs = [(paths[i], paths[j]) for i in range(len(paths)) for j in range(i+1, len(paths))]
    if max_pairs > 0 and len(pairs) > max_pairs:
        idx = rng.choice(len(pairs), size=max_pairs, replace=False)
        pairs = [pairs[i] for i in idx]
    tfm = T.Compose([T.ToTensor()])
    scores = []
    for p1, p2 in tqdm(pairs, desc=f"MS-SSIM: {folder.name}"):
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
                s = piq.multi_scale_ssim(x1, x2, data_range=1.0, reduction="none")
                score = float(s.item())
            elif ms_ssim_torch is not None:
                score = float(ms_ssim_torch(x1, x2, data_range=1.0).item())
            else:
                print("Neither piq nor pytorch_msssim installed; skip MS-SSIM.")
                return None
        scores.append(score)
    if not scores: return None
    return float(1.0 - np.mean(scores))

# --------- BRISQUE ----------
def calculate_brisque_quality(folder: Path, device: torch.device) -> Optional[float]:
    if piq is None:
        print("piq not installed; skip BRISQUE.")
        return None
    paths = list(folder.rglob("*.jpg")) + list(folder.rglob("*.png"))
    if not paths: return None
    tfm = T.Compose([T.ToTensor()])
    metric = piq.BRISQUELoss(data_range=1.0, reduction='none')
    scores = []
    for p in tqdm(paths, desc=f"BRISQUE: {folder.name}"):
        try:
            img = Image.open(p).convert("RGB")
        except Exception:
            continue
        x = tfm(img).unsqueeze(0).to(device)
        with torch.inference_mode():
            s = metric(x)
        scores.append(float(s.item()))
    return float(np.mean(scores)) if scores else None

# --------- Evaluate one method_concept ----------
def evaluate_one(gen_root: Path, method: str, concept: str, real_dir: Path,
                 device: torch.device, fid_mode: str, match_real_count: bool,
                 batch_size: int, num_workers: int, max_pairs: int,
                 clip_model, clip_preproc,
                 inception, real_features,
                 kid_subset_size: int, kid_subsets: int):
    if not gen_root.exists() or not gen_root.is_dir():
        print(f"[skip] {gen_root} not found.")
        return
    gen_dirs = [d for d in list_child_dirs(gen_root) if list_images(d)]
    if not gen_dirs:
        print(f"[skip] No image subfolders under: {gen_root}")
        return

    eval_dir = gen_root.parent / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    csv_path = eval_dir / f"{method}_{concept}.csv"
    print(f"[{method}] writing CSV -> {csv_path}")

    columns = [
        "method","concept","folder","prompt","seed","guidance","steps","num_images",
        "vendi_pixel","vendi_inception",
        "fid","fid_clean",
        # === KID ===
        "kid_mean","kid_std",
        "clip_score","one_minus_ms_ssim","brisque"
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()

        for gen_dir in gen_dirs:
            print(f"\n=== {method}/{concept}: {gen_dir.name} ===")
            paths = list_images(gen_dir)
            if not paths:
                print("  no images, skip.")
                continue

            meta = parse_meta_from_name(gen_dir.name)

            # Vendi
            try:
                imgs = [Image.open(p).convert("RGB") for p in paths]
            except Exception:
                imgs = []
            if not imgs:
                print("  cannot load images, skip.")
                continue
            vendi_scores = compute_vendi_for_images(imgs, device=str(device))

            # Features for FID/KID (extract once if inception available)
            fake_feats = torch.empty(0)
            if inception is not None:
                fake_feats = _extract_inception_features(paths, inception, device,
                                                         num_workers=num_workers, batch_size=batch_size)

            # FID (piq)
            fid_value = None
            if fid_mode in ("piq","both") and (inception is not None) and (real_features is not None) and (real_features.numel()>0):
                real_feats_used = real_features
                if match_real_count and fake_feats.numel() > 0 and real_features.shape[0] >= fake_feats.shape[0]:
                    rng = np.random.default_rng(abs(hash(gen_dir.name)) % (2**32))
                    idx = rng.choice(real_features.shape[0], size=fake_feats.shape[0], replace=False)
                    real_feats_used = real_features[idx]
                if fake_feats.numel() > 0:
                    fid_value = compute_fid_from_features(real_feats_used, fake_feats)

            # FID (clean-fid)
            fid_clean = None
            if fid_mode in ("clean","both"):
                fid_clean = compute_fid_cleanfid(real_dir, gen_dir, mode="clean")

            # === KID ===
            kid_mean, kid_std = None, None
            if (kid_subsets > 0) and (inception is not None) and (real_features is not None) and (fake_feats.numel() > 0):
                rng = np.random.default_rng(abs(hash(gen_dir.name)) % (2**32))
                kid_ret = compute_kid_from_features(real_features, fake_feats,
                                                    subset_size=kid_subset_size,
                                                    n_subsets=kid_subsets,
                                                    rng=rng)
                if kid_ret is not None:
                    kid_mean, kid_std = kid_ret

            # CLIP cosine (JIT only)
            clip_cos = None
            if (clip_model is not None) and (clip_preproc is not None):
                clip_text = meta.get("prompt")
                clip_cos = calculate_clip_cosine(
                    gen_dir, clip_model, clip_preproc, device, batch_size,
                    text=clip_text, tokenizer=None  # tokenizer will be resolved inside
                )

            # MS-SSIM diversity
            msssim_div = calculate_ms_ssim_diversity(gen_dir, max_pairs=max_pairs, device=device)

            # BRISQUE
            brisque_quality = calculate_brisque_quality(gen_dir, device=device)

            w.writerow({
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
                # === KID ===
                "kid_mean": kid_mean,
                "kid_std": kid_std,
                "clip_score": clip_cos,
                "one_minus_ms_ssim": msssim_div,
                "brisque": brisque_quality,
            })
    print(f"[{method}] done -> {csv_path}")

def main():
    ap = argparse.ArgumentParser(description="Evaluate dpp/pg/cads for one concept (CSV-only).")
    ap.add_argument("--outputs-root", required=True, help="Root outputs dir containing <method>_<concept> folders.")
    ap.add_argument("--concept", required=True, help="Concept name, e.g., truck")
    ap.add_argument("--real-root", required=True, help="Root of real sets; real dir will be <real-root>/<concept>")

    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--max-pairs", type=int, default=100)

    ap.add_argument("--fid-mode", choices=["piq","clean","both","none"], default="both")
    ap.add_argument("--match-real-count", action="store_true")

    ap.add_argument("--clip-jit", type=str, default=os.path.expanduser("~/.cache/clip/ViT-B-32.pt"))
    ap.add_argument("--clip-image-size", type=int, default=224,
                    help="Match pipeline CLIP image size (e.g., 224 or 336).")

    # 可选：自定义方法列表，默认只跑 dpp/pg/cads
    ap.add_argument("--methods", nargs="+", default=["dpp","pg","cads","ourmethod"],
                    help="Methods to run (default: dpp pg cads). 'ours' will be ignored if included.")

    # === KID ===
    ap.add_argument("--kid-subset-size", type=int, default=1000,
                    help="Subset size per KID estimate (min(#real,#fake,#this)).")
    ap.add_argument("--kid-subsets", type=int, default=10,
                    help="Number of random subsets for KID (set 0 to disable KID).")

    args = ap.parse_args()

    device = torch.device(args.device if ("cuda" in args.device and torch.cuda.is_available()) else "cpu")
    print(f"Using device: {device}")

    outputs_root = Path(args.outputs_root)
    concept = args.concept
    real_dir = Path(args.real_root) / concept
    if not real_dir.exists():
        raise SystemExit(f"Real dir not found: {real_dir}")

    # Build shared tools/models (reused across methods)
    # NOTE: also build inception if we plan to compute KID.
    need_inception = (args.fid_mode in ("piq","both")) or (args.kid_subsets > 0)
    inception = _build_inception_extractor(device) if need_inception else None

    real_features = None
    if inception is not None:
        real_paths = list_images(real_dir)
        if real_paths:
            real_features = _extract_inception_features(real_paths, inception, device,
                                                        num_workers=args.num_workers, batch_size=args.batch_size)
        else:
            print("No real images found; FID(piq)/KID will be skipped.")
            if args.fid_mode in ("piq","both"):
                inception = None  # disable feature-based metrics if no real set

    # CLIP setup: use OpenAI JIT only (no model fallback)
    clip_model = None
    clip_preproc = None
    jit_path = Path(args.clip_jit) if args.clip_jit else None
    if jit_path and jit_path.exists():
        try:
            clip_model, clip_preproc = load_clip_from_jit(jit_path, image_size=args.clip_image_size)
            clip_model = clip_model.to(device)
            print(f"Loaded CLIP JIT from {jit_path} (image_size={args.clip_image_size})")
        except Exception as e:
            print(f"Failed to load CLIP JIT: {e}")
    else:
        print(f"CLIP JIT not found at {jit_path}; CLIP cosine will be skipped.")

    # Run each method (skip 'ours' explicitly)
    for method in args.methods:
        if method.lower() == "ours":
            print("[skip] 'ours' is under debugging; skipping.")
            continue
        gen_root = outputs_root / f"{method}_{concept}" / "imgs"
        evaluate_one(
            gen_root=gen_root, method=method, concept=concept, real_dir=real_dir,
            device=device, fid_mode=args.fid_mode, match_real_count=args.match_real_count,
            batch_size=args.batch_size, num_workers=args.num_workers, max_pairs=args.max_pairs,
            clip_model=clip_model, clip_preproc=clip_preproc,
            inception=inception, real_features=real_features,
            kid_subset_size=args.kid_subset_size, kid_subsets=args.kid_subsets
        )

if __name__ == "__main__":
    main()