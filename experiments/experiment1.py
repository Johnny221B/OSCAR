#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Experiment 1 (PRD & AUC-PRD) evaluator (no prompts_json filtering).

K-set folder name (fixed):
  <prompt>_seed<seed>_g<guidance>_s<steps>
e.g.:
  a_photo_of_one_truck_seed3333_g3.0_s30

Changes in this version
-----------------------
- CHANGED: No longer loads/uses prompts_json to select prompts.
- CHANGED: Scans ALL K-sets under imgs_root that match the regex.
- Everything else remains the same (CLIP features, PRD/AUC-PRD, recall@precision,
  per-guidance aggregation, CSV outputs under {outputs_root}/{method}_{concept}/eval).
"""

import argparse, json, re, gc, sys
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

# ===== CLIP =====
import torch
import torch.nn.functional as F
import clip  # pip install git+https://github.com/openai/CLIP.git
from sklearn.cluster import KMeans

# ----------------- utils -----------------
IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
def is_image(p: Path) -> bool: return p.suffix.lower() in IMG_EXTS
def list_images(d: Path) -> List[Path]:
    return sorted([p for p in d.iterdir() if p.is_file() and is_image(p)]) if d.is_dir() else []

def free_mem():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def norm_prompt(s: str) -> str:
    s = s.replace(" ", "_")
    s = re.sub(r"_+", "_", s)
    return s

# --------- K-set: <prompt>_seed<seed>_g<guidance>_s<steps> ---------
KSET_RE = re.compile(
    r"^(?P<prompt>.+)_seed(?P<seed>\d+)_g(?P<guidance>[-+]?\d*\.?\d+)_s(?P<steps>\d+)$"
)

# CHANGED: scan all K-sets under imgs_root (no allowed_prompts filtering)
def scan_ksets_all(imgs_root: Path, print_found: bool=False) -> List[Dict]:
    found_prompts = []
    out = []
    if not imgs_root.exists():
        return out
    for d in sorted(imgs_root.iterdir()):
        if not d.is_dir():
            continue
        m = KSET_RE.match(d.name)
        if not m:
            continue
        prompt = m.group("prompt")
        found_prompts.append(prompt)
        imgs = list_images(d)
        if not imgs:
            continue
        out.append({
            "dir": d,
            "prompt": prompt,
            "seed": int(m.group("seed")),
            "guidance": float(m.group("guidance")),
            "steps": int(m.group("steps")),
            "images": imgs
        })
    if print_found:
        uniq = sorted(set(found_prompts))
        print(f"[DEBUG] Found {len(uniq)} prompt prefixes under {imgs_root}:")
        for s in uniq[:100]:
            print("  -", s)
        if len(uniq) > 100:
            print("  ... (+ more)")
    return out

# --------- CLIP Featurizer (local weights supported) ---------
class CLIPFeaturizer:
    def __init__(self, model_name="ViT-B/32", device=None, batch_size=64,
                 download_root: str=None, jit: bool=True):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        try:
            self.model, self.preprocess = clip.load(
                model_name, device=self.device, jit=jit, download_root=download_root
            )
        except Exception as e:
            print(
                f"[ERROR] clip.load('{model_name}', download_root={download_root}) failed.\n"
                f"Ensure local .pt exists (e.g., ~/.cache/clip/ViT-B-32.pt) and model_name matches. Error: {e}",
                file=sys.stderr
            )
            raise
        self.model.eval()
        self.bs = batch_size
        print(f"[INFO] CLIP = {model_name} | device={self.device} | jit={jit} | root={download_root}")

    @torch.no_grad()
    def encode_paths(self, paths: List[Path]) -> np.ndarray:
        feats = []
        for i in range(0, len(paths), self.bs):
            batch = []
            for p in paths[i:i+self.bs]:
                im = Image.open(p).convert("RGB")
                batch.append(self.preprocess(im))
            x = torch.stack(batch).to(self.device)
            f = self.model.encode_image(x)
            f = F.normalize(f.float(), dim=-1)
            feats.append(f.cpu().numpy())
        free_mem()
        return np.concatenate(feats, axis=0) if feats else np.zeros((0, 512), dtype=np.float32)

# --------- PRD (Sajjadi) ---------
def prd_from_hist(p: np.ndarray, q: np.ndarray, lambdas: np.ndarray):
    p = np.clip(p, 1e-12, 1.0); p /= p.sum()
    q = np.clip(q, 1e-12, 1.0); q /= q.sum()
    prec, reca = [], []
    for lam in lambdas:
        prec.append(np.minimum(lam * p, q).sum())
        reca.append(np.minimum(p, q / lam).sum())
    return np.asarray(prec), np.asarray(reca)

def compute_prd_curve(real_feats: np.ndarray, gen_feats: np.ndarray,
                      n_bins: int = 10, n_lambdas: int = 151, random_state: int = 0):
    if real_feats.size == 0 or gen_feats.size == 0:
        # Edge case: empty feats -> degenerate PRD
        lambdas = np.exp(np.linspace(np.log(1e-3), np.log(1e3), n_lambdas))
        return np.zeros_like(lambdas), np.zeros_like(lambdas)
    X = np.concatenate([real_feats, gen_feats], axis=0)
    km = KMeans(n_clusters=n_bins, random_state=random_state, n_init=10)
    labels = km.fit_predict(X)
    r_labels = labels[:len(real_feats)]
    g_labels = labels[len(real_feats):]
    p = np.bincount(r_labels, minlength=n_bins).astype(np.float64); p /= p.sum()
    q = np.bincount(g_labels, minlength=n_bins).astype(np.float64); q /= q.sum()
    lambdas = np.exp(np.linspace(np.log(1e-3), np.log(1e3), n_lambdas))
    P, R = prd_from_hist(p, q, lambdas)
    return P, R

def auc_prd(P: np.ndarray, R: np.ndarray) -> float:
    order = np.argsort(R)
    x = R[order]; y = P[order]
    mask = np.concatenate([[True], np.diff(x) > 1e-9])
    x = x[mask]; y = y[mask]
    return float(np.trapz(y, x)) if x.size and y.size else 0.0

def recall_at_precision(P: np.ndarray, R: np.ndarray, targets=(0.60,0.70,0.76)) -> Dict[float,float]:
    if P.size == 0 or R.size == 0:
        return {float(t): 0.0 for t in targets}
    order = np.argsort(P)
    x = P[order]; y = R[order]
    out = {}
    for t in targets:
        t = float(np.clip(t, 0.0, 1.0))
        if t <= x[0]: out[t] = float(y[0]); continue
        if t >= x[-1]: out[t] = float(y[-1]); continue
        j = np.searchsorted(x, t)
        x0,x1 = x[j-1],x[j]; y0,y1 = y[j-1],y[j]
        out[t] = float(y0 + (y1 - y0) * ((t - x0) / max(x1 - x0, 1e-12)))
    return out

# --------- Core (one method) ---------
def run_for_method(
    concept: str,
    real_dir: Path,
    imgs_root: Path,
    method: str,
    steps: int,
    batch_size: int,
    device: str,
    out_dir: Path,
    append_csv: bool,
    print_found: bool,
    clip_model_name: str,
    clip_download_root: str,
    clip_jit: bool,
):
    out_dir.mkdir(parents=True, exist_ok=True)

    # CHANGED: no prompts loading; we just scan all K-sets
    clipper = CLIPFeaturizer(model_name=clip_model_name, device=device, batch_size=batch_size,
                             download_root=clip_download_root, jit=clip_jit)

    real_imgs = list_images(real_dir)
    assert real_imgs, f"No real images in {real_dir}"
    real_feats = clipper.encode_paths(real_imgs)

    ksets = scan_ksets_all(imgs_root, print_found=print_found)  # CHANGED
    if not ksets:
        print(f"[WARN] No K-sets under {imgs_root} (regex match failed or empty).")
        return

    by_g_p: Dict[float, Dict[str, List[Tuple[np.ndarray,np.ndarray,float,Dict[float,float]]]]] = defaultdict(lambda: defaultdict(list))

    for ks in tqdm(ksets, desc=f"[{concept}|{method}] K-sets"):
        if ks["steps"] != steps:
            continue
        gen_feats = clipper.encode_paths(ks["images"])
        P, R = compute_prd_curve(real_feats, gen_feats, n_bins=10, n_lambdas=151, random_state=0)
        auc = auc_prd(P, R)
        iso = recall_at_precision(P, R, targets=(0.60,0.70,0.76))
        by_g_p[ks["guidance"]][ks["prompt"]].append((P,R,auc,iso))

    for g, mp in by_g_p.items():
        if not mp:
            continue
        prompt_stats = []
        for p, curves in mp.items():
            if not curves:
                continue
            Ps = np.stack([c[0] for c in curves], 0)
            R_ref = curves[0][1]
            P_mu = Ps.mean(0); P_sd = Ps.std(0)
            auc_mu = float(np.mean([c[2] for c in curves]))
            iso_mu = {k: float(np.mean([c[3][k] for c in curves])) for k in (0.60,0.70,0.76)}
            prompt_stats.append((R_ref, P_mu, P_sd, auc_mu, iso_mu))

        if not prompt_stats:
            continue

        R_ref = prompt_stats[0][0]
        P_mu = np.stack([t[1] for t in prompt_stats], 0).mean(0)
        P_sd = np.stack([t[2] for t in prompt_stats], 0).mean(0)
        auc_mu = float(np.mean([t[3] for t in prompt_stats]))
        r60 = float(np.mean([t[4][0.60] for t in prompt_stats]))
        r70 = float(np.mean([t[4][0.70] for t in prompt_stats]))
        r76 = float(np.mean([t[4][0.76] for t in prompt_stats]))

        rows = []
        for rr, pp_mu, pp_sd in zip(R_ref, P_mu, P_sd):
            rows.append({
                "method": method,
                "concept": concept,
                "guidance": g,
                "recall": float(rr),
                "precision_mu": float(pp_mu),
                "precision_sd": float(pp_sd),
                "auc_prd_mu": auc_mu,
                "rec_at_p0.60_mu": r60,
                "rec_at_p0.70_mu": r70,
                "rec_at_p0.76_mu": r76,
            })
        df = pd.DataFrame(rows).sort_values(by=["method","recall"]).reset_index(drop=True)
        out_csv = out_dir / f"exp1_{g}_{concept}.csv"
        if append_csv and out_csv.exists():
            df.to_csv(out_csv, mode="a", header=False, index=False)
            print(f"[APPEND] {out_csv} (+{len(df)} rows, method={method})")
        else:
            out_dir.mkdir(parents=True, exist_ok=True)
            df.to_csv(out_csv, index=False)
            print(f"[SAVE] {out_csv} (rows={len(df)}, method={method})")

# --------- path helpers ---------
def resolve_real_dir(dataset_root: Path, concept: str) -> Path:
    cands = [dataset_root / concept, dataset_root / concept.replace(" ", "_")]
    for p in cands:
        if list_images(p):
            return p
    return cands[0]

def autodetect_paths(outputs_root: Path, dataset_root: Path, method: str, concept: str):
    imgs_root    = outputs_root / f"{method}_{concept}" / "imgs"
    out_dir      = outputs_root / f"{method}_{concept}" / "eval"
    real_dir     = resolve_real_dir(dataset_root, concept)
    return real_dir, imgs_root, out_dir

# ----------------- CLI -----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--concept", type=str, required=True, help="e.g., bus")
    ap.add_argument("--methods", type=str, required=True, help="comma-separated, e.g., 'dpp,baseline1,baseline2'")

    ap.add_argument("--real_dir", type=Path, default=None)
    ap.add_argument("--imgs_root", type=Path, default=None)
    
    ap.add_argument("--prompts_json", type=Path, default=None, help="(ignored in this version)")
    ap.add_argument("--out_dir", type=Path, default=None)

    ap.add_argument("--outputs_root", type=Path, default=None, help="Base outputs root")
    ap.add_argument("--dataset_root", type=Path, default=None, help="Base dataset root")

    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--append_csv", type=int, default=1, help="1=append across methods in their own eval dirs")
    ap.add_argument("--print_found", type=int, default=0, help="1=print all parsed prompt prefixes under imgs_root")

    # CLIP local
    ap.add_argument("--clip_model_name", type=str, default="ViT-B/32")
    ap.add_argument("--clip_download_root", type=str, default="~/.cache/clip")
    ap.add_argument("--clip_jit", type=int, default=1)

    args = ap.parse_args()
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    assert methods, "No methods given"

    clip_root = str(Path(args.clip_download_root).expanduser()) if args.clip_download_root else None

    if args.prompts_json is not None:
        print("[INFO] --prompts_json is provided but ignored in this version.")

    for method in methods:
        if args.real_dir and args.imgs_root and args.out_dir:
            real_dir  = args.real_dir
            imgs_root = args.imgs_root
            out_dir   = args.out_dir
        else:
            assert args.outputs_root is not None and args.dataset_root is not None, \
                "未提供显式路径时，必须给 --outputs_root 与 --dataset_root 以自动推断。"
            real_dir, imgs_root, out_dir = autodetect_paths(
                args.outputs_root, args.dataset_root, method, args.concept
            )

        print(f"\n[RUN] concept={args.concept} | method={method}\n"
              f"      real_dir     = {real_dir}\n"
              f"      imgs_root    = {imgs_root}\n"
              f"      out_dir      = {out_dir}\n")

        run_for_method(
            concept=args.concept,
            real_dir=real_dir,
            imgs_root=imgs_root,
            method=method,
            steps=args.steps,
            batch_size=args.batch_size,
            device=args.device,
            out_dir=out_dir,
            append_csv=bool(args.append_csv),
            print_found=bool(args.print_found),
            clip_model_name=args.clip_model_name,
            clip_download_root=clip_root,
            clip_jit=bool(args.clip_jit),
        )

if __name__ == "__main__":
    main()
