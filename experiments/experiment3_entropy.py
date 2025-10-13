#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Export normalized cluster entropy (H/log k) for each {method, guidance} under one concept.

This version:
- CHANGED: No prompts_json filtering; scan ALL K-sets under imgs/.
- CHANGED: You can pass either:
    A) --mc_roots {outputs_root}/{method}_{concept}[, ...]   (or their .../imgs)
    B) --outputs_root + --methods + --concept  (auto paths)
- CHANGED: If --out_dir is omitted, each method writes to its own {method}_{concept}/eval/.
- K-set dirname formats supported:
    * new: {prompt}_seed{seed}_g{guidance}_s{steps}
    * old: {prompt}_{seed}_{guidance}_{steps}

Aggregation
-----------
For each guidance:
  per-Kset entropy -> (seed-mean per prompt) -> macro-mean over prompts
Output columns: method, concept, guidance, k, entropy_mean, entropy_std
"""

import argparse, re, gc
from pathlib import Path
from typing import Dict, List
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

# ===== CLIP features =====
import torch
import torch.nn.functional as F
import clip  # pip install git+https://github.com/openai/CLIP.git
from sklearn.cluster import KMeans

# ---------- utils ----------
IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
def is_image(p: Path) -> bool: return p.suffix.lower() in IMG_EXTS
def list_images(d: Path) -> List[Path]:
    return sorted([p for p in d.iterdir() if p.is_file() and is_image(p)]) if d.is_dir() else []

def free_mem():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def _normalize_prompt(s: str) -> str:
    # for grouping same prompt across seeds
    s = s.replace(" ", "_")
    s = re.sub(r"_+", "_", s).strip("_")
    return s.lower()

# ---- parse K-set dirnames (new & old formats) ----
NEW_RE = re.compile(r"^(?P<prompt>.+)_seed(?P<seed>\d+)_g(?P<guidance>[-+]?\d*\.?\d+)_s(?P<steps>\d+)$")
OLD_RE = re.compile(r"^(?P<prompt>.+)_(?P<seed>\d+?)_(?P<guidance>[-+]?\d*\.?\d+?)_(?P<steps>\d+?)$")

def parse_kset_dirname(name: str):
    for pat in (NEW_RE, OLD_RE):
        m = pat.match(name)
        if m:
            return {
                "prompt_raw": m.group("prompt"),
                "seed": int(m.group("seed")),
                "guidance": float(m.group("guidance")),
                "steps": int(m.group("steps")),
            }
    return None

# ---- CHANGED: scan ALL K-sets under imgs_root (no prompt filtering) ----
def scan_ksets_all(imgs_root: Path, target_guidances: List[float], steps: int, debug: bool=False) -> List[Dict]:
    out, unmatched = [], []
    gd_set = set(float(g) for g in target_guidances) if target_guidances else None
    if not imgs_root.exists():
        if debug: print(f"[DEBUG] imgs_root not found: {imgs_root}")
        return out
    for d in sorted(imgs_root.iterdir()):
        if not d.is_dir():
            continue
        info = parse_kset_dirname(d.name)
        if not info:
            unmatched.append((d.name, "regex_mismatch"))
            continue
        if gd_set is not None and info["guidance"] not in gd_set:
            continue
        if steps is not None and info["steps"] != steps:
            continue
        imgs = list_images(d)
        if not imgs:
            unmatched.append((d.name, "no_images"))
            continue
        out.append({
            "dir": d,
            "folder": d.name,
            "prompt_raw": info["prompt_raw"],
            "prompt": _normalize_prompt(info["prompt_raw"]),  # grouping key
            "seed": info["seed"],
            "guidance": info["guidance"],
            "steps": info["steps"],
            "images": imgs,
            "n_images": len(imgs),
        })
    if debug:
        print(f"[DEBUG] scan_ksets_all: matched={len(out)}, unmatched={len(unmatched)}")
        for name, why in unmatched[:30]:
            print(f"  - UNMATCHED: {name}  | {why}")
        if len(unmatched) > 30:
            print(f"  ... ({len(unmatched)-30} more)")
    return out

# ---- CLIP featurizer & kmeans (fit on REAL only) ----
class CLIPFeaturizer:
    def __init__(self, model_name="ViT-L/14@336px", device=None, batch_size=64):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        self.model.eval()
        self.bs = batch_size
    @torch.no_grad()
    def encode_paths(self, paths: List[Path]) -> np.ndarray:
        if not paths:
            return np.zeros((0, 512), dtype=np.float32)
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
        return np.concatenate(feats, axis=0)

def kmeans_centers(real_feats: np.ndarray, k: int, seed: int = 0) -> np.ndarray:
    km = KMeans(n_clusters=k, random_state=seed, n_init=10)
    km.fit(real_feats)
    C = km.cluster_centers_
    C = C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-12)
    return C

# ---- entropy on one K-set ----
def entropy_for_kset(gen_feats: np.ndarray, centers: np.ndarray) -> float:
    if gen_feats.size == 0:
        return 0.0
    sims = gen_feats @ centers.T
    nearest = np.argmax(sims, axis=1)  # same as argmin of (1 - sims)
    k = centers.shape[0]
    counts = np.bincount(nearest, minlength=k).astype(np.float64)
    p = counts / max(counts.sum(), 1.0)
    eps = 1e-12
    H = -(p * np.log(p + eps)).sum()
    H_norm = H / (np.log(k) + eps)
    return float(H_norm)

# ---- aggregate: seed-mean per prompt -> macro over prompts ----
def aggregate_entropy(by_g_p_seed: Dict[float, Dict[str, List[float]]]) -> Dict[float, Dict[str, float]]:
    out = {}
    for g, mp in by_g_p_seed.items():
        per_prompt_mean = []
        for p, lst in mp.items():
            per_prompt_mean.append(np.mean(lst))
        ent_mu = float(np.mean(per_prompt_mean)) if per_prompt_mean else 0.0
        ent_sd = float(np.std (per_prompt_mean)) if per_prompt_mean else 0.0
        out[g] = {"entropy_mean": ent_mu, "entropy_std": ent_sd}
    return out

# ---- helpers: locate imgs/ and default eval/ ----
def resolve_imgs_root(method_concept_root: Path) -> Path:
    if (method_concept_root / "imgs").is_dir():
        return method_concept_root / "imgs"
    if method_concept_root.name == "imgs" and method_concept_root.is_dir():
        return method_concept_root
    raise FileNotFoundError(f"Cannot locate imgs/ under {method_concept_root}")

def default_out_dir(method_concept_root: Path) -> Path:
    return method_concept_root / "eval"

def infer_method_name(method_concept_root: Path, concept: str) -> str:
    name = method_concept_root.name
    suffix = f"_{concept}"
    if name.endswith(suffix):
        return name[: -len(suffix)]
    return name  # fallback

# ---- main driver ----
def run(
    real_dir: Path,
    # Mode A: explicit list of {method}_{concept} roots (or their imgs/)
    method_concept_roots: List[Path],
    # Mode B: outputs_root + methods + concept
    outputs_root: Path,
    methods: List[str],
    concept: str,
    guidances: List[float],
    steps: int,
    k: int,
    batch_size: int,
    device: str,
    out_dir: Path,   # if provided -> single combined CSV; else per-method CSV in {method}_{concept}/eval
    debug: bool,
    show_matched: bool,
    dump_matched: Path,
):
    feat = CLIPFeaturizer(device=device, batch_size=batch_size)

    real_imgs = list_images(real_dir)
    assert len(real_imgs) > 0, f"No real images in {real_dir}"
    real_feats = feat.encode_paths(real_imgs)
    centers = kmeans_centers(real_feats, k=k, seed=0)

    # Build task list
    tasks = []

    # A: explicit roots
    for mc_root in method_concept_roots or []:
        mc_root = mc_root.resolve()
        imgs_root = resolve_imgs_root(mc_root)
        method = infer_method_name(mc_root, concept)
        odir = (out_dir if out_dir is not None else default_out_dir(mc_root))
        odir.mkdir(parents=True, exist_ok=True)
        tasks.append((method, mc_root, imgs_root, odir))

    # B: outputs_root + methods
    if outputs_root is not None and methods:
        for m in methods:
            mc_root = (outputs_root / f"{m}_{concept}").resolve()
            imgs_root = resolve_imgs_root(mc_root)
            odir = (out_dir if out_dir is not None else default_out_dir(mc_root))
            odir.mkdir(parents=True, exist_ok=True)
            tasks.append((m, mc_root, imgs_root, odir))

    if not tasks:
        raise ValueError("No method/concept roots to process. Provide --mc_roots or --outputs_root+--methods+--concept.")

    combined_rows = []  # if --out_dir provided, we write a single CSV later

    # Process each task
    for method, mc_root, imgs_root, odir in tasks:
        print(f"\n[RUN] concept={concept} | method={method}\n"
              f"      real_dir  = {real_dir}\n"
              f"      imgs_root = {imgs_root}\n"
              f"      out_dir   = {odir}\n")

        ksets = scan_ksets_all(imgs_root, guidances, steps, debug=debug)
        if not ksets:
            print(f"[WARN] no K-sets under {imgs_root} that match guidances/steps")
            continue

        # stats about matches
        c_guid = Counter([ks["guidance"] for ks in ksets])
        total_imgs = sum(ks["n_images"] for ks in ksets)
        print(f"[INFO] {concept}|{method}: matched K-sets = {len(ksets)} "
              f"(images={total_imgs}, by_guidance={dict(sorted(c_guid.items()))})")

        if show_matched:
            print("[MATCHED] folder | prompt_raw(norm) | seed | g | s | #imgs")
            for ks in sorted(ksets, key=lambda x: (x["guidance"], x["prompt"], x["seed"], x["folder"])):
                print(f"  - {ks['folder']} | {ks['prompt_raw']} ({ks['prompt']}) | "
                      f"{ks['seed']} | {ks['guidance']} | {ks['steps']} | {ks['n_images']}")

        if dump_matched is not None:
            dump_dir = Path(dump_matched); dump_dir.mkdir(parents=True, exist_ok=True)
            _dfm = pd.DataFrame([{
                "method": method,
                "concept": concept,
                "folder": ks["folder"],
                "prompt_raw": ks["prompt_raw"],
                "prompt_norm": ks["prompt"],
                "seed": ks["seed"],
                "guidance": ks["guidance"],
                "steps": ks["steps"],
                "n_images": ks["n_images"],
                "abs_dir": str(ks["dir"]),
            } for ks in ksets]).sort_values(["guidance","prompt_norm","seed","folder"])
            out_csv_list = dump_dir / f"matched_{method}_{concept}.csv"
            _dfm.to_csv(out_csv_list, index=False)
            print(f"[SAVE] matched list -> {out_csv_list}")

        # g -> prompt_norm -> [entropy over seeds]
        by_g_p_seed: Dict[float, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))

        for ks in tqdm(ksets, desc=f"[{concept}|{method}] K-sets"):
            gen_feats = feat.encode_paths(ks["images"])
            ent = entropy_for_kset(gen_feats, centers)
            by_g_p_seed[ks["guidance"]][ks["prompt"]].append(ent)

        stats = aggregate_entropy(by_g_p_seed)
        rows = [{
            "method": method,
            "concept": concept,
            "guidance": float(g),
            "k": int(k),
            "entropy_mean": float(st["entropy_mean"]),
            "entropy_std":  float(st["entropy_std"]),
        } for g, st in stats.items()]

        if out_dir is None:
            # per-method CSV to its own eval/
            df = pd.DataFrame(rows).sort_values(by=["method","guidance"]).reset_index(drop=True)
            out_csv = odir / f"exp3_entropy_{concept}.csv"
            df.to_csv(out_csv, index=False)
            print(f"[SAVE] {out_csv} (rows={len(df)})")
        else:
            combined_rows.extend(rows)

    if out_dir is not None and combined_rows:
        df_all = pd.DataFrame(combined_rows).sort_values(by=["method","guidance"]).reset_index(drop=True)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_csv = out_dir / f"exp3_entropy_{concept}.csv"
        df_all.to_csv(out_csv, index=False)
        print(f"[SAVE] {out_csv} (rows={len(df_all)})")

# ---- CLI ----
def parse_floats(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real_dir", type=Path, required=True, help="real reference folder for ONE concept")

    # Mode A: pass one or more {outputs_root}/{method}_{concept} roots (or their imgs)
    ap.add_argument("--mc_roots", type=str, default="",
                    help="comma-separated paths to {method}_{concept} roots (or .../imgs)")

    # Mode B: auto paths
    ap.add_argument("--outputs_root", type=Path, default=None, help="root containing {method}_{concept}/")
    ap.add_argument("--concept", type=str, default=None)
    ap.add_argument("--methods", type=str, default="", help="comma-separated methods")

    ap.add_argument("--guidances", type=str, required=True, help="e.g. '3.0,5.0,7.5'")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--device", type=str, default=None)

    # Output: if omitted -> per-method {method}_{concept}/eval/
    ap.add_argument("--out_dir", type=Path, default=None,
                    help="if provided, write ONE combined CSV here; else per-method CSVs")

    ap.add_argument("--debug", type=int, default=0)
    ap.add_argument("--show_matched", type=int, default=1)
    ap.add_argument("--dump_matched", type=Path, default=None)
    args = ap.parse_args()

    # Parse inputs
    mc_roots = [Path(x.strip()) for x in args.mc_roots.split(",") if x.strip()]
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    guidances = parse_floats(args.guidances)

    if not mc_roots and (args.outputs_root is None or not methods or not args.concept):
        raise ValueError("Provide either --mc_roots, or --outputs_root + --methods + --concept.")

    run(
        real_dir=args.real_dir,
        method_concept_roots=mc_roots,
        outputs_root=args.outputs_root,
        methods=methods,
        concept=args.concept,
        guidances=guidances,
        steps=args.steps,
        k=args.k,
        batch_size=args.batch_size,
        device=args.device,
        out_dir=args.out_dir,
        debug=bool(args.debug),
        show_matched=bool(args.show_matched),
        dump_matched=args.dump_matched,
    )

if __name__ == "__main__":
    main()