#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Batch guidance-wise metrics aggregator.

Given a concept and multiple methods, this script will locate each CSV at:
  {ROOT}/<method>_<concept>/eval/<method>_<concept>.csv

For each file, it computes mean & std for all numeric metric columns (excluding obvious identifiers)
grouped by "guidance". It saves:
  - One combined CSV across methods: summary_<concept>_all_methods.csv
  - One CSV per method: summary_<method>_<concept>.csv
"""

import argparse
from pathlib import Path
from typing import List, Sequence, Dict
import pandas as pd


DEFAULT_EXCLUDES = {
    "guidance", "seed", "steps", "k", "K", "n", "N",
    "prompt", "prompt_slug", "concept", "method", "model",
    "folder", "path"
}


def find_metrics(df: pd.DataFrame, excludes: Sequence[str] = DEFAULT_EXCLUDES) -> List[str]:
    """Pick numeric metric columns, excluding obvious identifiers."""
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    metric_cols = [c for c in numeric_cols if c not in set(excludes)]
    if not metric_cols:
        metric_cols = [c for c in numeric_cols if c != "guidance"]
    return metric_cols


def summarize_one(df: pd.DataFrame, metric_cols: List[str]) -> pd.DataFrame:
    """Group by guidance and compute mean & std for the specified metric columns."""
    if "guidance" not in df.columns:
        raise KeyError("Missing required column: 'guidance'")

    df = df.copy()
    df["guidance"] = pd.to_numeric(df["guidance"], errors="coerce")
    df = df.dropna(subset=["guidance"])

    if not metric_cols:
        raise ValueError("No numeric metric columns found.")

    summary_mean = df.groupby("guidance", dropna=False)[
        metric_cols].mean().add_suffix("_mean")
    summary_std = df.groupby("guidance", dropna=False)[
        metric_cols].std(ddof=1).add_suffix("_std")
    summary = pd.concat([summary_mean, summary_std], axis=1).reset_index()

    ordered_cols = ["guidance"]
    for c in metric_cols:
        ordered_cols += [f"{c}_mean", f"{c}_std"]
    summary = summary[ordered_cols].sort_values(
        "guidance").reset_index(drop=True)
    return summary


def process(
    root: Path,
    concept: str,
    methods: Sequence[str],
    outdir: Path,
    excludes: Sequence[str] = DEFAULT_EXCLUDES,
) -> Dict[str, Path]:
    """Read each method CSV, write per-method summary and a combined summary."""
    root = Path(root)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    combined_rows = []
    per_method_paths: Dict[str, Path] = {}

    for m in methods:
        csv_path = root / f"{m}_{concept}" / "eval" / f"{m}_{concept}.csv"
        if not csv_path.exists():
            print(f"[WARN] Missing CSV for method '{m}': {csv_path}")
            continue

        df = pd.read_csv(csv_path)
        metric_cols = find_metrics(df, excludes)
        summary = summarize_one(df, metric_cols)
        summary.insert(0, "method", m)
        summary.insert(1, "concept", concept)

        per_path = outdir / f"summary_{m}_{concept}.csv"
        summary.to_csv(per_path, index=False)
        per_method_paths[m] = per_path

        combined_rows.append(summary)

    comb_path = outdir / f"summary_{concept}_all_methods.csv"
    if combined_rows:
        combined = pd.concat(combined_rows, ignore_index=True)
        combined.to_csv(comb_path, index=False)
    else:
        print(
            "[WARN] No method summaries were generated; combined file will not be written.")

    return {"combined": comb_path, **per_method_paths}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True,
                    help="Root containing <method>_<concept>/eval/<method>_<concept>.csv")
    ap.add_argument("--concept", type=str, required=True,
                    help="Concept name, e.g., truck")
    ap.add_argument("--methods", nargs="+", required=True,
                    help="List of methods, e.g., dpp pg cads ours")
    ap.add_argument("--outdir", type=Path, default=Path("."),
                    help="Directory to save the summary CSVs")
    ap.add_argument("--excludes", type=str, default="",
                    help="Extra comma-separated non-metric columns to exclude")
    args = ap.parse_args()

    excludes = set(DEFAULT_EXCLUDES)
    if args.excludes.strip():
        excludes |= {x.strip() for x in args.excludes.split(",") if x.strip()}

    paths = process(args.root, args.concept,
                    args.methods, args.outdir, excludes)
    print("Done. Outputs:")
    for k, v in paths.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
