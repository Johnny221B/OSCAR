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

NEW: Optional prompt filtering
  - Use --prompts to provide a list in CLI (exact or substring match)
  - Or --prompts-file to read prompts (one per line)
  - Auto-detect prompt column among ["prompt", "prompt_slug"] unless --prompt-col is given

Usage example:
  python guidance_summary_batch.py \
      --root /mnt/data6t/yyz/flow_grpo/flow_base/outputs \
      --concept truck \
      --methods dpp pg cads ours \
      --outdir /mnt/data6t/yyz/flow_grpo/flow_base/outputs \
      --prompts "a red truck on road" "blue truck in snow" \
      --match exact

  python guidance_summary_batch.py \
      --root /mnt/data6t/yyz/flow_grpo/flow_base/outputs \
      --concept truck \
      --methods dpp pg cads \
      --outdir . \
      --prompts-file ./keep_prompts.txt \
      --match contains --prompt-col prompt_slug
"""

import argparse
from pathlib import Path
from typing import List, Sequence, Dict, Optional
import pandas as pd
import re

DEFAULT_EXCLUDES = {
    "guidance", "seed", "steps", "k", "K", "n", "N",
    "prompt", "prompt_slug", "concept", "method", "model",
    "folder", "path"
}

AUTO_PROMPT_COL_CANDIDATES = ["prompt", "prompt_slug"]


def find_metrics(df: pd.DataFrame, excludes: Sequence[str] = DEFAULT_EXCLUDES) -> List[str]:
    """Pick numeric metric columns, excluding obvious identifiers."""
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    metric_cols = [c for c in numeric_cols if c not in set(excludes)]
    if not metric_cols:
        # Fallback: everything numeric except guidance
        metric_cols = [c for c in numeric_cols if c != "guidance"]
    return metric_cols


def summarize_one(df: pd.DataFrame, metric_cols: List[str]) -> pd.DataFrame:
    """Group by guidance and compute mean & std for the specified metric columns."""
    if "guidance" not in df.columns:
        raise KeyError("CSV缺少必需列：'guidance'")

    df = df.copy()
    df["guidance"] = pd.to_numeric(df["guidance"], errors="coerce")
    df = df.dropna(subset=["guidance"])

    if not metric_cols:
        raise ValueError("未找到任何可用的指标列（数值列）。")

    summary_mean = df.groupby("guidance", dropna=False)[metric_cols].mean().add_suffix("_mean")
    summary_std  = df.groupby("guidance", dropna=False)[metric_cols].std(ddof=1).add_suffix("_std")
    summary = pd.concat([summary_mean, summary_std], axis=1).reset_index()

    # Interleave mean/std per metric
    ordered_cols = ["guidance"]
    for c in metric_cols:
        ordered_cols += [f"{c}_mean", f"{c}_std"]
    summary = summary[ordered_cols].sort_values("guidance").reset_index(drop=True)
    return summary


def _auto_pick_prompt_col(df: pd.DataFrame) -> Optional[str]:
    for c in AUTO_PROMPT_COL_CANDIDATES:
        if c in df.columns:
            return c
    return None


def _normalize_case(s: pd.Series) -> pd.Series:
    # safer than lower() for some unicode cases
    return s.astype(str).str.casefold()


def filter_df_by_prompts(
    df: pd.DataFrame,
    prompts: Optional[Sequence[str]] = None,
    prompt_col: Optional[str] = None,
    match: str = "exact",
    case_sensitive: bool = False,
) -> pd.DataFrame:
    """
    Filter rows to only those whose `prompt_col` matches any in `prompts`.
    - match="exact": exact string match
    - match="contains": substring match (literal, non-regex)
    """
    if not prompts:
        return df

    if prompt_col is None:
        prompt_col = _auto_pick_prompt_col(df)
        if prompt_col is None:
            raise KeyError("开启了 prompt 过滤，但数据中找不到用于匹配的列（期望: 'prompt' 或 'prompt_slug'）。"
                           "可通过 --prompt-col 指定。")

    series = df[prompt_col].astype(str)
    if match not in {"exact", "contains"}:
        raise ValueError(f"--match 仅支持 'exact' 或 'contains'，收到: {match}")

    if not case_sensitive:
        series_cmp = _normalize_case(series)
        # normalize prompts list too
        prompt_set = [str(p).casefold() for p in prompts]
    else:
        series_cmp = series
        prompt_set = [str(p) for p in prompts]

    if match == "exact":
        mask = series_cmp.isin(prompt_set)
    else:  # contains
        # Build a single regex union of escaped prompts for efficient vectorized contains.
        # When case-insensitive, case parameter handles it.
        escaped = [re.escape(p) for p in prompts]
        if not escaped:
            return df.iloc[0:0]  # no prompts => empty
        pattern = "|".join(escaped)
        mask = series.str.contains(pattern, case=case_sensitive, regex=True)

    filtered = df[mask].copy()
    return filtered


def process(
    root: Path,
    concept: str,
    methods: Sequence[str],
    outdir: Path,
    excludes: Sequence[str] = DEFAULT_EXCLUDES,
    prompts: Optional[Sequence[str]] = None,
    prompt_col: Optional[str] = None,
    match: str = "exact",
    case_sensitive: bool = False,
) -> Dict[str, Path]:
    """
    For each method, read its CSV, optionally filter by prompts, and write a per-method summary.
    Also write a combined summary across all methods.
    """
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

        # Optional prompt filtering
        if prompts:
            before = len(df)
            try:
                df = filter_df_by_prompts(
                    df, prompts=prompts, prompt_col=prompt_col,
                    match=match, case_sensitive=case_sensitive
                )
            except KeyError as e:
                print(f"[WARN] {e} -> 跳过 method '{m}' 的过滤，按未过滤数据继续。")
            after = len(df)
            print(f"[INFO] Method='{m}': prompt 过滤前 {before} 行，过滤后 {after} 行。")
            if after == 0:
                print(f"[WARN] Method '{m}' 过滤后无样本，跳过该方法的汇总。")
                continue

        metric_cols = find_metrics(df, excludes)
        if not metric_cols:
            print(f"[WARN] Method '{m}' 未找到数值型指标列，跳过。")
            continue

        summary = summarize_one(df, metric_cols)
        summary.insert(0, "method", m)
        summary.insert(1, "concept", concept)

        # Save per-method summary
        per_path = outdir / f"summary_{m}_{concept}.csv"
        summary.to_csv(per_path, index=False)
        per_method_paths[m] = per_path

        combined_rows.append(summary)

    # Combined summary across methods
    comb_path = outdir / f"summary_{concept}_all_methods.csv"
    if combined_rows:
        combined = pd.concat(combined_rows, ignore_index=True)
        combined.to_csv(comb_path, index=False)
    else:
        print("[WARN] No method summaries were generated; combined file will not be written.")

    return {"combined": comb_path, **per_method_paths}


def _read_prompts_file(path: Path) -> List[str]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip("\n\r")
            if s:
                items.append(s)
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True,
                    help="Root containing <method>_<concept>/eval/<method>_<concept>.csv")
    ap.add_argument("--concept", type=str, required=True, help="Concept name, e.g., truck")
    ap.add_argument("--methods", nargs="+", required=True,
                    help="List of methods, e.g., dpp pg cads ours")
    ap.add_argument("--outdir", type=Path, default=Path("."),
                    help="Directory to save the summary CSVs")
    ap.add_argument("--excludes", type=str, default="",
                    help="Extra comma-separated non-metric columns to exclude")

    # ---- New: prompt filtering options ----
    ap.add_argument("--prompts", nargs="+", default=None,
                    help="Only keep rows whose prompt matches any of these (default exact match).")
    ap.add_argument("--prompts-file", type=Path, default=None,
                    help="Path to a text file with one prompt per line (combined with --prompts if both are given).")
    ap.add_argument("--prompt-col", type=str, default=None,
                    help="Which column to use as prompt key (auto-detect among 'prompt','prompt_slug' if omitted).")
    ap.add_argument("--match", type=str, choices=["exact", "contains"], default="exact",
                    help="Matching mode for prompts: exact or contains (substring).")
    ap.add_argument("--case-sensitive", action="store_true",
                    help="Use case-sensitive matching (default is case-insensitive).")

    args = ap.parse_args()

    excludes = set(DEFAULT_EXCLUDES)
    if args.excludes.strip():
        excludes |= {x.strip() for x in args.excludes.split(",") if x.strip()}

    # Build prompt list
    prompt_list: Optional[List[str]] = None
    if args.prompts or args.prompts_file:
        prompt_list = []
        if args.prompts:
            prompt_list.extend(args.prompts)
        if args.prompts_file:
            if not args.prompts_file.exists():
                raise FileNotFoundError(f"--prompts-file not found: {args.prompts_file}")
            prompt_list.extend(_read_prompts_file(args.prompts_file))

    paths = process(
        args.root, args.concept, args.methods, args.outdir, excludes,
        prompts=prompt_list, prompt_col=args.prompt_col,
        match=args.match, case_sensitive=args.case_sensitive
    )
    print("Done. Outputs:")
    for k, v in paths.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
