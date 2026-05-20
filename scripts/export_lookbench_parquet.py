#!/usr/bin/env python3
"""
Export local LookBench Arrow trees to Parquet for the official look-bench repo
(``data/raw/lookbench/look-bench``) BaseDataset loader.

Adds an integer ``label`` column (consistent across query and gallery) from ``item_ID``.

Usage:
  python scripts/export_lookbench_parquet.py
  python scripts/export_lookbench_parquet.py --subsets real_studio_flat
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from datasets import Dataset, load_from_disk

_REPO = Path(__file__).resolve().parent.parent
SRC = _REPO / "data" / "raw" / "lookbench" / "datasets"
DST = _REPO / "data" / "raw" / "lookbench" / "parquet_export"

DEFAULT_CFGS = (
    "real_studio_flat",
    "aigen_studio",
    "real_streetlook",
    "aigen_streetlook",
)


def export_one(cfg: str, dst_root: Path) -> None:
    path = SRC / cfg
    if not path.is_dir():
        raise FileNotFoundError(path)

    d = load_from_disk(str(path))
    q_df = d["query"].to_pandas()
    g_df = d["gallery"].to_pandas()

    all_ids = pd.unique(pd.concat([q_df["item_ID"], g_df["item_ID"]], ignore_index=True))
    lab_map = {x: i for i, x in enumerate(all_ids)}
    q_df = q_df.copy()
    g_df = g_df.copy()
    q_df["label"] = q_df["item_ID"].map(lab_map)
    g_df["label"] = g_df["item_ID"].map(lab_map)

    out_dir = dst_root / cfg
    out_dir.mkdir(parents=True, exist_ok=True)
    Dataset.from_pandas(q_df).to_parquet(str(out_dir / "query.parquet"))
    Dataset.from_pandas(g_df).to_parquet(str(out_dir / "gallery.parquet"))
    print(f"Wrote {out_dir}/query.parquet ({len(q_df)} rows)")
    print(f"Wrote {out_dir}/gallery.parquet ({len(g_df)} rows)")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--subsets", nargs="*", default=list(DEFAULT_CFGS))
    p.add_argument("--dst", type=Path, default=DST)
    args = p.parse_args()
    args.dst.mkdir(parents=True, exist_ok=True)
    for cfg in args.subsets:
        export_one(cfg, args.dst)
    print("\nPoint look-bench configs/datasets YAML data_root at:", args.dst)


if __name__ == "__main__":
    main()
