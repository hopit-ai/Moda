"""
Phase A.4 — Sample 5K stratified anchor indices for drift regularization.

Picks 5,000 pair_ids stratified across the v4 buckets, writes indices into the
student image cache (positions in the cache, not pair_ids) so the training
loop can drop them in directly.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PAIRS = REPO_ROOT / "data" / "processed" / "v5_multifield" / "pairs_50k.jsonl"
DEFAULT_INDEX = REPO_ROOT / "data" / "processed" / "v5_multifield" / "student_image_index.json"
DEFAULT_OUT = REPO_ROOT / "data" / "processed" / "v5_multifield" / "anchor_indices.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=Path, default=DEFAULT_PAIRS)
    ap.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    pairs = []
    with args.pairs.open() as f:
        for line in f:
            line = line.strip()
            if line:
                pairs.append(json.loads(line))

    image_index: dict[str, int] = json.loads(args.index.read_text())
    rng = random.Random(args.seed)

    # Stratify by primary bucket
    by_bucket: dict[str, list[str]] = defaultdict(list)
    for p in pairs:
        if p["pair_id"] not in image_index:
            continue
        b = (p.get("buckets") or ["other"])[0]
        by_bucket[b].append(p["pair_id"])

    selected: list[str] = []
    n_buckets = len(by_bucket)
    per_bucket = max(1, args.n // n_buckets)
    for b, ids in by_bucket.items():
        rng.shuffle(ids)
        selected.extend(ids[:per_bucket])
    rng.shuffle(selected)
    selected = selected[: args.n]

    cache_indices = sorted(image_index[pid] for pid in selected)
    args.out.write_text(json.dumps({
        "indices": cache_indices,
        "pair_ids": selected,
        "n": len(cache_indices),
        "seed": args.seed,
    }))
    print(f"Wrote {len(cache_indices)} anchor indices to {args.out}")


if __name__ == "__main__":
    main()
