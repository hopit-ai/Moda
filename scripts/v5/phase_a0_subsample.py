"""
Phase A.0 — Stratified 50K subsample from v4 pairs.

Picks N pairs balanced across the 9 v4 buckets, prioritizing high-score_linear
pairs within each bucket. Writes a new JSONL the v5 pipeline operates on.

Usage:
    python scripts/v5/phase_a0_subsample.py --n 50000
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = REPO_ROOT / "data" / "processed" / "v4_pattern_targeted" / "pairs.jsonl"
DEFAULT_OUTPUT = REPO_ROOT / "data" / "processed" / "v5_multifield" / "pairs_50k.jsonl"

# Target proportions across 9 v4 buckets (must sum to 1.0).
# Weighted toward fashion-relevant buckets, with a meaningful slice for
# long_description (the v4 bucket that was severely underfilled at 52 records)
# and non_fashion (helps generalization to atlas/polyvore lifestyle items).
BUCKET_TARGETS = {
    "apparel":          0.30,
    "short_title":      0.20,
    "accessories":      0.10,
    "footwear":         0.08,
    "color_centric":    0.08,
    "compound_attr":    0.08,
    "non_fashion":      0.08,
    "brand_product":    0.06,
    "long_description": 0.02,   # only 52 available; floor is whatever exists
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--n", type=int, default=50000)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.input} ...")
    pairs_by_bucket: dict[str, list[dict]] = defaultdict(list)
    n_total = 0
    with args.input.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            n_total += 1
            # Each pair has a `buckets` list — assign to the FIRST bucket
            # that matches our target keys, preferring narrower buckets first.
            buckets = r.get("buckets") or []
            chosen = None
            # Priority order: pick the bucket with the smallest target share first
            # so we don't fill long_description with pairs that also fit "apparel"
            for b in sorted(BUCKET_TARGETS.keys(), key=lambda k: BUCKET_TARGETS[k]):
                if b in buckets:
                    chosen = b
                    break
            if chosen is None:
                # Fall back to category1 or skip
                chosen = r.get("category1", "other")
                if chosen not in BUCKET_TARGETS:
                    continue
            pairs_by_bucket[chosen].append(r)

    print(f"Loaded {n_total:,} total pairs, {len(pairs_by_bucket)} usable buckets")
    for b in sorted(BUCKET_TARGETS, key=lambda k: -len(pairs_by_bucket[k])):
        print(f"  {b}: {len(pairs_by_bucket[b]):,} available")

    rng = random.Random(args.seed)

    # Sort each bucket by score_linear desc, then take the top per-bucket quota.
    # If a bucket is short, fill the deficit from over-supplied buckets at end.
    selected: list[dict] = []
    bucket_counts: Counter[str] = Counter()
    deficit = 0
    for b, target_frac in BUCKET_TARGETS.items():
        target_n = int(args.n * target_frac)
        avail = pairs_by_bucket.get(b, [])
        # Sort by score_linear desc, then shuffle ties for diversity
        avail.sort(key=lambda r: (-r.get("score_linear", 0), rng.random()))
        take = avail[:target_n]
        selected.extend(take)
        bucket_counts[b] = len(take)
        deficit += target_n - len(take)

    # Fill deficit from the remaining pool of buckets that had surplus
    if deficit > 0:
        leftover: list[dict] = []
        used_ids = {r["pair_id"] for r in selected}
        for b, lst in pairs_by_bucket.items():
            for r in lst:
                if r["pair_id"] not in used_ids:
                    leftover.append(r)
        leftover.sort(key=lambda r: (-r.get("score_linear", 0), rng.random()))
        fill = leftover[:deficit]
        selected.extend(fill)
        for r in fill:
            buckets = r.get("buckets") or []
            for b in BUCKET_TARGETS:
                if b in buckets:
                    bucket_counts[b] += 1
                    break

    # Trim/round to exactly args.n
    if len(selected) > args.n:
        rng.shuffle(selected)
        selected = selected[: args.n]

    print(f"\nSelected {len(selected):,} pairs:")
    for b, c in bucket_counts.most_common():
        print(f"  {b}: {c:,} ({100*c/len(selected):.1f}%)")

    # Shuffle the final list so the labeling order is bucket-mixed
    rng.shuffle(selected)

    print(f"\nWriting {args.output} ...")
    with args.output.open("w") as f:
        for r in selected:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Done. {len(selected):,} pairs written.")

    # Stats sidecar
    stats = {
        "n_total_input": n_total,
        "n_selected": len(selected),
        "seed": args.seed,
        "bucket_counts": dict(bucket_counts),
        "bucket_targets": BUCKET_TARGETS,
    }
    stats_path = args.output.with_suffix(".stats.json")
    stats_path.write_text(json.dumps(stats, indent=2))
    print(f"Stats: {stats_path}")


if __name__ == "__main__":
    main()
