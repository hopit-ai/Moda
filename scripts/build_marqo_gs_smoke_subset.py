"""
Sample a small (image, query, weight) training subset from
``Marqo/marqo-GS-10M`` (in_domain split) for SigLIP L/16-384 fine-tuning.

Why this approach (HF datasets streaming) instead of the S3 tar:
  - The S3 tar (marqo-gs-dataset.tar) contains only METADATA + relative paths
    like /MarqoData/google_shopping/images_wfash/<id>.webp; the actual images
    are in a separate ~20 GB tar that doesn't have a public direct URL.
  - The HuggingFace mirror Marqo/marqo-GS-10M ships parquets with images
    embedded as PIL objects, so we can stream a single ~100 MB shard, decode
    images in-memory, save the ones we keep, and produce triplets.jsonl with
    zero 404s. Far simpler and faster than the tar+URL fetching dance.

Inputs:
  - HuggingFace dataset Marqo/marqo-GS-10M (auto-cached under data/hf_cache)

Outputs:
  - data/processed/marqo_gs_wfash_subset/triplets.jsonl
      one record per kept pair: {query, image_path, weight, title, position}
  - data/processed/marqo_gs_wfash_subset/images/<sha256>.jpg

Usage:
  .venv/bin/python scripts/build_marqo_gs_smoke_subset.py --num-pairs 5000
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import random
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("build-marqo-gs-subset")

REPO_ROOT = Path(__file__).resolve().parents[1]
HF_CACHE = REPO_ROOT / "data" / "hf_cache"
OUT_DIR = REPO_ROOT / "data" / "processed" / "marqo_gs_wfash_subset"
IMG_DIR = OUT_DIR / "images"
TRIPLETS_PATH = OUT_DIR / "triplets.jsonl"


def position_to_weight(position: int, score_linear: int | None = None) -> float:
    """Convert Marqo's per-pair ranking signal into a [0,1] training weight.

    Marqo-GS exposes two ranking signals per (query, product) pair:
      - position: 1-based Google Shopping rank (lower = more relevant)
      - score_linear: int 0..99, where higher is more relevant in their
        normalised scheme

    We prefer score_linear when present (it's already normalised for top-K).
    Falls back to 1/position. Result clamped to [0.05, 1.0] so even weak
    pairs contribute a non-negligible gradient.
    """
    if score_linear is not None and score_linear > 0:
        w = score_linear / 99.0
    else:
        w = 1.0 / max(position, 1)
    return max(0.05, min(1.0, float(w)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--num-pairs", type=int, default=5000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--max-examined",
        type=int,
        default=200_000,
        help="Stop after streaming this many rows (a single in_domain shard ~100k rows).",
    )
    p.add_argument(
        "--shards",
        type=int,
        default=2,
        help="How many in_domain-*.parquet shards to stream from (each ~100 MB).",
    )
    p.add_argument(
        "--repo-id",
        default="Marqo/marqo-GS-10M",
        help="HuggingFace dataset repo (Marqo/marqo-GS-10M is the full 10M, "
        "Marqo/marqo-gs-woman-fashion is wfash but only ships zero_shot split).",
    )
    p.add_argument(
        "--query-min-len",
        type=int,
        default=2,
        help="Reject queries shorter than this many chars.",
    )
    p.add_argument(
        "--use-title",
        action="store_true",
        help="Also write the product 'title' to the triplet so the trainer "
        "can mix in the longer text field if it wants.",
    )
    args = p.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    IMG_DIR.mkdir(parents=True, exist_ok=True)

    from datasets import load_dataset

    rng = random.Random(args.seed)

    log.info("streaming %s (in_domain split, %d shard(s))", args.repo_id, args.shards)
    data_files = [f"data/in_domain-{i}.parquet" for i in range(args.shards)]
    ds = load_dataset(
        args.repo_id,
        data_files=data_files,
        split="train",
        streaming=True,
        cache_dir=str(HF_CACHE),
    )

    # Reservoir sample over the streamed rows. We keep the rare full record
    # (image is a PIL object so each kept record holds memory; that's fine for
    # n<=10k, would tighten if scaling to 100k+).
    target = args.num_pairs
    kept: list[dict] = []
    examined = 0
    rejected = 0
    t0 = time.time()
    for row in ds:
        examined += 1
        q = (row.get("query") or "").strip()
        if len(q) < args.query_min_len or row.get("image") is None:
            rejected += 1
            if examined >= args.max_examined:
                break
            continue

        rec = {
            "query": q,
            "title": (row.get("title") or "").strip() if args.use_title else None,
            "position": int(row.get("position", 0)),
            "score_linear": int(row.get("score_linear", 0)),
            "image": row["image"],  # PIL Image
        }
        if len(kept) < target:
            kept.append(rec)
        else:
            j = rng.randrange(examined)
            if j < target:
                kept[j] = rec

        if examined % 5000 == 0:
            log.info(
                "  examined=%d  kept=%d  rejected=%d  elapsed=%.1fs",
                examined, len(kept), rejected, time.time() - t0,
            )
        if examined >= args.max_examined:
            break

    log.info(
        "stream done: examined=%d, kept=%d, rejected=%d in %.1fs",
        examined, len(kept), rejected, time.time() - t0,
    )

    if not kept:
        log.error("no records sampled — inspect dataset.")
        sys.exit(2)

    # Save images + write triplets.jsonl. Images go to JPEG (smaller, faster
    # to load than WebP for training) keyed by content hash for dedup.
    log.info("saving %d images and writing triplets ...", len(kept))
    written = 0
    tmp_path = TRIPLETS_PATH.with_suffix(".jsonl.tmp")
    with tmp_path.open("w") as f:
        for rec in kept:
            try:
                img = rec["image"].convert("RGB")
            except Exception as e:
                log.warning("skip image (%s)", e)
                continue
            # Hash on (query, position) pair so duplicates with same product
            # but different rank still get unique files (rare but possible).
            key = f"{rec['query']}|{rec['position']}|{rec.get('score_linear')}"
            h = hashlib.sha256(key.encode("utf-8")).hexdigest()[:24]
            img_path = IMG_DIR / f"{h}.jpg"
            if not img_path.exists():
                img.save(img_path, "JPEG", quality=92)
            out = {
                "query": rec["query"],
                "image_path": str(img_path),
                "weight": position_to_weight(rec["position"], rec.get("score_linear")),
                "position": rec["position"],
                "score_linear": rec.get("score_linear"),
                "split": "train",
            }
            if args.use_title and rec.get("title"):
                out["title"] = rec["title"]
            f.write(json.dumps(out) + "\n")
            written += 1
    os.replace(tmp_path, TRIPLETS_PATH)
    log.info("wrote %d triplets -> %s", written, TRIPLETS_PATH)
    log.info("images dir size: %.1f MB", sum(p.stat().st_size for p in IMG_DIR.glob("*.jpg")) / 1e6)


if __name__ == "__main__":
    main()
