"""Build a non-leaky, stratified fashion training set.

Sources:
  1. Marqo/marqo-GS-10M, sampled by query and rank bucket.
  2. Optional DeepFashion-InShop / DeepFashion-Multimodal auxiliary rows.

This deliberately does NOT use Marqo/fashion200k, so fashion200k remains a
clean benchmark.

Output schema, one JSON line per image-text pair:
  {
    "source": "gs10m" | "deepfashion-inshop" | "deepfashion-multimodal",
    "query": "...",
    "title": "...",
    "image_path": "...",
    "position": 7,
    "score_linear": 94,
    "rank_bucket": "top",
    "weight": 0.94
  }

GS stratification:
  - top: positions 1-10
  - mid: positions 11-40
  - tail: positions 41-100

For each query we keep up to N examples from each bucket, which prevents the
first few query clusters in the stream from dominating the training data.
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
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("stratified-fashion-set")

REPO_ROOT = Path(__file__).resolve().parents[1]
HF_CACHE = REPO_ROOT / "data" / "hf_cache"
OUT_DIR = REPO_ROOT / "data" / "processed" / "fashion_stratified_gs_df"
IMG_DIR = OUT_DIR / "images"
JSONL_PATH = OUT_DIR / "pairs.jsonl"

RANK_BUCKETS = {
    "top": (1, 10),
    "mid": (11, 40),
    "tail": (41, 100),
}

FASHION_KEYWORDS = {
    "activewear",
    "bag",
    "belt",
    "blazer",
    "blouse",
    "boots",
    "bra",
    "cardigan",
    "coat",
    "denim",
    "dress",
    "earmuffs",
    "fashion",
    "gloves",
    "hat",
    "heel",
    "hoodie",
    "jacket",
    "jeans",
    "jumpsuit",
    "leggings",
    "loafers",
    "pants",
    "purse",
    "sandal",
    "scarf",
    "shirt",
    "shoe",
    "shorts",
    "skirt",
    "sneaker",
    "sweater",
    "sweatshirt",
    "swimwear",
    "top",
    "trouser",
    "wallet",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-id", default="Marqo/marqo-GS-10M")
    p.add_argument("--split", default="in_domain")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-examined", type=int, default=500_000)
    p.add_argument("--max-queries", type=int, default=2_000)
    p.add_argument("--per-bucket", type=int, default=3)
    p.add_argument(
        "--min-fashion-keyword",
        action="store_true",
        help="Require query/title to include a broad fashion keyword.",
    )
    p.add_argument("--deepfashion-inshop", type=int, default=5_000)
    p.add_argument("--deepfashion-multimodal", type=int, default=5_000)
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    return p.parse_args()


def rank_bucket(position: int) -> str | None:
    for name, (lo, hi) in RANK_BUCKETS.items():
        if lo <= position <= hi:
            return name
    return None


def score_to_weight(position: int, score_linear: int | None) -> float:
    if score_linear is not None and score_linear > 0:
        return max(0.01, min(1.0, score_linear / 100.0))
    return max(0.01, min(1.0, 1.0 / max(position, 1)))


def is_fashion_row(query: str, title: str) -> bool:
    text = f"{query} {title}".lower()
    return any(keyword in text for keyword in FASHION_KEYWORDS)


def save_image(image: Any, key: str, img_dir: Path) -> str | None:
    try:
        rgb = image.convert("RGB")
    except Exception as exc:
        log.debug("skip undecodable image: %s", exc)
        return None

    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:24]
    img_path = img_dir / f"{digest}.jpg"
    if not img_path.exists():
        rgb.save(img_path, "JPEG", quality=92)
    return str(img_path)


def flush_query(
    query: str,
    rows_by_bucket: dict[str, list[dict[str, Any]]],
    rng: random.Random,
    per_bucket: int,
) -> list[dict[str, Any]]:
    if not query:
        return []

    kept: list[dict[str, Any]] = []
    for bucket in ("top", "mid", "tail"):
        rows = rows_by_bucket.get(bucket, [])
        if len(rows) > per_bucket:
            rows = rng.sample(rows, per_bucket)
        kept.extend(rows)
    return kept


def build_gs_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    from datasets import load_dataset

    rng = random.Random(args.seed)
    log.info("streaming %s split=%s", args.repo_id, args.split)
    ds = load_dataset(
        args.repo_id,
        split=args.split,
        streaming=True,
        cache_dir=str(HF_CACHE),
    )

    selected: list[dict[str, Any]] = []
    current_query = ""
    current_buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen_queries: set[str] = set()
    examined = 0
    rejected = Counter()
    t0 = time.time()

    for row in ds:
        examined += 1
        query = (row.get("query") or "").strip()
        title = (row.get("title") or "").strip()
        image = row.get("image")
        position = int(row.get("position") or 0)
        bucket = rank_bucket(position)

        if query != current_query and current_query:
            selected.extend(flush_query(current_query, current_buckets, rng, args.per_bucket))
            seen_queries.add(current_query)
            current_buckets = defaultdict(list)
            if len(seen_queries) >= args.max_queries:
                break

        current_query = query

        if not query or image is None or bucket is None:
            rejected["missing_or_bad_rank"] += 1
        elif args.min_fashion_keyword and not is_fashion_row(query, title):
            rejected["non_fashion_keyword"] += 1
        else:
            score_linear = row.get("score_linear")
            score_linear = int(score_linear) if score_linear is not None else None
            current_buckets[bucket].append(
                {
                    "source": "gs10m",
                    "query": query,
                    "title": title,
                    "position": position,
                    "score_linear": score_linear,
                    "rank_bucket": bucket,
                    "weight": score_to_weight(position, score_linear),
                    "image": image,
                    "product_id": str(row.get("product_id") or ""),
                }
            )

        if examined % 25_000 == 0:
            log.info(
                "examined=%d queries=%d selected_so_far=%d rejected=%s elapsed=%.1fs",
                examined,
                len(seen_queries),
                len(selected),
                dict(rejected),
                time.time() - t0,
            )
        if examined >= args.max_examined:
            break

    selected.extend(flush_query(current_query, current_buckets, rng, args.per_bucket))
    log.info(
        "GS stream done: examined=%d queries=%d selected=%d rejected=%s in %.1fs",
        examined,
        len(seen_queries),
        len(selected),
        dict(rejected),
        time.time() - t0,
    )
    return selected


def deepfashion_query(row: dict[str, Any]) -> str:
    parts = []
    for key in ("color", "category3", "category2", "category1"):
        value = row.get(key)
        if value and str(value).lower() != "none":
            parts.append(str(value).strip())
    if parts:
        return " ".join(parts)
    text = str(row.get("text") or "").strip()
    return text[:120] if text else ""


def add_deepfashion_rows(repo_id: str, limit: int, source: str) -> list[dict[str, Any]]:
    if limit <= 0:
        return []

    from datasets import load_dataset

    rows: list[dict[str, Any]] = []
    log.info("streaming %s limit=%d", repo_id, limit)
    ds = load_dataset(repo_id, split="data", streaming=True, cache_dir=str(HF_CACHE))
    for row in ds:
        if len(rows) >= limit:
            break
        image = row.get("image")
        query = deepfashion_query(row)
        if image is None or not query:
            continue
        rows.append(
            {
                "source": source,
                "query": query,
                "title": (row.get("text") or "").strip(),
                "position": 1,
                "score_linear": 100,
                "rank_bucket": "deepfashion",
                "weight": 1.0,
                "image": image,
                "product_id": str(row.get("item_ID") or ""),
            }
        )
    log.info("loaded %d rows from %s", len(rows), repo_id)
    return rows


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir
    img_dir = out_dir / "images"
    jsonl_path = out_dir / "pairs.jsonl"
    stats_path = out_dir / "stats.json"

    out_dir.mkdir(parents=True, exist_ok=True)
    img_dir.mkdir(parents=True, exist_ok=True)

    rows = build_gs_rows(args)
    rows.extend(
        add_deepfashion_rows(
            "Marqo/deepfashion-inshop",
            args.deepfashion_inshop,
            "deepfashion-inshop",
        )
    )
    rows.extend(
        add_deepfashion_rows(
            "Marqo/deepfashion-multimodal",
            args.deepfashion_multimodal,
            "deepfashion-multimodal",
        )
    )

    rng = random.Random(args.seed)
    rng.shuffle(rows)

    log.info("saving %d sampled rows", len(rows))
    written = 0
    source_counts = Counter()
    bucket_counts = Counter()
    query_counts = Counter()
    tmp_path = jsonl_path.with_suffix(".jsonl.tmp")

    with tmp_path.open("w") as f:
        for rec in rows:
            key = "|".join(
                [
                    rec["source"],
                    rec["query"],
                    rec.get("product_id") or "",
                    str(rec.get("position") or ""),
                    str(rec.get("score_linear") or ""),
                ]
            )
            image_path = save_image(rec["image"], key, img_dir)
            if image_path is None:
                continue

            out = {
                "source": rec["source"],
                "query": rec["query"],
                "title": rec.get("title") or "",
                "image_path": image_path,
                "position": rec["position"],
                "score_linear": rec["score_linear"],
                "rank_bucket": rec["rank_bucket"],
                "weight": rec["weight"],
            }
            f.write(json.dumps(out) + "\n")
            written += 1
            source_counts[out["source"]] += 1
            bucket_counts[out["rank_bucket"]] += 1
            query_counts[out["query"]] += 1

    os.replace(tmp_path, jsonl_path)

    stats = {
        "written": written,
        "sources": dict(source_counts),
        "rank_buckets": dict(bucket_counts),
        "unique_queries": len(query_counts),
        "top_queries": query_counts.most_common(20),
        "jsonl_path": str(jsonl_path),
        "images_dir": str(img_dir),
        "images_mb": sum(p.stat().st_size for p in img_dir.glob("*.jpg")) / 1e6,
        "args": vars(args) | {"out_dir": str(out_dir)},
    }
    with stats_path.open("w") as f:
        json.dump(stats, f, indent=2)

    log.info("wrote %d pairs -> %s", written, jsonl_path)
    log.info("stats -> %s", stats_path)
    log.info("source counts: %s", dict(source_counts))
    log.info("bucket counts: %s", dict(bucket_counts))
    log.info("unique queries: %d", len(query_counts))
    log.info("images dir size: %.1f MB", stats["images_mb"])


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
