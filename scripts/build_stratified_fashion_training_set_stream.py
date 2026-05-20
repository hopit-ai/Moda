"""Memory-safe stratified builder for GS-10M + DeepFashion.

This is the streaming version of build_stratified_fashion_training_set.py. It
writes sampled rows immediately instead of storing PIL images in memory.

No fashion200k data is used.
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
log = logging.getLogger("stream-fashion-set")

REPO_ROOT = Path(__file__).resolve().parents[1]
HF_CACHE = REPO_ROOT / "data" / "hf_cache"
OUT_DIR = REPO_ROOT / "data" / "processed" / "fashion_stratified_gs_df"

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
    "boot",
    "bra",
    "cardigan",
    "coat",
    "denim",
    "dress",
    "earmuff",
    "fashion",
    "glove",
    "hat",
    "heel",
    "hoodie",
    "jacket",
    "jean",
    "legging",
    "loafer",
    "pant",
    "purse",
    "sandal",
    "scarf",
    "shirt",
    "shoe",
    "short",
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
    p.add_argument("--max-examined", type=int, default=250_000)
    p.add_argument("--max-queries", type=int, default=5_000)
    p.add_argument("--per-bucket", type=int, default=5)
    p.add_argument("--deepfashion-inshop", type=int, default=10_000)
    p.add_argument("--deepfashion-multimodal", type=int, default=10_000)
    p.add_argument("--min-fashion-keyword", action="store_true")
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


def deepfashion_query(row: dict[str, Any]) -> str:
    parts: list[str] = []
    for key in ("color", "category3", "category2", "category1"):
        value = row.get(key)
        if value and str(value).lower() != "none":
            parts.append(str(value).strip())
    if parts:
        return " ".join(parts)
    text = str(row.get("text") or "").strip()
    return text[:120] if text else ""


class PairWriter:
    def __init__(self, out_dir: Path) -> None:
        self.out_dir = out_dir
        self.img_dir = out_dir / "images"
        self.jsonl_path = out_dir / "pairs.jsonl"
        self.stats_path = out_dir / "stats.json"
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.img_dir.mkdir(parents=True, exist_ok=True)
        self.tmp_path = self.jsonl_path.with_suffix(".jsonl.tmp")
        self.handle = self.tmp_path.open("w")
        self.written = 0
        self.source_counts: Counter[str] = Counter()
        self.bucket_counts: Counter[str] = Counter()
        self.query_counts: Counter[str] = Counter()

    def save_image(self, image: Any, key: str) -> str | None:
        try:
            rgb = image.convert("RGB")
        except Exception as exc:
            log.debug("skip undecodable image: %s", exc)
            return None
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:24]
        img_path = self.img_dir / f"{digest}.jpg"
        if not img_path.exists():
            rgb.save(img_path, "JPEG", quality=92)
        return str(img_path)

    def write(self, rec: dict[str, Any]) -> None:
        key = "|".join(
            [
                rec["source"],
                rec["query"],
                str(rec.get("product_id") or ""),
                str(rec.get("position") or ""),
                str(rec.get("score_linear") or ""),
            ]
        )
        image_path = self.save_image(rec["image"], key)
        if image_path is None:
            return
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
        self.handle.write(json.dumps(out) + "\n")
        self.written += 1
        self.source_counts[out["source"]] += 1
        self.bucket_counts[out["rank_bucket"]] += 1
        self.query_counts[out["query"]] += 1

    def close(self, args: argparse.Namespace) -> None:
        self.handle.close()
        os.replace(self.tmp_path, self.jsonl_path)
        stats = {
            "written": self.written,
            "sources": dict(self.source_counts),
            "rank_buckets": dict(self.bucket_counts),
            "unique_queries": len(self.query_counts),
            "top_queries": self.query_counts.most_common(20),
            "jsonl_path": str(self.jsonl_path),
            "images_dir": str(self.img_dir),
            "images_mb": sum(p.stat().st_size for p in self.img_dir.glob("*.jpg")) / 1e6,
            "args": vars(args) | {"out_dir": str(args.out_dir)},
        }
        with self.stats_path.open("w") as f:
            json.dump(stats, f, indent=2)
        log.info("wrote %d pairs -> %s", self.written, self.jsonl_path)
        log.info("stats -> %s", self.stats_path)
        log.info("source counts: %s", dict(self.source_counts))
        log.info("bucket counts: %s", dict(self.bucket_counts))
        log.info("unique queries: %d", len(self.query_counts))
        log.info("images dir size: %.1f MB", stats["images_mb"])


def flush_query(
    writer: PairWriter,
    query: str,
    rows_by_bucket: dict[str, list[dict[str, Any]]],
    rng: random.Random,
    per_bucket: int,
) -> int:
    if not query:
        return 0
    n = 0
    for bucket in ("top", "mid", "tail"):
        rows = rows_by_bucket.get(bucket, [])
        if len(rows) > per_bucket:
            rows = rng.sample(rows, per_bucket)
        for row in rows:
            writer.write(row)
            n += 1
    return n


def stream_gs(args: argparse.Namespace, writer: PairWriter) -> None:
    from datasets import load_dataset

    rng = random.Random(args.seed)
    log.info("streaming %s split=%s", args.repo_id, args.split)
    ds = load_dataset(
        args.repo_id,
        split=args.split,
        streaming=True,
        cache_dir=str(HF_CACHE),
    )

    current_query = ""
    current_buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen_queries = 0
    examined = 0
    rejected: Counter[str] = Counter()
    t0 = time.time()

    for row in ds:
        examined += 1
        query = (row.get("query") or "").strip()
        title = (row.get("title") or "").strip()
        image = row.get("image")
        position = int(row.get("position") or 0)
        bucket = rank_bucket(position)

        if query != current_query and current_query:
            flush_query(writer, current_query, current_buckets, rng, args.per_bucket)
            seen_queries += 1
            current_buckets = defaultdict(list)
            if seen_queries >= args.max_queries:
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
                "examined=%d queries=%d written=%d rejected=%s elapsed=%.1fs",
                examined,
                seen_queries,
                writer.written,
                dict(rejected),
                time.time() - t0,
            )
        if examined >= args.max_examined:
            break

    if current_query and seen_queries < args.max_queries:
        flush_query(writer, current_query, current_buckets, rng, args.per_bucket)
        seen_queries += 1

    log.info(
        "GS stream done: examined=%d queries=%d written=%d rejected=%s in %.1fs",
        examined,
        seen_queries,
        writer.written,
        dict(rejected),
        time.time() - t0,
    )


def stream_deepfashion(repo_id: str, limit: int, source: str, writer: PairWriter) -> None:
    if limit <= 0:
        return
    from datasets import load_dataset

    log.info("streaming %s limit=%d", repo_id, limit)
    ds = load_dataset(repo_id, split="data", streaming=True, cache_dir=str(HF_CACHE))
    loaded = 0
    for row in ds:
        if loaded >= limit:
            break
        image = row.get("image")
        query = deepfashion_query(row)
        if image is None or not query:
            continue
        writer.write(
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
        loaded += 1
        if loaded % 2_500 == 0:
            log.info("  %s loaded=%d written=%d", source, loaded, writer.written)
    log.info("loaded %d rows from %s", loaded, repo_id)


def main() -> None:
    args = parse_args()
    writer = PairWriter(args.out_dir)
    try:
        stream_gs(args, writer)
        stream_deepfashion(
            "Marqo/deepfashion-inshop",
            args.deepfashion_inshop,
            "deepfashion-inshop",
            writer,
        )
        stream_deepfashion(
            "Marqo/deepfashion-multimodal",
            args.deepfashion_multimodal,
            "deepfashion-multimodal",
            writer,
        )
    finally:
        writer.close(args)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
