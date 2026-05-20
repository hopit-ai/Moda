"""
v6 Step 1 — Mine GS-10M for long queries (≥10 words).

Standard GS-10M mining used short catalog titles as the text signal. This pass
streams the same dataset but filters for items where the **query** itself is ≥10
words — these are natural prose-style search queries that match fashion200k's
distribution, with zero API cost and zero LLM generation needed.

Also captures items with a description field if one exists (some GS-10M subsets
may carry product descriptions beyond the title).

Outputs (data/processed/v6/):
  pairs_gs10m_long_query.jsonl  — (pair_id, query, image_file, title, score)
  images/                       — 224px product images (same format as v4 pipeline)
  stats_long_query.json         — counts per query-length bucket

Usage:
    python scripts/v6/step1_mine_gs10m_long_queries.py
    python scripts/v6/step1_mine_gs10m_long_queries.py --target 80000 --min_words 8
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from PIL import Image
from tqdm import tqdm

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = REPO / "data" / "processed" / "v6"
IMG_DIR = OUT_DIR / "images"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=80000,
                    help="Max long-query pairs to collect")
    ap.add_argument("--min_words", type=int, default=10,
                    help="Minimum query word count to qualify as 'long'")
    ap.add_argument("--min_score", type=float, default=3.0,
                    help="Minimum GS-10M relevance score (1-10)")
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    IMG_DIR.mkdir(parents=True, exist_ok=True)

    out_path = OUT_DIR / "pairs_gs10m_long_query.jsonl"
    stats_path = OUT_DIR / "stats_long_query.json"

    if out_path.exists():
        existing = sum(1 for _ in out_path.open())
        print(f"Output already has {existing:,} pairs — resuming (will append new pairs)")
        seen_ids = {json.loads(l)["pair_id"] for l in out_path.open() if l.strip()}
    else:
        seen_ids = set()

    print(f"Mining GS-10M for queries with ≥{args.min_words} words, score ≥{args.min_score}")
    print(f"Target: {args.target:,} pairs | Output: {out_path}")

    from datasets import load_dataset, get_dataset_split_names
    from itertools import chain

    # GS-10M split names changed — discover and stream all available splits
    try:
        available_splits = get_dataset_split_names("Marqo/marqo-GS-10M")
    except Exception:
        available_splits = ["train"]
    print(f"GS-10M available splits: {available_splits}")

    split_iters = []
    for sp in available_splits:
        try:
            split_iters.append(load_dataset("Marqo/marqo-GS-10M", split=sp, streaming=True))
            print(f"  Added split: {sp}")
        except Exception as e:
            print(f"  Skipped split {sp}: {e}")

    if not split_iters:
        raise RuntimeError("No GS-10M splits could be loaded.")

    ds = chain(*split_iters)

    # Sample first item to discover available fields
    sample = next(ds)
    print(f"GS-10M fields: {list(sample.keys())}")
    # Put sample back by re-chaining — re-open the first split for full coverage
    split_iters2 = []
    for sp in available_splits:
        try:
            split_iters2.append(load_dataset("Marqo/marqo-GS-10M", split=sp, streaming=True))
        except Exception:
            pass
    ds = chain(*split_iters2)

    length_buckets = {f"{i*5}-{i*5+4}w": 0 for i in range(0, 20)}
    n_written = len(seen_ids)
    n_scanned = 0
    n_skipped_score = 0
    n_skipped_image = 0
    has_description_field = "description" in sample

    t0 = time.time()
    with out_path.open("a") as fout:
        pbar = tqdm(ds, desc="streaming GS-10M", unit=" items")
        for item in pbar:
            n_scanned += 1
            if n_written >= args.target:
                break

            query = (item.get("query") or "").strip()
            n_words = len(query.split())

            # Bucket all lengths for stats (even those we skip)
            bucket_key = f"{(n_words // 5)*5}-{(n_words // 5)*5+4}w"
            if bucket_key in length_buckets:
                length_buckets[bucket_key] += 1

            if n_words < args.min_words:
                continue

            score = float(item.get("score", 0))
            if score < args.min_score:
                n_skipped_score += 1
                continue

            # Use query as pair_id since it's the key signal here
            pid = f"gs10m_long_{n_scanned}"
            if pid in seen_ids:
                continue

            # Save image
            img_file = f"{pid}.jpg"
            img_path = IMG_DIR / img_file
            if not img_path.exists():
                pil = item.get("image")
                if pil is None:
                    n_skipped_image += 1
                    continue
                try:
                    if not isinstance(pil, Image.Image):
                        pil = Image.fromarray(pil)
                    pil = pil.convert("RGB")
                    w, h = pil.size
                    s = args.img_size
                    scale = s / min(w, h)
                    pil = pil.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
                    left = (pil.width - s) // 2
                    top = (pil.height - s) // 2
                    pil = pil.crop((left, top, left + s, top + s))
                    pil.save(img_path, "JPEG", quality=85)
                except Exception as e:
                    n_skipped_image += 1
                    continue

            record = {
                "pair_id": pid,
                "query": query,
                "title": (item.get("title") or "").strip(),
                "image_file": img_file,
                "score": score,
                "n_words": n_words,
            }
            if has_description_field:
                record["description"] = (item.get("description") or "").strip()

            fout.write(json.dumps(record) + "\n")
            seen_ids.add(pid)
            n_written += 1

            if n_written % 1000 == 0:
                elapsed = time.time() - t0
                rate = n_written / elapsed
                pbar.set_postfix(written=n_written, scanned=n_scanned,
                                 rate=f"{rate:.1f}/s")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s")
    print(f"  Scanned: {n_scanned:,} | Written: {n_written:,}")
    print(f"  Skipped (low score): {n_skipped_score:,} | Skipped (no image): {n_skipped_image:,}")
    print(f"  Long-query rate: {n_written/max(n_scanned,1)*100:.2f}% of all items")

    # Query length distribution of what we collected
    collected = [json.loads(l) for l in out_path.open() if l.strip()]
    word_dist = {}
    for r in collected:
        nw = r.get("n_words", len(r.get("query","").split()))
        k = f"{nw}w"
        word_dist[k] = word_dist.get(k, 0) + 1

    stats = {
        "total_pairs": n_written,
        "total_scanned": n_scanned,
        "min_words": args.min_words,
        "min_score": args.min_score,
        "has_description_field": has_description_field,
        "length_buckets_all": length_buckets,
        "word_count_distribution": dict(sorted(word_dist.items(),
                                               key=lambda x: int(x[0].rstrip("w")))),
    }
    stats_path.write_text(json.dumps(stats, indent=2))
    print(f"\nStats written to {stats_path}")
    print(f"Has description field: {has_description_field}")
    if has_description_field:
        n_with_desc = sum(1 for r in collected if r.get("description", "").strip())
        print(f"Items with non-empty description: {n_with_desc:,} / {n_written:,}")


if __name__ == "__main__":
    main()
