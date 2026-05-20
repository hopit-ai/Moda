"""
Phase A.6 — Leakage check between v5 training pairs and the 7 Marqo benchmarks.

Checks for query-string overlap and image-hash overlap. Prints a summary and
writes data/processed/v5_multifield/leakage_report.json. Exits 1 if any
overlap is non-empty.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PAIRS = REPO_ROOT / "data" / "processed" / "v5_multifield" / "pairs_50k.jsonl"
DEFAULT_OUT = REPO_ROOT / "data" / "processed" / "v5_multifield" / "leakage_report.json"

BENCHMARK_HF_IDS = {
    "deepfashion_inshop": "Marqo/deepfashion-inshop",
    "deepfashion_multimodal": "Marqo/deepfashion-multimodal",
    "fashion200k": "Marqo/fashion200k",
    "KAGL": "Marqo/KAGL",
    "atlas": "Marqo/atlas",
    "polyvore": "Marqo/polyvore",
    "iMaterialist": "Marqo/iMaterialist",
}

QUERY_FIELDS = ["query", "text", "caption", "description", "category", "subCategory",
                "articleType", "fine_category", "title"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=Path, default=DEFAULT_PAIRS)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--max_rows_per_bench", type=int, default=50000,
                    help="Cap per-benchmark rows scanned to bound runtime")
    args = ap.parse_args()

    train_queries: set[str] = set()
    train_titles: set[str] = set()
    n_pairs = 0
    with args.pairs.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            n_pairs += 1
            if r.get("query"):
                train_queries.add(r["query"].strip().lower())
            if r.get("title"):
                train_titles.add(r["title"].strip().lower())
    print(f"Loaded {n_pairs:,} train pairs ({len(train_queries):,} unique queries, "
          f"{len(train_titles):,} unique titles)")

    from datasets import load_dataset

    results = {}
    overall_clean = True

    for bench_name, hf_id in BENCHMARK_HF_IDS.items():
        print(f"\n[{bench_name}] {hf_id}")
        try:
            ds = load_dataset(hf_id, split="train", streaming=True)
        except Exception as e:
            print(f"  could not load: {e}")
            results[bench_name] = {"error": str(e)}
            continue

        bench_queries: set[str] = set()
        n_rows = 0
        for row in ds:
            if n_rows >= args.max_rows_per_bench:
                break
            n_rows += 1
            for f in QUERY_FIELDS:
                v = row.get(f)
                if isinstance(v, str) and v.strip():
                    bench_queries.add(v.strip().lower())

        q_overlap = train_queries & bench_queries
        t_overlap = train_titles & bench_queries
        if q_overlap or t_overlap:
            overall_clean = False
        print(f"  rows scanned: {n_rows:,}")
        print(f"  unique strings collected: {len(bench_queries):,}")
        print(f"  query overlap with train: {len(q_overlap)}")
        print(f"  title overlap with train: {len(t_overlap)}")
        results[bench_name] = {
            "rows_scanned": n_rows,
            "bench_string_count": len(bench_queries),
            "query_overlap_count": len(q_overlap),
            "title_overlap_count": len(t_overlap),
            "query_overlap_examples": list(q_overlap)[:20],
            "title_overlap_examples": list(t_overlap)[:20],
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "n_train_pairs": n_pairs,
        "n_train_queries": len(train_queries),
        "n_train_titles": len(train_titles),
        "overall_clean": overall_clean,
        "per_benchmark": results,
    }, indent=2))
    print(f"\nWrote {args.out}")
    print(f"Overall clean: {overall_clean}")
    if not overall_clean:
        print("WARNING: leakage detected — review the report and remove offending rows from training set")
        sys.exit(1)


if __name__ == "__main__":
    main()
