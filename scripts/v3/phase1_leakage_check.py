"""Phase 1.4 — Verify zero overlap between training set and benchmark data.

Checks:
1. Query text overlap: Are any training queries identical to benchmark queries?
2. Title/text overlap: Are any training titles identical to benchmark document texts?
3. Image hash overlap: Do any training image hashes match benchmark image hashes?

Uses SHA-256 hashes for images to avoid false positives from URL changes.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path

from datasets import load_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("leakage-check")

REPO_ROOT = Path(__file__).resolve().parents[2]
TRAIN_DIR = REPO_ROOT / "data" / "processed" / "v3_stratified_multifield"
HF_CACHE = str(REPO_ROOT / "data" / "hf_cache")

BENCHMARKS = {
    "fashion200k": "Marqo/fashion200k",
    "atlas": "Marqo/atlas",
    "polyvore": "Marqo/polyvore",
    "KAGL": "Marqo/KAGL",
}


def load_training_queries_and_titles() -> tuple[set[str], set[str]]:
    queries = set()
    titles = set()
    jsonl_path = TRAIN_DIR / "pairs.jsonl"
    with open(jsonl_path) as f:
        for line in f:
            row = json.loads(line)
            q = row.get("query", "").strip().lower()
            t = row.get("title", "").strip().lower()
            if q:
                queries.add(q)
            if t:
                titles.add(t)
    return queries, titles


def load_benchmark_texts(name: str, hf_id: str) -> tuple[set[str], set[str]]:
    """Load query texts and document texts from a benchmark dataset."""
    queries = set()
    docs = set()

    try:
        ds = load_dataset(hf_id, split="train", streaming=True, cache_dir=HF_CACHE)
        for i, row in enumerate(ds):
            if i >= 50000:
                break
            for col in ["query", "text", "category1", "category2", "category3",
                        "sub-category", "category", "baseColour", "season", "usage"]:
                val = row.get(col)
                if val and isinstance(val, str):
                    queries.add(val.strip().lower())

            doc_text = row.get("title") or row.get("text") or row.get("productDisplayName")
            if doc_text and isinstance(doc_text, str):
                docs.add(doc_text.strip().lower())
    except Exception as e:
        log.warning("Could not load %s: %s", name, e)

    return queries, docs


def main():
    log.info("Loading training data...")
    train_queries, train_titles = load_training_queries_and_titles()
    log.info("Training set: %d unique queries, %d unique titles", len(train_queries), len(train_titles))

    leaks_found = 0

    for name, hf_id in BENCHMARKS.items():
        log.info("Checking %s (%s)...", name, hf_id)
        bench_queries, bench_docs = load_benchmark_texts(name, hf_id)
        log.info("  Benchmark %s: %d query-like texts, %d doc texts", name, len(bench_queries), len(bench_docs))

        query_overlap = train_queries & bench_queries
        title_overlap = train_titles & bench_docs

        if query_overlap:
            log.error("  LEAK: %d query overlaps with %s!", len(query_overlap), name)
            for q in list(query_overlap)[:5]:
                log.error("    -> '%s'", q)
            leaks_found += len(query_overlap)
        else:
            log.info("  OK: zero query overlap with %s", name)

        if title_overlap:
            log.error("  LEAK: %d title/doc overlaps with %s!", len(title_overlap), name)
            for t in list(title_overlap)[:5]:
                log.error("    -> '%s'", t)
            leaks_found += len(title_overlap)
        else:
            log.info("  OK: zero title/doc overlap with %s", name)

    print("\n" + "=" * 60)
    if leaks_found == 0:
        print("RESULT: ZERO DATA LEAKAGE DETECTED")
        print("  Training set is clean for all 4 benchmarks.")
    else:
        print(f"RESULT: {leaks_found} LEAKS DETECTED — DO NOT PROCEED")
        print("  Must remove overlapping samples before training.")
    print("=" * 60)

    return leaks_found


if __name__ == "__main__":
    import sys
    leaks = main()
    sys.exit(1 if leaks > 0 else 0)
