"""
MODA Phase 4E — Image Hard Negative Labels via LLM

Same strategy as Phase 3C (generate_biencoder_labels.py) but mines hard
negatives from the *image retrieval channel* instead of text retrieval.

For each train query, we:
  1. Use FashionCLIP text encoder to search the IMAGE FAISS index
  2. Retrieve top-K image candidates (text-to-image cross-modal search)
  3. Ask GPT-4o-mini to judge relevance of each (query, product) pair
  4. Products scored 0 by LLM but highly ranked by image retrieval = hard negatives

These labels, combined with the text-retriever labels from Phase 3C,
will be used for joint text+image fine-tuning (Phase 4F).

Output: data/processed/image_retriever_labels.jsonl

Usage:
  export PALEBLUEDOT_API_KEY=your_key
  python benchmark/generate_image_labels.py
  python benchmark/generate_image_labels.py --max_queries 100 --report
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from benchmark.generate_biencoder_labels import (
    build_article_texts,
    label_batch,
    verify_no_leakage,
    RELEVANCE_PROMPT,
    BULK_MODEL,
    SPLIT_PATH,
    HNM_DIR,
    PROCESSED_DIR,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

EMBEDDINGS_DIR = PROCESSED_DIR / "embeddings"
IMAGE_FAISS_PATH = EMBEDDINGS_DIR / "fashion-clip-visual_faiss.index"
IMAGE_IDS_PATH = EMBEDDINGS_DIR / "fashion-clip-visual_article_ids.json"
LABELS_PATH = PROCESSED_DIR / "image_retriever_labels.jsonl"

RANDOM_SEED = 42
TOP_K_RETRIEVE = 20
MAX_CONCURRENT = 30


def retrieve_image_candidates(
    queries: list[tuple[str, str]],
    top_k: int = TOP_K_RETRIEVE,
) -> dict[str, list[tuple[str, int]]]:
    """Use FashionCLIP text encoder to search the IMAGE FAISS index.

    This is cross-modal: text query → image embedding space.
    CLIP's shared embedding space makes this possible zero-shot.
    Uses subprocess isolation for FAISS to avoid PyTorch+BLAS segfaults.
    """
    import subprocess
    import tempfile
    import torch
    from benchmark.models import load_clip_model, encode_texts_clip

    assert IMAGE_FAISS_PATH.exists(), f"Image FAISS index not found: {IMAGE_FAISS_PATH}"

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    log.info("Encoding %d queries with FashionCLIP text encoder on %s...", len(queries), device)
    model, _, tokenizer = load_clip_model("fashion-clip", device=device)
    texts = [q[1] for q in queries]
    q_embs = encode_texts_clip(texts, model, tokenizer, device, batch_size=128)
    del model
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    with tempfile.TemporaryDirectory() as tmp:
        q_path = Path(tmp) / "queries.npy"
        out_path = Path(tmp) / "results.json"
        np.save(str(q_path), q_embs.astype("float32"))

        worker = Path(__file__).parent / "_faiss_search_worker.py"
        cmd = [
            sys.executable, str(worker),
            str(q_path), str(IMAGE_FAISS_PATH), str(IMAGE_IDS_PATH),
            str(out_path), str(top_k),
        ]
        log.info("Running FAISS image search subprocess (top_k=%d)...", top_k)
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            log.error("FAISS image worker failed:\nstdout: %s\nstderr: %s",
                      result.stdout[-1000:], result.stderr[-1000:])
            raise RuntimeError("FAISS image search failed")

        with open(out_path) as f:
            raw_results = json.load(f)

    qids = [q[0] for q in queries]
    results: dict[str, list[tuple[str, int]]] = {}
    for i, qid in enumerate(qids):
        results[qid] = [
            (aid, rank + 1)
            for rank, aid in enumerate(raw_results[i])
        ]

    return results


def build_pairs(
    queries: list[tuple[str, str]],
    retrieval_results: dict[str, list[tuple[str, int]]],
    article_texts: dict[str, str],
) -> list[dict]:
    pairs = []
    query_dict = {qid: qt for qid, qt in queries}

    for qid, candidates in retrieval_results.items():
        query_text = query_dict.get(qid, "")
        if not query_text:
            continue
        for aid, rank in candidates:
            product_text = article_texts.get(aid, "")
            if not product_text:
                continue
            pairs.append({
                "query_id": qid,
                "article_id": aid,
                "query_text": query_text,
                "product_text": product_text,
                "retriever_rank": rank,
                "source": "image_retriever_mined",
            })

    log.info("Built %d image-retriever pairs from %d queries", len(pairs), len(retrieval_results))
    return pairs


def report_distribution():
    if not LABELS_PATH.exists():
        print(f"No labels file: {LABELS_PATH}")
        return

    counts = {0: 0, 1: 0, 2: 0, 3: 0}
    rank_by_score: dict[int, list[int]] = {0: [], 1: [], 2: [], 3: []}
    total = 0
    with open(LABELS_PATH) as f:
        for line in f:
            if line.strip():
                obj = json.loads(line)
                s = obj["score"]
                counts[s] = counts.get(s, 0) + 1
                rank_by_score[s].append(obj.get("retriever_rank", 0))
                total += 1

    print(f"\nTotal image-retriever labels: {total:,}")
    print(f"\nScore distribution:")
    for s in range(4):
        pct = 100 * counts[s] / total if total else 0
        avg_rank = np.mean(rank_by_score[s]) if rank_by_score[s] else 0
        bar = "#" * int(pct / 2)
        print(f"  {s}: {counts[s]:>6,}  ({pct:5.1f}%)  avg_rank={avg_rank:.1f}  {bar}")

    usable_pos = counts[2] + counts[3]
    usable_neg = counts[0]
    print(f"\nUsable for contrastive training:")
    print(f"  Positives (score 2-3): {usable_pos:,}")
    print(f"  Hard negatives (score 0): {usable_neg:,}")
    print(f"  Ambiguous (score 1): {counts[1]:,}")


def main():
    parser = argparse.ArgumentParser(description="Generate image-retriever training labels")
    parser.add_argument("--max_queries", type=int, default=5000)
    parser.add_argument("--top_k", type=int, default=TOP_K_RETRIEVE)
    parser.add_argument("--concurrency", type=int, default=MAX_CONCURRENT)
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--model", type=str, default=BULK_MODEL)
    parser.add_argument("--output", type=str, default=None,
                        help="Custom output path (default: image_retriever_labels.jsonl)")
    args = parser.parse_args()

    if args.report:
        report_distribution()
        return

    if not IMAGE_FAISS_PATH.exists():
        log.error("Image FAISS index not found! Run embed_hnm_images.py first.")
        return

    log.info("=" * 60)
    log.info("MODA Phase 4E — Image Hard Negative Label Generation")
    log.info("Strategy: text→image retrieval → GPT-4o-mini labeling")
    log.info("=" * 60)

    splits = json.loads(SPLIT_PATH.read_text())
    train_qids = set(splits["train"])
    log.info("Train split: %d query IDs", len(train_qids))

    all_queries: list[tuple[str, str]] = []
    with open(HNM_DIR / "queries.csv", newline="") as f:
        for row in csv.DictReader(f):
            qid = row["query_id"].strip()
            if qid in train_qids:
                all_queries.append((qid, row["query_text"].strip()))

    rng = random.Random(RANDOM_SEED)
    rng.shuffle(all_queries)
    seen_texts: set[str] = set()
    unique_queries: list[tuple[str, str]] = []
    for qid, qt in all_queries:
        if qt not in seen_texts:
            seen_texts.add(qt)
            unique_queries.append((qid, qt))
    log.info("Unique train query texts: %d", len(unique_queries))

    queries = unique_queries[:args.max_queries]
    log.info("Selected %d queries for image retrieval mining", len(queries))

    verify_no_leakage(queries)

    log.info("Loading H&M articles...")
    articles_df = pd.read_csv(HNM_DIR / "articles.csv", dtype=str).fillna("")
    article_texts = build_article_texts(articles_df)
    log.info("Articles loaded: %d", len(article_texts))

    retrieval_results = retrieve_image_candidates(queries, top_k=args.top_k)
    pairs = build_pairs(queries, retrieval_results, article_texts)
    log.info("Total pairs to label: %d (~$%.1f at GPT-4o-mini rates)",
             len(pairs), len(pairs) * 0.00008)

    output_path = Path(args.output) if args.output else LABELS_PATH
    asyncio.run(label_batch(
        pairs, output_path,
        model=args.model,
        concurrency=args.concurrency,
    ))

    report_distribution()


if __name__ == "__main__":
    main()
