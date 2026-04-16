"""
MODA Phase 4 — Cross-Encoder Distillation to Retriever

Generates soft labels by scoring top-K retrieval candidates with the
LLM-trained cross-encoder. These scores become teacher signals for
training the three-tower retriever via MarginMSE loss.

Strategy:
  1. For each train query, retrieve top-K candidates with current best retriever
  2. Score all K candidates with the LLM-trained CE (pointwise scores)
  3. Output (query, doc_i, doc_j, ce_margin) pairs for MarginMSE training:
     margin = ce_score(q, d_i) - ce_score(q, d_j)

Output: data/processed/ce_distillation_pairs.jsonl
Each line: {"query_id", "query_text", "doc_i_id", "doc_j_id",
            "ce_score_i", "ce_score_j", "ce_margin"}

Usage:
  python -m benchmark.distill_ce_to_retriever
  python -m benchmark.distill_ce_to_retriever --max_queries 500 --top_k 50
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from benchmark.article_text import build_article_texts_from_df

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

PROCESSED_DIR = _REPO_ROOT / "data" / "processed"
HNM_DIR = _REPO_ROOT / "data" / "raw" / "hnm_real"
MODEL_DIR = _REPO_ROOT / "models"
SPLIT_PATH = PROCESSED_DIR / "query_splits.json"
OUTPUT_PATH = PROCESSED_DIR / "ce_distillation_pairs.jsonl"

CE_MODEL_PATH = str(MODEL_DIR / "moda-fashion-ce-llm-best")
EMBED_DIR = PROCESSED_DIR / "embeddings"
FAISS_PATH = EMBED_DIR / "fashion-clip_faiss.index"
IDS_PATH = EMBED_DIR / "fashion-clip_article_ids.json"

RANDOM_SEED = 42
TOP_K = 100
MAX_PAIRS_PER_QUERY = 20


def retrieve_candidates(
    queries: list[tuple[str, str]],
    top_k: int = TOP_K,
) -> dict[str, list[str]]:
    """Use FashionCLIP to retrieve top-K candidates per query."""
    import subprocess
    from benchmark.models import load_clip_model, encode_texts_clip

    assert FAISS_PATH.exists(), f"FAISS index not found: {FAISS_PATH}"

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    log.info("Encoding %d queries on %s...", len(queries), device)
    model, _, tokenizer = load_clip_model("fashion-clip", device=device)
    texts = [q[1] for q in queries]
    q_emb = encode_texts_clip(texts, model, tokenizer, device, batch_size=128)
    del model
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    q_path = PROCESSED_DIR / "distill_q_emb.npy"
    out_path = PROCESSED_DIR / "distill_faiss_results.json"
    np.save(str(q_path), q_emb.astype("float32"))

    worker = Path(__file__).parent / "_faiss_search_worker.py"
    cmd = [
        sys.executable, str(worker),
        str(q_path), str(FAISS_PATH), str(IDS_PATH),
        str(out_path), str(top_k),
    ]
    log.info("Running FAISS search (top_k=%d)...", top_k)
    subprocess.run(cmd, check=True)

    with open(out_path) as f:
        raw_results = json.load(f)

    results = {}
    for (qid, _), candidates in zip(queries, raw_results):
        results[qid] = candidates

    q_path.unlink(missing_ok=True)
    out_path.unlink(missing_ok=True)
    return results


def score_with_ce(
    queries: list[tuple[str, str]],
    candidates: dict[str, list[str]],
    article_texts: dict[str, str],
    ce_model_path: str,
    batch_size: int = 64,
) -> dict[str, list[tuple[str, float]]]:
    """Score all (query, candidate) pairs with the cross-encoder."""
    from sentence_transformers.cross_encoder import CrossEncoder

    log.info("Loading CE model from %s...", ce_model_path)
    ce = CrossEncoder(ce_model_path, max_length=512)

    scored: dict[str, list[tuple[str, float]]] = {}
    query_dict = {qid: qt for qid, qt in queries}

    for qid in tqdm(candidates, desc="CE scoring", ncols=80):
        qt = query_dict.get(qid, "")
        cands = candidates[qid]
        if not qt or not cands:
            continue

        pairs = []
        valid_cands = []
        for aid in cands:
            at = article_texts.get(aid, "")
            if at:
                pairs.append((qt, at))
                valid_cands.append(aid)

        if not pairs:
            continue

        scores = ce.predict(pairs, batch_size=batch_size, show_progress_bar=False)
        scored[qid] = list(zip(valid_cands, scores.tolist()))

    return scored


def generate_margin_pairs(
    scored: dict[str, list[tuple[str, float]]],
    queries: list[tuple[str, str]],
    max_pairs_per_query: int = MAX_PAIRS_PER_QUERY,
    seed: int = RANDOM_SEED,
) -> list[dict]:
    """Generate (doc_i, doc_j, margin) pairs from CE scores.

    Samples pairs where doc_i has higher CE score than doc_j, creating
    teacher signals for MarginMSE distillation.
    """
    rng = random.Random(seed)
    query_dict = {qid: qt for qid, qt in queries}
    pairs = []

    for qid, scored_docs in scored.items():
        if len(scored_docs) < 2:
            continue

        sorted_docs = sorted(scored_docs, key=lambda x: -x[1])
        qt = query_dict.get(qid, "")
        query_pairs = []

        for i in range(len(sorted_docs)):
            for j in range(i + 1, len(sorted_docs)):
                aid_i, score_i = sorted_docs[i]
                aid_j, score_j = sorted_docs[j]
                margin = score_i - score_j
                if margin > 0.05:
                    query_pairs.append({
                        "query_id": qid,
                        "query_text": qt,
                        "doc_i_id": aid_i,
                        "doc_j_id": aid_j,
                        "ce_score_i": round(score_i, 4),
                        "ce_score_j": round(score_j, 4),
                        "ce_margin": round(margin, 4),
                    })

        if len(query_pairs) > max_pairs_per_query:
            query_pairs = rng.sample(query_pairs, max_pairs_per_query)
        pairs.extend(query_pairs)

    return pairs


def main():
    parser = argparse.ArgumentParser(description="CE distillation data generation")
    parser.add_argument("--max_queries", type=int, default=5000)
    parser.add_argument("--top_k", type=int, default=TOP_K)
    parser.add_argument("--ce_model", type=str, default=CE_MODEL_PATH)
    parser.add_argument("--max_pairs_per_query", type=int, default=MAX_PAIRS_PER_QUERY)
    args = parser.parse_args()

    if not Path(args.ce_model).exists():
        log.error("CE model not found at %s. Train it first.", args.ce_model)
        return

    t0 = time.time()

    log.info("=" * 60)
    log.info("MODA — Cross-Encoder Distillation Data Generation")
    log.info("  CE model: %s", args.ce_model)
    log.info("  Top-K candidates: %d", args.top_k)
    log.info("=" * 60)

    splits = json.loads(SPLIT_PATH.read_text())
    train_qids = set(splits["train"])

    all_queries: list[tuple[str, str]] = []
    with open(HNM_DIR / "queries.csv", newline="") as f:
        for row in csv.DictReader(f):
            qid = row["query_id"].strip()
            if qid in train_qids:
                all_queries.append((qid, row["query_text"].strip()))

    rng = random.Random(RANDOM_SEED)
    rng.shuffle(all_queries)
    seen: set[str] = set()
    unique_queries: list[tuple[str, str]] = []
    for qid, qt in all_queries:
        if qt not in seen:
            seen.add(qt)
            unique_queries.append((qid, qt))

    queries = unique_queries[:args.max_queries]
    log.info("Selected %d queries", len(queries))

    import pandas as pd
    articles_df = pd.read_csv(HNM_DIR / "articles.csv", dtype=str).fillna("")
    article_texts = build_article_texts_from_df(articles_df)

    candidates = retrieve_candidates(queries, top_k=args.top_k)
    scored = score_with_ce(queries, candidates, article_texts, args.ce_model)

    pairs = generate_margin_pairs(scored, queries, args.max_pairs_per_query)

    with open(OUTPUT_PATH, "w") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")

    elapsed = time.time() - t0
    log.info("Generated %d margin pairs in %.1f min → %s",
             len(pairs), elapsed / 60, OUTPUT_PATH)

    margins = [p["ce_margin"] for p in pairs]
    log.info("Margin stats: mean=%.3f std=%.3f min=%.3f max=%.3f",
             np.mean(margins), np.std(margins), np.min(margins), np.max(margins))


if __name__ == "__main__":
    main()
