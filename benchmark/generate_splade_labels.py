"""
MODA Phase 3C — SPLADE Training Data via Retriever-Mined Hard Negatives

Mines hard negatives from the **off-shelf** SPLADE model's own retrieval
failures, then labels them with an LLM (Claude Sonnet via PaleblueDot).

DATA LEAKAGE SAFEGUARDS:
  - Only train-split queries are used (never val/test)
  - Hard negatives are mined from the OFF-SHELF SPLADE, not a fine-tuned one
  - Article texts use the canonical builder (same at train and eval time)
  - Query text deduplication prevents the same surface form from appearing
    in both train and val/test splits
  - Assertions verify split disjointness at startup

Strategy:
  1. Load train-split queries (verified disjoint from val/test)
  2. Run off-shelf SPLADE retrieval → top-K candidates per query
  3. Also include purchase positives from qrels (ensures relevant docs)
  4. Label each (query, candidate) pair with LLM (0-3 relevance)
  5. Output: score 2-3 = positive, score 0 by LLM but ranked high = hard neg

Output: data/processed/splade_training_labels.jsonl
Each line: {"query_id", "article_id", "query_text", "product_text",
            "score", "reason", "retriever_rank", "source"}

Usage:
  export PALEBLUEDOT_API_KEY=your_key
  python -m benchmark.generate_splade_labels                    # full run
  python -m benchmark.generate_splade_labels --max_queries 100  # smoke test
  python -m benchmark.generate_splade_labels --report           # stats only
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
from openai import AsyncOpenAI

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from benchmark.article_text import build_article_texts_from_df as build_article_texts

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

HNM_DIR = _REPO_ROOT / "data" / "raw" / "hnm_real"
PROCESSED_DIR = _REPO_ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

LABELS_PATH = PROCESSED_DIR / "splade_training_labels.jsonl"
SPLIT_PATH = PROCESSED_DIR / "query_splits.json"

BASE_URL = "https://open.palebluedot.ai/v1"
LLM_MODEL = "anthropic/claude-sonnet-4.6"

RANDOM_SEED = 42
RETRIEVAL_TOP_K = 20
RANDOM_NEGS_PER_QUERY = 3
MAX_CONCURRENT = 15
MAX_RETRIES = 3


# ─── Leakage checks ──────────────────────────────────────────────────────────

def load_and_verify_splits() -> dict[str, set[str]]:
    """Load query splits and verify they are disjoint."""
    splits = json.loads(SPLIT_PATH.read_text())
    train = set(splits["train"])
    val = set(splits["val"])
    test = set(splits["test"])

    assert len(train & val) == 0, "LEAKAGE: train ∩ val is non-empty!"
    assert len(train & test) == 0, "LEAKAGE: train ∩ test is non-empty!"
    assert len(val & test) == 0, "LEAKAGE: val ∩ test is non-empty!"
    log.info("Split integrity OK: train=%d, val=%d, test=%d (all disjoint)",
             len(train), len(val), len(test))
    return {"train": train, "val": val, "test": test}


def verify_no_query_text_leakage(
    train_queries: list[tuple[str, str]],
    all_queries_by_split: dict[str, list[tuple[str, str]]],
) -> None:
    """Assert that no query TEXT in train also appears in val or test."""
    train_texts = {qt for _, qt in train_queries}
    for split_name in ("val", "test"):
        other_texts = {qt for _, qt in all_queries_by_split.get(split_name, [])}
        overlap = train_texts & other_texts
        assert len(overlap) == 0, (
            f"LEAKAGE: {len(overlap)} query texts appear in both train and {split_name}! "
            f"Examples: {list(overlap)[:5]}"
        )
    log.info("Query text leakage check passed (no text overlap between splits)")


# ─── Data loading ─────────────────────────────────────────────────────────────

def load_train_queries(
    splits: dict[str, set[str]], max_queries: int = 0
) -> tuple[list[tuple[str, str]], dict[str, list[tuple[str, str]]]]:
    """Load queries, filter to train split, deduplicate by text."""
    train_qids = splits["train"]

    all_by_split: dict[str, list[tuple[str, str]]] = {"train": [], "val": [], "test": []}
    with open(HNM_DIR / "queries.csv", newline="") as f:
        for row in csv.DictReader(f):
            qid = row["query_id"].strip()
            qt = row["query_text"].strip()
            for sname, sids in splits.items():
                if qid in sids:
                    all_by_split[sname].append((qid, qt))
                    break

    # Deduplicate train by query text (keep first occurrence)
    seen_texts: set[str] = set()
    unique_train: list[tuple[str, str]] = []
    for qid, qt in all_by_split["train"]:
        if qt not in seen_texts:
            seen_texts.add(qt)
            unique_train.append((qid, qt))

    rng = random.Random(RANDOM_SEED)
    rng.shuffle(unique_train)
    if max_queries > 0:
        unique_train = unique_train[:max_queries]

    log.info("Train queries: %d unique texts (from %d IDs)",
             len(unique_train), len(all_by_split["train"]))
    return unique_train, all_by_split


def load_qrels_train(train_qids: set[str]) -> dict[str, list[str]]:
    """Load purchase positives for train queries."""
    qrels: dict[str, list[str]] = {}
    with open(HNM_DIR / "qrels.csv", newline="") as f:
        for row in csv.DictReader(f):
            qid = row["query_id"].strip()
            if qid not in train_qids:
                continue
            pos_ids = [a.strip() for a in row.get("positive_ids", "").split(",") if a.strip()]
            if pos_ids:
                qrels[qid] = pos_ids
    return qrels


# ─── SPLADE retrieval ─────────────────────────────────────────────────────────

def retrieve_with_offshelf_splade(
    queries: list[tuple[str, str]],
    article_ids: list[str],
    article_texts: dict[str, str],
    top_k: int = RETRIEVAL_TOP_K,
) -> dict[str, list[str]]:
    """Run off-shelf SPLADE retrieval on train queries."""
    import torch
    from benchmark.splade_retriever import SpladeRetriever

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    splade = SpladeRetriever(device=device)
    splade.encode_articles(article_ids, article_texts, batch_size=32, force=False)

    texts = [qt for _, qt in queries]
    results_lists = splade.search_batch(texts, top_k=top_k, query_chunk=1000)

    splade.free_model()

    results: dict[str, list[str]] = {}
    for (qid, _), res in zip(queries, results_lists):
        results[qid] = res
    return results


# ─── LLM labeling ─────────────────────────────────────────────────────────────

LABEL_PROMPT = """Rate how relevant this product is to the search query.

Query: "{query}"
Product: "{product}"

Score 0-3:
  0 = Not relevant at all (wrong category, gender, or completely different item)
  1 = Partially relevant (same broad category but wrong specific attributes)
  2 = Good match (right category and most attributes match)
  3 = Exact match (would satisfy the search intent perfectly)

Respond with ONLY a JSON object: {{"score": <0-3>, "reason": "<brief reason>"}}"""


async def label_pair(
    client: AsyncOpenAI,
    query_text: str,
    product_text: str,
    semaphore: asyncio.Semaphore,
) -> dict | None:
    async with semaphore:
        prompt = LABEL_PROMPT.format(query=query_text, product=product_text)
        for attempt in range(MAX_RETRIES):
            try:
                resp = await client.chat.completions.create(
                    model=LLM_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=100,
                    temperature=0.0,
                )
                text = resp.choices[0].message.content.strip()
                text = text.replace("```json", "").replace("```", "").strip()
                result = json.loads(text)
                score = int(result.get("score", -1))
                if 0 <= score <= 3:
                    return {"score": score, "reason": result.get("reason", "")}
            except Exception as e:
                if attempt == MAX_RETRIES - 1:
                    log.warning("Failed to label pair after %d attempts: %s", MAX_RETRIES, e)
                await asyncio.sleep(1.0 * (attempt + 1))
    return None


async def label_batch(
    pairs: list[dict],
    article_texts: dict[str, str],
    existing_labels: dict[str, dict],
    output_path: Path | None = None,
) -> list[dict]:
    """Label query-product pairs with LLM, skipping already-labeled ones.

    Writes results incrementally to output_path so progress survives crashes.
    """
    api_key = os.environ.get("PALEBLUEDOT_API_KEY", "")
    if not api_key:
        log.error("PALEBLUEDOT_API_KEY not set — cannot label")
        return []

    client = AsyncOpenAI(api_key=api_key, base_url=BASE_URL)
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)

    to_label = []
    results = []
    for pair in pairs:
        key = f"{pair['query_id']}_{pair['article_id']}"
        if key in existing_labels:
            cached = existing_labels[key]
            pair["score"] = cached["score"]
            pair["reason"] = cached.get("reason", "")
            results.append(pair)
        else:
            to_label.append(pair)

    log.info("Labeling %d pairs (%d cached)", len(to_label), len(results))

    # Write cached results first
    if output_path is not None:
        with open(output_path, "w") as f:
            for row in results:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        log.info("Wrote %d cached labels to %s", len(results), output_path)

    tasks = []
    for pair in to_label:
        pt = article_texts.get(pair["article_id"], "")
        if not pt:
            continue
        tasks.append((pair, label_pair(client, pair["query_text"], pt, semaphore)))

    out_f = open(output_path, "a") if output_path else None
    try:
        for i in range(0, len(tasks), 50):
            batch_tasks = tasks[i:i + 50]
            batch_results = await asyncio.gather(*[t[1] for t in batch_tasks])
            for (pair, _), result in zip(batch_tasks, batch_results):
                if result is not None:
                    pair["score"] = result["score"]
                    pair["reason"] = result["reason"]
                    pair["product_text"] = article_texts.get(pair["article_id"], "")
                    results.append(pair)
                    if out_f:
                        out_f.write(json.dumps(pair, ensure_ascii=False) + "\n")

            if out_f:
                out_f.flush()

            if i > 0 and i % 200 == 0:
                log.info("  Labeled %d / %d pairs...", len(results), len(pairs))
    finally:
        if out_f:
            out_f.close()

    return results


# ─── Main ─────────────────────────────────────────────────────────────────────

def build_candidate_pairs(
    queries: list[tuple[str, str]],
    splade_results: dict[str, list[str]],
    qrels: dict[str, list[str]],
    article_ids_set: set[str],
    article_texts: dict[str, str],
) -> list[dict]:
    """Build (query, article) pairs from SPLADE results + purchase positives + random negs."""
    rng = random.Random(RANDOM_SEED)
    all_article_ids = list(article_ids_set)
    pairs = []

    for qid, qt in queries:
        seen_aids: set[str] = set()

        # SPLADE top-K candidates
        for rank, aid in enumerate(splade_results.get(qid, []), start=1):
            if aid not in seen_aids:
                pairs.append({
                    "query_id": qid, "article_id": aid, "query_text": qt,
                    "retriever_rank": rank, "source": "splade_topk",
                })
                seen_aids.add(aid)

        # Purchase positives (may not be in SPLADE top-K)
        for aid in qrels.get(qid, []):
            if aid not in seen_aids and aid in article_ids_set:
                pairs.append({
                    "query_id": qid, "article_id": aid, "query_text": qt,
                    "retriever_rank": -1, "source": "qrel_positive",
                })
                seen_aids.add(aid)

        # Random negatives (very likely irrelevant, provides easy negatives)
        added = 0
        while added < RANDOM_NEGS_PER_QUERY:
            rid = rng.choice(all_article_ids)
            if rid not in seen_aids and rid in article_texts:
                pairs.append({
                    "query_id": qid, "article_id": rid, "query_text": qt,
                    "retriever_rank": -1, "source": "random_neg",
                })
                seen_aids.add(rid)
                added += 1

    log.info("Built %d candidate pairs for %d queries", len(pairs), len(queries))
    return pairs


def report_stats(labels_path: Path) -> None:
    """Print distribution stats for existing labels."""
    if not labels_path.exists():
        log.info("No labels file found at %s", labels_path)
        return

    scores = {0: 0, 1: 0, 2: 0, 3: 0}
    sources = {}
    n = 0
    with open(labels_path) as f:
        for line in f:
            row = json.loads(line)
            s = row.get("score", -1)
            if s in scores:
                scores[s] += 1
            src = row.get("source", "unknown")
            sources[src] = sources.get(src, 0) + 1
            n += 1

    log.info("Label distribution (%d total):", n)
    for s, c in sorted(scores.items()):
        log.info("  score=%d: %d (%.1f%%)", s, c, 100.0 * c / max(n, 1))
    log.info("Source distribution:")
    for src, c in sorted(sources.items()):
        log.info("  %s: %d", src, c)

    hard_negs = scores[0]
    positives = scores[2] + scores[3]
    log.info("Usable: %d positives (score 2-3), %d hard negatives (score 0)", positives, hard_negs)


def main():
    parser = argparse.ArgumentParser(description="Generate SPLADE fine-tuning labels")
    parser.add_argument("--max_queries", type=int, default=5000)
    parser.add_argument("--top_k", type=int, default=RETRIEVAL_TOP_K)
    parser.add_argument("--report", action="store_true", help="Print stats for existing labels")
    parser.add_argument("--skip_retrieval", action="store_true",
                        help="Skip SPLADE retrieval (use existing candidates from labels file)")
    args = parser.parse_args()

    if args.report:
        report_stats(LABELS_PATH)
        return

    t0 = time.time()

    # ── 1. Verify splits (leakage check) ─────────────────────────────────
    from benchmark.leakage_guard import run_all_checks, get_forbidden_train_texts
    run_all_checks(split_path=SPLIT_PATH)

    splits = load_and_verify_splits()
    train_queries, all_by_split = load_train_queries(splits, max_queries=args.max_queries)

    # Remove train queries whose text also appears in val/test
    forbidden_texts = get_forbidden_train_texts(splits)
    if forbidden_texts:
        before = len(train_queries)
        train_queries = [
            (qid, qt) for qid, qt in train_queries
            if qt.strip().lower() not in forbidden_texts
        ]
        log.info("Excluded %d train queries with text overlap in val/test",
                 before - len(train_queries))

    # ── 2. Load articles ─────────────────────────────────────────────────
    articles_df = pd.read_csv(HNM_DIR / "articles.csv", dtype=str).fillna("")
    article_texts = build_article_texts(articles_df)
    article_ids = list(article_texts.keys())
    article_ids_set = set(article_ids)
    log.info("Loaded %d articles", len(article_ids))

    # ── 3. Load qrels (train only) ───────────────────────────────────────
    train_qids = {qid for qid, _ in train_queries}
    qrels = load_qrels_train(train_qids)
    log.info("Loaded qrels for %d train queries", len(qrels))

    # ── 4. SPLADE retrieval (off-shelf only!) ────────────────────────────
    if not args.skip_retrieval:
        log.info("Running off-shelf SPLADE retrieval (top_k=%d)...", args.top_k)
        splade_results = retrieve_with_offshelf_splade(
            train_queries, article_ids, article_texts, top_k=args.top_k,
        )
    else:
        log.info("Skipping retrieval — building pairs from existing labels")
        splade_results = {}

    # ── 5. Build candidate pairs ─────────────────────────────────────────
    pairs = build_candidate_pairs(
        train_queries, splade_results, qrels, article_ids_set, article_texts,
    )

    # ── 6. Load existing labels (for incremental runs) ───────────────────
    existing: dict[str, dict] = {}
    if LABELS_PATH.exists():
        with open(LABELS_PATH) as f:
            for line in f:
                row = json.loads(line)
                key = f"{row['query_id']}_{row['article_id']}"
                existing[key] = row
        log.info("Loaded %d existing labels", len(existing))

    # ── 7. Label with LLM (incremental writes) ─────────────────────────
    labeled = asyncio.run(label_batch(pairs, article_texts, existing, output_path=LABELS_PATH))

    elapsed = time.time() - t0
    log.info("Wrote %d labels to %s (%.1f min)", len(labeled), LABELS_PATH, elapsed / 60)
    report_stats(LABELS_PATH)


if __name__ == "__main__":
    main()
