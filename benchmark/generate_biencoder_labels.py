"""
MODA Phase 3C — Bi-Encoder Training Data via Retriever-Mined Hard Negatives

Generates LLM-judged labels for query-product pairs where the products are
retrieved by FashionCLIP (the current bi-encoder). This creates hard negatives
that teach the bi-encoder exactly where it's currently failing.

Strategy:
  1. Sample train-split queries (leakage-free — same splits as Phase 3B)
  2. Run FashionCLIP retrieval to get top-K candidates per query
  3. Label each (query, candidate) pair with GPT-4o-mini (0-3 relevance)
  4. Output triplets: score 2-3 = positive, score 0-1 = hard negative

Output: data/processed/biencoder_retriever_labels.jsonl
Each line: {"query_id", "article_id", "query_text", "product_text",
            "score", "reason", "retriever_rank", "source"}

Usage:
  export PALEBLUEDOT_API_KEY=your_key
  python benchmark/generate_biencoder_labels.py                      # full run
  python benchmark/generate_biencoder_labels.py --max_queries 100    # smoke test
  python benchmark/generate_biencoder_labels.py --report             # distribution
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

HNM_DIR = _REPO_ROOT / "data" / "raw" / "hnm_real"
PROCESSED_DIR = _REPO_ROOT / "data" / "processed"
EMBEDDINGS_DIR = PROCESSED_DIR / "embeddings"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

LABELS_PATH = PROCESSED_DIR / "biencoder_retriever_labels.jsonl"
SPLIT_PATH = PROCESSED_DIR / "query_splits.json"

BASE_URL = "https://open.palebluedot.ai/v1"
BULK_MODEL = "openai/gpt-4o-mini"

RANDOM_SEED = 42
TOP_K_RETRIEVE = 20
MAX_CONCURRENT = 30
MAX_RETRIES = 3

RELEVANCE_PROMPT = """\
You are a fashion search relevance judge. Given a user's search query and a product description, rate how relevant the product is to the query.

Rating scale:
3 = Exact match — product clearly matches all aspects of the query (category, color, style, gender)
2 = Good match — right product category, minor attribute mismatch (e.g. slightly different color or style)
1 = Partial match — same general category but significant mismatches (e.g. right gender but wrong garment type)
0 = Not relevant — completely different product type or category

Query: "{query}"
Product: "{product}"

Respond ONLY with JSON: {{"score": <0-3>, "reason": "<10 words max>"}}"""


def build_article_texts(articles_df: pd.DataFrame) -> dict[str, str]:
    texts = {}
    for _, row in articles_df.iterrows():
        aid = str(row.get("article_id", "")).strip()
        if not aid:
            continue
        parts = []
        for field, limit in [
            ("prod_name", None),
            ("product_type_name", None),
            ("colour_group_name", None),
            ("section_name", None),
            ("garment_group_name", None),
            ("detail_desc", 200),
        ]:
            val = str(row.get(field, "")).strip()
            if val and val.lower() not in ("nan", "none", ""):
                parts.append(val[:limit] if limit else val)
        texts[aid] = " | ".join(parts)
    return texts


def retrieve_candidates(
    queries: list[tuple[str, str]],
    top_k: int = TOP_K_RETRIEVE,
) -> dict[str, list[tuple[str, int]]]:
    """Run FashionCLIP retrieval to get top-K candidates per query.

    Returns {query_id: [(article_id, rank), ...]}
    """
    import torch
    from benchmark.models import load_clip_model, encode_texts_clip

    model_name = "fashion-clip"
    safe_name = model_name.replace("/", "_").replace(":", "_")
    faiss_path = EMBEDDINGS_DIR / f"{safe_name}_faiss.index"
    ids_path = EMBEDDINGS_DIR / f"{safe_name}_article_ids.json"
    assert faiss_path.exists(), f"FAISS index not found: {faiss_path}"

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    log.info("Encoding %d queries with FashionCLIP on %s...", len(queries), device)
    model, _, tokenizer = load_clip_model(model_name, device=device)
    texts = [q[1] for q in queries]
    q_emb = encode_texts_clip(texts, model, tokenizer, device, batch_size=128)
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    q_path = PROCESSED_DIR / "bienc_q_emb.npy"
    out_path = PROCESSED_DIR / "bienc_faiss_results.json"
    np.save(str(q_path), q_emb.astype("float32"))

    worker = Path(__file__).parent / "_faiss_search_worker.py"
    import subprocess
    cmd = [
        sys.executable, str(worker),
        str(q_path), str(faiss_path), str(ids_path),
        str(out_path), str(top_k),
    ]
    log.info("Running FAISS search subprocess (top_k=%d)...", top_k)
    subprocess.run(cmd, check=True)

    with open(out_path) as f:
        raw_results = json.load(f)

    qids = [q[0] for q in queries]
    results: dict[str, list[tuple[str, int]]] = {}
    for qid, candidates in zip(qids, raw_results):
        results[qid] = [(aid, rank + 1) for rank, aid in enumerate(candidates)]

    q_path.unlink(missing_ok=True)
    out_path.unlink(missing_ok=True)

    return results


def build_pairs(
    queries: list[tuple[str, str]],
    retrieval_results: dict[str, list[tuple[str, int]]],
    article_texts: dict[str, str],
) -> list[dict]:
    """Build query-product pairs from retrieval results for LLM labeling."""
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
                "source": "retriever_mined",
            })

    log.info("Built %d retriever-mined pairs from %d queries", len(pairs), len(retrieval_results))
    return pairs


def load_checkpoint(output_path: Path) -> set[str]:
    done = set()
    if output_path.exists():
        with open(output_path) as f:
            for line in f:
                if line.strip():
                    obj = json.loads(line)
                    done.add(f"{obj['query_id']}_{obj['article_id']}")
    log.info("Checkpoint: %d pairs already labeled", len(done))
    return done


async def label_one(
    client: AsyncOpenAI,
    pair: dict,
    model: str,
    semaphore: asyncio.Semaphore,
) -> dict | None:
    prompt = RELEVANCE_PROMPT.format(
        query=pair["query_text"],
        product=pair["product_text"][:300],
    )

    use_json_mode = "gpt-4o" in model

    for attempt in range(MAX_RETRIES):
        try:
            kwargs: dict = dict(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=200,
            )
            if use_json_mode:
                kwargs["response_format"] = {"type": "json_object"}

            async with semaphore:
                resp = await client.chat.completions.create(**kwargs)

            content = resp.choices[0].message.content
            if content is None:
                raise ValueError(f"Empty response (finish={resp.choices[0].finish_reason})")
            text = content.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            obj = json.loads(text)
            score = int(obj.get("score", -1))
            if score < 0 or score > 3:
                raise ValueError(f"Invalid score: {score}")

            return {
                "query_id": pair["query_id"],
                "article_id": pair["article_id"],
                "query_text": pair["query_text"],
                "product_text": pair["product_text"][:300],
                "score": score,
                "reason": str(obj.get("reason", ""))[:100],
                "retriever_rank": pair.get("retriever_rank", 0),
                "source": pair.get("source", "retriever_mined"),
                "model": model,
            }
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(2 ** attempt)
            else:
                log.warning("Failed after %d retries: %s — %s",
                            MAX_RETRIES, pair["article_id"], e)
                return None


async def label_batch(
    pairs: list[dict],
    output_path: Path,
    model: str = BULK_MODEL,
    concurrency: int = MAX_CONCURRENT,
):
    api_key = os.environ.get("PALEBLUEDOT_API_KEY", "")
    if not api_key:
        raise ValueError("Set PALEBLUEDOT_API_KEY environment variable.")

    client = AsyncOpenAI(base_url=BASE_URL, api_key=api_key)
    semaphore = asyncio.Semaphore(concurrency)

    done_keys = load_checkpoint(output_path)
    todo = [p for p in pairs if f"{p['query_id']}_{p['article_id']}" not in done_keys]
    log.info("To label: %d pairs (skipping %d already done)", len(todo), len(done_keys))

    if not todo:
        log.info("All pairs already labeled!")
        return

    t0 = time.time()
    completed = 0
    failed = 0

    CHUNK_SIZE = 200
    for chunk_start in range(0, len(todo), CHUNK_SIZE):
        chunk = todo[chunk_start:chunk_start + CHUNK_SIZE]
        tasks = [label_one(client, p, model, semaphore) for p in chunk]
        results = await asyncio.gather(*tasks)

        with open(output_path, "a") as f:
            for r in results:
                if r is not None:
                    f.write(json.dumps(r) + "\n")
                    completed += 1
                else:
                    failed += 1

        total_done = len(done_keys) + completed
        elapsed = time.time() - t0
        rate = completed / elapsed if elapsed > 0 else 0
        remaining = (len(todo) - completed - failed) / rate if rate > 0 else 0
        log.info(
            "  Progress: %d/%d done (%.1f/s, ~%.0fs remaining, %d failed)",
            total_done, len(pairs), rate, remaining, failed,
        )

    elapsed = time.time() - t0
    log.info(
        "Labeling complete: %d labeled, %d failed in %.1f min (%.1f pairs/s)",
        completed, failed, elapsed / 60, completed / elapsed if elapsed > 0 else 0,
    )


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

    print(f"\nTotal labels: {total:,}")
    print(f"\nScore distribution:")
    for s in range(4):
        pct = 100 * counts[s] / total if total else 0
        bar = "#" * int(pct / 2)
        avg_rank = np.mean(rank_by_score[s]) if rank_by_score[s] else 0
        print(f"  {s}: {counts[s]:>6,}  ({pct:5.1f}%)  avg_rank={avg_rank:.1f}  {bar}")

    usable_pos = counts[2] + counts[3]
    usable_neg = counts[0]
    print(f"\nUsable for contrastive training:")
    print(f"  Positives (score 2-3): {usable_pos:,}")
    print(f"  Hard negatives (score 0): {usable_neg:,}")
    print(f"  Ambiguous (score 1): {counts[1]:,} (can be used as soft negatives)")


def verify_no_leakage(queries: list[tuple[str, str]]):
    """Assert that no query in our set belongs to the test split."""
    splits = json.loads(SPLIT_PATH.read_text())
    test_qids = set(splits["test"])
    val_qids = set(splits["val"])

    query_qids = {qid for qid, _ in queries}
    test_overlap = query_qids & test_qids
    val_overlap = query_qids & val_qids

    assert len(test_overlap) == 0, \
        f"LEAK DETECTED: {len(test_overlap)} queries overlap with test split!"
    assert len(val_overlap) == 0, \
        f"LEAK DETECTED: {len(val_overlap)} queries overlap with val split!"

    qid_to_text = {}
    with open(HNM_DIR / "queries.csv", newline="") as f:
        for row in csv.DictReader(f):
            qid_to_text[row["query_id"].strip()] = row["query_text"].strip()

    test_texts = {qid_to_text.get(qid, "") for qid in test_qids}
    query_texts = {qt for _, qt in queries}
    text_overlap = query_texts & test_texts

    assert len(text_overlap) == 0, \
        f"TEXT LEAK DETECTED: {len(text_overlap)} query texts overlap with test split!"

    log.info("Leakage check PASSED: 0 ID overlap, 0 text overlap with test/val splits")


def main():
    parser = argparse.ArgumentParser(description="Generate bi-encoder training labels")
    parser.add_argument("--max_queries", type=int, default=5000,
                        help="Number of train queries to mine (default: 5000)")
    parser.add_argument("--top_k", type=int, default=TOP_K_RETRIEVE,
                        help="Candidates per query from retriever (default: 20)")
    parser.add_argument("--concurrency", type=int, default=MAX_CONCURRENT)
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--model", type=str, default=BULK_MODEL)
    args = parser.parse_args()

    if args.report:
        report_distribution()
        return

    log.info("=" * 60)
    log.info("MODA Phase 3C — Bi-Encoder Training Data Generation")
    log.info("Strategy: FashionCLIP retrieval → GPT-4o-mini labeling")
    log.info("=" * 60)

    # Load splits and select train queries
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
    # Deduplicate by query text to maximize diversity
    seen_texts: set[str] = set()
    unique_queries: list[tuple[str, str]] = []
    for qid, qt in all_queries:
        if qt not in seen_texts:
            seen_texts.add(qt)
            unique_queries.append((qid, qt))
    log.info("Unique train query texts: %d", len(unique_queries))

    queries = unique_queries[:args.max_queries]
    log.info("Selected %d queries for retrieval mining", len(queries))

    # Leakage verification
    verify_no_leakage(queries)

    # Load articles
    log.info("Loading H&M articles...")
    articles_df = pd.read_csv(HNM_DIR / "articles.csv", dtype=str).fillna("")
    article_texts = build_article_texts(articles_df)
    log.info("Articles loaded: %d", len(article_texts))

    # Run FashionCLIP retrieval
    retrieval_results = retrieve_candidates(queries, top_k=args.top_k)

    # Build pairs for labeling
    pairs = build_pairs(queries, retrieval_results, article_texts)
    log.info("Total pairs to label: %d (~$%.1f at GPT-4o-mini rates)",
             len(pairs), len(pairs) * 0.00008)

    # Label with LLM
    asyncio.run(label_batch(
        pairs, LABELS_PATH,
        model=args.model,
        concurrency=args.concurrency,
    ))

    report_distribution()


if __name__ == "__main__":
    main()
