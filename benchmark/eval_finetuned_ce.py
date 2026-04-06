"""
MODA Phase 3 — Cross-Encoder Evaluation (Leak-Free)

Compares off-the-shelf, Phase 3A fine-tuned, and Phase 3B LLM-trained CE
rerankers using ONLY the held-out test split from train_cross_encoder.py.

Pipeline (same as Phase 2 Config 8):
  NER-boosted BM25 × 0.4 + FashionCLIP × 0.6 → RRF top-100 → CE rerank → top-50

Configs evaluated:
  1. Hybrid NER baseline (no rerank)        — retrieval quality reference
  2. Hybrid NER + off-the-shelf CE@50       — ms-marco-MiniLM-L-6-v2
  3. Hybrid NER + fine-tuned CE@50          — moda-fashion-ce-best (Phase 3A)
  4. Hybrid NER + LLM-trained CE@50         — moda-fashion-ce-llm-best (Phase 3B)

Usage:
  python benchmark/eval_finetuned_ce.py
  python benchmark/eval_finetuned_ce.py --n_queries 5000
  python benchmark/eval_finetuned_ce.py --n_queries 0   # use ALL test queries (~22K)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
from opensearchpy import OpenSearch
from tqdm import tqdm

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from benchmark.eval_full_pipeline import (
    load_benchmark,
    load_articles,
    load_or_compute_ner,
    bm25_ner_search,
    dense_search_batch,
    rrf_fusion,
    evaluate,
    ce_rerank_batch,
    _build_article_text,
    RESULTS_DIR,
    DENSE_MODEL,
    BM25_WEIGHT,
    DENSE_WEIGHT,
    TOP_K_RERANK,
    TOP_K_FINAL,
    HNM_DIR,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

SPLIT_PATH = _REPO_ROOT / "data" / "processed" / "query_splits.json"
FINETUNED_CE = str(_REPO_ROOT / "models" / "moda-fashion-ce-best")
LLM_TRAINED_CE = str(_REPO_ROOT / "models" / "moda-fashion-ce-llm-best")
OFFSHELF_CE = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RESULTS_PATH = RESULTS_DIR / "hnm_finetuned_ce_eval.json"


def load_test_query_ids() -> set[str]:
    if not SPLIT_PATH.exists():
        raise FileNotFoundError(
            f"Query split file not found: {SPLIT_PATH}\n"
            "Run train_cross_encoder.py first to generate train/val/test splits."
        )
    splits = json.loads(SPLIT_PATH.read_text())
    test_qids = set(splits["test"])
    log.info("Loaded %d test query IDs from %s", len(test_qids), SPLIT_PATH)

    train_qids = set(splits["train"])
    val_qids = set(splits["val"])
    overlap = test_qids & (train_qids | val_qids)
    assert len(overlap) == 0, f"LEAK DETECTED: {len(overlap)} test qids overlap with train/val!"

    return test_qids


def main():
    p = argparse.ArgumentParser(
        description="MODA Phase 3 — Fine-tuned CE evaluation (leak-free)")
    p.add_argument("--n_queries", type=int, default=0,
                   help="Sample N test queries (0 = use all ~22K test queries)")
    args = p.parse_args()

    t_start = time.time()

    # ── 1. Load test split ──────────────────────────────────────────────────
    test_qids = load_test_query_ids()

    # ── 2. Load full benchmark, then filter to test-only ────────────────────
    all_queries, all_qrels = load_benchmark(n_queries=None)

    queries = [(qid, qt) for qid, qt in all_queries if qid in test_qids]
    qrels = {qid: v for qid, v in all_qrels.items() if qid in test_qids}
    log.info("Filtered to test split: %d → %d queries", len(all_queries), len(queries))

    if args.n_queries > 0 and len(queries) > args.n_queries:
        import random
        rng = random.Random(42)
        queries = rng.sample(queries, args.n_queries)
        log.info("Sampled %d test queries (seed=42)", args.n_queries)

    articles = load_articles()
    qids = [q[0] for q in queries]
    texts = [q[1] for q in queries]

    # ── 3. NER ──────────────────────────────────────────────────────────────
    ner_cache = load_or_compute_ner(queries)

    # ── 4. BM25 (NER-boosted) ──────────────────────────────────────────────
    client = OpenSearch(
        hosts=[{"host": "localhost", "port": 9200}],
        http_compress=True, timeout=30,
    )
    log.info("Running NER-boosted BM25 retrieval...")
    bm25_results: dict[str, list[str]] = {}
    for qid, text in tqdm(queries, desc="BM25+NER", ncols=80):
        ner_entities = ner_cache.get(qid, {})
        bm25_results[qid] = bm25_ner_search(
            client, text, ner_entities, top_k=TOP_K_RERANK)

    # ── 5. Dense retrieval ──────────────────────────────────────────────────
    dense_results = dense_search_batch(
        queries, model_name=DENSE_MODEL, top_k=TOP_K_RERANK)

    # ── 6. Hybrid RRF ──────────────────────────────────────────────────────
    bm25_lists = [bm25_results.get(qid, []) for qid in qids]
    dense_lists = [dense_results.get(qid, []) for qid in qids]
    hybrid_lists = rrf_fusion(bm25_lists, dense_lists)
    hybrid_results = {qid: lst for qid, lst in zip(qids, hybrid_lists)}

    res_baseline = evaluate(hybrid_results, qrels, label="Hybrid_NER_baseline")

    # ── 7. Off-the-shelf CE reranking ───────────────────────────────────────
    log.info("CE reranking with OFF-THE-SHELF model: %s", OFFSHELF_CE)
    offshelf_lists = ce_rerank_batch(
        texts, hybrid_lists, articles, model_name=OFFSHELF_CE)
    offshelf_results = {qid: lst for qid, lst in zip(qids, offshelf_lists)}
    res_offshelf = evaluate(
        offshelf_results, qrels, label="Hybrid_NER_CE_offshelf@50")

    # ── 8. Fine-tuned CE reranking (Phase 3A) ───────────────────────────────
    log.info("CE reranking with FINE-TUNED model: %s", FINETUNED_CE)
    finetuned_lists = ce_rerank_batch(
        texts, hybrid_lists, articles, model_name=FINETUNED_CE)
    finetuned_results = {qid: lst for qid, lst in zip(qids, finetuned_lists)}
    res_finetuned = evaluate(
        finetuned_results, qrels, label="Hybrid_NER_CE_finetuned@50")

    # ── 9. LLM-trained CE reranking (Phase 3B) ──────────────────────────
    all_results = {
        "Hybrid_NER_baseline": res_baseline,
        "Hybrid_NER_CE_offshelf@50": res_offshelf,
        "Hybrid_NER_CE_finetuned@50": res_finetuned,
    }

    llm_ce_path = Path(LLM_TRAINED_CE)
    if llm_ce_path.exists():
        log.info("CE reranking with LLM-TRAINED model: %s", LLM_TRAINED_CE)
        llm_lists = ce_rerank_batch(
            texts, hybrid_lists, articles, model_name=LLM_TRAINED_CE)
        llm_results = {qid: lst for qid, lst in zip(qids, llm_lists)}
        res_llm = evaluate(
            llm_results, qrels, label="Hybrid_NER_CE_llm_trained@50")
        all_results["Hybrid_NER_CE_llm_trained@50"] = res_llm
    else:
        log.warning("LLM-trained CE not found at %s — skipping", LLM_TRAINED_CE)

    # ── 10. Save results ────────────────────────────────────────────────────
    output = {
        "configs": all_results,
        "settings": {
            "n_queries": len(queries),
            "split": "test_only",
            "test_qids_total": len(test_qids),
            "offshelf_model": OFFSHELF_CE,
            "finetuned_model": FINETUNED_CE,
            "llm_trained_model": LLM_TRAINED_CE,
            "pool_size": TOP_K_RERANK,
            "rerank_top_k": TOP_K_FINAL,
        },
    }
    with open(RESULTS_PATH, "w") as f:
        json.dump(output, f, indent=2)
    log.info("Results saved → %s", RESULTS_PATH)

    # ── 11. Print comparison ────────────────────────────────────────────────
    ref_ndcg = res_offshelf["metrics"]["ndcg@10"]

    print("\n" + "=" * 90)
    print("PHASE 3 — CE Evaluation (TEST SPLIT ONLY — Zero Leakage)")
    print(f"  {len(queries):,} test queries  |  Pool: {TOP_K_RERANK} → Top-{TOP_K_FINAL}"
          f"  |  Split: {len(test_qids):,} held-out test qids")
    print("=" * 90)
    print(f"{'Config':<40} {'nDCG@10':>9} {'MRR':>9} {'R@10':>9}"
          f"  vs off-shelf CE")
    print("-" * 90)
    for name, res in all_results.items():
        m = res["metrics"]
        ndcg = m["ndcg@10"]
        delta = (ndcg / ref_ndcg - 1) * 100 if ref_ndcg > 0 else 0
        sign = "+" if delta >= 0 else ""
        icon = "✅" if delta > 1 else ("❌" if delta < -1 else "≈")
        print(
            f"  {name:<38} {ndcg:>9.4f} {m['mrr']:>9.4f}"
            f" {m['recall@10']:>9.4f}  {sign}{delta:.1f}% {icon}"
        )
    print("=" * 90)
    print(f"\nTotal elapsed: {(time.time() - t_start) / 60:.1f} min\n")

    return all_results


if __name__ == "__main__":
    main()
