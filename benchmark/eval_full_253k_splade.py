"""
MODA — 253K SPLADE Evaluation

Runs top SPLADE configs on the full 253,685-query benchmark with bootstrap CIs.
Appends results to results/full/full_ablation.json alongside existing BM25/Dense configs.

Stages:
  1. SPLADE encode articles + query search (cached)
  2. Dense (FashionCLIP) retrieval (reuse existing cache if available)
  3. Evaluate: SPLADE-only, SPLADE+Dense RRF (multiple weights)
  4. (Optional) CE rerank best hybrid → full pipeline

Usage:
  python -m benchmark.eval_full_253k_splade
  python -m benchmark.eval_full_253k_splade --with_ce   # adds CE reranking (~8h extra)
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from benchmark.metrics import compute_all_metrics, aggregate_metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

HNM_DIR = _REPO_ROOT / "data" / "raw" / "hnm_real"
OUT_DIR = _REPO_ROOT / "results" / "full"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SPLADE_CACHE = OUT_DIR / "splade_top100.json"
DENSE_CACHE = OUT_DIR / "dense_top100.json"

SPLADE_MODEL = "naver/splade-cocondenser-ensembledistil"
TOP_K = 100
KS = [5, 10, 20, 50]
RRF_K = 60
RANDOM_SEED = 42


def load_benchmark() -> tuple[list[tuple[str, str]], dict[str, dict[str, int]]]:
    log.info("Loading full H&M benchmark...")
    qrels: dict[str, dict[str, int]] = {}
    with open(HNM_DIR / "qrels.csv", newline="") as f:
        for row in csv.DictReader(f):
            qid = row["query_id"].strip()
            pos_ids = [x.strip() for x in row.get("positive_ids", "").split() if x.strip()]
            neg_ids = [x.strip() for x in row.get("negative_ids", "").split() if x.strip()]
            grades = {aid: 2 for aid in pos_ids}
            grades.update({aid: 1 for aid in neg_ids if aid not in grades})
            qrels[qid] = grades

    df = pd.read_csv(HNM_DIR / "queries.csv")
    valid = set(qrels.keys())
    df = df[df["query_id"].astype(str).isin(valid)]
    queries = [(str(r["query_id"]), str(r["query_text"])) for _, r in df.iterrows()]
    log.info("Loaded %d queries, %d qrels entries", len(queries), len(qrels))
    return queries, qrels


def stage_splade(queries: list[tuple[str, str]], force: bool = False) -> dict[str, list[str]]:
    """SPLADE retrieval for all queries. Caches to disk."""
    if SPLADE_CACHE.exists() and not force:
        log.info("Loading cached SPLADE results from %s", SPLADE_CACHE)
        return json.loads(SPLADE_CACHE.read_text())

    from benchmark.splade_retriever import SpladeRetriever

    log.info("Initializing SPLADE retriever (%s)...", SPLADE_MODEL)
    splade = SpladeRetriever(model_name=SPLADE_MODEL)

    from benchmark.eval_splade_pipeline import load_article_ids_and_texts
    article_ids, article_texts = load_article_ids_and_texts()
    log.info("Encoding %d articles...", len(article_ids))
    splade.encode_articles(article_ids, [article_texts[aid] for aid in article_ids])

    texts = [qt for _, qt in queries]
    qids = [qid for qid, _ in queries]

    log.info("Searching %d queries (chunked)...", len(queries))
    result_lists = splade.search_batch(texts, top_k=TOP_K, query_chunk=500)

    results = {qid: lst for qid, lst in zip(qids, result_lists)}

    splade.free_model()

    SPLADE_CACHE.write_text(json.dumps(results))
    log.info("SPLADE results cached → %s", SPLADE_CACHE)
    return results


def stage_dense(queries: list[tuple[str, str]], force: bool = False) -> dict[str, list[str]]:
    """Dense (FashionCLIP) retrieval. Reuses existing cache if available."""
    if DENSE_CACHE.exists() and not force:
        log.info("Loading cached dense results from %s", DENSE_CACHE)
        return json.loads(DENSE_CACHE.read_text())

    from benchmark.eval_splade_pipeline import (
        load_article_ids_and_texts,
        dense_retrieval_fashionclip,
    )

    article_ids, article_texts = load_article_ids_and_texts()
    log.info("Running FashionCLIP dense retrieval for %d queries...", len(queries))
    dense_lists = dense_retrieval_fashionclip(
        queries, article_ids, article_texts,
        top_k=TOP_K, faiss_query_chunk=512,
    )
    qids = [qid for qid, _ in queries]
    results = {qid: lst for qid, lst in zip(qids, dense_lists)}

    DENSE_CACHE.write_text(json.dumps(results))
    log.info("Dense results cached → %s", DENSE_CACHE)
    return results


def rrf_fusion(
    list_a: list[list[str]], list_b: list[list[str]],
    k: int = RRF_K, w_a: float = 0.4, w_b: float = 0.6,
    top_k: int = TOP_K,
) -> list[list[str]]:
    fused = []
    for a, b in zip(list_a, list_b):
        scores: dict[str, float] = {}
        for rank, doc in enumerate(a, 1):
            scores[doc] = scores.get(doc, 0.0) + w_a / (k + rank)
        for rank, doc in enumerate(b, 1):
            scores[doc] = scores.get(doc, 0.0) + w_b / (k + rank)
        fused.append([d for d, _ in sorted(scores.items(), key=lambda x: -x[1])[:top_k]])
    return fused


def bootstrap_ci(
    values: list[float], n_boot: int = 5000, ci: float = 0.95,
) -> tuple[float, float]:
    rng = np.random.default_rng(RANDOM_SEED)
    boot_means = [rng.choice(values, size=len(values), replace=True).mean()
                  for _ in range(n_boot)]
    alpha = (1 - ci) / 2
    return float(np.quantile(boot_means, alpha)), float(np.quantile(boot_means, 1 - alpha))


def evaluate_configs(
    qids: list[str],
    qrels: dict[str, dict[str, int]],
    configs: dict[str, list[list[str]]],
) -> dict[str, dict]:
    results = {}
    for name, retrieved_lists in configs.items():
        if retrieved_lists is None:
            log.info("  Skipping %s (no data)", name)
            continue
        per_q = []
        ndcg10_vals = []
        for qid, retrieved in zip(qids, retrieved_lists):
            q_qrels = qrels.get(qid, {})
            if not q_qrels:
                continue
            m = compute_all_metrics(retrieved, q_qrels, ks=KS)
            per_q.append(m)
            ndcg10_vals.append(m.get("ndcg@10", 0))

        agg = aggregate_metrics(per_q)
        lo, hi = bootstrap_ci(ndcg10_vals)
        results[name] = {
            "n_queries": len(per_q),
            "metrics": agg,
            "ndcg10_ci95": [round(lo, 4), round(hi, 4)],
        }
        log.info(
            "  %-35s nDCG@10=%.4f [%.4f–%.4f]  MRR=%.4f  R@10=%.4f",
            name, agg.get("ndcg@10", 0), lo, hi,
            agg.get("mrr", 0), agg.get("recall@10", 0),
        )
    return results


def main():
    parser = argparse.ArgumentParser(description="253K SPLADE eval")
    parser.add_argument("--force", action="store_true", help="Recompute even if cache exists")
    parser.add_argument("--with_ce", action="store_true", help="Add CE reranking (slow)")
    args = parser.parse_args()

    t0 = time.time()
    queries, qrels = load_benchmark()
    qids = [qid for qid, _ in queries]

    splade_results = stage_splade(queries, force=args.force)
    dense_results = stage_dense(queries, force=args.force)

    splade_lists = [splade_results.get(qid, []) for qid in qids]
    dense_lists = [dense_results.get(qid, []) for qid in qids]

    configs = {
        "9_SPLADE_only": splade_lists,
        "10_SPLADE_Dense_03_07": rrf_fusion(splade_lists, dense_lists, w_a=0.3, w_b=0.7),
        "11_SPLADE_Dense_04_06": rrf_fusion(splade_lists, dense_lists, w_a=0.4, w_b=0.6),
        "12_SPLADE_Dense_05_05": rrf_fusion(splade_lists, dense_lists, w_a=0.5, w_b=0.5),
    }

    log.info("Evaluating %d SPLADE configs on %d queries...", len(configs), len(qids))
    splade_eval = evaluate_configs(qids, qrels, configs)

    ablation_path = OUT_DIR / "full_ablation.json"
    if ablation_path.exists():
        existing = json.loads(ablation_path.read_text())
    else:
        existing = {}

    existing.update(splade_eval)
    ablation_path.write_text(json.dumps(existing, indent=2))
    log.info("Updated %s with %d SPLADE configs", ablation_path, len(splade_eval))

    elapsed = time.time() - t0
    print("\n" + "=" * 90)
    print("MODA — 253K SPLADE Evaluation Results")
    print("=" * 90)
    print(f"{'Config':<35} {'nDCG@10':>9} {'95% CI':>16} {'MRR':>9} {'R@10':>9}")
    print("-" * 90)
    for name, res in splade_eval.items():
        m = res["metrics"]
        ci = res.get("ndcg10_ci95", [0, 0])
        print(
            f"{name:<35} {m.get('ndcg@10', 0):>9.4f} "
            f"[{ci[0]:.4f}–{ci[1]:.4f}] "
            f"{m.get('mrr', 0):>9.4f} {m.get('recall@10', 0):>9.4f}"
        )
    print("=" * 90)
    print(f"Total elapsed: {elapsed / 60:.1f} min")
    print()


if __name__ == "__main__":
    main()
