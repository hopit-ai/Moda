"""
MODA — Recall Boost Evaluation

Tests strategies to improve retrieval recall and push nDCG@10 toward 0.1:
  1. Larger pool sizes (200, 500) with direct CE reranking
  2. Two-stage reranking: bi-encoder filter (500→100) then CE rerank (100→top-k)
  3. Asymmetric retrieval: Dense@500 + BM25@100 → RRF
  4. Fine-tuned bi-encoder as the dense channel

Usage:
  python -m benchmark.eval_recall_boost
  python -m benchmark.eval_recall_boost --configs pool200,twostage500
  python -m benchmark.eval_recall_boost --n_queries 5000  # quick test
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
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
    DENSE_MODEL,
    BM25_WEIGHT,
    DENSE_WEIGHT,
    RRF_K,
    RESULTS_DIR,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

SPLIT_PATH = _REPO_ROOT / "data" / "processed" / "query_splits.json"
LLM_CE = str(_REPO_ROOT / "models" / "moda-fashion-ce-llm-best")
FINETUNED_BIENC = _REPO_ROOT / "models" / "moda-fashionclip-finetuned" / "best"
EMBEDDINGS_DIR = _REPO_ROOT / "data" / "processed" / "embeddings"
RESULTS_PATH = RESULTS_DIR / "recall_boost_eval.json"


# ── Bi-encoder re-scoring for two-stage reranking ────────────────────────────

def load_finetuned_article_embeddings() -> tuple[np.ndarray, list[str]]:
    """Encode all articles with fine-tuned FashionCLIP. Cache to disk."""
    cache_path = EMBEDDINGS_DIR / "finetuned_article_emb.npy"
    ids_path = EMBEDDINGS_DIR / "fashion-clip_article_ids.json"

    with open(ids_path) as f:
        article_ids = json.load(f)

    if cache_path.exists():
        log.info("Loading cached fine-tuned article embeddings from %s", cache_path)
        return np.load(str(cache_path)), article_ids

    log.info("Encoding %d articles with fine-tuned FashionCLIP (one-time)...", len(article_ids))
    from benchmark.eval_finetuned_biencoder import load_articles as load_article_texts, encode_with_finetuned
    article_texts = load_article_texts()
    ordered_texts = [article_texts.get(aid, "") for aid in article_ids]

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    emb = encode_with_finetuned(ordered_texts, FINETUNED_BIENC, device)
    np.save(str(cache_path), emb)
    log.info("Cached fine-tuned article embeddings → %s", cache_path)
    return emb, article_ids


def encode_queries_finetuned(texts: list[str]) -> np.ndarray:
    """Encode query texts with fine-tuned FashionCLIP."""
    from benchmark.eval_finetuned_biencoder import encode_with_finetuned
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    return encode_with_finetuned(texts, FINETUNED_BIENC, device)


def biencoder_rescore(
    query_embs: np.ndarray,
    candidate_lists: list[list[str]],
    article_emb: np.ndarray,
    article_id_to_idx: dict[str, int],
    top_k: int = 100,
) -> list[list[str]]:
    """Re-score candidates using fine-tuned bi-encoder dot product, keep top-k.

    This is the Stage-1 filter: fast dot-product scoring to prune
    500 candidates down to top-k before expensive CE reranking.
    """
    results = []
    for i, candidates in enumerate(candidate_lists):
        if not candidates:
            results.append([])
            continue
        q_emb = query_embs[i]
        idxs = [article_id_to_idx.get(cid) for cid in candidates]
        valid = [(cid, idx) for cid, idx in zip(candidates, idxs) if idx is not None]
        if not valid:
            results.append(candidates[:top_k])
            continue
        cids, idx_arr = zip(*valid)
        cand_emb = article_emb[list(idx_arr)]
        scores = cand_emb @ q_emb
        ranked_idx = np.argsort(-scores)[:top_k]
        results.append([cids[j] for j in ranked_idx])
    return results


# ── Main ─────────────────────────────────────────────────────────────────────

ALL_CONFIGS = [
    "baseline100",
    "pool200",
    "pool500",
    "twostage500",
    "asymmetric",
    "ft_bienc_pool200",
    "ft_bienc_twostage500",
]


def main():
    p = argparse.ArgumentParser(description="MODA — Recall Boost Evaluation")
    p.add_argument("--n_queries", type=int, default=0,
                   help="Sample N test queries (0 = all ~22K)")
    p.add_argument("--configs", type=str, default="all",
                   help=f"Comma-separated configs to run. Options: {','.join(ALL_CONFIGS)} or 'all'")
    args = p.parse_args()

    configs_to_run = ALL_CONFIGS if args.configs == "all" else args.configs.split(",")
    t_start = time.time()

    # ── Load data ────────────────────────────────────────────────────────────
    splits = json.loads(SPLIT_PATH.read_text())
    test_qids = set(splits["test"])

    all_queries, all_qrels = load_benchmark(n_queries=None)
    queries = [(qid, qt) for qid, qt in all_queries if qid in test_qids]
    qrels = {qid: v for qid, v in all_qrels.items() if qid in test_qids}

    if args.n_queries > 0 and len(queries) > args.n_queries:
        import random
        rng = random.Random(42)
        queries = rng.sample(queries, args.n_queries)

    articles = load_articles()
    qids = [q[0] for q in queries]
    texts = [q[1] for q in queries]
    log.info("Loaded %d test queries", len(queries))

    # ── NER ──────────────────────────────────────────────────────────────────
    ner_cache = load_or_compute_ner(queries)

    # ── OpenSearch client ────────────────────────────────────────────────────
    client = OpenSearch(
        hosts=[{"host": "localhost", "port": 9200}],
        http_compress=True, timeout=30,
    )

    # ── Pre-compute retrievals at various depths ─────────────────────────────
    bm25_cache: dict[int, dict[str, list[str]]] = {}
    dense_cache: dict[int, dict[str, list[str]]] = {}

    needed_bm25_depths = set()
    needed_dense_depths = set()

    for cfg in configs_to_run:
        if cfg in ("baseline100", "pool200", "pool500", "twostage500"):
            if cfg == "baseline100":
                needed_bm25_depths.add(100); needed_dense_depths.add(100)
            elif cfg == "pool200":
                needed_bm25_depths.add(200); needed_dense_depths.add(200)
            elif cfg in ("pool500", "twostage500"):
                needed_bm25_depths.add(500); needed_dense_depths.add(500)
        elif cfg == "asymmetric":
            needed_bm25_depths.add(100); needed_dense_depths.add(500)
        elif cfg in ("ft_bienc_pool200", "ft_bienc_twostage500"):
            if cfg == "ft_bienc_pool200":
                needed_bm25_depths.add(200)
            else:
                needed_bm25_depths.add(500)

    max_bm25 = max(needed_bm25_depths) if needed_bm25_depths else 0
    max_dense = max(needed_dense_depths) if needed_dense_depths else 0

    if max_bm25 > 0:
        log.info("BM25 retrieval (top-%d)...", max_bm25)
        bm25_all: dict[str, list[str]] = {}
        for qid, text in tqdm(queries, desc=f"BM25 top-{max_bm25}", ncols=80):
            ner_entities = ner_cache.get(qid, {})
            bm25_all[qid] = bm25_ner_search(client, text, ner_entities, top_k=max_bm25)
        for depth in needed_bm25_depths:
            bm25_cache[depth] = {qid: lst[:depth] for qid, lst in bm25_all.items()}

    if max_dense > 0:
        log.info("Dense retrieval (top-%d)...", max_dense)
        dense_all = dense_search_batch(queries, model_name=DENSE_MODEL, top_k=max_dense)
        for depth in needed_dense_depths:
            dense_cache[depth] = {qid: lst[:depth] for qid, lst in dense_all.items()}

    # ── Pre-load fine-tuned bi-encoder if needed ─────────────────────────────
    ft_configs = [c for c in configs_to_run if c.startswith("ft_bienc") or c == "twostage500"
                  or c == "ft_bienc_twostage500"]
    article_emb = None
    article_id_to_idx = None
    query_embs_ft = None

    if ft_configs:
        article_emb, article_ids = load_finetuned_article_embeddings()
        article_id_to_idx = {aid: i for i, aid in enumerate(article_ids)}

        log.info("Encoding %d queries with fine-tuned FashionCLIP...", len(texts))
        query_embs_ft = encode_queries_finetuned(texts)

    # ── Fine-tuned bi-encoder dense retrieval ────────────────────────────────
    ft_dense_cache: dict[int, dict[str, list[str]]] = {}
    ft_needed = set()
    for cfg in configs_to_run:
        if cfg == "ft_bienc_pool200":
            ft_needed.add(200)
        elif cfg == "ft_bienc_twostage500":
            ft_needed.add(500)

    if ft_needed and article_emb is not None:
        max_ft = max(ft_needed)
        log.info("Fine-tuned bi-encoder FAISS search (top-%d)...", max_ft)
        from benchmark.eval_finetuned_biencoder import build_and_search_faiss
        raw_lists = build_and_search_faiss(query_embs_ft, article_emb, article_ids, top_k=max_ft)
        ft_dense_all = {qid: lst for qid, lst in zip(qids, raw_lists)}
        for depth in ft_needed:
            ft_dense_cache[depth] = {qid: lst[:depth] for qid, lst in ft_dense_all.items()}

    # ── Run each config ──────────────────────────────────────────────────────
    all_results = {}

    def make_hybrid(bm25_depth, dense_depth, pool_size, dense_source="vanilla"):
        bm25_d = bm25_cache[bm25_depth]
        if dense_source == "vanilla":
            dense_d = dense_cache[dense_depth]
        else:
            dense_d = ft_dense_cache[dense_depth]
        bm25_lists = [bm25_d.get(qid, []) for qid in qids]
        dense_lists = [dense_d.get(qid, []) for qid in qids]
        return rrf_fusion(bm25_lists, dense_lists, top_k=pool_size)

    for cfg in configs_to_run:
        log.info("=" * 60)
        log.info("CONFIG: %s", cfg)
        log.info("=" * 60)

        if cfg == "baseline100":
            hybrid_lists = make_hybrid(100, 100, 100)
            hybrid_results = {qid: lst for qid, lst in zip(qids, hybrid_lists)}
            all_results["Baseline@100"] = evaluate(hybrid_results, qrels, label="Baseline@100")
            ce_lists = ce_rerank_batch(texts, hybrid_lists, articles, model_name=LLM_CE, top_k=50)
            ce_results = {qid: lst for qid, lst in zip(qids, ce_lists)}
            all_results["CE_pool100"] = evaluate(ce_results, qrels, label="CE_pool100")

        elif cfg == "pool200":
            hybrid_lists = make_hybrid(200, 200, 200)
            hybrid_results = {qid: lst for qid, lst in zip(qids, hybrid_lists)}
            all_results["Baseline@200"] = evaluate(hybrid_results, qrels, label="Baseline@200")
            ce_lists = ce_rerank_batch(texts, hybrid_lists, articles, model_name=LLM_CE, top_k=50)
            ce_results = {qid: lst for qid, lst in zip(qids, ce_lists)}
            all_results["CE_pool200"] = evaluate(ce_results, qrels, label="CE_pool200")

        elif cfg == "pool500":
            hybrid_lists = make_hybrid(500, 500, 500)
            hybrid_results = {qid: lst for qid, lst in zip(qids, hybrid_lists)}
            all_results["Baseline@500"] = evaluate(hybrid_results, qrels, label="Baseline@500")
            log.info("CE reranking 500 candidates (this will be slow)...")
            ce_lists = ce_rerank_batch(texts, hybrid_lists, articles, model_name=LLM_CE, top_k=50)
            ce_results = {qid: lst for qid, lst in zip(qids, ce_lists)}
            all_results["CE_pool500"] = evaluate(ce_results, qrels, label="CE_pool500")

        elif cfg == "twostage500":
            hybrid_lists = make_hybrid(500, 500, 500)
            hybrid_results = {qid: lst for qid, lst in zip(qids, hybrid_lists)}
            all_results["Baseline@500_ts"] = evaluate(hybrid_results, qrels, label="Baseline@500")

            log.info("Stage 1: Bi-encoder re-scoring 500 → 100...")
            filtered_lists = biencoder_rescore(
                query_embs_ft, hybrid_lists, article_emb, article_id_to_idx, top_k=100)
            filtered_results = {qid: lst for qid, lst in zip(qids, filtered_lists)}
            all_results["BiEnc_filter@100"] = evaluate(
                filtered_results, qrels, label="BiEnc_filter_500→100")

            log.info("Stage 2: CE reranking 100 → top-50...")
            ce_lists = ce_rerank_batch(texts, filtered_lists, articles, model_name=LLM_CE, top_k=50)
            ce_results = {qid: lst for qid, lst in zip(qids, ce_lists)}
            all_results["TwoStage_500_100_CE"] = evaluate(
                ce_results, qrels, label="TwoStage_500→100→CE")

        elif cfg == "asymmetric":
            bm25_lists = [bm25_cache[100].get(qid, []) for qid in qids]
            dense_lists = [dense_cache[500].get(qid, []) for qid in qids]
            hybrid_lists = rrf_fusion(bm25_lists, dense_lists, top_k=500)
            hybrid_results = {qid: lst for qid, lst in zip(qids, hybrid_lists)}
            all_results["Asymmetric_B100_D500"] = evaluate(
                hybrid_results, qrels, label="Asymmetric_BM25@100+Dense@500")

            log.info("Stage 1: Bi-encoder re-scoring 500 → 100...")
            filtered_lists = biencoder_rescore(
                query_embs_ft, hybrid_lists, article_emb, article_id_to_idx, top_k=100)

            log.info("Stage 2: CE reranking 100 → top-50...")
            ce_lists = ce_rerank_batch(texts, filtered_lists, articles, model_name=LLM_CE, top_k=50)
            ce_results = {qid: lst for qid, lst in zip(qids, ce_lists)}
            all_results["Asymmetric_TwoStage_CE"] = evaluate(
                ce_results, qrels, label="Asymmetric_TwoStage_CE")

        elif cfg == "ft_bienc_pool200":
            hybrid_lists = make_hybrid(200, 200, 200, dense_source="finetuned")
            hybrid_results = {qid: lst for qid, lst in zip(qids, hybrid_lists)}
            all_results["FT_Baseline@200"] = evaluate(
                hybrid_results, qrels, label="FT_Bienc_Baseline@200")
            ce_lists = ce_rerank_batch(texts, hybrid_lists, articles, model_name=LLM_CE, top_k=50)
            ce_results = {qid: lst for qid, lst in zip(qids, ce_lists)}
            all_results["FT_CE_pool200"] = evaluate(
                ce_results, qrels, label="FT_Bienc_CE_pool200")

        elif cfg == "ft_bienc_twostage500":
            hybrid_lists = make_hybrid(500, 500, 500, dense_source="finetuned")
            hybrid_results = {qid: lst for qid, lst in zip(qids, hybrid_lists)}
            all_results["FT_Baseline@500"] = evaluate(
                hybrid_results, qrels, label="FT_Bienc_Baseline@500")

            log.info("Stage 1: Bi-encoder re-scoring 500 → 100...")
            filtered_lists = biencoder_rescore(
                query_embs_ft, hybrid_lists, article_emb, article_id_to_idx, top_k=100)

            log.info("Stage 2: CE reranking 100 → top-50...")
            ce_lists = ce_rerank_batch(texts, filtered_lists, articles, model_name=LLM_CE, top_k=50)
            ce_results = {qid: lst for qid, lst in zip(qids, ce_lists)}
            all_results["FT_TwoStage_500_CE"] = evaluate(
                ce_results, qrels, label="FT_TwoStage_500→100→CE")

    # ── Save results ─────────────────────────────────────────────────────────
    output = {
        "configs": all_results,
        "settings": {
            "n_queries": len(queries),
            "split": "test_only",
            "configs_run": configs_to_run,
            "llm_ce_model": LLM_CE,
            "dense_model": DENSE_MODEL,
            "finetuned_bienc": str(FINETUNED_BIENC),
        },
    }
    with open(RESULTS_PATH, "w") as f:
        json.dump(output, f, indent=2)
    log.info("Results saved → %s", RESULTS_PATH)

    # ── Print summary ────────────────────────────────────────────────────────
    print("\n" + "=" * 95)
    print("RECALL BOOST EVALUATION — RESULTS SUMMARY")
    print(f"  {len(queries):,} test queries")
    print("=" * 95)
    print(f"{'Config':<40} {'nDCG@10':>9} {'MRR':>9} {'R@10':>9} {'R@50':>9}")
    print("-" * 95)

    baseline_ndcg = all_results.get("CE_pool100", {}).get("metrics", {}).get("ndcg@10", 0.0735)
    for name, res in all_results.items():
        m = res["metrics"]
        ndcg = m["ndcg@10"]
        delta = (ndcg / baseline_ndcg - 1) * 100 if baseline_ndcg > 0 else 0
        sign = "+" if delta >= 0 else ""
        print(f"  {name:<38} {ndcg:>9.4f} {m['mrr']:>9.4f}"
              f" {m['recall@10']:>9.4f} {m.get('recall@50', 0):>9.4f}"
              f"  {sign}{delta:.1f}%")
    print("=" * 95)
    elapsed = (time.time() - t_start) / 60
    print(f"\nTotal elapsed: {elapsed:.1f} min")

    best_name = max(all_results, key=lambda k: all_results[k]["metrics"]["ndcg@10"])
    best_ndcg = all_results[best_name]["metrics"]["ndcg@10"]
    print(f"BEST: {best_name} → nDCG@10 = {best_ndcg:.4f}")
    if best_ndcg >= 0.1:
        print("*** TARGET 0.1 REACHED! ***")
    else:
        print(f"Gap to 0.1: {0.1 - best_ndcg:.4f}")

    return all_results


if __name__ == "__main__":
    main()
