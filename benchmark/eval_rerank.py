"""
MODA Phase 2 — Step 3: Cross-Encoder Re-ranking

Pipeline:
  1. BM25 (or Hybrid) retrieves top-K candidates
  2. Fetch candidate article text from OpenSearch
  3. Cross-encoder scores every (query, candidate) pair
  4. Re-rank by cross-encoder score → top-50

Compares all stages:
  BM25@100  →  CE-reranked@50
  Hybrid@100 →  CE-reranked@50

Usage:
  python benchmark/eval_rerank.py
  python benchmark/eval_rerank.py --pool_size 100 --rerank_top_k 50
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from opensearchpy import OpenSearch

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from benchmark.eval_hybrid import (
    BM25Retriever,
    DenseRetriever,
    load_queries,
    load_qrels,
    rrf_fusion,
    evaluate,
    save_results,
    PHASE1_BASELINES,
    INDEX_NAME,
    RESULTS_DIR,
)
from benchmark.metrics import compute_all_metrics, aggregate_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Article text fetcher (cache to avoid N*K OpenSearch calls)
# ─────────────────────────────────────────────────────────────────────────────

class ArticleTextCache:
    """Fetch and cache article text from OpenSearch for cross-encoder scoring."""

    _FIELDS = ["prod_name", "product_type_name", "colour_group_name", "section_name",
               "garment_group_name", "detail_desc"]

    def __init__(self, client: OpenSearch, max_cached: int = 200_000):
        self.client = client
        self._cache: dict[str, str] = {}
        self._max_cached = max_cached

    def _build_text(self, src: dict) -> str:
        parts = []
        if src.get("prod_name"):
            parts.append(src["prod_name"])
        if src.get("product_type_name"):
            parts.append(src["product_type_name"])
        if src.get("colour_group_name"):
            parts.append(src["colour_group_name"])
        if src.get("section_name"):
            parts.append(src["section_name"])
        if src.get("detail_desc"):
            parts.append(src["detail_desc"][:200])
        return " | ".join(p for p in parts if p)

    def prefetch(self, article_ids: list[str]) -> None:
        """Batch-fetch article texts not yet in cache."""
        to_fetch = [aid for aid in article_ids if aid not in self._cache]
        if not to_fetch:
            return
        # Batch in chunks of 500 (OpenSearch mget limit)
        chunk_size = 500
        for i in range(0, len(to_fetch), chunk_size):
            chunk = to_fetch[i:i + chunk_size]
            resp = self.client.mget(index=INDEX_NAME, body={"ids": chunk})
            for doc in resp["docs"]:
                if doc.get("found"):
                    self._cache[doc["_id"]] = self._build_text(doc["_source"])
                else:
                    self._cache[doc["_id"]] = ""

    def get(self, article_id: str) -> str:
        return self._cache.get(article_id, "")


# ─────────────────────────────────────────────────────────────────────────────
# Cross-encoder re-ranker
# ─────────────────────────────────────────────────────────────────────────────

class CrossEncoderReranker:
    """Re-rank candidate lists using a cross-encoder."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
                 batch_size: int = 64, device: str = "cpu"):
        from sentence_transformers import CrossEncoder
        log.info("Loading cross-encoder: %s on %s", model_name, device)
        self.ce = CrossEncoder(model_name, device=device)
        self.batch_size = batch_size
        self.model_name = model_name
        log.info("Cross-encoder loaded ✓")

    def rerank(
        self,
        query: str,
        candidates: list[str],
        article_texts: dict[str, str],
        top_k: int,
    ) -> list[str]:
        """Score each (query, candidate_text) pair, return top_k by score."""
        if not candidates:
            return []
        pairs = [(query, article_texts.get(cid, cid)) for cid in candidates]
        # Score in batches
        all_scores = self.ce.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)
        ranked = sorted(zip(candidates, all_scores), key=lambda x: x[1], reverse=True)
        return [cid for cid, _ in ranked[:top_k]]

    def rerank_batch(
        self,
        queries: list[str],
        candidate_lists: list[list[str]],
        text_cache: ArticleTextCache,
        top_k: int = 50,
    ) -> list[list[str]]:
        """Re-rank all queries, prefetching all unique article texts first."""
        # Prefetch all unique candidates
        all_ids = list({cid for clist in candidate_lists for cid in clist})
        log.info("Prefetching %d unique article texts from OpenSearch...", len(all_ids))
        t0 = time.time()
        text_cache.prefetch(all_ids)
        log.info("Prefetch done in %.1fs", time.time() - t0)

        log.info("Re-ranking %d queries (pool=%d)...", len(queries),
                 max(len(c) for c in candidate_lists) if candidate_lists else 0)
        results = []
        t0 = time.time()
        for i, (query, candidates) in enumerate(zip(queries, candidate_lists)):
            reranked = self.rerank(query, candidates, text_cache._cache, top_k)
            results.append(reranked)
            if (i + 1) % 100 == 0:
                elapsed = time.time() - t0
                log.info("  Re-ranked %d/%d (%.1f q/s)", i + 1, len(queries),
                         (i + 1) / elapsed)
        log.info("Re-ranking done in %.1fs", time.time() - t0)
        return results


# ─────────────────────────────────────────────────────────────────────────────
# Full comparison table with all methods
# ─────────────────────────────────────────────────────────────────────────────

def print_full_leaderboard(all_results: dict[str, dict]):
    metrics = ["ndcg@5", "ndcg@10", "mrr", "recall@10", "p@10"]
    best_p1_ndcg = PHASE1_BASELINES["clip"]["ndcg@10"]

    header = f"\n{'Method':<42}" + "".join(f"{m:>12}" for m in metrics) + "   vs P1 Best"
    sep = "─" * (42 + 12 * len(metrics) + 14)

    print(f"\n{'='*80}")
    print("MODA PHASE 2 — FULL RETRIEVAL LEADERBOARD (H&M, 4,078 queries)")
    print(f"{'='*80}")
    print(header)
    print(sep)

    for method, agg in all_results.items():
        delta = (agg.get("ndcg@10", 0) - best_p1_ndcg) / best_p1_ndcg * 100
        sign = "+" if delta >= 0 else ""
        icon = "✅" if delta > 0 else "🔴"
        row = f"  {method:<40}" + "".join(f"{agg.get(m, 0):>12.4f}" for m in metrics)
        print(f"{row}  {sign}{delta:.1f}% {icon}")

    print(sep)
    print(f"\n  Phase 1 best (dense CLIP): nDCG@10={best_p1_ndcg:.4f}")
    best_method = max(all_results, key=lambda k: all_results[k].get("ndcg@10", 0))
    best_ndcg = all_results[best_method]["ndcg@10"]
    print(f"  Phase 2 best:              nDCG@10={best_ndcg:.4f} ({best_method})")
    print(f"  Overall gain:              {(best_ndcg - best_p1_ndcg)/best_p1_ndcg*100:+.1f}%\n")


def save_full_results(all_results: dict, method_configs: dict):
    RESULTS_DIR.mkdir(exist_ok=True)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "phase1_baselines": PHASE1_BASELINES,
        "all_methods": all_results,
        "method_configs": method_configs,
    }
    out = RESULTS_DIR / "phase2_full_results.json"
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
    log.info("Full results saved to %s", out)

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    metrics = ["ndcg@5", "ndcg@10", "mrr", "recall@10", "p@10"]
    best_p1 = PHASE1_BASELINES["clip"]["ndcg@10"]

    lines = [
        "# MODA Phase 2 — Full Retrieval Leaderboard",
        f"_Generated: {ts}_",
        "",
        f"**Dataset:** H&M fashion articles (105,542 docs, 4,078 queries)",
        f"**Phase 1 best baseline:** Dense CLIP ViT-B/32 — nDCG@10 = {best_p1:.4f}",
        "",
        "## Results",
        "",
        "| Method | nDCG@5 | nDCG@10 | MRR | Recall@10 | P@10 | vs Phase 1 Best |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]

    for method, agg in all_results.items():
        delta = (agg.get("ndcg@10", 0) - best_p1) / best_p1 * 100
        sign = "+" if delta >= 0 else ""
        icon = "✅" if delta > 0 else "🔴"
        lines.append(
            f"| **{method}** "
            f"| {agg.get('ndcg@5', 0):.4f} "
            f"| {agg.get('ndcg@10', 0):.4f} "
            f"| {agg.get('mrr', 0):.4f} "
            f"| {agg.get('recall@10', 0):.4f} "
            f"| {agg.get('p@10', 0):.4f} "
            f"| {sign}{delta:.1f}% {icon} |"
        )

    best_method = max(all_results, key=lambda k: all_results[k].get("ndcg@10", 0))
    best_ndcg = all_results[best_method]["ndcg@10"]
    lines += [
        "",
        "## Phase 1 Baselines (for reference)",
        "",
        "| Model | nDCG@10 | MRR | Recall@10 | P@10 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for model, b in PHASE1_BASELINES.items():
        lines.append(f"| Dense ({model}) | {b['ndcg@10']:.4f} | {b['mrr']:.4f} | {b['recall@10']:.4f} | {b['p@10']:.4f} |")

    lines += [
        "",
        "## Summary",
        "",
        f"- **Best Phase 2 method:** {best_method}",
        f"- **nDCG@10:** {best_ndcg:.4f} (+{(best_ndcg - best_p1)/best_p1*100:.1f}% vs Phase 1)",
        "",
        "### Why does hybrid re-ranking work?",
        "",
        "1. **BM25** excels at exact lexical matching of product names",
        "2. **Dense (CLIP/SigLIP)** adds semantic recall for paraphrases and category-level queries",
        "3. **RRF fusion** balances both signals without requiring score normalization",
        "4. **Cross-encoder re-ranking** refines the top candidates using a richer interaction model",
        "   that attends to the full query-document pair (vs. independent bi-encoder embeddings)",
    ]

    out_md = RESULTS_DIR / "PHASE2_FULL_LEADERBOARD.md"
    out_md.write_text("\n".join(lines))
    log.info("Full leaderboard saved to %s", out_md)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="MODA Phase 2 — Cross-Encoder Re-ranking Eval")
    p.add_argument("--ce_model", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    p.add_argument("--pool_size", type=int, default=100,
                   help="BM25/hybrid candidate pool size for re-ranking")
    p.add_argument("--rerank_top_k", type=int, default=50)
    p.add_argument("--dense_model", default="clip",
                   help="Dense model for hybrid branch")
    p.add_argument("--bm25_weight", type=float, default=1.0)
    p.add_argument("--dense_weight", type=float, default=1.0)
    p.add_argument("--rrf_k", type=int, default=60)
    p.add_argument("--ce_batch_size", type=int, default=64)
    p.add_argument("--device", default="cpu")
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=9200)
    p.add_argument("--sample_queries", type=int, default=0)
    p.add_argument("--ks", default="5,10,20,50")
    args = p.parse_args()

    ks = [int(k) for k in args.ks.split(",")]

    log.info("=" * 60)
    log.info("MODA Phase 2 — Step 3: Cross-Encoder Re-ranking")
    log.info("CE model: %s | Pool size: %d", args.ce_model, args.pool_size)
    log.info("=" * 60)

    queries = load_queries(args.sample_queries)
    qrels = load_qrels()
    query_texts = [q["query_text"] for q in queries]
    log.info("Loaded %d queries", len(queries))

    # OpenSearch client
    client = OpenSearch(
        hosts=[{"host": args.host, "port": args.port}],
        http_compress=True, use_ssl=False, verify_certs=False,
    )
    text_cache = ArticleTextCache(client)

    # Load previously computed Phase 2 Step 2 results if available
    all_results: dict[str, dict] = {}

    # Load Phase 2 hybrid results for reference
    phase2_json = RESULTS_DIR / "phase2_hybrid_results.json"
    if phase2_json.exists():
        with open(phase2_json) as f:
            prev = json.load(f)
        all_results.update(prev.get("phase2_results", {}))
        log.info("Loaded %d results from Phase 2 Step 2", len(all_results))

    # ── BM25 (large pool) ─────────────────────────────────────────────────────
    log.info("\nRunning BM25 with pool_size=%d...", args.pool_size)
    bm25 = BM25Retriever(host=args.host, port=args.port, top_k=args.pool_size)
    t0 = time.time()
    bm25_pool: list[list[str]] = []
    for i, q in enumerate(queries):
        bm25_pool.append(bm25.retrieve(q["query_text"]))
        if (i + 1) % 1000 == 0:
            log.info("  BM25: %d/%d (%.1f q/s)", i + 1, len(queries),
                     (i + 1) / (time.time() - t0))
    log.info("BM25 pool done in %.1fs", time.time() - t0)

    # Truncate to rerank_top_k for BM25-only baseline
    bm25_top50 = [lst[:args.rerank_top_k] for lst in bm25_pool]
    all_results[f"BM25 top-{args.rerank_top_k}"] = evaluate(
        queries, bm25_top50, qrels, ks, f"BM25 top-{args.rerank_top_k}")

    # ── Dense (for hybrid) ────────────────────────────────────────────────────
    log.info("\nBuilding dense retrieval for hybrid...")
    dense_r = DenseRetriever(args.dense_model, top_k=args.pool_size, device=args.device)
    q_embeddings = dense_r.encode_queries(query_texts)
    dense_pool = dense_r.batch_search(q_embeddings)

    # ── Hybrid pool ───────────────────────────────────────────────────────────
    hybrid_pool = rrf_fusion(
        bm25_pool, dense_pool,
        k=args.rrf_k,
        bm25_weight=args.bm25_weight,
        dense_weight=args.dense_weight,
        top_k=args.pool_size,
    )
    hybrid_top50 = [lst[:args.rerank_top_k] for lst in hybrid_pool]
    all_results[f"Hybrid BM25+Dense ({args.dense_model}) top-{args.rerank_top_k}"] = evaluate(
        queries, hybrid_top50, qrels, ks, f"Hybrid top-{args.rerank_top_k}")

    # ── Cross-encoder ─────────────────────────────────────────────────────────
    ce = CrossEncoderReranker(
        model_name=args.ce_model,
        batch_size=args.ce_batch_size,
        device=args.device,
    )

    # BM25 → CE reranked
    log.info("\nBM25 → CE re-ranking (pool=%d → top-%d)...", args.pool_size, args.rerank_top_k)
    bm25_ce = ce.rerank_batch(query_texts, bm25_pool, text_cache, top_k=args.rerank_top_k)
    all_results[f"BM25@{args.pool_size} → CE@{args.rerank_top_k}"] = evaluate(
        queries, bm25_ce, qrels, ks, f"BM25@{args.pool_size} → CE@{args.rerank_top_k}")

    # Hybrid → CE reranked
    log.info("\nHybrid → CE re-ranking (pool=%d → top-%d)...", args.pool_size, args.rerank_top_k)
    # Re-prefetch if needed (text_cache already warm)
    hybrid_ce = ce.rerank_batch(query_texts, hybrid_pool, text_cache, top_k=args.rerank_top_k)
    label = f"Hybrid@{args.pool_size} → CE@{args.rerank_top_k}"
    all_results[label] = evaluate(queries, hybrid_ce, qrels, ks, label)

    # ── Print and save ────────────────────────────────────────────────────────
    print_full_leaderboard(all_results)
    save_full_results(all_results, vars(args))


if __name__ == "__main__":
    main()
