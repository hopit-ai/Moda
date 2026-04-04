"""
MODA Phase 2 — Step 2: Hybrid Retrieval Evaluation
BM25 (OpenSearch) + Dense (FAISS) + RRF Fusion + Cross-Encoder Reranking

Retrieval methods evaluated:
  1. bm25        — OpenSearch BM25 only
  2. dense       — FAISS dense only (pre-built embeddings)
  3. hybrid      — BM25 + Dense fused via Reciprocal Rank Fusion (RRF)
  4. hybrid+ce   — Hybrid followed by cross-encoder re-ranking (Phase 2 Step 3)

Usage:
  python benchmark/eval_hybrid.py --models fashion-siglip clip
  python benchmark/eval_hybrid.py --methods bm25 dense hybrid
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from opensearchpy import OpenSearch

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from benchmark.metrics import compute_all_metrics, aggregate_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

HNM_DIR = _REPO_ROOT / "data" / "raw" / "hnm"
EMBEDDINGS_DIR = _REPO_ROOT / "data" / "processed" / "embeddings"
RESULTS_DIR = _REPO_ROOT / "results"
INDEX_NAME = "moda_hnm"

# Phase 1 baselines for comparison
PHASE1_BASELINES = {
    "clip":          {"ndcg@10": 0.0966, "mrr": 0.1154, "recall@10": 0.0125, "p@10": 0.0485},
    "fashion-clip":  {"ndcg@10": 0.0886, "mrr": 0.1088, "recall@10": 0.0125, "p@10": 0.0477},
    "fashion-siglip":{"ndcg@10": 0.0765, "mrr": 0.0904, "recall@10": 0.0100, "p@10": 0.0377},
}


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_queries(sample_n: int = 0) -> list[dict]:
    import pandas as pd
    df = pd.read_csv(HNM_DIR / "queries.csv", dtype=str)
    if sample_n and len(df) > sample_n:
        df = df.sample(n=sample_n, random_state=42).reset_index(drop=True)
    return df.to_dict("records")


def load_qrels() -> dict[str, dict[str, int]]:
    qrels: dict[str, dict[str, int]] = {}
    with open(HNM_DIR / "qrels.csv", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            qid = row["query_id"].strip()
            pos_ids = [x.strip() for x in row.get("positive_ids", "").split() if x.strip()]
            neg_ids = [x.strip() for x in row.get("negative_ids", "").split() if x.strip()]
            grades: dict[str, int] = {}
            for aid in pos_ids:
                grades[aid] = 2
            for aid in neg_ids:
                if aid not in grades:
                    grades[aid] = 1
            qrels[qid] = grades
    return qrels


# ─────────────────────────────────────────────────────────────────────────────
# Retrievers
# ─────────────────────────────────────────────────────────────────────────────

class BM25Retriever:
    """OpenSearch BM25 retriever."""

    def __init__(self, host: str = "localhost", port: int = 9200, top_k: int = 50):
        self.client = OpenSearch(
            hosts=[{"host": host, "port": port}],
            http_compress=True, use_ssl=False, verify_certs=False,
        )
        self.top_k = top_k
        # Verify index
        count = self.client.count(index=INDEX_NAME)["count"]
        log.info("BM25Retriever: index '%s' has %d docs", INDEX_NAME, count)

    def retrieve(self, query_text: str) -> list[str]:
        """Return ranked list of article_ids."""
        body = {
            "query": {
                "multi_match": {
                    "query": query_text,
                    "fields": [
                        "prod_name^4",
                        "product_type_name^3",
                        "colour_group_name^2",
                        "section_name^1.5",
                        "garment_group_name^1.5",
                        "detail_desc^1",
                        "search_text^1",
                    ],
                    "type": "best_fields",
                    "tie_breaker": 0.3,
                }
            },
            "size": self.top_k,
            "_source": False,
        }
        resp = self.client.search(index=INDEX_NAME, body=body)
        return [hit["_id"] for hit in resp["hits"]["hits"]]


class DenseRetriever:
    """FAISS dense retriever using pre-built embeddings and subprocess-isolated search."""

    def __init__(self, model_name: str, top_k: int = 50, device: str = "cpu"):
        safe_name = model_name.replace("/", "_").replace(":", "_")
        self.faiss_path = EMBEDDINGS_DIR / f"{safe_name}_faiss.index"
        self.ids_path = EMBEDDINGS_DIR / f"{safe_name}_article_ids.json"
        self.model_name = model_name
        self.top_k = top_k
        self.device = device
        self._embeddings: np.ndarray | None = None  # cached query embeddings
        assert self.faiss_path.exists(), f"FAISS index not found: {self.faiss_path}"

    def encode_queries(self, queries: list[str]) -> np.ndarray:
        """Encode all query texts into embeddings (no FAISS loaded in this process)."""
        from benchmark.models import load_clip_model, encode_texts_clip
        log.info("DenseRetriever: loading model '%s' on %s...", self.model_name, self.device)
        model, _, tokenizer = load_clip_model(self.model_name, device=self.device)
        embeddings = encode_texts_clip(queries, model, tokenizer, self.device, batch_size=128)
        del model
        import torch
        if self.device == "mps":
            torch.mps.empty_cache()
        return embeddings

    def batch_search(self, query_embeddings: np.ndarray) -> list[list[str]]:
        """Run FAISS search in a subprocess (avoids PyTorch+FAISS BLAS conflict)."""
        import tempfile, os
        tmp = Path(tempfile.gettempdir())
        q_path = tmp / "moda_hybrid_query_emb.npy"
        r_path = tmp / "moda_hybrid_faiss_results.json"
        np.save(str(q_path), query_embeddings)
        worker = Path(__file__).parent / "_faiss_search_worker.py"
        cmd = [sys.executable, str(worker),
               str(q_path), str(self.faiss_path), str(self.ids_path),
               str(r_path), str(self.top_k)]
        result = subprocess.run(cmd, capture_output=False, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"FAISS worker failed: {result.returncode}")
        with open(r_path) as f:
            results = json.load(f)
        for p in [q_path, r_path]:
            try:
                os.unlink(p)
            except OSError:
                pass
        return results


def rrf_fusion(
    bm25_lists: list[list[str]],
    dense_lists: list[list[str]],
    k: int = 60,
    bm25_weight: float = 1.0,
    dense_weight: float = 1.0,
    top_k: int = 50,
) -> list[list[str]]:
    """
    Reciprocal Rank Fusion.
    score(d) = bm25_weight * 1/(k + rank_bm25(d)) + dense_weight * 1/(k + rank_dense(d))
    """
    fused = []
    for bm25_ranked, dense_ranked in zip(bm25_lists, dense_lists):
        scores: dict[str, float] = defaultdict(float)
        for rank, doc_id in enumerate(bm25_ranked, start=1):
            scores[doc_id] += bm25_weight / (k + rank)
        for rank, doc_id in enumerate(dense_ranked, start=1):
            scores[doc_id] += dense_weight / (k + rank)
        fused.append(sorted(scores, key=lambda x: scores[x], reverse=True)[:top_k])
    return fused


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(
    queries: list[dict],
    retrieved_lists: list[list[str]],
    qrels: dict[str, dict[str, int]],
    ks: list[int],
    method_name: str,
) -> dict:
    per_query = []
    latencies = []
    for q, retrieved in zip(queries, retrieved_lists):
        qid = str(q["query_id"])
        q_qrels = qrels.get(qid, {})
        t0 = time.perf_counter()
        m = compute_all_metrics(retrieved, q_qrels, ks=ks)
        latencies.append((time.perf_counter() - t0) * 1000)
        per_query.append({"query_id": qid, **m})

    agg = aggregate_metrics([
        {k: v for k, v in r.items() if isinstance(v, float)}
        for r in per_query
    ])
    agg["mean_latency_ms"] = float(np.mean(latencies))
    agg["n_queries"] = len(queries)

    log.info("── %s ──", method_name.upper())
    for metric in ["ndcg@10", "mrr", "recall@10", "p@10"]:
        log.info("  %s: %.4f", metric, agg.get(metric, 0))

    return agg


def print_comparison_table(results: dict[str, dict], ks: list[int]):
    """Print a markdown comparison table vs Phase 1 baselines."""
    metrics = ["ndcg@10", "mrr", "recall@10", "p@10"]
    header = f"{'Method':<35}" + "".join(f"{m:>12}" for m in metrics)
    sep = "─" * len(header)
    print(f"\n{'='*70}")
    print("PHASE 2 HYBRID RETRIEVAL RESULTS vs PHASE 1 BASELINES")
    print(f"{'='*70}")
    print(header)
    print(sep)

    # Print Phase 1 baselines first (greyed)
    for model, baseline in PHASE1_BASELINES.items():
        if any(f"dense_{model}" in k or f"clip" in k.lower() for k in results):
            name = f"[Phase 1] Dense ({model})"
            row = f"  {name:<33}" + "".join(f"{baseline.get(m, 0):>12.4f}" for m in metrics)
            print(row)
    print(sep)

    # Print new results
    for method_name, agg in results.items():
        row = f"  {method_name:<33}" + "".join(f"{agg.get(m, 0):>12.4f}" for m in metrics)
        # Show delta vs best Phase 1 baseline (CLIP)
        best_baseline = PHASE1_BASELINES["clip"]
        ndcg_delta = (agg.get("ndcg@10", 0) - best_baseline["ndcg@10"]) / best_baseline["ndcg@10"] * 100
        sign = "+" if ndcg_delta >= 0 else ""
        beat = "✅" if ndcg_delta > 0 else "🔴"
        print(f"{row}   {sign}{ndcg_delta:.1f}% {beat}")

    print(sep)
    print(f"\n  Phase 1 best (CLIP dense): nDCG@10={PHASE1_BASELINES['clip']['ndcg@10']:.4f}")


def save_results(all_results: dict, method_configs: dict):
    RESULTS_DIR.mkdir(exist_ok=True)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "phase1_baselines": PHASE1_BASELINES,
        "phase2_results": all_results,
        "method_configs": method_configs,
    }
    out = RESULTS_DIR / "phase2_hybrid_results.json"
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
    log.info("Results saved to %s", out)

    # Markdown report
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "# MODA Phase 2 — Hybrid Retrieval Results",
        f"_Generated: {ts}_",
        "",
        "## Method Comparison (H&M Benchmark, 4,078 queries × 105,542 articles)",
        "",
        "| Method | nDCG@5 | nDCG@10 | MRR | Recall@10 | P@10 | vs Phase 1 Best |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    best_p1 = PHASE1_BASELINES["clip"]["ndcg@10"]
    for method, agg in all_results.items():
        delta = (agg.get("ndcg@10", 0) - best_p1) / best_p1 * 100
        sign = "+" if delta >= 0 else ""
        icon = "✅" if delta > 0 else "🔴"
        lines.append(
            f"| {method} "
            f"| {agg.get('ndcg@5', 0):.4f} "
            f"| {agg.get('ndcg@10', 0):.4f} "
            f"| {agg.get('mrr', 0):.4f} "
            f"| {agg.get('recall@10', 0):.4f} "
            f"| {agg.get('p@10', 0):.4f} "
            f"| {sign}{delta:.1f}% {icon} |"
        )

    # Add Phase 1 baselines for reference
    lines += ["", "### Phase 1 Baselines (for reference)", "",
              "| Model | nDCG@10 | MRR | Recall@10 | P@10 |",
              "| --- | --- | --- | --- | --- |"]
    for model, b in PHASE1_BASELINES.items():
        lines.append(f"| Dense ({model}) | {b['ndcg@10']:.4f} | {b['mrr']:.4f} | {b['recall@10']:.4f} | {b['p@10']:.4f} |")

    out_md = RESULTS_DIR / "PHASE2_HYBRID_RESULTS.md"
    out_md.write_text("\n".join(lines))
    log.info("Markdown report saved to %s", out_md)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="MODA Phase 2 — Hybrid Retrieval Eval")
    p.add_argument("--models", nargs="+", default=["fashion-siglip", "clip"],
                   help="Dense models to include")
    p.add_argument("--methods", nargs="+",
                   default=["bm25", "dense", "hybrid"],
                   choices=["bm25", "dense", "hybrid"],
                   help="Retrieval methods to evaluate")
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--rrf_k", type=int, default=60, help="RRF k parameter")
    p.add_argument("--bm25_weight", type=float, default=1.0)
    p.add_argument("--dense_weight", type=float, default=1.0)
    p.add_argument("--sample_queries", type=int, default=0)
    p.add_argument("--ks", default="5,10,20,50")
    p.add_argument("--device", default="cpu")
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=9200)
    args = p.parse_args()

    ks = [int(k) for k in args.ks.split(",")]

    log.info("=" * 60)
    log.info("MODA Phase 2 — Hybrid Retrieval Evaluation")
    log.info("Methods: %s | Models: %s", args.methods, args.models)
    log.info("=" * 60)

    # Load data
    queries = load_queries(args.sample_queries)
    qrels = load_qrels()
    query_texts = [q["query_text"] for q in queries]
    log.info("Loaded %d queries", len(queries))

    all_results: dict[str, dict] = {}
    method_configs = vars(args)

    # ── BM25 ─────────────────────────────────────────────────────────────────
    bm25_retrieved: list[list[str]] | None = None
    if "bm25" in args.methods:
        log.info("\n%s\nRunning BM25 retrieval...\n%s", "─"*50, "─"*50)
        bm25 = BM25Retriever(host=args.host, port=args.port, top_k=args.top_k)
        t0 = time.time()
        bm25_retrieved = []
        for i, q in enumerate(queries):
            bm25_retrieved.append(bm25.retrieve(q["query_text"]))
            if (i + 1) % 500 == 0:
                log.info("  BM25: %d/%d queries (%.1f q/s)", i + 1, len(queries),
                         (i + 1) / (time.time() - t0))
        log.info("BM25 retrieval done in %.1fs", time.time() - t0)
        all_results["BM25 only"] = evaluate(queries, bm25_retrieved, qrels, ks, "BM25 only")

    # ── Dense + Hybrid per model ──────────────────────────────────────────────
    for model_name in args.models:
        log.info("\n%s\nModel: %s\n%s", "─"*50, model_name, "─"*50)

        dense_r = DenseRetriever(model_name, top_k=args.top_k, device=args.device)

        # Encode queries (no FAISS yet)
        log.info("Encoding %d queries with %s...", len(queries), model_name)
        q_embeddings = dense_r.encode_queries(query_texts)

        # FAISS search in subprocess
        log.info("Running FAISS search...")
        dense_retrieved = dense_r.batch_search(q_embeddings)

        if "dense" in args.methods:
            label = f"Dense ({model_name})"
            all_results[label] = evaluate(queries, dense_retrieved, qrels, ks, label)

        if "hybrid" in args.methods and bm25_retrieved is not None:
            label = f"Hybrid BM25+Dense ({model_name})"
            log.info("Fusing BM25 + Dense with RRF (k=%d, bm25_w=%.1f, dense_w=%.1f)...",
                     args.rrf_k, args.bm25_weight, args.dense_weight)
            hybrid_retrieved = rrf_fusion(
                bm25_retrieved, dense_retrieved,
                k=args.rrf_k,
                bm25_weight=args.bm25_weight,
                dense_weight=args.dense_weight,
                top_k=args.top_k,
            )
            all_results[label] = evaluate(queries, hybrid_retrieved, qrels, ks, label)

    # ── Print and save ────────────────────────────────────────────────────────
    print_comparison_table(all_results, ks)
    save_results(all_results, method_configs)

    # Final verdict
    best_method = max(all_results, key=lambda k: all_results[k].get("ndcg@10", 0))
    best_ndcg = all_results[best_method]["ndcg@10"]
    baseline_ndcg = PHASE1_BASELINES["clip"]["ndcg@10"]
    improvement = (best_ndcg - baseline_ndcg) / baseline_ndcg * 100
    print(f"\n{'='*60}")
    print(f"BEST METHOD: {best_method}")
    print(f"nDCG@10: {best_ndcg:.4f} vs Phase 1 best {baseline_ndcg:.4f}")
    print(f"Improvement: {improvement:+.1f}% {'✅ BEATS BASELINE' if improvement > 0 else '🔴 Below baseline'}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
