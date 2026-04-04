"""
MODA Phase 1 — H&M Full-Pipeline Benchmark Harness  (Tier 2)

This is MODA's original contribution: a reproducible end-to-end benchmark
harness on the H&M dataset with pluggable retrieval methods.

Data expected in data/raw/hnm/:
  articles.csv  — 105K products (article_id, prod_name, detail_desc,
                   product_type_name, colour_group_name, garment_group_name, …)
  queries.csv   — query_id, query_text
  qrels.csv     — query_id, positive_ids (space-separated), negative_ids

Relevance grades:
  positive_ids → 2  (purchased)
  negative_ids → 1  (viewed but not purchased)
  absent       → 0

Output files (--output_dir):
  hnm_results_{method}_{model}.json
  hnm_results_{method}_{model}.md
  hnm_results_{method}_{model}_per_query.csv

Usage:
  # Dense retrieval with fashion-clip (requires embed_hnm.py to have run first)
  python benchmark/eval_hnm.py --retrieval_method dense --model fashion-clip

  # BM25 stub (Phase 2)
  python benchmark/eval_hnm.py --retrieval_method bm25 --model none
"""

from __future__ import annotations

import abc
import argparse
import csv
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from benchmark.metrics import compute_all_metrics, aggregate_metrics
from benchmark.models import MODEL_REGISTRY, encode_texts_clip, load_clip_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Retriever interface
# ---------------------------------------------------------------------------


class BaseRetriever(abc.ABC):
    """Abstract base class for all retrieval backends."""

    @abc.abstractmethod
    def retrieve(self, query: str, top_k: int) -> list[str]:
        """Return up to top_k article_ids ranked by relevance.

        Args:
            query: Free-text query string.
            top_k: Maximum number of results to return.

        Returns:
            Ordered list of article_id strings (most relevant first).
        """

    @abc.abstractmethod
    def name(self) -> str:
        """Human-readable name for this retriever."""


# ---------------------------------------------------------------------------
# Dense retriever (FAISS)
# ---------------------------------------------------------------------------


class DenseRetriever(BaseRetriever):
    """FAISS-based dense retrieval using pre-built embeddings.

    The index and article_ids must be created by embed_hnm.py first.

    Args:
        faiss_index_path: Path to the .index file produced by embed_hnm.py.
        article_ids_path: Path to the .json mapping (list of article_ids).
        model_name: Model used to encode queries at retrieval time.
        device: Torch device.
    """

    def __init__(
        self,
        faiss_index_path: Path,
        article_ids_path: Path,
        model_name: str,
        device: str = "cpu",
    ) -> None:
        try:
            import faiss  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "faiss-cpu is required. Install with: pip install faiss-cpu"
            ) from exc

        if not faiss_index_path.exists():
            raise FileNotFoundError(
                f"FAISS index not found: {faiss_index_path}\n"
                "Run embed_hnm.py first to build the index."
            )
        if not article_ids_path.exists():
            raise FileNotFoundError(f"Article ID mapping not found: {article_ids_path}")

        self._index = faiss.read_index(str(faiss_index_path))
        with open(article_ids_path) as f:
            self._article_ids: list[str] = json.load(f)
        self._model_name = model_name
        self._device = device

        logger.info(
            "DenseRetriever: loaded FAISS index with %d vectors, model=%s",
            self._index.ntotal, model_name,
        )

        # Load query encoder (lazy caching via instance)
        self._clip_model, _, self._tokenizer = load_clip_model(model_name, device=device)
        self._clip_model.eval()

    def retrieve(self, query: str, top_k: int) -> list[str]:
        import torch  # type: ignore

        with torch.no_grad():
            tokens = self._tokenizer([query]).to(self._device)
            q_emb = self._clip_model.encode_text(tokens)
            q_emb = q_emb / q_emb.norm(dim=-1, keepdim=True)
        q_np = q_emb.cpu().numpy().astype(np.float32)

        _, indices = self._index.search(q_np, top_k)
        return [self._article_ids[i] for i in indices[0] if i >= 0]

    def name(self) -> str:
        safe = self._model_name.replace("/", "_")
        return f"dense_{safe}"


# ---------------------------------------------------------------------------
# BM25 retriever stub (Phase 2)
# ---------------------------------------------------------------------------


class BM25Retriever(BaseRetriever):
    """BM25 retrieval via OpenSearch — stub for Phase 2.

    The OpenSearch calls are marked TODO and will be implemented in Phase 2
    once the OpenSearch index is set up by Agent 1's infrastructure work.

    Args:
        opensearch_host: OpenSearch endpoint URL.
        index_name: Name of the OpenSearch articles index.
    """

    def __init__(
        self,
        opensearch_host: str = "http://localhost:9200",
        index_name: str = "hnm_articles",
    ) -> None:
        self._host = opensearch_host
        self._index = index_name
        logger.warning(
            "BM25Retriever: OpenSearch integration not yet implemented (Phase 2). "
            "Returning empty results."
        )

    def retrieve(self, query: str, top_k: int) -> list[str]:
        # TODO (Phase 2): Send BM25 query to OpenSearch
        #   POST {self._host}/{self._index}/_search
        #   body: { "query": { "multi_match": { "query": query,
        #           "fields": ["prod_name^3", "detail_desc", "product_type_name"] } },
        #           "size": top_k }
        #   Parse hits[].["_source"]["article_id"]
        raise NotImplementedError(
            "BM25Retriever is a Phase 2 feature. "
            "OpenSearch index setup is handled by the infrastructure pipeline."
        )

    def name(self) -> str:
        return "bm25_opensearch"


# ---------------------------------------------------------------------------
# Hybrid retriever stub (Phase 2)
# ---------------------------------------------------------------------------


class HybridRetriever(BaseRetriever):
    """Hybrid BM25 + Dense retrieval — stub for Phase 2."""

    def __init__(self, dense: DenseRetriever, bm25: BM25Retriever, alpha: float = 0.5) -> None:
        self._dense = dense
        self._bm25 = bm25
        self._alpha = alpha

    def retrieve(self, query: str, top_k: int) -> list[str]:
        # TODO (Phase 2): reciprocal rank fusion or score interpolation
        raise NotImplementedError("HybridRetriever is a Phase 2 feature.")

    def name(self) -> str:
        return f"hybrid_alpha{self._alpha}"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_articles(articles_csv: Path) -> dict[str, dict]:
    """Load articles.csv into a dict keyed by article_id."""
    import pandas as pd  # type: ignore

    if not articles_csv.exists():
        raise FileNotFoundError(
            f"articles.csv not found at {articles_csv}. "
            "Ensure Agent 1 has downloaded the H&M dataset."
        )
    df = pd.read_csv(articles_csv, dtype=str)
    return {row["article_id"]: row for row in df.to_dict("records")}


def load_queries(queries_csv: Path, sample_n: Optional[int] = None) -> list[dict]:
    """Load queries.csv, optionally sampling up to sample_n queries."""
    import pandas as pd  # type: ignore

    if not queries_csv.exists():
        raise FileNotFoundError(f"queries.csv not found at {queries_csv}")
    df = pd.read_csv(queries_csv, dtype=str)
    if sample_n and len(df) > sample_n:
        df = df.sample(n=sample_n, random_state=42).reset_index(drop=True)
        logger.info("Sampled %d queries from %d total.", sample_n, len(df))
    return df.to_dict("records")


def load_qrels(qrels_csv: Path) -> dict[str, dict[str, int]]:
    """Load qrels.csv → dict[query_id → dict[article_id → grade]].

    Positive IDs get grade 2 (purchased), negative IDs get grade 1 (viewed).
    """
    if not qrels_csv.exists():
        raise FileNotFoundError(f"qrels.csv not found at {qrels_csv}")

    qrels: dict[str, dict[str, int]] = {}
    with open(qrels_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            qid = row["query_id"].strip()
            pos_ids = [x.strip() for x in row.get("positive_ids", "").split() if x.strip()]
            neg_ids = [x.strip() for x in row.get("negative_ids", "").split() if x.strip()]
            grades: dict[str, int] = {}
            for aid in pos_ids:
                grades[aid] = 2
            for aid in neg_ids:
                if aid not in grades:  # positives win if overlap
                    grades[aid] = 1
            qrels[qid] = grades
    return qrels


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------


def run_eval(
    retriever: BaseRetriever,
    queries: list[dict],
    qrels: dict[str, dict[str, int]],
    articles: dict[str, dict],
    top_k: int = 50,
    ks: list[int] | None = None,
) -> tuple[dict, list[dict]]:
    """Run the evaluation loop over all queries.

    Args:
        retriever: The retrieval backend to evaluate.
        queries: List of query dicts with keys ``query_id`` and ``query_text``.
        qrels: Nested dict from load_qrels.
        articles: Article dict from load_articles.
        top_k: Maximum results to retrieve per query.
        ks: Cut-off ranks for metric computation.

    Returns:
        Tuple of (aggregated_metrics_dict, per_query_rows).
    """
    from tqdm import tqdm  # type: ignore

    if ks is None:
        ks = [5, 10, 20, 50]

    per_query_rows: list[dict] = []
    latencies_ms: list[float] = []

    for q in tqdm(queries, desc=f"Evaluating [{retriever.name()}]"):
        qid = str(q["query_id"])
        qtext = str(q["query_text"])
        q_qrels = qrels.get(qid, {})

        t0 = time.perf_counter()
        retrieved = retriever.retrieve(qtext, top_k)
        latency_ms = (time.perf_counter() - t0) * 1000
        latencies_ms.append(latency_ms)

        m = compute_all_metrics(retrieved, q_qrels, ks=ks)

        # Product group for breakdown analysis
        # Use first retrieved article's group as proxy (or first positive)
        group = "-"
        if retrieved and retrieved[0] in articles:
            group = articles[retrieved[0]].get("product_group_name", "-") or "-"

        row = {
            "query_id": qid,
            "query_text": qtext,
            "n_retrieved": len(retrieved),
            "n_positive": sum(1 for v in q_qrels.values() if v >= 2),
            "latency_ms": round(latency_ms, 2),
            "product_group": group,
            **m,
        }
        per_query_rows.append(row)

    agg = aggregate_metrics([
        {k: v for k, v in r.items() if isinstance(v, float) and k != "latency_ms"}
        for r in per_query_rows
    ])
    agg["mean_latency_ms"] = float(np.mean(latencies_ms))
    agg["p95_latency_ms"] = float(np.percentile(latencies_ms, 95))
    agg["n_queries"] = len(queries)

    return agg, per_query_rows


# ---------------------------------------------------------------------------
# Per-group breakdown
# ---------------------------------------------------------------------------


def group_breakdown(per_query_rows: list[dict], metric: str = "ndcg@10") -> dict[str, float]:
    """Compute mean of a metric broken down by product_group."""
    from collections import defaultdict

    groups: dict[str, list[float]] = defaultdict(list)
    for row in per_query_rows:
        g = row.get("product_group", "-")
        if metric in row:
            groups[g].append(row[metric])
    return {g: float(np.mean(vals)) for g, vals in groups.items()}


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def _save_results(
    agg: dict,
    per_query_rows: list[dict],
    breakdown: dict,
    output_dir: Path,
    file_stem: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # JSON
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "aggregated": agg,
        "breakdown_by_product_group": breakdown,
        "per_query": per_query_rows,
    }
    json_path = output_dir / f"{file_stem}.json"
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)
    logger.info("Results JSON saved to %s", json_path)

    # Markdown table (aggregated)
    md_path = output_dir / f"{file_stem}.md"
    lines = [
        f"# H&M Benchmark Results — {file_stem}",
        "",
        f"_Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}_",
        "",
        "## Aggregated Metrics",
        "",
        "| Metric | Value |",
        "| --- | --- |",
    ]
    for k, v in sorted(agg.items()):
        val = f"{v:.4f}" if isinstance(v, float) else str(v)
        lines.append(f"| {k} | {val} |")
    lines += [
        "",
        "## Breakdown by Product Group (nDCG@10)",
        "",
        "| Product Group | nDCG@10 |",
        "| --- | --- |",
    ]
    for g, v in sorted(breakdown.items(), key=lambda x: x[1], reverse=True):
        lines.append(f"| {g} | {v:.4f} |")
    md_path.write_text("\n".join(lines))
    logger.info("Results Markdown saved to %s", md_path)

    # Per-query CSV
    csv_path = output_dir / f"{file_stem}_per_query.csv"
    if per_query_rows:
        fieldnames = list(per_query_rows[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(per_query_rows)
    logger.info("Per-query CSV saved to %s", csv_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="H&M full-pipeline benchmark harness (MODA Tier 2)."
    )
    parser.add_argument(
        "--retrieval_method",
        choices=["dense", "bm25", "hybrid"],
        default="dense",
        help="Retrieval backend to evaluate",
    )
    parser.add_argument(
        "--model",
        default="fashion-clip",
        help="Embedding model (used by dense/hybrid retrievers)",
    )
    parser.add_argument(
        "--top_k", type=int, default=50,
        help="Number of results to retrieve per query (default: 50)",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("results"),
        help="Directory to save result files",
    )
    parser.add_argument(
        "--sample_queries", type=int, default=5000,
        help="Number of queries to evaluate (default: 5000, 0 = all)",
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=Path("data/raw/hnm"),
        help="Directory containing articles.csv, queries.csv, qrels.csv",
    )
    parser.add_argument(
        "--embeddings_dir",
        type=Path,
        default=Path("data/processed/embeddings"),
        help="Directory with pre-built FAISS index (for dense retrieval)",
    )
    parser.add_argument(
        "--device", default="cpu",
        help="Torch device (cpu, cuda, mps)",
    )
    parser.add_argument(
        "--ks",
        default="5,10,20,50",
        help="Comma-separated k values for metric computation",
    )
    return parser.parse_args()


def _build_retriever(args: argparse.Namespace) -> BaseRetriever:
    if args.retrieval_method == "dense":
        safe_name = args.model.replace("/", "_").replace(":", "_")
        faiss_path = args.embeddings_dir / f"{safe_name}_faiss.index"
        ids_path = args.embeddings_dir / f"{safe_name}_article_ids.json"
        return DenseRetriever(faiss_path, ids_path, args.model, device=args.device)
    elif args.retrieval_method == "bm25":
        return BM25Retriever()
    elif args.retrieval_method == "hybrid":
        dense = DenseRetriever(
            args.embeddings_dir / f"{args.model.replace('/', '_')}_faiss.index",
            args.embeddings_dir / f"{args.model.replace('/', '_')}_article_ids.json",
            args.model,
            device=args.device,
        )
        return HybridRetriever(dense, BM25Retriever())
    else:
        raise ValueError(f"Unknown retrieval method: {args.retrieval_method}")


if __name__ == "__main__":
    args = _parse_args()
    ks = [int(k) for k in args.ks.split(",")]
    sample_n = args.sample_queries if args.sample_queries > 0 else None

    logger.info("Loading dataset from %s …", args.data_dir)
    articles = load_articles(args.data_dir / "articles.csv")
    queries = load_queries(args.data_dir / "queries.csv", sample_n=sample_n)
    qrels = load_qrels(args.data_dir / "qrels.csv")

    logger.info("Building retriever: method=%s, model=%s", args.retrieval_method, args.model)
    retriever = _build_retriever(args)

    logger.info("Starting evaluation: %d queries, top_k=%d …", len(queries), args.top_k)
    agg, per_query_rows = run_eval(retriever, queries, qrels, articles, top_k=args.top_k, ks=ks)
    breakdown = group_breakdown(per_query_rows)

    safe_model = args.model.replace("/", "_").replace(":", "_")
    file_stem = f"hnm_{args.retrieval_method}_{safe_model}"
    _save_results(agg, per_query_rows, breakdown, args.output_dir, file_stem)

    print("\n=== Aggregated Results ===")
    for k in sorted(agg.keys()):
        v = agg[k]
        print(f"  {k:25s}: {v:.4f}" if isinstance(v, float) else f"  {k:25s}: {v}")
    print()
