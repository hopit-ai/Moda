"""
MODA Phase 1 — Core Metrics Library

Implements standard IR metrics used across all benchmark tiers:
  nDCG@k, MRR, Recall@k, Precision@k, Average Precision, MAP.

All functions follow the convention that a higher score is always better.
"""

from __future__ import annotations

import math
from typing import Optional


# ---------------------------------------------------------------------------
# Individual metric functions
# ---------------------------------------------------------------------------


def ndcg_at_k(relevance_scores: list[float], k: int) -> float:
    """Compute normalised Discounted Cumulative Gain at rank k.

    **Ideal DCG convention:** The ideal ranking is the descending sort of
    *relevance scores at retrieved ranks only* (not "best possible over the
    whole catalogue"). So nDCG measures how well the observed order matches
    the best reordering of *this* candidate list. Documents not in the
    retrieved list do not enter the ideal vector (standard limitation when
    labels exist only for a session subset).

    Args:
        relevance_scores: Ordered relevance scores for retrieved documents
            (index 0 = rank 1). Each score is a non-negative float; higher
            means more relevant (e.g. 2=purchased, 1=shown-negative, 0=absent).
        k: Cut-off rank.

    Returns:
        nDCG@k in [0, 1].
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    if not relevance_scores:
        return 0.0

    truncated = relevance_scores[:k]

    def dcg(scores: list[float]) -> float:
        return sum(rel / math.log2(idx + 2) for idx, rel in enumerate(scores))

    actual_dcg = dcg(truncated)
    ideal_dcg = dcg(sorted(relevance_scores, reverse=True)[:k])

    if ideal_dcg == 0.0:
        return 0.0
    return actual_dcg / ideal_dcg


def mrr(relevance_scores: list[float]) -> float:
    """Mean Reciprocal Rank — reciprocal of the rank of the first relevant doc.

    Args:
        relevance_scores: Ordered relevance scores (index 0 = rank 1).
            A document is considered relevant if its score > 0.

    Returns:
        MRR in (0, 1], or 0.0 if no relevant document is found.
    """
    for rank, score in enumerate(relevance_scores, start=1):
        if score > 0:
            return 1.0 / rank
    return 0.0


def recall_at_k(retrieved_ids: list, relevant_ids: set, k: int) -> float:
    """Recall@k — fraction of relevant documents retrieved in top-k.

    Args:
        retrieved_ids: Ordered list of retrieved document IDs (rank 1 first).
        relevant_ids: Set of all truly relevant document IDs.
        k: Cut-off rank.

    Returns:
        Recall@k in [0, 1], or 0.0 when relevant_ids is empty.
    """
    if not relevant_ids:
        return 0.0
    hits = sum(1 for doc_id in retrieved_ids[:k] if doc_id in relevant_ids)
    return hits / len(relevant_ids)


def precision_at_k(retrieved_ids: list, relevant_ids: set, k: int) -> float:
    """Precision@k — fraction of top-k retrieved docs that are relevant.

    Args:
        retrieved_ids: Ordered list of retrieved document IDs (rank 1 first).
        relevant_ids: Set of relevant document IDs.
        k: Cut-off rank.

    Returns:
        Precision@k in [0, 1].
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    top_k = retrieved_ids[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for doc_id in top_k if doc_id in relevant_ids)
    return hits / k


def average_precision(retrieved_ids: list, relevant_ids: set) -> float:
    """Average Precision — area under the precision–recall curve.

    Args:
        retrieved_ids: Ordered list of retrieved document IDs (rank 1 first).
        relevant_ids: Set of relevant document IDs.

    Returns:
        AP in [0, 1], or 0.0 when relevant_ids is empty.
    """
    if not relevant_ids:
        return 0.0
    hits = 0
    cumulative_precision = 0.0
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_ids:
            hits += 1
            cumulative_precision += hits / rank
    if hits == 0:
        return 0.0
    return cumulative_precision / len(relevant_ids)


# ---------------------------------------------------------------------------
# Combined metric computation
# ---------------------------------------------------------------------------


def compute_all_metrics(
    retrieved_ids: list,
    qrels: dict,
    ks: list[int] | None = None,
) -> dict[str, float]:
    """Compute all standard IR metrics for a single query.

    Args:
        retrieved_ids: Ordered list of retrieved document IDs (rank 1 first).
        qrels: Mapping from document ID → relevance grade (non-negative float).
            Documents absent from qrels are assumed relevance 0.
        ks: List of cut-off ranks to evaluate. Defaults to [5, 10, 20, 50].

    Returns:
        Dict with keys like ``"ndcg@10"``, ``"recall@20"``, ``"mrr"``, etc.
    """
    if ks is None:
        ks = [5, 10, 20, 50]

    # Build per-rank relevance score list for nDCG / MRR
    relevance_scores = [float(qrels.get(doc_id, 0)) for doc_id in retrieved_ids]

    # All labelled session SKUs (grade > 0: purchased or impression-negative).
    # Used for recall / P@k / AP / MRR "first hit"; see NDCG note on grade 1 vs 2.
    relevant_ids: set = {doc_id for doc_id, grade in qrels.items() if grade > 0}

    results: dict[str, float] = {}

    # MRR and MAP don't depend on k
    results["mrr"] = mrr(relevance_scores)
    results["ap"] = average_precision(retrieved_ids, relevant_ids)

    for k in ks:
        results[f"ndcg@{k}"] = ndcg_at_k(relevance_scores, k)
        results[f"recall@{k}"] = recall_at_k(retrieved_ids, relevant_ids, k)
        results[f"p@{k}"] = precision_at_k(retrieved_ids, relevant_ids, k)

    return results


def aggregate_metrics(per_query_metrics: list[dict[str, float]]) -> dict[str, float]:
    """Macro-average per-query metric dicts across all queries.

    Args:
        per_query_metrics: List of metric dicts as returned by
            :func:`compute_all_metrics`.

    Returns:
        Single dict with macro-averaged values (same keys).
    """
    if not per_query_metrics:
        return {}
    keys = per_query_metrics[0].keys()
    return {
        key: sum(m[key] for m in per_query_metrics) / len(per_query_metrics)
        for key in keys
    }


# ---------------------------------------------------------------------------
# Marqo-style retrieval metrics (Recall@k and MRR for T2I / image retrieval)
# ---------------------------------------------------------------------------


def recall_at_k_binary(
    retrieved_ids: list, positive_id: str, k: int
) -> float:
    """Recall@k for single-positive retrieval (text-to-image style).

    Args:
        retrieved_ids: Ordered list of retrieved IDs.
        positive_id: The single correct match.
        k: Cut-off rank.

    Returns:
        1.0 if positive_id appears in top-k, else 0.0.
    """
    return 1.0 if positive_id in retrieved_ids[:k] else 0.0


def mrr_binary(retrieved_ids: list, positive_id: str) -> float:
    """MRR for single-positive retrieval.

    Args:
        retrieved_ids: Ordered list of retrieved IDs.
        positive_id: The single correct match.

    Returns:
        Reciprocal rank of positive_id, or 0.0 if not found.
    """
    try:
        rank = retrieved_ids.index(positive_id) + 1
        return 1.0 / rank
    except ValueError:
        return 0.0


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    import sys

    def _assert_close(a: float, b: float, tol: float = 1e-6, msg: str = "") -> None:
        assert abs(a - b) < tol, f"Expected {b:.6f}, got {a:.6f}. {msg}"

    print("Running metrics unit tests …")

    # --- nDCG ---
    # Perfect ranking: [2, 1, 0]
    _assert_close(ndcg_at_k([2, 1, 0], k=3), 1.0, msg="nDCG perfect")
    # Reversed: [0, 1, 2] → should be < 1
    assert ndcg_at_k([0, 1, 2], k=3) < 1.0, "nDCG reversed < 1"
    # All zeros
    _assert_close(ndcg_at_k([0, 0, 0], k=3), 0.0, msg="nDCG all-zero")
    # k larger than list
    _assert_close(ndcg_at_k([1], k=10), 1.0, msg="nDCG k>len")

    # --- MRR ---
    _assert_close(mrr([1, 0, 0]), 1.0, msg="MRR first relevant at rank 1")
    _assert_close(mrr([0, 0, 1]), 1 / 3, msg="MRR first relevant at rank 3")
    _assert_close(mrr([0, 0, 0]), 0.0, msg="MRR no relevant")

    # --- Recall@k ---
    rel = {"a", "b", "c"}
    _assert_close(recall_at_k(["a", "b", "c", "d"], rel, k=3), 1.0)
    _assert_close(recall_at_k(["d", "e", "f"], rel, k=3), 0.0)
    _assert_close(recall_at_k(["a", "d", "e"], rel, k=3), 1 / 3)
    _assert_close(recall_at_k(["a", "b"], set(), k=2), 0.0, msg="Recall empty rel")

    # --- Precision@k ---
    _assert_close(precision_at_k(["a", "b", "c"], {"a", "b"}, k=3), 2 / 3)
    _assert_close(precision_at_k(["a", "b", "c"], {"a", "b"}, k=2), 1.0)
    _assert_close(precision_at_k(["x", "y"], {"a"}, k=2), 0.0)

    # --- AP ---
    _assert_close(average_precision(["a", "b", "c"], {"a", "c"}), (1 + 2 / 3) / 2)
    _assert_close(average_precision([], {"a"}), 0.0)
    _assert_close(average_precision(["x"], {"a"}), 0.0)

    # --- compute_all_metrics ---
    qrels_sample = {"a": 2, "b": 1, "c": 0}
    metrics = compute_all_metrics(["a", "x", "b"], qrels_sample, ks=[1, 2])
    assert "ndcg@1" in metrics
    assert "recall@2" in metrics
    _assert_close(metrics["mrr"], 1.0, msg="MRR with grade-2 at rank 1")

    # --- aggregate_metrics ---
    agg = aggregate_metrics([{"mrr": 1.0, "ndcg@10": 0.8}, {"mrr": 0.5, "ndcg@10": 0.4}])
    _assert_close(agg["mrr"], 0.75)
    _assert_close(agg["ndcg@10"], 0.6)

    # --- Binary helpers ---
    _assert_close(recall_at_k_binary(["a", "b", "c"], "b", k=2), 1.0)
    _assert_close(recall_at_k_binary(["a", "b", "c"], "c", k=2), 0.0)
    _assert_close(mrr_binary(["a", "b", "c"], "b"), 0.5)
    _assert_close(mrr_binary(["a", "b", "c"], "z"), 0.0)

    print("All tests passed.")
    sys.exit(0)
