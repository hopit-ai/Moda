"""
Compute bootstrap confidence intervals for all experiment results.

Loads each eval JSON, reconstructs per-query nDCG@10 scores from the
saved retrieval lists + qrels, then computes 95% bootstrap CIs.

Usage:
  python -m benchmark.compute_confidence_intervals
  python -m benchmark.compute_confidence_intervals --n_bootstrap 5000
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

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from benchmark.metrics import compute_all_metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

RESULTS_DIR = _REPO_ROOT / "results" / "real"
HNM_DIR = _REPO_ROOT / "data" / "raw" / "hnm_real"
SPLIT_PATH = _REPO_ROOT / "data" / "processed" / "query_splits.json"


def load_qrels() -> dict[str, dict[str, int]]:
    """Load test-split qrels."""
    splits = json.loads(SPLIT_PATH.read_text())
    test_qids = set(splits["test"])

    qrels: dict[str, dict[str, int]] = {}
    with open(HNM_DIR / "qrels.csv", newline="") as f:
        for row in csv.DictReader(f):
            qid = row["query_id"].strip()
            if qid not in test_qids:
                continue
            pos_ids = [a.strip() for a in row.get("positive_ids", "").split(",") if a.strip()]
            neg_ids = [a.strip() for a in row.get("negative_ids", "").split(",") if a.strip()]
            q = {}
            for aid in pos_ids:
                q[aid] = 2
            for aid in neg_ids:
                q[aid] = 1
            if q:
                qrels[qid] = q
    return qrels


def bootstrap_ci(
    per_query_scores: np.ndarray,
    n_bootstrap: int = 10000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Compute bootstrap confidence interval.

    Returns (mean, ci_lower, ci_upper).
    """
    rng = np.random.RandomState(seed)
    n = len(per_query_scores)
    means = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        sample = rng.choice(per_query_scores, size=n, replace=True)
        means[i] = sample.mean()

    alpha = (1 - ci) / 2
    lower = np.percentile(means, 100 * alpha)
    upper = np.percentile(means, 100 * (1 - alpha))
    return float(per_query_scores.mean()), float(lower), float(upper)


def compute_per_query_ndcg(
    retrieved_dict: dict[str, list[str]],
    qrels: dict[str, dict[str, int]],
) -> np.ndarray:
    """Compute per-query nDCG@10 scores."""
    scores = []
    for qid, retrieved in retrieved_dict.items():
        q_qrels = qrels.get(qid, {})
        if not q_qrels:
            continue
        m = compute_all_metrics(retrieved, q_qrels, ks=[10])
        scores.append(m["ndcg@10"])
    return np.array(scores)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_bootstrap", type=int, default=10000)
    args = parser.parse_args()

    t0 = time.time()
    qrels = load_qrels()
    log.info("Loaded qrels for %d queries", len(qrels))

    eval_files = [
        "splade_pipeline_eval.json",
        "finetuned_splade_eval.json",
        "phase1_2_splade_eval.json",
        "phase3_9_comprehensive_eval.json",
        "phase3c_biencoder_eval.json",
        "phase3_fused_item_tower_eval.json",
        "phase3_8_moe_eval.json",
    ]

    all_ci_results: dict[str, dict] = {}

    for fname in eval_files:
        fpath = RESULTS_DIR / fname
        if not fpath.exists():
            log.warning("Skipping %s (not found)", fname)
            continue

        data = json.load(open(fpath))
        log.info("Processing %s (%d configs)...", fname, len(data))

        for config_name, config_data in data.items():
            if not isinstance(config_data, dict):
                continue
            metrics = config_data.get("metrics")
            if not metrics or "ndcg@10" not in metrics:
                continue

            retrieved = config_data.get("retrieved")
            if retrieved:
                scores = compute_per_query_ndcg(retrieved, qrels)
                if len(scores) > 0:
                    mean, lower, upper = bootstrap_ci(scores, n_bootstrap=args.n_bootstrap)
                    all_ci_results[config_name] = {
                        "ndcg@10": mean,
                        "ci_lower": lower,
                        "ci_upper": upper,
                        "n_queries": len(scores),
                        "mrr": metrics.get("mrr", 0),
                        "recall@10": metrics.get("recall@10", 0),
                        "source": fname,
                    }
                    continue

            n_queries = config_data.get("n_queries", 22855)
            ndcg = metrics["ndcg@10"]
            se = np.sqrt(ndcg * (1 - ndcg) / n_queries) if n_queries > 0 else 0
            ci_half = 1.96 * se
            all_ci_results[config_name] = {
                "ndcg@10": ndcg,
                "ci_lower": ndcg - ci_half,
                "ci_upper": ndcg + ci_half,
                "n_queries": n_queries,
                "mrr": metrics.get("mrr", 0),
                "recall@10": metrics.get("recall@10", 0),
                "source": fname,
                "ci_method": "normal_approx",
            }

    # Also handle three-tower (nested structure)
    tt_path = RESULTS_DIR / "phase4g_three_tower_eval.json"
    if tt_path.exists():
        data = json.load(open(tt_path))
        configs = data.get("configs", {})
        for config_name, config_data in configs.items():
            if not isinstance(config_data, dict):
                continue
            metrics = config_data.get("metrics")
            if not metrics or "ndcg@10" not in metrics:
                continue
            n_queries = config_data.get("n_queries", 22855)
            ndcg = metrics["ndcg@10"]
            se = np.sqrt(ndcg * (1 - ndcg) / n_queries) if n_queries > 0 else 0
            ci_half = 1.96 * se
            all_ci_results[config_name] = {
                "ndcg@10": ndcg,
                "ci_lower": ndcg - ci_half,
                "ci_upper": ndcg + ci_half,
                "n_queries": n_queries,
                "mrr": metrics.get("mrr", 0),
                "recall@10": metrics.get("recall@10", 0),
                "source": "phase4g_three_tower_eval.json",
                "ci_method": "normal_approx",
            }

    # Sort by nDCG descending
    sorted_results = sorted(all_ci_results.items(), key=lambda x: -x[1]["ndcg@10"])

    # Print results
    log.info("")
    log.info("=" * 95)
    log.info("  EXPERIMENT RESULTS WITH 95%% CONFIDENCE INTERVALS")
    log.info("=" * 95)
    log.info("%-40s  %10s  %22s  %8s  %8s", "Config", "nDCG@10", "95% CI", "MRR", "R@10")
    log.info("-" * 95)
    for name, r in sorted_results:
        log.info(
            "%-40s  %10.4f  [%8.4f, %8.4f]  %8.4f  %8.4f",
            name[:40], r["ndcg@10"], r["ci_lower"], r["ci_upper"],
            r["mrr"], r["recall@10"],
        )
    log.info("=" * 95)

    # Save
    out_path = RESULTS_DIR / "all_experiments_with_ci.json"
    with open(out_path, "w") as f:
        json.dump(dict(sorted_results), f, indent=2)
    log.info("Saved to %s", out_path)

    elapsed = time.time() - t0
    log.info("Done in %.1fs", elapsed)


if __name__ == "__main__":
    main()
