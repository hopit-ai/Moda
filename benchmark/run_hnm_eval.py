"""
MODA Phase 1 — H&M Dense Retrieval Evaluation (Segfault-safe version)

Avoids the FAISS + PyTorch model loading conflict by separating:
  Phase A: Encode all queries in one batch (model loaded, no FAISS)
  Phase B: Batch FAISS search (model unloaded, FAISS loaded)
  Phase C: Compute metrics

Usage:
  python benchmark/run_hnm_eval.py --model fashion-siglip
  python benchmark/run_hnm_eval.py --model fashion-clip --sample_queries 1000
  python benchmark/run_hnm_eval.py --model clip
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from benchmark.metrics import compute_all_metrics, aggregate_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

HNM_DIR = _REPO_ROOT / "data" / "raw" / "hnm"
EMBEDDINGS_DIR = _REPO_ROOT / "data" / "processed" / "embeddings"
RESULTS_DIR = _REPO_ROOT / "results"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_queries(queries_csv: Path, sample_n: int = 0) -> list[dict]:
    import pandas as pd
    df = pd.read_csv(queries_csv, dtype=str)
    if sample_n and len(df) > sample_n:
        df = df.sample(n=sample_n, random_state=42).reset_index(drop=True)
        log.info("Sampled %d queries from %d total", sample_n, len(df))
    return df.to_dict("records")


def load_qrels(qrels_csv: Path) -> dict[str, dict[str, int]]:
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
                if aid not in grades:
                    grades[aid] = 1
            qrels[qid] = grades
    return qrels


def load_articles(articles_csv: Path) -> dict[str, dict]:
    import pandas as pd
    df = pd.read_csv(articles_csv, dtype=str)
    return {row["article_id"]: row for row in df.to_dict("records")}


# ---------------------------------------------------------------------------
# Phase A: Batch query encoding (no FAISS)
# ---------------------------------------------------------------------------

def encode_queries(
    queries: list[dict],
    model_name: str,
    device: str = "cpu",
    batch_size: int = 128,
) -> np.ndarray:
    """Encode all query texts into normalized embeddings. No FAISS imported."""
    from benchmark.models import load_clip_model, encode_texts_clip

    log.info("Loading model '%s' on device '%s' …", model_name, device)
    model, _, tokenizer = load_clip_model(model_name, device=device)

    texts = [str(q["query_text"]) for q in queries]
    log.info("Encoding %d queries …", len(texts))
    embeddings = encode_texts_clip(texts, model, tokenizer, device, batch_size=batch_size)
    log.info("Query embeddings shape: %s", embeddings.shape)

    # Free model memory before loading FAISS
    del model
    import torch
    if device == "mps":
        torch.mps.empty_cache()
    elif device == "cuda":
        torch.cuda.empty_cache()

    return embeddings


# ---------------------------------------------------------------------------
# Phase B: Batch FAISS search (subprocess — avoids PyTorch+FAISS BLAS conflict)
# ---------------------------------------------------------------------------

def batch_faiss_search(
    query_embeddings: np.ndarray,
    faiss_index_path: Path,
    article_ids_path: Path,
    top_k: int = 50,
    tmp_dir: Path | None = None,
) -> list[list[str]]:
    """Run FAISS search in a fresh subprocess (no PyTorch in that process)."""
    import subprocess
    import tempfile
    import os

    tmp = tmp_dir or Path(tempfile.gettempdir())
    q_emb_path = tmp / "moda_query_embeddings.npy"
    results_path = tmp / "moda_faiss_results.json"

    np.save(str(q_emb_path), query_embeddings)
    log.info("Saved query embeddings to %s, launching FAISS worker …", q_emb_path)

    worker = Path(__file__).parent / "_faiss_search_worker.py"
    python_exe = sys.executable
    cmd = [
        python_exe, str(worker),
        str(q_emb_path), str(faiss_index_path), str(article_ids_path),
        str(results_path), str(top_k),
    ]
    result = subprocess.run(cmd, capture_output=False, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FAISS worker failed with code {result.returncode}")

    with open(results_path) as f:
        all_results: list[list[str]] = json.load(f)

    # Cleanup temp files
    for p in [q_emb_path, results_path]:
        try:
            os.unlink(p)
        except OSError:
            pass

    log.info("FAISS search complete: %d query results loaded.", len(all_results))
    return all_results


# ---------------------------------------------------------------------------
# Phase C: Metrics computation
# ---------------------------------------------------------------------------

def compute_metrics(
    queries: list[dict],
    retrieved_lists: list[list[str]],
    qrels: dict[str, dict[str, int]],
    articles: dict[str, dict],
    ks: list[int],
) -> tuple[dict, list[dict]]:
    per_query_rows: list[dict] = []
    latencies_ms: list[float] = []

    for q, retrieved in zip(queries, retrieved_lists):
        qid = str(q["query_id"])
        qtext = str(q["query_text"])
        q_qrels = qrels.get(qid, {})

        t0 = time.perf_counter()
        m = compute_all_metrics(retrieved, q_qrels, ks=ks)
        latency_ms = (time.perf_counter() - t0) * 1000
        latencies_ms.append(latency_ms)

        group = "-"
        if retrieved and retrieved[0] in articles:
            group = articles[retrieved[0]].get("product_group_name", "-") or "-"

        per_query_rows.append({
            "query_id": qid,
            "query_text": qtext,
            "n_retrieved": len(retrieved),
            "n_positive": sum(1 for v in q_qrels.values() if v >= 2),
            "latency_ms": round(latency_ms, 2),
            "product_group": group,
            **m,
        })

    agg = aggregate_metrics([
        {k: v for k, v in r.items() if isinstance(v, float) and k != "latency_ms"}
        for r in per_query_rows
    ])
    agg["mean_latency_ms"] = float(np.mean(latencies_ms))
    agg["p95_latency_ms"] = float(np.percentile(latencies_ms, 95))
    agg["n_queries"] = len(queries)
    return agg, per_query_rows


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def save_results(agg: dict, per_query_rows: list[dict], breakdown: dict,
                 output_dir: Path, file_stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "aggregated": agg,
        "breakdown_by_product_group": breakdown,
    }
    json_path = output_dir / f"{file_stem}.json"
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)
    log.info("Results saved to %s", json_path)

    md_lines = [
        f"# H&M Benchmark — {file_stem}",
        f"_Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}_",
        "",
        "## Aggregated Metrics",
        "| Metric | Value |",
        "| --- | --- |",
    ]
    for k, v in sorted(agg.items()):
        val = f"{v:.4f}" if isinstance(v, float) else str(v)
        md_lines.append(f"| {k} | {val} |")
    md_lines += ["", "## Breakdown by Product Group (nDCG@10)", "| Group | nDCG@10 |", "| --- | --- |"]
    for g, v in sorted(breakdown.items(), key=lambda x: x[1], reverse=True)[:20]:
        md_lines.append(f"| {g} | {v:.4f} |")
    (output_dir / f"{file_stem}.md").write_text("\n".join(md_lines))

    if per_query_rows:
        csv_path = output_dir / f"{file_stem}_per_query.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(per_query_rows[0].keys()))
            writer.writeheader()
            writer.writerows(per_query_rows)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="H&M dense retrieval eval (segfault-safe)")
    p.add_argument("--model", default="fashion-siglip")
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--sample_queries", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--ks", default="5,10,20,50")
    p.add_argument("--data_dir", type=Path, default=HNM_DIR)
    p.add_argument("--embeddings_dir", type=Path, default=EMBEDDINGS_DIR)
    p.add_argument("--output_dir", type=Path, default=RESULTS_DIR)
    return p.parse_args()


def main():
    args = _parse_args()
    ks = [int(k) for k in args.ks.split(",")]
    safe_model = args.model.replace("/", "_").replace(":", "_")

    log.info("=" * 60)
    log.info("MODA H&M Dense Retrieval Eval — %s", args.model)
    log.info("=" * 60)

    # Load data
    queries = load_queries(args.data_dir / "queries.csv", args.sample_queries)
    qrels = load_qrels(args.data_dir / "qrels.csv")
    articles = load_articles(args.data_dir / "articles.csv")
    log.info("Loaded %d queries, %d articles", len(queries), len(articles))

    # Phase A: Encode queries (no FAISS in memory)
    query_embeddings = encode_queries(queries, args.model, device=args.device, batch_size=args.batch_size)

    # Phase B: Batch FAISS search
    faiss_path = args.embeddings_dir / f"{safe_model}_faiss.index"
    ids_path = args.embeddings_dir / f"{safe_model}_article_ids.json"
    retrieved_lists = batch_faiss_search(query_embeddings, faiss_path, ids_path, top_k=args.top_k)

    # Phase C: Metrics
    log.info("Computing metrics …")
    agg, per_query_rows = compute_metrics(queries, retrieved_lists, qrels, articles, ks)

    # Product group breakdown
    from collections import defaultdict
    groups: dict[str, list[float]] = defaultdict(list)
    for row in per_query_rows:
        groups[row.get("product_group", "-")].append(row.get("ndcg@10", 0.0))
    breakdown = {g: float(np.mean(v)) for g, v in groups.items()}

    # Save
    file_stem = f"hnm_dense_{safe_model}"
    save_results(agg, per_query_rows, breakdown, args.output_dir, file_stem)

    # Print summary
    print("\n=== H&M Dense Retrieval Results ===")
    print(f"Model: {args.model}")
    key_metrics = ["ndcg@5", "ndcg@10", "ndcg@20", "mrr", "recall@10", "recall@20", "p@10", "mean_latency_ms"]
    for k in key_metrics:
        if k in agg:
            v = agg[k]
            print(f"  {k:25s}: {v:.4f}" if isinstance(v, float) else f"  {k:25s}: {v}")
    print()


if __name__ == "__main__":
    main()
