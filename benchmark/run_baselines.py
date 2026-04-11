"""
MODA Phase 1 — Run All Baselines & Generate Leaderboard

Orchestrates the full Phase 1 evaluation pipeline:
  1. Embed H&M articles with all 3 models (FashionSigLIP, FashionCLIP, CLIP)
  2. Run H&M dense retrieval eval for each model
  3. Collect Tier 1 results (if available from eval_marqo_7dataset.py)
  4. Output combined leaderboard tables (Tier 1 + Tier 2)

Usage:
  # Run everything end-to-end (sequential to avoid MPS contention)
  python benchmark/run_baselines.py

  # Only H&M eval (skip if embeddings already exist)
  python benchmark/run_baselines.py --skip_embed

  # Only leaderboard generation from existing results
  python benchmark/run_baselines.py --leaderboard_only
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

_REPO_ROOT = Path(__file__).parent.parent
_PYTHON = _REPO_ROOT / ".venv" / "bin" / "python"
_RESULTS_DIR = _REPO_ROOT / "results"
_EMBEDDINGS_DIR = _REPO_ROOT / "data" / "processed" / "embeddings"
_HNM_DIR = _REPO_ROOT / "data" / "raw" / "hnm"
_LOGS_DIR = _REPO_ROOT / "logs"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

MODELS = ["fashion-siglip", "fashion-clip", "clip"]
MODEL_LABELS = {
    "clip":           "CLIP ViT-B/32 (OpenAI)",
    "fashion-clip":   "Marqo-FashionCLIP",
    "fashion-siglip": "Marqo-FashionSigLIP",
}

# Marqo published numbers for Tier 1 reference
PUBLISHED = {
    "fashion-siglip": {"t2i_recall1": 0.121, "t2i_mrr": 0.237, "cat_p1": 0.758, "subcat_p1": 0.767},
    "fashion-clip":   {"t2i_recall1": 0.094, "t2i_mrr": 0.200, "cat_p1": 0.734, "subcat_p1": 0.767},
}


# ---------------------------------------------------------------------------
# Step 1: Embed H&M articles
# ---------------------------------------------------------------------------

def embed_hnm_articles(model: str, device: str, batch_size: int, force: bool) -> bool:
    """Run embed_hnm.py for a model. Returns True on success."""
    safe = model.replace("/", "_").replace(":", "_")
    index_path = _EMBEDDINGS_DIR / f"{safe}_faiss.index"

    if index_path.exists() and not force:
        log.info("Embeddings already exist for %s — skipping (use --force to re-run)", model)
        return True

    cmd = [
        str(_PYTHON), "-u", "benchmark/embed_hnm.py",
        "--model", model,
        "--batch_size", str(batch_size),
        "--device", device,
        "--articles_csv", str(_HNM_DIR / "articles.csv"),
        "--output_dir", str(_EMBEDDINGS_DIR),
    ]
    log_path = _LOGS_DIR / f"embed_hnm_{safe}.log"
    log.info("Embedding H&M articles with %s …", model)
    log.info("Log: %s", log_path)

    with open(log_path, "w") as log_f:
        result = subprocess.run(cmd, cwd=str(_REPO_ROOT), stdout=log_f, stderr=subprocess.STDOUT)

    if result.returncode == 0:
        log.info("✓ Embedding complete for %s", model)
        return True
    else:
        log.error("✗ Embedding FAILED for %s (see %s)", model, log_path)
        return False


# ---------------------------------------------------------------------------
# Step 2: Run H&M eval (Tier 2)
# ---------------------------------------------------------------------------

def eval_hnm(model: str, device: str, top_k: int, sample_queries: int) -> Optional[dict]:
    """Run eval_hnm.py for a model. Returns aggregated metrics or None on failure."""
    safe = model.replace("/", "_").replace(":", "_")
    result_json = _RESULTS_DIR / f"hnm_dense_{safe}.json"

    cmd = [
        str(_PYTHON), "-u", "benchmark/eval_hnm.py",
        "--retrieval_method", "dense",
        "--model", model,
        "--top_k", str(top_k),
        "--sample_queries", str(sample_queries),
        "--output_dir", str(_RESULTS_DIR),
        "--data_dir", str(_HNM_DIR),
        "--embeddings_dir", str(_EMBEDDINGS_DIR),
        "--device", device,
    ]
    log_path = _LOGS_DIR / f"eval_hnm_{safe}.log"
    log.info("Running H&M eval for %s …", model)

    with open(log_path, "w") as log_f:
        result = subprocess.run(cmd, cwd=str(_REPO_ROOT), stdout=log_f, stderr=subprocess.STDOUT)

    if result.returncode != 0:
        log.error("✗ H&M eval FAILED for %s (see %s)", model, log_path)
        return None

    if not result_json.exists():
        log.error("✗ Result JSON not found: %s", result_json)
        return None

    with open(result_json) as f:
        data = json.load(f)

    log.info("✓ H&M eval complete for %s", model)
    return data.get("aggregated", {})


# ---------------------------------------------------------------------------
# Step 3: Build leaderboard
# ---------------------------------------------------------------------------

def load_tier1_results() -> dict[str, dict]:
    """Load Tier 1 results from eval_marqo_7dataset.py output."""
    raw_path = _RESULTS_DIR / "tier1" / "tier1_raw_results.json"
    if not raw_path.exists():
        return {}
    with open(raw_path) as f:
        raw = json.load(f)
    by_model: dict[str, dict] = {}
    for res in raw:
        if res is None:
            continue
        model = res["model"]
        tasks = res.get("tasks", {})
        if model not in by_model:
            by_model[model] = {"t2i_recall1": [], "t2i_mrr": [], "cat_p1": [], "subcat_p1": []}
        if "text-to-image" in tasks:
            t = tasks["text-to-image"]
            by_model[model]["t2i_recall1"].append(t.get("Recall@1", 0))
            by_model[model]["t2i_mrr"].append(t.get("MRR", 0))
        if "category-to-product" in tasks:
            by_model[model]["cat_p1"].append(tasks["category-to-product"].get("P@1", 0))
        if "sub-category-to-product" in tasks:
            by_model[model]["subcat_p1"].append(tasks["sub-category-to-product"].get("P@1", 0))
    return {m: {k: float(np.mean(v)) if v else None for k, v in agg.items()} for m, agg in by_model.items()}


def build_leaderboard(tier2_results: dict[str, Optional[dict]]) -> str:
    """Build combined Tier 1 + Tier 2 leaderboard markdown."""
    tier1 = load_tier1_results()

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "# MODA Phase 1 Leaderboard",
        "",
        f"_Generated: {now}_",
        "",
        "## Tier 1 — Marqo 7-Dataset Embedding Benchmark",
        "",
        "_(Text-to-Image, Category-to-Product metrics averaged across available datasets)_",
        "",
        "| Model | T2I Recall@1 | T2I MRR | Cat P@1 | SubCat P@1 | Source |",
        "| --- | --- | --- | --- | --- | --- |",
    ]

    for model in MODELS:
        label = MODEL_LABELS[model]
        pub = PUBLISHED.get(model, {})
        t1 = tier1.get(model, {})

        def _fmt(our_val, pub_val):
            if our_val is not None:
                s = f"{our_val:.3f}"
                if pub_val:
                    s += f" _(pub: {pub_val:.3f})_"
                return s
            elif pub_val:
                return f"_pub: {pub_val:.3f}_"
            return "—"

        lines.append(
            f"| {label} "
            f"| {_fmt(t1.get('t2i_recall1'), pub.get('t2i_recall1'))} "
            f"| {_fmt(t1.get('t2i_mrr'), pub.get('t2i_mrr'))} "
            f"| {_fmt(t1.get('cat_p1'), pub.get('cat_p1'))} "
            f"| {_fmt(t1.get('subcat_p1'), pub.get('subcat_p1'))} "
            f"| {'Our eval' if t1 else 'Published only'} |"
        )

    lines += [
        "",
        "## Tier 2 — H&M Full-Pipeline Benchmark (MODA Original)",
        "",
        "_Dense retrieval only (Phase 1 baseline). Phase 2 adds BM25 + hybrid + reranking._",
        "_Relevance: positives=same product_type_name (grade 2), negatives=same garment_group (grade 1)._",
        "",
        "| Model | nDCG@5 | nDCG@10 | MRR | Recall@10 | Recall@20 | P@10 | Latency (ms) |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]

    for model in MODELS:
        label = MODEL_LABELS[model]
        m = tier2_results.get(model)
        if m:
            lines.append(
                f"| {label} "
                f"| {m.get('ndcg@5', 0):.3f} "
                f"| {m.get('ndcg@10', 0):.3f} "
                f"| {m.get('mrr', 0):.3f} "
                f"| {m.get('recall@10', 0):.3f} "
                f"| {m.get('recall@20', 0):.3f} "
                f"| {m.get('p@10', 0):.3f} "
                f"| {m.get('mean_latency_ms', 0):.1f} |"
            )
        else:
            lines.append(f"| {label} | — | — | — | — | — | — | — |")

    lines += [
        "",
        "---",
        "",
        "## Notes",
        "",
        "- **Tier 1** uses Marqo's official eval harness on DeepFashion-InShop, DeepFashion-Multimodal,",
        "  Fashion200k, Atlas, Polyvore (iMaterialist deferred to Phase 3).",
        "- **Tier 2** uses the H&M articles catalog (105K products) with category-based relevance.",
        "  Phase 2 will add transaction-based relevance (purchase signals).",
        "- All evals run on Apple MPS (M-series GPU). GPU A100 results pending Phase 3.",
    ]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MODA Phase 1 baseline orchestrator")
    p.add_argument("--device", default="mps", help="Torch device (mps/cpu/cuda)")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--sample_queries", type=int, default=0,
                   help="Queries to eval (0=all, default=all)")
    p.add_argument("--models", nargs="+", default=MODELS)
    p.add_argument("--skip_embed", action="store_true",
                   help="Skip embedding step (use existing FAISS indices)")
    p.add_argument("--force", action="store_true",
                   help="Re-embed even if FAISS index already exists")
    p.add_argument("--leaderboard_only", action="store_true",
                   help="Skip all evals, just build leaderboard from existing results")
    return p.parse_args()


def main():
    args = _parse_args()
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    _LOGS_DIR.mkdir(parents=True, exist_ok=True)

    log.info("=" * 60)
    log.info("MODA Phase 1 — Baseline Evaluation Pipeline")
    log.info("Models : %s", args.models)
    log.info("Device : %s", args.device)
    log.info("=" * 60)

    tier2_results: dict[str, Optional[dict]] = {}

    if not args.leaderboard_only:
        for model in args.models:
            log.info("\n--- Model: %s ---", model)

            # Step 1: Embed
            if not args.skip_embed:
                ok = embed_hnm_articles(model, args.device, args.batch_size, args.force)
                if not ok:
                    log.warning("Skipping eval for %s due to embedding failure", model)
                    continue

            # Step 2: Eval
            metrics = eval_hnm(model, args.device, args.top_k, args.sample_queries)
            tier2_results[model] = metrics

            if metrics:
                log.info("Results for %s:", model)
                for k, v in sorted(metrics.items()):
                    if isinstance(v, float):
                        log.info("  %-25s: %.4f", k, v)
    else:
        # Load existing Tier 2 results
        for model in args.models:
            safe = model.replace("/", "_").replace(":", "_")
            result_json = _RESULTS_DIR / f"hnm_dense_{safe}.json"
            if result_json.exists():
                with open(result_json) as f:
                    data = json.load(f)
                tier2_results[model] = data.get("aggregated")
                log.info("Loaded existing results for %s", model)
            else:
                tier2_results[model] = None

    # Step 3: Build leaderboard
    leaderboard_md = build_leaderboard(tier2_results)
    lb_path = _RESULTS_DIR / "phase1_leaderboard.md"
    lb_path.write_text(leaderboard_md)
    log.info("\nLeaderboard saved to %s", lb_path)

    # Also save tier2 summary JSON
    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tier2": tier2_results,
    }
    with open(_RESULTS_DIR / "phase1_tier2_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print("\n" + leaderboard_md)


if __name__ == "__main__":
    main()
