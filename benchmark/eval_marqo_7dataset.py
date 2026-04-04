"""
MODA Phase 1 — Tier 1: Marqo 7-Dataset Benchmark Wrapper

Runs Marqo's official eval harness (repos/marqo-FashionCLIP/eval.py) for
each model × dataset combination, then aggregates results into a leaderboard.

Models evaluated:
  - clip          : ViT-B-32 (openai) — generic baseline
  - fashion-clip  : Marqo/marqo-fashionCLIP
  - fashion-siglip: Marqo/marqo-fashionSigLIP

Datasets (Batch 1, skipping iMaterialist):
  deepfashion_inshop, deepfashion_multimodal, fashion200k, atlas, polyvore
  (KAGL is also in the repo — included if available)

Usage:
  python benchmark/eval_marqo_7dataset.py --models fashion-siglip
  python benchmark/eval_marqo_7dataset.py --models clip fashion-clip fashion-siglip
  python benchmark/eval_marqo_7dataset.py --datasets deepfashion_inshop deepfashion_multimodal
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
_MARQO_DIR = _REPO_ROOT / "repos" / "marqo-FashionCLIP"
_HF_CACHE = _REPO_ROOT / "data" / "hf_cache"
_RESULTS_DIR = _REPO_ROOT / "results" / "tier1"
_PYTHON = _REPO_ROOT / ".venv" / "bin" / "python"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model configs — maps friendly name → Marqo eval.py arguments
# ---------------------------------------------------------------------------

MODEL_CONFIGS: dict[str, dict] = {
    "clip": {
        "model_name": "ViT-B-32",
        "pretrained": "openai",
        "run_name": "CLIP-ViT-B-32-openai",
        "label": "CLIP ViT-B/32 (OpenAI)",
    },
    "fashion-clip": {
        "model_name": "Marqo/marqo-fashionCLIP",
        "pretrained": None,
        "run_name": "Marqo-FashionCLIP",
        "label": "Marqo-FashionCLIP",
    },
    "fashion-siglip": {
        "model_name": "Marqo/marqo-fashionSigLIP",
        "pretrained": None,
        "run_name": "Marqo-FashionSigLIP",
        "label": "Marqo-FashionSigLIP",
    },
}

# Published numbers from Marqo paper (for verification)
MARQO_PUBLISHED = {
    "fashion-clip": {"text-to-image": {"Recall@1": 0.094, "MRR": 0.200}},
    "fashion-siglip": {"text-to-image": {"Recall@1": 0.121, "MRR": 0.237}},
}

# Dataset configs available in the Marqo repo
DATASET_CONFIGS = {
    "deepfashion_inshop": "deepfashion_inshop.json",
    "deepfashion_multimodal": "deepfashion_multimodal.json",
    "fashion200k": "fashion200k.json",
    "atlas": "atlas.json",
    "polyvore": "polyvore.json",
    "KAGL": "KAGL.json",
}

BATCH1_DATASETS = [
    "deepfashion_inshop",
    "deepfashion_multimodal",
    "fashion200k",
    "atlas",
    "polyvore",
]


# ---------------------------------------------------------------------------
# Marqo eval runner
# ---------------------------------------------------------------------------

def run_single_eval(
    model_key: str,
    dataset_key: str,
    batch_size: int = 256,
    device: str = "cpu",
    overwrite: bool = False,
) -> Optional[dict]:
    """Run Marqo's eval.py for one model × dataset combination.

    Returns the parsed results dict, or None on failure.
    """
    cfg = MODEL_CONFIGS[model_key]
    config_file = _MARQO_DIR / "configs" / DATASET_CONFIGS[dataset_key]

    if not config_file.exists():
        log.warning("Config not found: %s — skipping %s", config_file, dataset_key)
        return None

    # Check if HF cache has this dataset (avoid re-download)
    hf_cache_flag = []
    if _HF_CACHE.exists():
        hf_cache_flag = ["--cache-dir", str(_HF_CACHE)]

    # Build command
    cmd = [
        str(_PYTHON),
        "eval.py",
        "--dataset-config", str(config_file.resolve()),
        "--model-name", cfg["model_name"],
        "--run-name", cfg["run_name"],
        "--batch-size", str(batch_size),
        "--device", device,
        "--output-dir", str((_MARQO_DIR / "results").resolve()),
        "--data-dir", str((_MARQO_DIR / "data").resolve()),
    ] + hf_cache_flag

    if cfg["pretrained"]:
        cmd += ["--pretrained", cfg["pretrained"]]

    if overwrite:
        cmd += ["--overwrite-embeddings", "--overwrite-retrieval"]

    log.info("Running eval: model=%s, dataset=%s", model_key, dataset_key)
    log.info("Command: %s", " ".join(cmd))

    try:
        result = subprocess.run(
            cmd,
            cwd=str(_MARQO_DIR),
            capture_output=False,
            timeout=3600,
        )
        if result.returncode != 0:
            log.error("Eval failed for %s × %s (exit %d)", model_key, dataset_key, result.returncode)
            return None
    except subprocess.TimeoutExpired:
        log.error("Eval timed out for %s × %s", model_key, dataset_key)
        return None
    except Exception as e:
        log.error("Eval error for %s × %s: %s", model_key, dataset_key, e)
        return None

    return collect_results(model_key, dataset_key)


def collect_results(model_key: str, dataset_key: str) -> Optional[dict]:
    """Parse result JSON files produced by Marqo's eval.py."""
    cfg = MODEL_CONFIGS[model_key]
    results_base = _MARQO_DIR / "results" / dataset_key / cfg["run_name"]

    if not results_base.exists():
        log.warning("Results dir not found: %s", results_base)
        return None

    task_results: dict[str, dict] = {}
    for task_dir in results_base.iterdir():
        if not task_dir.is_dir():
            continue
        task_name = task_dir.name
        result_files = list(task_dir.glob("result*.json"))
        if not result_files:
            continue
        with open(result_files[0]) as f:
            raw = json.load(f)
        # Flatten nested metric dicts
        flat: dict[str, float] = {}
        for k, v in raw.items():
            if isinstance(v, dict):
                for sub_k, sub_v in v.items():
                    flat[sub_k] = float(sub_v)
            elif isinstance(v, (int, float)):
                flat[k] = float(v)
        task_results[task_name] = flat

    return {"model": model_key, "dataset": dataset_key, "tasks": task_results}


# ---------------------------------------------------------------------------
# Leaderboard table builder
# ---------------------------------------------------------------------------

def build_tier1_leaderboard(all_results: list[dict]) -> str:
    """Build a markdown leaderboard table from all eval results."""
    # Aggregate: compute average Recall@1, Recall@10, MRR for text-to-image
    # and average P@1, P@10, MRR for category/sub-category tasks

    by_model: dict[str, dict] = {}
    for res in all_results:
        if res is None:
            continue
        model = res["model"]
        if model not in by_model:
            by_model[model] = {
                "t2i_recall1": [], "t2i_recall10": [], "t2i_mrr": [],
                "cat_p1": [], "cat_p10": [], "cat_mrr": [],
                "subcat_p1": [], "subcat_p10": [], "subcat_mrr": [],
            }
        tasks = res.get("tasks", {})
        if "text-to-image" in tasks:
            t = tasks["text-to-image"]
            by_model[model]["t2i_recall1"].append(t.get("Recall@1", t.get("Recall@10", 0)))
            by_model[model]["t2i_recall10"].append(t.get("Recall@10", 0))
            by_model[model]["t2i_mrr"].append(t.get("MRR", 0))
        for task_key in ["category-to-product", "sub-category-to-product"]:
            agg_key = "cat" if "category-to-product" == task_key else "subcat"
            if task_key in tasks:
                t = tasks[task_key]
                by_model[model][f"{agg_key}_p1"].append(t.get("P@1", 0))
                by_model[model][f"{agg_key}_p10"].append(t.get("P@10", 0))
                by_model[model][f"{agg_key}_mrr"].append(t.get("MRR", 0))

    def _avg(vals: list) -> str:
        return f"{np.mean(vals):.3f}" if vals else "—"

    lines = [
        "# MODA Tier 1 Leaderboard — Marqo 7-Dataset Benchmark",
        "",
        f"_Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}_",
        "",
        "## Text-to-Image Retrieval",
        "",
        "| Model | Avg Recall@1 | Avg Recall@10 | Avg MRR | Note |",
        "| --- | --- | --- | --- | --- |",
    ]
    for model_key, agg in by_model.items():
        label = MODEL_CONFIGS[model_key]["label"]
        pub = MARQO_PUBLISHED.get(model_key, {}).get("text-to-image", {})
        note = f"Marqo published: R@1={pub.get('Recall@1', '?')}" if pub else ""
        lines.append(f"| {label} | {_avg(agg['t2i_recall1'])} | {_avg(agg['t2i_recall10'])} | {_avg(agg['t2i_mrr'])} | {note} |")

    lines += [
        "",
        "## Category-to-Product Retrieval",
        "",
        "| Model | Avg P@1 | Avg P@10 | Avg MRR |",
        "| --- | --- | --- | --- |",
    ]
    for model_key, agg in by_model.items():
        label = MODEL_CONFIGS[model_key]["label"]
        lines.append(f"| {label} | {_avg(agg['cat_p1'])} | {_avg(agg['cat_p10'])} | {_avg(agg['cat_mrr'])} |")

    lines += [
        "",
        "## Sub-Category-to-Product Retrieval",
        "",
        "| Model | Avg P@1 | Avg P@10 | Avg MRR |",
        "| --- | --- | --- | --- |",
    ]
    for model_key, agg in by_model.items():
        label = MODEL_CONFIGS[model_key]["label"]
        lines.append(f"| {label} | {_avg(agg['subcat_p1'])} | {_avg(agg['subcat_p10'])} | {_avg(agg['subcat_mrr'])} |")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MODA Tier 1: Run Marqo 7-dataset benchmark for all models."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(MODEL_CONFIGS.keys()),
        choices=list(MODEL_CONFIGS.keys()),
        help="Models to evaluate",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=BATCH1_DATASETS,
        choices=list(DATASET_CONFIGS.keys()),
        help="Datasets to evaluate on",
    )
    parser.add_argument(
        "--batch_size", type=int, default=256,
        help="Embedding batch size (default: 256 for CPU)",
    )
    parser.add_argument(
        "--device", default="cpu",
        help="Torch device (cpu/cuda/mps)",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Re-compute even if embeddings/results exist",
    )
    parser.add_argument(
        "--collect_only", action="store_true",
        help="Skip eval, just collect existing results and build leaderboard",
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    log.info("=" * 60)
    log.info("MODA Phase 1 — Tier 1: Marqo 7-Dataset Benchmark")
    log.info("Models   : %s", args.models)
    log.info("Datasets : %s", args.datasets)
    log.info("=" * 60)

    all_results = []

    for model_key in args.models:
        for dataset_key in args.datasets:
            if args.collect_only:
                res = collect_results(model_key, dataset_key)
            else:
                res = run_single_eval(
                    model_key, dataset_key,
                    batch_size=args.batch_size,
                    device=args.device,
                    overwrite=args.overwrite,
                )
            if res:
                all_results.append(res)
                log.info("✓ %s × %s: tasks=%s", model_key, dataset_key, list(res["tasks"].keys()))
            else:
                log.warning("✗ %s × %s: no results", model_key, dataset_key)

    # Save raw results
    raw_path = _RESULTS_DIR / "tier1_raw_results.json"
    with open(raw_path, "w") as f:
        json.dump(all_results, f, indent=2)
    log.info("Raw results saved to %s", raw_path)

    # Build and save leaderboard
    if all_results:
        md = build_tier1_leaderboard(all_results)
        md_path = _RESULTS_DIR / "tier1_leaderboard.md"
        md_path.write_text(md)
        log.info("Leaderboard saved to %s", md_path)
        print("\n" + md)
    else:
        log.warning("No results collected — leaderboard empty.")


if __name__ == "__main__":
    main()
