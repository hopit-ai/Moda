"""MoDA Phase 1/2 — 10K-corpus screener wrapper for the Marqo benchmark.

Wraps ``repos/marqo-FashionCLIP/eval_subsample.py`` (which adds stratified
corpus subsampling on top of Marqo's eval.py).

Why this exists:
  Full atlas evaluation takes ~8h on MPS for SO400M-384. A 10K-corpus
  stratified screener takes ~1h, letting us compare 6+ candidate models
  in a single overnight run.

Usage:
  # Phase 1c — calibration (we know the full-corpus answer, verify
  # the screener preserves rank order):
  python benchmark/eval_marqo_subsample.py \
    --models fashion-siglip google-siglip-so400m-384 \
    --datasets atlas --corpus-size 10000

  # Phase 2 — screen all 6 candidates on atlas-10K:
  python benchmark/eval_marqo_subsample.py \
    --models google-siglip-so400m-384 google-siglip2-b16-384 google-siglip2-l16-384 \
              google-siglip2-so400m-384 jinaclip-v2 \
    --datasets atlas --corpus-size 10000

Output:
  - Per-model results land under
    ``repos/marqo-FashionCLIP/results/<dataset>/<run_name>_subsample10K/``
  - A consolidated screener leaderboard is written to
    ``results/screener/screener_<dataset>_<size>.md``
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).parent.parent
_MARQO_DIR = _REPO_ROOT / "repos" / "marqo-FashionCLIP"
_HF_CACHE = _REPO_ROOT / "data" / "hf_cache"
_RESULTS_DIR = _REPO_ROOT / "results" / "screener"
_PYTHON = _REPO_ROOT / ".venv" / "bin" / "python"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# Reuse the model + dataset configs from the existing wrapper to stay
# single-source-of-truth. Add JinaCLIP-v2 here only if open_clip supports it.
sys.path.insert(0, str(_REPO_ROOT / "benchmark"))
from eval_marqo_7dataset import MODEL_CONFIGS, DATASET_CONFIGS  # noqa: E402


def run_single_eval_subsample(
    model_key: str,
    dataset_key: str,
    corpus_size: int,
    seed: int,
    batch_size: int,
    device: str,
    overwrite: bool,
) -> dict | None:
    cfg = MODEL_CONFIGS[model_key]
    config_file = _MARQO_DIR / "configs" / DATASET_CONFIGS[dataset_key]
    if not config_file.exists():
        log.warning("Config not found: %s — skipping", config_file)
        return None

    run_name = f"{cfg['run_name']}_subsample{corpus_size}_seed{seed}"

    cmd = [
        str(_PYTHON), "eval_subsample.py",
        "--dataset-config", str(config_file.resolve()),
        "--model-name", cfg["model_name"],
        "--run-name", run_name,
        "--batch-size", str(batch_size),
        "--device", device,
        "--output-dir", str((_MARQO_DIR / "results").resolve()),
        "--data-dir", str((_MARQO_DIR / "data").resolve()),
        "--corpus-size", str(corpus_size),
        "--subsample-seed", str(seed),
    ]
    if _HF_CACHE.exists():
        cmd += ["--cache-dir", str(_HF_CACHE)]
    if cfg["pretrained"]:
        cmd += ["--pretrained", cfg["pretrained"]]
    if overwrite:
        cmd += ["--overwrite-embeddings", "--overwrite-retrieval"]

    log.info("Running: %s × %s @ size=%d seed=%d", model_key, dataset_key, corpus_size, seed)
    log.info("CMD: %s", " ".join(cmd))

    try:
        r = subprocess.run(cmd, cwd=str(_MARQO_DIR), capture_output=False, timeout=21600)
        if r.returncode != 0:
            log.error("Eval failed: %s × %s (exit %d)", model_key, dataset_key, r.returncode)
            return None
    except subprocess.TimeoutExpired:
        log.error("Eval timed out: %s × %s", model_key, dataset_key)
        return None

    return collect_results(model_key, dataset_key, run_name)


def collect_results(model_key: str, dataset_key: str, run_name: str) -> dict | None:
    base = _MARQO_DIR / "results" / dataset_key / run_name
    if not base.exists():
        log.warning("Results dir missing: %s", base)
        return None
    out: dict[str, dict] = {}
    for task_dir in base.iterdir():
        if not task_dir.is_dir():
            continue
        files = list(task_dir.glob("result_*.json"))
        if not files:
            continue
        with open(files[0]) as f:
            raw = json.load(f)
        flat: dict[str, float] = {}
        for k, v in raw.items():
            if isinstance(v, dict) and k != "_subsample":
                for sub_k, sub_v in v.items():
                    flat[sub_k] = float(sub_v)
            elif isinstance(v, (int, float)):
                flat[k] = float(v)
        out[task_dir.name] = flat
    meta_path = base / "subsample_meta.json"
    meta = json.load(open(meta_path)) if meta_path.exists() else {}
    return {"model": model_key, "dataset": dataset_key, "run_name": run_name, "tasks": out, "meta": meta}


def build_leaderboard(all_results: list[dict], dataset_key: str, corpus_size: int) -> str:
    lines = [
        f"# MoDA Phase 1/2 Screener Leaderboard — {dataset_key.upper()} corpus={corpus_size}",
        "",
        f"_Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}_",
        "",
        "Stratified subsampling: every test query keeps all of its positives, ",
        "remainder filled with random non-positives (fixed seed). Absolute Recall@K ",
        "values are inflated vs full-corpus eval; the **relative ordering** between ",
        "models is what matters for screening.",
        "",
        "## Text-to-Image (the metric Marqo's GCL targets)",
        "",
        "| Rank | Model | MAP@10 | NDCG@10 | Recall@10 | MRR | Queries used |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    rows = []
    for r in all_results:
        if not r:
            continue
        t2i = r["tasks"].get("text-to-image", {})
        if not t2i:
            continue
        rows.append((
            MODEL_CONFIGS[r["model"]]["label"],
            t2i.get("MAP@10", 0),
            t2i.get("NDCG@10", 0),
            t2i.get("Recall@10", 0),
            t2i.get("MRR", 0),
            r["meta"].get("queries_dropped_per_task", {}),
        ))
    rows.sort(key=lambda x: x[1], reverse=True)
    for i, (label, mAP, ndcg, rec, mrr, drops) in enumerate(rows, 1):
        first_task_drops = next(iter(drops.values()), 0) if drops else 0
        lines.append(f"| {i} | {label} | {mAP:.4f} | {ndcg:.4f} | {rec:.4f} | {mrr:.4f} | dropped {first_task_drops} |")
    return "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", required=True, choices=list(MODEL_CONFIGS.keys()))
    p.add_argument("--datasets", nargs="+", default=["atlas"], choices=list(DATASET_CONFIGS.keys()))
    p.add_argument("--corpus-size", type=int, default=10000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--device", default="mps")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    all_results: list[dict] = []
    for ds in args.datasets:
        ds_results: list[dict] = []
        for m in args.models:
            r = run_single_eval_subsample(
                m, ds, args.corpus_size, args.seed, args.batch_size, args.device, args.overwrite,
            )
            if r:
                ds_results.append(r)
                all_results.append(r)

        leaderboard_md = build_leaderboard(ds_results, ds, args.corpus_size)
        out_md = _RESULTS_DIR / f"screener_{ds}_{args.corpus_size}_seed{args.seed}.md"
        with open(out_md, "w") as f:
            f.write(leaderboard_md)
        log.info("Leaderboard saved: %s", out_md)

    raw_path = _RESULTS_DIR / f"screener_raw_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
    with open(raw_path, "w") as f:
        json.dump(all_results, f, indent=2)
    log.info("Raw results saved: %s", raw_path)


if __name__ == "__main__":
    main()
