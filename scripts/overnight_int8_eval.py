"""
Overnight script: full-corpus evaluation of INT8-quantized SigLIP-2 L/16/384.

Loads the INT8 model (dequantizes on-the-fly), then runs Marqo's official
eval harness on all 4 clean datasets (fashion200k, atlas, polyvore, KAGL).

Usage:
    python scripts/overnight_int8_eval.py
"""

import gc
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

import open_clip
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
INT8_PATH = ROOT / "models" / "siglip2-l16-int8" / "model_int8.pt"
HF_CACHE = ROOT / "data" / "hf_cache"
MARQO_DIR = ROOT / "repos" / "marqo-FashionCLIP"
PYTHON = ROOT / ".venv" / "bin" / "python"
RESULTS_DIR = ROOT / "results" / "int8-fullcorpus"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = ["fashion200k", "atlas", "polyvore", "KAGL"]
DATASET_CONFIGS = {
    "fashion200k": "fashion200k.json",
    "atlas": "atlas.json",
    "polyvore": "polyvore.json",
    "KAGL": "KAGL.json",
}


def load_int8_model():
    """Load INT8 model, dequantize, return (model, preprocess, tokenizer)."""
    log.info("Loading INT8 model from %s", INT8_PATH)
    checkpoint = torch.load(INT8_PATH, map_location="cpu", weights_only=False)
    quant_state = checkpoint["state_dict"]
    quant_meta = checkpoint["quant_meta"]

    dequant_state = {}
    for name, tensor in quant_state.items():
        if name in quant_meta:
            dequant_state[name] = tensor.float() * quant_meta[name]
        else:
            dequant_state[name] = tensor

    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-L-16-SigLIP2-384", pretrained="webli", cache_dir=str(HF_CACHE)
    )
    model.load_state_dict(dequant_state)
    model.eval()
    tokenizer = open_clip.get_tokenizer("ViT-L-16-SigLIP2-384")

    del quant_state, dequant_state, checkpoint
    gc.collect()

    log.info("INT8 model loaded and dequantized successfully.")
    return model, preprocess, tokenizer


def save_dequantized_state_dict(model, path):
    """Save full FP32 state dict (after dequant) for Marqo eval harness."""
    torch.save(model.state_dict(), path)
    log.info("Saved dequantized state dict to %s (%.1f MB)", path, os.path.getsize(path) / 1e6)


def run_eval(dataset_key: str, pretrained_path: str, device: str = "mps"):
    """Run Marqo's eval.py for the INT8 model on one dataset."""
    config_file = MARQO_DIR / "configs" / DATASET_CONFIGS[dataset_key]
    if not config_file.exists():
        log.warning("Config not found: %s — skipping", config_file)
        return None

    cmd = [
        str(PYTHON),
        "eval.py",
        "--dataset-config", str(config_file.resolve()),
        "--model-name", "ViT-L-16-SigLIP2-384",
        "--pretrained", pretrained_path,
        "--run-name", "SigLIP2-L16-INT8",
        "--batch-size", "16",
        "--device", device,
        "--output-dir", str(RESULTS_DIR.resolve()),
        "--data-dir", str((MARQO_DIR / "data").resolve()),
        "--cache-dir", str(HF_CACHE),
        "--overwrite-embeddings",
        "--overwrite-retrieval",
    ]

    log.info("Running eval: dataset=%s", dataset_key)
    log.info("Command: %s", " ".join(cmd))

    try:
        result = subprocess.run(
            cmd,
            cwd=str(MARQO_DIR),
            timeout=14400,
        )
        if result.returncode != 0:
            log.error("Eval failed for %s (exit %d)", dataset_key, result.returncode)
            return None
    except subprocess.TimeoutExpired:
        log.error("Eval timed out for %s", dataset_key)
        return None
    except Exception as e:
        log.error("Eval error for %s: %s", dataset_key, e)
        return None

    return collect_results(dataset_key)


def collect_results(dataset_key: str):
    """Parse results from Marqo eval output."""
    results_base = RESULTS_DIR / dataset_key / "SigLIP2-L16-INT8"
    if not results_base.exists():
        log.warning("Results dir not found: %s", results_base)
        return None

    task_results = {}
    for task_dir in results_base.iterdir():
        if not task_dir.is_dir():
            continue
        metrics_file = task_dir / "metrics.json"
        if metrics_file.exists():
            with open(metrics_file) as f:
                task_results[task_dir.name] = json.load(f)

    return task_results


def main():
    log.info("=" * 60)
    log.info("OVERNIGHT INT8 FULL-CORPUS EVALUATION")
    log.info("=" * 60)

    # Load and save dequantized model for eval harness
    dequant_path = ROOT / "models" / "siglip2-l16-int8" / "model_dequantized.pt"
    if not dequant_path.exists():
        model, preprocess, tokenizer = load_int8_model()
        save_dequantized_state_dict(model, dequant_path)
        del model, preprocess, tokenizer
        gc.collect()
        torch.mps.empty_cache() if torch.backends.mps.is_available() else None
    else:
        log.info("Dequantized state dict already exists at %s", dequant_path)

    # Run eval on each dataset
    all_results = {}
    for ds in DATASETS:
        log.info("\n" + "=" * 40)
        log.info("DATASET: %s", ds)
        log.info("=" * 40)
        result = run_eval(ds, str(dequant_path))
        if result:
            all_results[ds] = result
            log.info("Results for %s: %s", ds, json.dumps(result, indent=2))
        else:
            log.error("No results for %s", ds)

    # Summary
    log.info("\n" + "=" * 60)
    log.info("FULL-CORPUS RESULTS SUMMARY")
    log.info("=" * 60)

    fsl_baseline = {
        "fashion200k": 0.1858,
        "atlas": 0.1826,
        "polyvore": 0.3665,
        "KAGL": 0.2769,
    }

    summary_file = RESULTS_DIR / "summary.json"
    with open(summary_file, "w") as f:
        json.dump(all_results, f, indent=2)
    log.info("Full results saved to %s", summary_file)

    for ds, res in all_results.items():
        if "text-to-image" in res:
            metrics = res["text-to-image"]
            map10 = metrics.get("MAP@10", metrics.get("mAP", "N/A"))
            log.info("  %s: MAP@10 = %s (FSL baseline: %.4f)", ds, map10, fsl_baseline.get(ds, 0))

    log.info("\nDONE — overnight evaluation complete.")


if __name__ == "__main__":
    main()
