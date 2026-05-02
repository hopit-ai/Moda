"""
MODA — LookBench DINOv2 Evaluation

Evaluates DINOv2 vision foundation models on LookBench image-to-image
retrieval. DINOv2 produces pure visual features via self-supervised learning
(no text involved), making it ideal for image-to-image similarity.

Models:
  - facebook/dinov2-large  (ViT-L/14, 300M params, 1024-dim)
  - facebook/dinov2-giant  (ViT-g/14, 1.1B params, 1536-dim)

Usage:
  python benchmark/eval_lookbench_dinov2.py --model dinov2-large
  python benchmark/eval_lookbench_dinov2.py --model dinov2-giant
  python benchmark/eval_lookbench_dinov2.py --model all
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from benchmark.eval_lookbench_baseline import (
    SUBSETS, NOISE_LABEL, NOISE_CAT,
    extract_labels, compute_all_metrics, compute_mrr,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

RESULTS_DIR = REPO_ROOT / "results" / "lookbench"

DINOV2_MODELS = {
    "dinov2-base": {
        "hf_id": "facebook/dinov2-base",
        "display": "DINOv2-Base (86M)",
        "emb_dim": 768,
    },
    "dinov2-large": {
        "hf_id": "facebook/dinov2-large",
        "display": "DINOv2-Large (300M)",
        "emb_dim": 1024,
    },
    "dinov2-giant": {
        "hf_id": "facebook/dinov2-giant",
        "display": "DINOv2-Giant (1.1B)",
        "emb_dim": 1536,
    },
}


def load_dinov2(model_key: str, device: str):
    cfg = DINOV2_MODELS[model_key]
    log.info("Loading %s ...", cfg["display"])
    processor = AutoImageProcessor.from_pretrained(cfg["hf_id"])
    model = AutoModel.from_pretrained(cfg["hf_id"]).to(device).eval()
    n_params = sum(p.numel() for p in model.parameters())
    log.info("Loaded %s on %s (%d params, %d-dim)",
             cfg["display"], device, n_params, cfg["emb_dim"])
    return model, processor


@torch.no_grad()
def encode_images_dinov2(data, model, processor, device, batch_size=32, desc="Encoding"):
    """Encode images with DINOv2, using CLS token as embedding."""
    all_feats = []
    all_ids = []

    for start in tqdm(range(0, len(data), batch_size), desc=desc):
        batch = data[start:start + batch_size]
        images = batch["image"]
        ids = batch.get("item_ID", batch.get("item_id",
                        list(range(start, start + len(images)))))

        pil_images = [img.convert("RGB") for img in images]
        inputs = processor(images=pil_images, return_tensors="pt").to(device)

        outputs = model(**inputs)
        # CLS token is the first token in last_hidden_state
        feats = outputs.last_hidden_state[:, 0, :].float()
        feats = F.normalize(feats, p=2, dim=-1)

        all_feats.append(feats.cpu())
        if isinstance(ids, (list, np.ndarray)):
            all_ids.extend(ids)
        else:
            all_ids.extend(ids.tolist() if hasattr(ids, "tolist") else [ids])

    return torch.cat(all_feats, dim=0).numpy(), all_ids


def evaluate_model(model_key: str, device: str, batch_size: int):
    """Run full LookBench evaluation for a DINOv2 model."""
    cfg = DINOV2_MODELS[model_key]
    log.info("\n" + "=" * 70)
    log.info("MODEL: %s", cfg["display"])
    log.info("=" * 70)

    model, processor = load_dinov2(model_key, device)

    # Encode shared noise gallery
    log.info("Encoding noise gallery...")
    noise_ds = load_dataset("srpone/look-bench", "noise")["gallery"]
    noise_feats, noise_ids = encode_images_dinov2(
        noise_ds, model, processor, device, batch_size, "noise")
    noise_labels = [NOISE_LABEL] * len(noise_ids)
    noise_cats = [NOISE_CAT] * len(noise_ids)
    log.info("Noise encoded: %d items", len(noise_ids))

    results = {}
    for subset in SUBSETS:
        log.info("Evaluating subset: %s", subset)
        ds = load_dataset("srpone/look-bench", subset)
        q_data, g_data = ds["query"], ds["gallery"]

        q_cats, _, q_labels = extract_labels(q_data)
        g_cats_sub, _, g_labels_sub = extract_labels(g_data)

        q_feats, q_ids = encode_images_dinov2(
            q_data, model, processor, device, batch_size, f"{subset}-queries")
        g_feats_sub, g_ids_sub = encode_images_dinov2(
            g_data, model, processor, device, batch_size, f"{subset}-gallery")

        g_feats = np.concatenate([g_feats_sub, noise_feats], axis=0)
        g_ids = list(g_ids_sub) + list(noise_ids)
        g_labels = g_labels_sub + noise_labels
        g_cats = g_cats_sub + noise_cats

        t0 = time.time()
        metrics = compute_all_metrics(
            q_feats, q_labels, q_cats, q_ids,
            g_feats, g_labels, g_cats, g_ids,
        )
        mrr = compute_mrr(q_feats, q_ids, g_feats, g_ids)

        log.info(
            "  %s  Fine_R@1=%.2f  Coarse_R@1=%.2f  ID_R@1=%.2f  nDCG@5=%.2f  MRR=%.2f",
            subset,
            metrics["fine_recall"]["recall@1"],
            metrics["coarse_recall"]["recall@1"],
            metrics["id_recall"]["recall@1"],
            metrics["ndcg@5"], mrr,
        )
        results[subset] = {
            "subset": subset,
            "n_queries": len(q_ids),
            **metrics,
            "mrr": mrr,
            "elapsed_sec": round(time.time() - t0, 1),
        }

    # Overall
    total_q = sum(r["n_queries"] for r in results.values())
    overall = {
        "fine_recall@1": round(sum(r["fine_recall"]["recall@1"] * r["n_queries"]
                                   for r in results.values()) / total_q, 2),
        "coarse_recall@1": round(sum(r["coarse_recall"]["recall@1"] * r["n_queries"]
                                     for r in results.values()) / total_q, 2),
        "id_recall@1": round(sum(r["id_recall"]["recall@1"] * r["n_queries"]
                                 for r in results.values()) / total_q, 2),
        "ndcg@5": round(sum(r["ndcg@5"] * r["n_queries"]
                            for r in results.values()) / total_q, 2),
        "total_queries": total_q,
    }
    results["overall"] = overall

    log.info("\n%s  Fine_R@1=%.2f  Coarse_R@1=%.2f  ID_R@1=%.2f  nDCG@5=%.2f",
             cfg["display"], overall["fine_recall@1"], overall["coarse_recall@1"],
             overall["id_recall@1"], overall["ndcg@5"])

    return {
        "model": cfg["display"],
        "hf_id": cfg["hf_id"],
        "emb_dim": cfg["emb_dim"],
        "device": device,
        "subsets": results,
    }


def main():
    parser = argparse.ArgumentParser(description="LookBench DINOv2 evaluation")
    parser.add_argument("--model", default="all",
                        choices=list(DINOV2_MODELS.keys()) + ["all"])
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = args.device
    if not device:
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
    log.info("Device: %s", device)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    models = list(DINOV2_MODELS.keys()) if args.model == "all" else [args.model]
    all_results = {}

    for model_key in models:
        t0 = time.time()
        result = evaluate_model(model_key, device, args.batch_size)
        elapsed = time.time() - t0
        log.info("%s completed in %.1f seconds", model_key, elapsed)
        all_results[model_key] = result

    # Summary table
    print("\n" + "=" * 90)
    print("SUMMARY: Fine Recall@1 (%)")
    print("=" * 90)
    header = f"  {'Model':<30} {'RealStudio':>10} {'AIGenStu':>10} {'RealStr':>10} {'AIGenStr':>10} {'Overall':>10}"
    print(header)
    print("  " + "-" * 80)
    print(f"  {'FashionSigLIP (baseline)':<30} {'66.96':>10} {'76.68':>10} {'56.37':>10} {'74.38':>10} {'63.84':>10}")
    for key, result in all_results.items():
        s = result["subsets"]
        print(f"  {result['model']:<30} "
              f"{s['real_studio_flat']['fine_recall']['recall@1']:>9.2f}% "
              f"{s['aigen_studio']['fine_recall']['recall@1']:>9.2f}% "
              f"{s['real_streetlook']['fine_recall']['recall@1']:>9.2f}% "
              f"{s['aigen_streetlook']['fine_recall']['recall@1']:>9.2f}% "
              f"{s['overall']['fine_recall@1']:>9.2f}%")
    print()

    out_path = RESULTS_DIR / "dinov2_eval.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    log.info("Results saved to %s", out_path)


if __name__ == "__main__":
    main()
