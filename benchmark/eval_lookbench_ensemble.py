"""
MODA — LookBench Ensemble & TTA Evaluation

Zero-training approaches to beat FashionSigLIP baseline on LookBench:

1. **Embedding Ensemble**: Combine FashionSigLIP + FashionCLIP embeddings.
   Since they have different dimensionalities (768 vs 512), we L2-normalize
   each model's embeddings independently, then concatenate and re-normalize.
   Cosine similarity on the concatenated space treats both models equally.

2. **Test-Time Augmentation (TTA)**: Encode each image with multiple
   augmentations (original + horizontal flip) and average the embeddings
   before normalization. This creates more robust representations.

3. **Ensemble + TTA**: Both combined.

Usage:
  python benchmark/eval_lookbench_ensemble.py --mode ensemble
  python benchmark/eval_lookbench_ensemble.py --mode tta
  python benchmark/eval_lookbench_ensemble.py --mode ensemble_tta
  python benchmark/eval_lookbench_ensemble.py --mode all
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

import open_clip

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from benchmark.eval_lookbench_baseline import (
    SUBSETS, NOISE_LABEL, NOISE_CAT,
    extract_labels, compute_all_metrics, compute_mrr,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(H:%M:%S)s %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)

RESULTS_DIR = REPO_ROOT / "results" / "lookbench"


MODELS = {
    "fashionsiglip": {
        "hf_hub": "hf-hub:Marqo/marqo-fashionSigLIP",
        "display": "FashionSigLIP",
        "emb_dim": 768,
    },
    "fashionclip": {
        "hf_hub": "hf-hub:Marqo/marqo-fashionCLIP",
        "display": "FashionCLIP",
        "emb_dim": 512,
    },
    "siglip-deepfashion2": {
        "hf_hub": "hf-hub:Marqo/marqo-fashionSigLIP",
        "display": "SigLIP-DeepFashion2",
        "emb_dim": 768,
        "checkpoint": "models/moda-siglip-deepfashion2/best/model_state_dict.pt",
    },
}


def load_model(key: str, device: str):
    cfg = MODELS[key]
    log.info("Loading %s ...", cfg["display"])
    model, _, preprocess = open_clip.create_model_and_transforms(cfg["hf_hub"])
    ckpt = cfg.get("checkpoint")
    if ckpt:
        ckpt_full = REPO_ROOT / ckpt
        log.info("  applying checkpoint: %s", ckpt_full)
        state = torch.load(ckpt_full, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=False)
    model = model.to(device).eval()
    return model, preprocess


@torch.no_grad()
def encode_images(data, model, preprocess, device, batch_size=32, desc="Encoding"):
    """Encode images, return L2-normalized features and item_IDs."""
    all_feats = []
    all_ids = []
    for start in tqdm(range(0, len(data), batch_size), desc=desc):
        batch = data[start:start + batch_size]
        images = batch["image"]
        ids = batch.get("item_ID", batch.get("item_id", list(range(start, start + len(images)))))
        tensors = torch.stack([preprocess(img.convert("RGB")) for img in images]).to(device)
        feats = model.encode_image(tensors).float()
        feats = F.normalize(feats, p=2, dim=-1)
        all_feats.append(feats.cpu())
        if isinstance(ids, (list, np.ndarray)):
            all_ids.extend(ids)
        else:
            all_ids.extend(ids.tolist() if hasattr(ids, "tolist") else [ids])
    return torch.cat(all_feats, dim=0).numpy(), all_ids


@torch.no_grad()
def encode_images_tta(data, model, preprocess, device, batch_size=32, desc="TTA"):
    """Encode with test-time augmentation: original + horizontal flip, averaged."""
    import torchvision.transforms.functional as TF

    all_feats = []
    all_ids = []
    for start in tqdm(range(0, len(data), batch_size), desc=desc):
        batch = data[start:start + batch_size]
        images = batch["image"]
        ids = batch.get("item_ID", batch.get("item_id", list(range(start, start + len(images)))))

        pil_images = [img.convert("RGB") for img in images]

        # Original
        t_orig = torch.stack([preprocess(img) for img in pil_images]).to(device)
        f_orig = model.encode_image(t_orig).float()

        # Horizontal flip
        flipped = [TF.hflip(img) for img in pil_images]
        t_flip = torch.stack([preprocess(img) for img in flipped]).to(device)
        f_flip = model.encode_image(t_flip).float()

        # Average and normalize
        feats = F.normalize((f_orig + f_flip) / 2, p=2, dim=-1)
        all_feats.append(feats.cpu())
        if isinstance(ids, (list, np.ndarray)):
            all_ids.extend(ids)
        else:
            all_ids.extend(ids.tolist() if hasattr(ids, "tolist") else [ids])

    return torch.cat(all_feats, dim=0).numpy(), all_ids


def evaluate_subset_with_feats(
    subset_name, q_feats, q_ids, g_feats, g_ids,
    query_data, gallery_data, noise_ids, noise_labels, noise_cats,
):
    """Evaluate using pre-computed features."""
    q_cats, _, q_labels = extract_labels(query_data)
    g_cats_subset, _, g_labels_subset = extract_labels(gallery_data)

    g_labels = g_labels_subset + noise_labels
    g_cats = g_cats_subset + noise_cats

    metrics = compute_all_metrics(
        q_feats, q_labels, q_cats, q_ids,
        g_feats, g_labels, g_cats, g_ids,
    )
    mrr = compute_mrr(q_feats, q_ids, g_feats, g_ids)

    log.info(
        "  %s  Fine_R@1=%.2f  Coarse_R@1=%.2f  nDCG@5=%.2f  MRR=%.2f",
        subset_name,
        metrics["fine_recall"]["recall@1"],
        metrics["coarse_recall"]["recall@1"],
        metrics["ndcg@5"], mrr,
    )
    return {
        "subset": subset_name,
        "n_queries": len(q_ids),
        **metrics,
        "mrr": mrr,
    }


def run_ensemble(device, batch_size, siglip_key="fashionsiglip"):
    """Evaluate SigLIP + FashionCLIP embedding concatenation."""
    log.info("=" * 60)
    log.info("ENSEMBLE: %s + FashionCLIP (concat + re-norm)",
             MODELS[siglip_key]["display"])
    log.info("=" * 60)

    siglip_model, siglip_pp = load_model(siglip_key, device)
    clip_model, clip_pp = load_model("fashionclip", device)

    noise_ds = load_dataset("srpone/look-bench", "noise")["gallery"]

    log.info("Encoding noise with FashionSigLIP...")
    noise_feats_s, noise_ids = encode_images(noise_ds, siglip_model, siglip_pp, device, batch_size, "noise-siglip")
    log.info("Encoding noise with FashionCLIP...")
    noise_feats_c, _ = encode_images(noise_ds, clip_model, clip_pp, device, batch_size, "noise-clip")

    # Concatenate and re-normalize
    noise_feats = np.concatenate([noise_feats_s, noise_feats_c], axis=1)
    noise_feats = noise_feats / np.linalg.norm(noise_feats, axis=1, keepdims=True)

    noise_labels = [NOISE_LABEL] * len(noise_ids)
    noise_cats = [NOISE_CAT] * len(noise_ids)

    results = {}
    for subset in SUBSETS:
        ds = load_dataset("srpone/look-bench", subset)
        q_data, g_data = ds["query"], ds["gallery"]

        q_s, q_ids = encode_images(q_data, siglip_model, siglip_pp, device, batch_size, f"{subset}-q-siglip")
        q_c, _ = encode_images(q_data, clip_model, clip_pp, device, batch_size, f"{subset}-q-clip")
        q_feats = np.concatenate([q_s, q_c], axis=1)
        q_feats = q_feats / np.linalg.norm(q_feats, axis=1, keepdims=True)

        g_s, g_ids_sub = encode_images(g_data, siglip_model, siglip_pp, device, batch_size, f"{subset}-g-siglip")
        g_c, _ = encode_images(g_data, clip_model, clip_pp, device, batch_size, f"{subset}-g-clip")
        g_feats_sub = np.concatenate([g_s, g_c], axis=1)
        g_feats_sub = g_feats_sub / np.linalg.norm(g_feats_sub, axis=1, keepdims=True)

        g_feats = np.concatenate([g_feats_sub, noise_feats], axis=0)
        g_ids = list(g_ids_sub) + list(noise_ids)

        results[subset] = evaluate_subset_with_feats(
            subset, q_feats, q_ids, g_feats, g_ids,
            q_data, g_data, noise_ids, noise_labels, noise_cats,
        )

    return results


def run_tta(device, batch_size):
    """Evaluate FashionSigLIP with test-time augmentation."""
    log.info("=" * 60)
    log.info("TTA: FashionSigLIP with horizontal flip augmentation")
    log.info("=" * 60)

    model, preprocess = load_model("fashionsiglip", device)

    noise_ds = load_dataset("srpone/look-bench", "noise")["gallery"]
    log.info("Encoding noise with TTA...")
    noise_feats, noise_ids = encode_images_tta(noise_ds, model, preprocess, device, batch_size, "noise-tta")
    noise_labels = [NOISE_LABEL] * len(noise_ids)
    noise_cats = [NOISE_CAT] * len(noise_ids)

    results = {}
    for subset in SUBSETS:
        ds = load_dataset("srpone/look-bench", subset)
        q_data, g_data = ds["query"], ds["gallery"]

        q_feats, q_ids = encode_images_tta(q_data, model, preprocess, device, batch_size, f"{subset}-q-tta")
        g_feats_sub, g_ids_sub = encode_images_tta(g_data, model, preprocess, device, batch_size, f"{subset}-g-tta")

        g_feats = np.concatenate([g_feats_sub, noise_feats], axis=0)
        g_ids = list(g_ids_sub) + list(noise_ids)

        results[subset] = evaluate_subset_with_feats(
            subset, q_feats, q_ids, g_feats, g_ids,
            q_data, g_data, noise_ids, noise_labels, noise_cats,
        )

    return results


def run_ensemble_tta(device, batch_size):
    """Evaluate ensemble (SigLIP+CLIP) with TTA on both models."""
    log.info("=" * 60)
    log.info("ENSEMBLE + TTA: SigLIP(tta) + CLIP(tta) concat")
    log.info("=" * 60)

    siglip_model, siglip_pp = load_model("fashionsiglip", device)
    clip_model, clip_pp = load_model("fashionclip", device)

    noise_ds = load_dataset("srpone/look-bench", "noise")["gallery"]

    noise_feats_s, noise_ids = encode_images_tta(noise_ds, siglip_model, siglip_pp, device, batch_size, "noise-siglip-tta")
    noise_feats_c, _ = encode_images_tta(noise_ds, clip_model, clip_pp, device, batch_size, "noise-clip-tta")
    noise_feats = np.concatenate([noise_feats_s, noise_feats_c], axis=1)
    noise_feats = noise_feats / np.linalg.norm(noise_feats, axis=1, keepdims=True)

    noise_labels = [NOISE_LABEL] * len(noise_ids)
    noise_cats = [NOISE_CAT] * len(noise_ids)

    results = {}
    for subset in SUBSETS:
        ds = load_dataset("srpone/look-bench", subset)
        q_data, g_data = ds["query"], ds["gallery"]

        q_s, q_ids = encode_images_tta(q_data, siglip_model, siglip_pp, device, batch_size, f"{subset}-q-siglip-tta")
        q_c, _ = encode_images_tta(q_data, clip_model, clip_pp, device, batch_size, f"{subset}-q-clip-tta")
        q_feats = np.concatenate([q_s, q_c], axis=1)
        q_feats = q_feats / np.linalg.norm(q_feats, axis=1, keepdims=True)

        g_s, g_ids_sub = encode_images_tta(g_data, siglip_model, siglip_pp, device, batch_size, f"{subset}-g-siglip-tta")
        g_c, _ = encode_images_tta(g_data, clip_model, clip_pp, device, batch_size, f"{subset}-g-clip-tta")
        g_feats_sub = np.concatenate([g_s, g_c], axis=1)
        g_feats_sub = g_feats_sub / np.linalg.norm(g_feats_sub, axis=1, keepdims=True)

        g_feats = np.concatenate([g_feats_sub, noise_feats], axis=0)
        g_ids = list(g_ids_sub) + list(noise_ids)

        results[subset] = evaluate_subset_with_feats(
            subset, q_feats, q_ids, g_feats, g_ids,
            q_data, g_data, noise_ids, noise_labels, noise_cats,
        )

    return results


def print_summary(all_results: dict):
    """Print comparison table."""
    print("\n" + "=" * 80)
    print("SUMMARY: Fine Recall@1 (%)")
    print("=" * 80)
    header = f"  {'Model':<35} {'RealStudio':>10} {'AIGenStu':>10} {'RealStr':>10} {'AIGenStr':>10} {'Overall':>10}"
    print(header)
    print("  " + "-" * 85)

    # Baseline reference
    baselines = {
        "FashionSigLIP (baseline)": {
            "real_studio_flat": 66.96, "aigen_studio": 76.68,
            "real_streetlook": 56.37, "aigen_streetlook": 74.38, "overall": 63.84,
        },
    }
    for name, vals in baselines.items():
        print(f"  {name:<35} {vals['real_studio_flat']:>9.2f}% {vals['aigen_studio']:>9.2f}% "
              f"{vals['real_streetlook']:>9.2f}% {vals['aigen_streetlook']:>9.2f}% {vals['overall']:>9.2f}%")

    for mode_name, results in all_results.items():
        overall_fine = 0
        total_q = 0
        vals = {}
        for subset, r in results.items():
            vals[subset] = r["fine_recall"]["recall@1"]
            overall_fine += r["fine_recall"]["recall@1"] * r["n_queries"]
            total_q += r["n_queries"]
        vals["overall"] = overall_fine / total_q if total_q else 0
        print(f"  {mode_name:<35} {vals.get('real_studio_flat', 0):>9.2f}% {vals.get('aigen_studio', 0):>9.2f}% "
              f"{vals.get('real_streetlook', 0):>9.2f}% {vals.get('aigen_streetlook', 0):>9.2f}% {vals['overall']:>9.2f}%")

    print()


def main():
    parser = argparse.ArgumentParser(description="LookBench ensemble & TTA evaluation")
    parser.add_argument("--mode", default="all",
                        choices=["ensemble", "tta", "ensemble_tta",
                                 "ensemble_df2", "all"])
    parser.add_argument("--batch_size", type=int, default=32)
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
    all_results = {}

    modes = [args.mode] if args.mode != "all" else ["ensemble", "tta", "ensemble_tta"]

    for mode in modes:
        t0 = time.time()
        if mode == "ensemble":
            results = run_ensemble(device, args.batch_size)
        elif mode == "ensemble_df2":
            results = run_ensemble(device, args.batch_size,
                                   siglip_key="siglip-deepfashion2")
        elif mode == "tta":
            results = run_tta(device, args.batch_size)
        elif mode == "ensemble_tta":
            results = run_ensemble_tta(device, args.batch_size)
        elapsed = time.time() - t0
        log.info("%s completed in %.1f seconds", mode, elapsed)
        all_results[mode] = results

    print_summary(all_results)

    out_path = RESULTS_DIR / "ensemble_tta_eval.json"
    serializable = {}
    for mode, results in all_results.items():
        serializable[mode] = results
    with open(out_path, "w") as f:
        json.dump(serializable, f, indent=2)
    log.info("Results saved to %s", out_path)


if __name__ == "__main__":
    main()
