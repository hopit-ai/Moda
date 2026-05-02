"""
MODA Phase 5 — LookBench Baseline Evaluation

Evaluates FashionSigLIP and FashionCLIP (zero-shot) on all four LookBench
subsets (RealStudioFlat, AIGen-Studio, RealStreetLook, AIGen-StreetLook).

Metrics aligned with paper arXiv:2601.14706 (validated empirically):
  - Fine Recall@K:   category AND main_attribute must both match
  - Coarse Recall@K: category must match
  - nDCG@K:          binary relevance from category+main_attribute match
  - ID Recall@K:     exact item_ID match (our additional sanity check)

NOTE: The paper's appendix describes Fine Recall as A_q ⊆ A_i (attribute
subset) and nDCG as graded |A_q ∩ A_i|/|A_q|. However, empirical validation
shows the paper's actual reported numbers use category+main_attribute equality
for both Fine Recall and nDCG relevance. Our reproduction matches the paper
within ~1% using this definition. See EXPERIMENT_LOG.md Phase 5 for details.

Paper reference numbers (query-weighted Overall):
  Table 3  Fine R@1:   FashionSigLIP=62.77  FashionCLIP=60.30
  Table 7  nDCG@5:     FashionSigLIP=49.44  FashionCLIP=48.63
  Table 8  Coarse R@1: FashionSigLIP=82.77  FashionCLIP=82.68

Usage:
  python benchmark/eval_lookbench_baseline.py
  python benchmark/eval_lookbench_baseline.py --models fashionsiglip
  python benchmark/eval_lookbench_baseline.py --subsets real_studio_flat aigen_studio
  python benchmark/eval_lookbench_baseline.py --batch_size 64
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import open_clip
import torch
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results" / "lookbench"

SUBSETS = [
    "real_studio_flat",
    "aigen_studio",
    "real_streetlook",
    "aigen_streetlook",
]

MODEL_CONFIGS = {
    "fashionsiglip": {
        "hf_hub": "hf-hub:Marqo/marqo-fashionSigLIP",
        "display": "Marqo-FashionSigLIP",
        "emb_dim": 768,
        "input_size": 224,
    },
    "fashionclip": {
        "hf_hub": "hf-hub:Marqo/marqo-fashionCLIP",
        "display": "Marqo-FashionCLIP",
        "emb_dim": 512,
        "input_size": 224,
    },
    "fashionclip-multimodal": {
        "hf_hub": "hf-hub:Marqo/marqo-fashionCLIP",
        "display": "MODA-FashionCLIP-Multimodal",
        "emb_dim": 512,
        "input_size": 224,
        "checkpoint": "models/moda-fashionclip-multimodal/best/model_state_dict.pt",
    },
    "siglip-deepfashion2": {
        "hf_hub": "hf-hub:Marqo/marqo-fashionSigLIP",
        "display": "MODA-SigLIP-DeepFashion2",
        "emb_dim": 768,
        "input_size": 224,
        "checkpoint": "models/moda-siglip-deepfashion2/best/model_state_dict.pt",
    },
    "siglip-vision-ft": {
        "hf_hub": "hf-hub:Marqo/marqo-fashionSigLIP",
        "display": "MODA-SigLIP-Vision-FT",
        "emb_dim": 768,
        "input_size": 224,
        "checkpoint": "models/moda-siglip-vision-finetuned/best/model_state_dict.pt",
    },
    "siglip-distilled": {
        "hf_hub": "hf-hub:Marqo/marqo-fashionSigLIP",
        "display": "MODA-SigLIP-Distilled",
        "emb_dim": 768,
        "input_size": 224,
        "checkpoint": "models/moda-siglip-distilled/best/model_state_dict.pt",
    },
    "siglip-recipe-z": {
        "hf_hub": "hf-hub:Marqo/marqo-fashionSigLIP",
        "display": "MODA-SigLIP-Recipe-Z",
        "emb_dim": 768,
        "input_size": 224,
        "checkpoint": "models/moda-siglip-recipe-z/best/model_state_dict.pt",
    },
}


# ---------------------------------------------------------------------------
# Attribute helpers
# ---------------------------------------------------------------------------

def build_item_label(item: dict) -> tuple[str, str, str]:
    """Return (category, main_attribute, cat_main_label) for matching."""
    cat = str(item.get("category", "")).strip().lower()
    main = str(item.get("main_attribute", "")).strip().lower()
    return (cat, main, f"{cat}|{main}")


def extract_labels(data) -> tuple[list[str], list[str], list[str]]:
    """Extract categories, main_attributes, and composite labels for a split."""
    cats, mains, labels = [], [], []
    for i in range(len(data)):
        cat, main, label = build_item_label(data[i])
        cats.append(cat)
        mains.append(main)
        labels.append(label)
    return cats, mains, labels


def fine_match(q_label: str, g_label: str) -> bool:
    """Paper's Fine Recall: category AND main_attribute must both match."""
    return q_label == g_label


def coarse_match(q_cat: str, g_cat: str) -> bool:
    """Paper's Coarse Recall: category must match."""
    return q_cat == g_cat


def ndcg_relevance(q_label: str, g_label: str) -> float:
    """Paper's nDCG relevance: binary match on category+main_attribute."""
    return 1.0 if q_label == g_label else 0.0


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_open_clip_model(model_key: str, device: str):
    """Load a model via OpenCLIP's hub integration, optionally applying a fine-tuned checkpoint."""
    cfg = MODEL_CONFIGS[model_key]
    log.info("Loading %s ...", cfg["display"])
    model, _, preprocess = open_clip.create_model_and_transforms(cfg["hf_hub"])

    ckpt_path = cfg.get("checkpoint")
    if ckpt_path:
        ckpt_full = REPO_ROOT / ckpt_path
        log.info("Applying fine-tuned checkpoint: %s", ckpt_full)
        state_dict = torch.load(ckpt_full, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict, strict=False)
        log.info("Checkpoint loaded (%d keys)", len(state_dict))

    model = model.to(device).eval()
    log.info("Loaded on %s  (emb_dim=%d)", device, cfg["emb_dim"])
    return model, preprocess


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_image_features(
    data,
    model,
    preprocess,
    device: str,
    batch_size: int = 32,
    desc: str = "Encoding",
) -> tuple[np.ndarray, list]:
    """Encode all images in a HF dataset split, return (features, item_IDs)."""
    all_feats: list[torch.Tensor] = []
    all_labels: list = []

    for start in tqdm(range(0, len(data), batch_size), desc=desc):
        batch = data[start : start + batch_size]
        images = batch["image"]
        labels = batch.get("item_ID", batch.get("item_id", list(range(start, start + len(images)))))

        tensors = torch.stack([preprocess(img.convert("RGB")) for img in images]).to(device)

        if device == "mps":
            feats = model.encode_image(tensors).float()
        else:
            feats = model.encode_image(tensors)

        feats = F.normalize(feats, p=2, dim=-1)
        all_feats.append(feats.cpu())
        if isinstance(labels, (list, np.ndarray)):
            all_labels.extend(labels)
        else:
            all_labels.extend(labels.tolist() if hasattr(labels, "tolist") else [labels])

    features = torch.cat(all_feats, dim=0).numpy()
    return features, all_labels


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def compute_all_metrics(
    query_feats: np.ndarray,
    query_labels: list[str],
    query_cats: list[str],
    query_ids: list,
    gallery_feats: np.ndarray,
    gallery_labels: list[str],
    gallery_cats: list[str],
    gallery_ids: list,
    k_values: list[int] = (1, 5, 10, 20),
    ndcg_k: int = 5,
    chunk_size: int = 256,
) -> dict:
    """Compute Fine/Coarse/ID Recall@K and nDCG@K.

    Fine Recall and nDCG use cat+main_attribute composite label matching,
    aligned with the paper's actual evaluation (see module docstring).
    """
    n_queries = len(query_ids)
    gallery_ids_arr = np.array(gallery_ids, dtype=str)
    gallery_labels_arr = np.array(gallery_labels)
    gallery_cats_arr = np.array(gallery_cats)
    query_ids_str = [str(l) for l in query_ids]
    max_k = max(max(k_values), ndcg_k)

    fine_recalls = {k: 0.0 for k in k_values}
    coarse_recalls = {k: 0.0 for k in k_values}
    id_recalls = {k: 0.0 for k in k_values}
    ndcg_sum = 0.0
    idcg = sum(1.0 / np.log2(r + 2) for r in range(ndcg_k))

    for start in tqdm(range(0, n_queries, chunk_size), desc="Metrics"):
        end = min(start + chunk_size, n_queries)
        q_batch = query_feats[start:end]
        sim = q_batch @ gallery_feats.T

        part_idx = np.argpartition(-sim, max_k, axis=1)[:, :max_k]
        part_sims = np.take_along_axis(sim, part_idx, axis=1)
        sorted_within = np.argsort(-part_sims, axis=1)
        topk_sorted = np.take_along_axis(part_idx, sorted_within, axis=1)

        for i in range(end - start):
            qi = start + i
            q_label = query_labels[qi]
            q_cat = query_cats[qi]
            q_id = query_ids_str[qi]
            top_indices = topk_sorted[i]

            fine_found_at = None
            coarse_found_at = None
            id_found_at = None

            dcg = 0.0
            for rank, g_idx in enumerate(top_indices):
                if fine_found_at is None and fine_match(q_label, gallery_labels_arr[g_idx]):
                    fine_found_at = rank
                if coarse_found_at is None and coarse_match(q_cat, gallery_cats_arr[g_idx]):
                    coarse_found_at = rank
                if id_found_at is None and gallery_ids_arr[g_idx] == q_id:
                    id_found_at = rank
                if rank < ndcg_k:
                    dcg += ndcg_relevance(q_label, gallery_labels_arr[g_idx]) / np.log2(rank + 2)

            ndcg_sum += dcg / idcg if idcg > 0 else 0.0

            for k in k_values:
                if fine_found_at is not None and fine_found_at < k:
                    fine_recalls[k] += 1.0
                if coarse_found_at is not None and coarse_found_at < k:
                    coarse_recalls[k] += 1.0
                if id_found_at is not None and id_found_at < k:
                    id_recalls[k] += 1.0

    results = {"fine_recall": {}, "coarse_recall": {}, "id_recall": {}}
    for k in k_values:
        results["fine_recall"][f"recall@{k}"] = round(fine_recalls[k] / n_queries * 100, 2)
        results["coarse_recall"][f"recall@{k}"] = round(coarse_recalls[k] / n_queries * 100, 2)
        results["id_recall"][f"recall@{k}"] = round(id_recalls[k] / n_queries * 100, 2)
    results[f"ndcg@{ndcg_k}"] = round(ndcg_sum / n_queries * 100, 2)
    return results


def compute_mrr(
    query_feats: np.ndarray,
    query_ids: list,
    gallery_feats: np.ndarray,
    gallery_ids: list,
    chunk_size: int = 256,
) -> float:
    """Compute Mean Reciprocal Rank (ID-based)."""
    n_queries = len(query_ids)
    gallery_ids_arr = np.array(gallery_ids, dtype=str)
    query_ids_str = [str(l) for l in query_ids]
    mrr_sum = 0.0

    for start in tqdm(range(0, n_queries, chunk_size), desc="MRR"):
        end = min(start + chunk_size, n_queries)
        q_batch = query_feats[start:end]
        sim = q_batch @ gallery_feats.T
        sorted_indices = np.argsort(-sim, axis=1)

        for i in range(end - start):
            q_id = query_ids_str[start + i]
            ranks = np.where(gallery_ids_arr[sorted_indices[i]] == q_id)[0]
            if len(ranks) > 0:
                mrr_sum += 1.0 / (ranks[0] + 1)

    return round(mrr_sum / n_queries * 100, 2)


# ---------------------------------------------------------------------------
# Subset evaluation
# ---------------------------------------------------------------------------

NOISE_LABEL = "noise|noise"
NOISE_CAT = "noise data"


def evaluate_subset(
    model,
    preprocess,
    subset_name: str,
    noise_feats: np.ndarray,
    noise_ids: list,
    noise_labels: list[str],
    noise_cats: list[str],
    device: str,
    batch_size: int,
) -> dict:
    """Run full evaluation on one LookBench subset."""
    log.info("Loading LookBench subset: %s", subset_name)
    ds = load_dataset("srpone/look-bench", subset_name)

    query_data = ds["query"]
    gallery_data = ds["gallery"]
    n_subset_gallery = len(gallery_data)
    n_total_gallery = n_subset_gallery + len(noise_ids)
    log.info(
        "  Queries: %d  Gallery: %d (subset) + %d (noise) = %d",
        len(query_data), n_subset_gallery, len(noise_ids), n_total_gallery,
    )

    q_cats, _, q_labels = extract_labels(query_data)
    g_cats_subset, _, g_labels_subset = extract_labels(gallery_data)

    log.info("  Encoding queries...")
    q_feats, q_ids = extract_image_features(
        query_data, model, preprocess, device, batch_size, desc=f"{subset_name} queries"
    )

    log.info("  Encoding subset gallery...")
    g_feats_subset, g_ids_subset = extract_image_features(
        gallery_data, model, preprocess, device, batch_size, desc=f"{subset_name} gallery"
    )

    g_feats = np.concatenate([g_feats_subset, noise_feats], axis=0)
    g_ids = g_ids_subset + noise_ids
    g_labels = g_labels_subset + noise_labels
    g_cats = g_cats_subset + noise_cats
    log.info("  Combined gallery: %d items", len(g_ids))

    log.info("  Computing metrics...")
    metrics = compute_all_metrics(
        q_feats, q_labels, q_cats, q_ids,
        g_feats, g_labels, g_cats, g_ids,
    )
    mrr = compute_mrr(q_feats, q_ids, g_feats, g_ids)

    log.info(
        "  %s  Fine_R@1=%.2f  Coarse_R@1=%.2f  ID_R@1=%.2f  nDCG@5=%.2f  MRR=%.2f",
        subset_name,
        metrics["fine_recall"]["recall@1"],
        metrics["coarse_recall"]["recall@1"],
        metrics["id_recall"]["recall@1"],
        metrics["ndcg@5"],
        mrr,
    )
    return {
        "subset": subset_name,
        "n_queries": len(query_data),
        "n_gallery_subset": n_subset_gallery,
        "n_gallery_noise": len(noise_ids),
        "n_gallery_total": n_total_gallery,
        **metrics,
        "mrr": mrr,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="LookBench baseline evaluation")
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(MODEL_CONFIGS.keys()),
        choices=list(MODEL_CONFIGS.keys()),
        help="Models to evaluate",
    )
    parser.add_argument(
        "--subsets",
        nargs="+",
        default=SUBSETS,
        help="LookBench subsets to evaluate on",
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", default=None, help="Force device (mps/cuda/cpu)")
    parser.add_argument(
        "--output", default=None,
        help="Output JSON filename (default: baseline_eval.json). Written under results/lookbench/.",
    )
    args = parser.parse_args()

    if args.device:
        device = args.device
    elif torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    log.info("Device: %s", device)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    all_results = {}

    for model_key in args.models:
        cfg = MODEL_CONFIGS[model_key]
        log.info("\n" + "=" * 70)
        log.info("MODEL: %s", cfg["display"])
        log.info("=" * 70)

        model, preprocess = load_open_clip_model(model_key, device)

        log.info("Encoding shared noise gallery (58K distractors)...")
        noise_ds = load_dataset("srpone/look-bench", "noise")
        noise_feats, noise_ids = extract_image_features(
            noise_ds["gallery"], model, preprocess, device, args.batch_size, desc="noise gallery"
        )
        noise_labels = [NOISE_LABEL] * len(noise_ids)
        noise_cats = [NOISE_CAT] * len(noise_ids)
        log.info("Noise gallery encoded: %d items", len(noise_ids))

        model_results = {}

        for subset in args.subsets:
            t0 = time.time()
            result = evaluate_subset(
                model, preprocess, subset,
                noise_feats, noise_ids, noise_labels, noise_cats,
                device, args.batch_size,
            )
            result["elapsed_sec"] = round(time.time() - t0, 1)
            model_results[subset] = result

        # Paper uses query-weighted average for "Overall"
        evaluated_subsets = [s for s in args.subsets if s in model_results]
        total_queries = sum(model_results[s]["n_queries"] for s in evaluated_subsets)

        def _weighted_avg(metric_path):
            return sum(
                model_results[s][metric_path[0]][metric_path[1]] * model_results[s]["n_queries"]
                for s in evaluated_subsets
            ) / total_queries

        weighted_fine = _weighted_avg(("fine_recall", "recall@1"))
        weighted_coarse = _weighted_avg(("coarse_recall", "recall@1"))
        weighted_id = _weighted_avg(("id_recall", "recall@1"))
        weighted_ndcg5 = sum(
            model_results[s]["ndcg@5"] * model_results[s]["n_queries"]
            for s in evaluated_subsets
        ) / total_queries

        model_results["overall"] = {
            "fine_recall@1": round(float(weighted_fine), 2),
            "coarse_recall@1": round(float(weighted_coarse), 2),
            "id_recall@1": round(float(weighted_id), 2),
            "ndcg@5": round(float(weighted_ndcg5), 2),
            "total_queries": total_queries,
        }

        log.info(
            "\n%s  Fine_R@1=%.2f  Coarse_R@1=%.2f  ID_R@1=%.2f  nDCG@5=%.2f",
            cfg["display"], weighted_fine, weighted_coarse, weighted_id, weighted_ndcg5,
        )

        all_results[model_key] = {
            "model": cfg["display"],
            "hf_hub": cfg["hf_hub"],
            "emb_dim": cfg["emb_dim"],
            "device": device,
            "subsets": model_results,
        }

        del model
        if device == "mps":
            torch.mps.empty_cache()
        elif device == "cuda":
            torch.cuda.empty_cache()

    out_path = RESULTS_DIR / (args.output or "baseline_eval.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    log.info("\nResults saved to %s", out_path)

    METRIC_TABLE = [
        ("FINE RECALL@1 (cat+main_attr)", "fine_recall", "recall@1", "fine_recall@1"),
        ("COARSE RECALL@1 (category only)", "coarse_recall", "recall@1", "coarse_recall@1"),
        ("nDCG@5 (binary cat+main_attr)", "ndcg@5", None, "ndcg@5"),
        ("ID RECALL@1 (item_ID match)", "id_recall", "recall@1", "id_recall@1"),
    ]

    print("\n" + "=" * 100)
    print("  LOOKBENCH BASELINE EVALUATION")
    print("=" * 100)

    for metric_name, mk, msk, ov_key in METRIC_TABLE:
        print(f"\n{'─' * 100}")
        print(f"  {metric_name}")
        print(f"{'─' * 100}")
        print(f"  {'Model':<25} {'RealStudio':>11} {'AIGen-Stu':>11} {'RealStreet':>11} {'AIGen-Str':>11} {'Overall':>11}")
        for _, data in all_results.items():
            vals = []
            for s in SUBSETS:
                if s in data["subsets"]:
                    v = data["subsets"][s][mk] if not msk else data["subsets"][s][mk][msk]
                    vals.append(f"{v:>10.2f}%")
                else:
                    vals.append(f"{'—':>11}")
            ov = data["subsets"]["overall"].get(ov_key)
            vals.append(f"{ov:>10.2f}%" if ov is not None else f"{'—':>11}")
            print(f"  {data['model']:<25} {'  '.join(vals)}")

    print("\n" + "=" * 100)
    print("  PAPER REFERENCE (arXiv:2601.14706)")
    print("─" * 100)
    print("  Table 3  Fine R@1 (cat+main_attr):")
    print("    FashionSigLIP:  66.17  74.09  55.15  74.38  Overall=62.77")
    print("    FashionCLIP:    63.80  71.50  52.40  68.75  Overall=60.30")
    print("  Table 7  nDCG@5 (binary cat+main_attr):")
    print("    FashionSigLIP:  51.86  58.53  42.43  66.27  Overall=49.44")
    print("    FashionCLIP:    51.68  54.93  41.87  63.22  Overall=48.63")
    print("  Table 8  Coarse R@1 (category only):")
    print("    FashionSigLIP:  88.63  93.78  73.39  90.00  Overall=82.77")
    print("    FashionCLIP:    88.72  87.05  75.33  84.38  Overall=82.68")
    print("=" * 100)


if __name__ == "__main__":
    main()
