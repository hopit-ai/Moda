"""Phase 10 — FSL Error Analysis on fashion200k.

Loads Marqo-FashionSigLIP, runs per-query retrieval on fashion200k,
and identifies the queries where FSL performs worst. Outputs:
  1. Per-query AP@10 scores
  2. Bottom-20% failure queries with their categories
  3. Category-level aggregates showing which categories FSL struggles with
  4. A "gap targets" file for synthetic data generation

Usage:
  python3 -u scripts/v3/phase10_fsl_error_analysis.py
  python3 -u scripts/v3/phase10_fsl_error_analysis.py --corpus-size 0  # full corpus
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("fsl-error-analysis")

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = REPO_ROOT / "results" / "fsl_error_analysis"


def load_fashion200k(corpus_size: int = 0):
    """Load fashion200k from HuggingFace with ground truth."""
    from datasets import load_dataset

    log.info("Loading Fashion200k dataset from HuggingFace...")
    ds = load_dataset("Marqo/fashion200k", split="data")
    log.info("Loaded %d items", len(ds))

    if corpus_size > 0 and corpus_size < len(ds):
        indices = list(range(len(ds)))
        rng = np.random.RandomState(42)
        rng.shuffle(indices)
        indices = sorted(indices[:corpus_size])
        ds = ds.select(indices)
        log.info("Subsampled to %d items", len(ds))

    return ds


def build_ground_truth(ds) -> dict:
    """Build ground truth: for each unique category3 query, find all matching doc indices."""
    gt = defaultdict(dict)

    for idx in range(len(ds)):
        item = ds[idx]
        cat3 = item.get("category3", "")
        if cat3:
            gt[cat3][str(idx)] = 1

    log.info("Ground truth: %d unique category3 queries", len(gt))
    return dict(gt)


def compute_per_query_ap(
    ranked_doc_ids: list[str],
    relevant_docs: set[str],
    k: int = 10,
) -> float:
    """Compute AP@k for a single query."""
    hits = 0
    precision_sum = 0.0
    for rank, doc_id in enumerate(ranked_doc_ids[:k], 1):
        if doc_id in relevant_docs:
            hits += 1
            precision_sum += hits / rank

    n_relevant = min(k, len(relevant_docs))
    if n_relevant == 0:
        return 0.0
    return precision_sum / n_relevant


def run_fsl_retrieval(ds, model, tokenizer, preprocess, device, k=10, task="fine-category-to-product"):
    """Run FSL retrieval for the fine-category-to-product task (category3 queries).

    This is the task most relevant to our training since we're teaching
    the model subcategory3-level discrimination.
    """
    log.info("Encoding corpus images...")
    from torch.utils.data import DataLoader, Dataset
    from PIL import Image
    import io

    class ImageDataset(Dataset):
        def __init__(self, hf_ds, preprocess):
            self.ds = hf_ds
            self.preprocess = preprocess

        def __len__(self):
            return len(self.ds)

        def __getitem__(self, idx):
            item = self.ds[idx]
            img = item["image"]
            if not isinstance(img, Image.Image):
                img = Image.open(io.BytesIO(img)).convert("RGB")
            else:
                img = img.convert("RGB")
            return self.preprocess(img)

    img_dataset = ImageDataset(ds, preprocess)
    loader = DataLoader(img_dataset, batch_size=128, num_workers=0, shuffle=False)

    all_img_embs = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            emb = model.encode_image(batch)
            emb = F.normalize(emb, dim=-1)
            all_img_embs.append(emb.cpu())
            if device.type == "mps":
                torch.mps.empty_cache()

    img_embs = torch.cat(all_img_embs, dim=0)  # (N, D)
    log.info("Encoded %d images -> shape %s", img_embs.shape[0], img_embs.shape)

    # Also encode text embeddings for doc_col fusion (image 0.9 + text 0.1)
    log.info("Encoding corpus text fields...")
    all_text_embs = []
    batch_texts = []
    for idx in range(len(ds)):
        text = ds[idx].get("text", "")
        batch_texts.append(text)
        if len(batch_texts) == 128 or idx == len(ds) - 1:
            with torch.no_grad():
                tok = tokenizer(batch_texts).to(device)
                emb = model.encode_text(tok)
                emb = F.normalize(emb, dim=-1)
                all_text_embs.append(emb.cpu())
            batch_texts = []
            if device.type == "mps":
                torch.mps.empty_cache()

    text_embs = torch.cat(all_text_embs, dim=0)  # (N, D)
    log.info("Encoded %d texts -> shape %s", text_embs.shape[0], text_embs.shape)

    # Fused doc embeddings: 0.9*image + 0.1*text (matching the benchmark config)
    doc_embs = F.normalize(0.9 * img_embs + 0.1 * text_embs, dim=-1)

    # Build ground truth for category3
    gt = build_ground_truth(ds)

    # Run retrieval for each category3 query
    log.info("Running retrieval for %d category3 queries...", len(gt))
    per_query_results = {}

    queries = list(gt.keys())
    batch_size = 64

    for i in range(0, len(queries), batch_size):
        batch_queries = queries[i:i+batch_size]
        with torch.no_grad():
            tok = tokenizer(batch_queries).to(device)
            q_embs = model.encode_text(tok)
            q_embs = F.normalize(q_embs, dim=-1).cpu()

        # Compute similarities
        sims = q_embs @ doc_embs.T  # (batch, N)

        for j, query in enumerate(batch_queries):
            scores, indices = torch.topk(sims[j], min(k * 5, sims.shape[1]))
            ranked_ids = [str(idx.item()) for idx in indices[:k*5]]

            relevant = set(gt[query].keys())
            ap = compute_per_query_ap(ranked_ids, relevant, k=k)

            per_query_results[query] = {
                "ap10": ap,
                "n_relevant": len(relevant),
                "top10_ids": ranked_ids[:10],
                "top10_scores": scores[:10].tolist(),
            }

        if (i // batch_size + 1) % 10 == 0:
            log.info("  Processed %d/%d queries", min(i + batch_size, len(queries)), len(queries))

    # Also run text-to-image task (the primary benchmark task)
    log.info("Running text-to-image retrieval...")
    text_queries_gt = defaultdict(dict)
    for idx in range(len(ds)):
        text_q = ds[idx].get("text", "")
        if text_q:
            text_queries_gt[text_q][str(idx)] = 1

    # For text-to-image, doc is just image embeddings
    text_queries = list(text_queries_gt.keys())
    text_per_query = {}

    for i in range(0, len(text_queries), batch_size):
        batch_queries = text_queries[i:i+batch_size]
        with torch.no_grad():
            tok = tokenizer(batch_queries).to(device)
            q_embs = model.encode_text(tok)
            q_embs = F.normalize(q_embs, dim=-1).cpu()

        sims = q_embs @ img_embs.T

        for j, query in enumerate(batch_queries):
            scores, indices = torch.topk(sims[j], min(k * 5, sims.shape[1]))
            ranked_ids = [str(idx.item()) for idx in indices[:k*5]]

            relevant = set(text_queries_gt[query].keys())
            ap = compute_per_query_ap(ranked_ids, relevant, k=k)

            text_per_query[query] = {
                "ap10": ap,
                "n_relevant": len(relevant),
            }

        if (i // batch_size + 1) % 20 == 0:
            log.info("  Text-to-image: %d/%d queries", min(i + batch_size, len(text_queries)), len(text_queries))

    return per_query_results, text_per_query, gt


def analyze_failures(per_query_results: dict, gt: dict, ds) -> dict:
    """Analyze where FSL fails — group by category1/2 and identify patterns."""

    # Sort by AP@10
    sorted_queries = sorted(per_query_results.items(), key=lambda x: x[1]["ap10"])
    n = len(sorted_queries)

    # Bottom 20% (worst performers)
    bottom_20_pct = sorted_queries[:n // 5]
    top_20_pct = sorted_queries[-(n // 5):]

    # Extract category hierarchy from queries
    # fashion200k category3 format is like "dresses/cocktail-and-party/midi"
    category_stats = defaultdict(lambda: {"aps": [], "count": 0})

    for query, result in per_query_results.items():
        parts = query.split("/")
        cat1 = parts[0] if len(parts) >= 1 else "unknown"
        cat2 = "/".join(parts[:2]) if len(parts) >= 2 else cat1

        category_stats[cat1]["aps"].append(result["ap10"])
        category_stats[cat1]["count"] += 1

    # Compute per-category mean AP
    cat_summary = {}
    for cat, stats in category_stats.items():
        cat_summary[cat] = {
            "mean_ap10": float(np.mean(stats["aps"])),
            "median_ap10": float(np.median(stats["aps"])),
            "count": stats["count"],
            "zero_ap_pct": float(np.mean([1 if ap == 0 else 0 for ap in stats["aps"]])),
        }

    cat_summary_sorted = dict(sorted(cat_summary.items(), key=lambda x: x[1]["mean_ap10"]))

    return {
        "overall_map10": float(np.mean([r["ap10"] for _, r in per_query_results.items()])),
        "n_queries": n,
        "n_zero_ap": sum(1 for _, r in per_query_results.items() if r["ap10"] == 0),
        "bottom_20pct": [
            {"query": q, "ap10": r["ap10"], "n_relevant": r["n_relevant"]}
            for q, r in bottom_20_pct
        ],
        "top_20pct_mean": float(np.mean([r["ap10"] for _, r in top_20_pct])),
        "bottom_20pct_mean": float(np.mean([r["ap10"] for _, r in bottom_20_pct])),
        "category_breakdown": cat_summary_sorted,
    }


def generate_gap_targets(analysis: dict, per_query_results: dict) -> list[dict]:
    """Generate a list of 'gap target' queries for synthetic data generation.

    These are the queries where FSL fails hardest — we'll generate synthetic
    training pairs specifically targeting these.
    """
    targets = []

    # All queries with AP@10 < 0.3 (below-average performance)
    for query, result in per_query_results.items():
        if result["ap10"] < 0.3:
            parts = query.split("/")
            targets.append({
                "query": query,
                "ap10": result["ap10"],
                "n_relevant": result["n_relevant"],
                "category1": parts[0] if parts else "",
                "category2": "/".join(parts[:2]) if len(parts) >= 2 else "",
                "category3_full": query,
            })

    targets.sort(key=lambda x: x["ap10"])
    log.info("Generated %d gap targets (queries with AP@10 < 0.3)", len(targets))
    return targets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-size", type=int, default=10000,
                        help="Corpus size for evaluation (0=full, default=10000)")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    t0 = time.time()

    if args.device:
        device = torch.device(args.device)
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    log.info("Device: %s", device)

    # Load FSL
    import open_clip
    log.info("Loading Marqo-FashionSigLIP...")
    model, _, preprocess_val = open_clip.create_model_and_transforms(
        "hf-hub:Marqo/marqo-fashionSigLIP"
    )
    tokenizer = open_clip.get_tokenizer("hf-hub:Marqo/marqo-fashionSigLIP")
    model = model.to(device)
    model.eval()
    log.info("FSL loaded on %s", device)

    # Load dataset
    ds = load_fashion200k(corpus_size=args.corpus_size)

    # Run retrieval
    per_query_cat3, per_query_text, gt = run_fsl_retrieval(
        ds, model, tokenizer, preprocess_val, device, k=10
    )

    # Analyze failures
    log.info("Analyzing failures...")
    analysis = analyze_failures(per_query_cat3, gt, ds)

    # Text-to-image stats
    text_map10 = float(np.mean([r["ap10"] for r in per_query_text.values()])) if per_query_text else 0.0
    analysis["text_to_image_map10"] = text_map10
    analysis["text_to_image_n_queries"] = len(per_query_text)

    # Generate gap targets
    gap_targets = generate_gap_targets(analysis, per_query_cat3)

    # Save results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_DIR / "per_query_cat3.json", "w") as f:
        json.dump(per_query_cat3, f, indent=2)

    with open(OUTPUT_DIR / "analysis.json", "w") as f:
        json.dump(analysis, f, indent=2)

    with open(OUTPUT_DIR / "gap_targets.json", "w") as f:
        json.dump(gap_targets, f, indent=2)

    # Print summary
    elapsed = time.time() - t0
    log.info("=" * 60)
    log.info("FSL Error Analysis Complete (%.1f min)", elapsed / 60)
    log.info("=" * 60)
    log.info("Task: fine-category-to-product (category3 queries)")
    log.info("  Overall MAP@10: %.4f", analysis["overall_map10"])
    log.info("  Total queries: %d", analysis["n_queries"])
    log.info("  Zero-AP queries: %d (%.1f%%)", analysis["n_zero_ap"],
             100 * analysis["n_zero_ap"] / max(analysis["n_queries"], 1))
    log.info("  Bottom 20%% mean: %.4f", analysis["bottom_20pct_mean"])
    log.info("  Top 20%% mean: %.4f", analysis["top_20pct_mean"])
    log.info("")
    log.info("Task: text-to-image")
    log.info("  MAP@10: %.4f (%d queries)", text_map10, len(per_query_text))
    log.info("")
    log.info("Category breakdown (worst first):")
    for cat, stats in list(analysis["category_breakdown"].items())[:8]:
        log.info("  %-20s MAP@10=%.4f  count=%d  zero_ap=%.0f%%",
                 cat, stats["mean_ap10"], stats["count"], stats["zero_ap_pct"] * 100)
    log.info("")
    log.info("Gap targets: %d queries with AP@10 < 0.3", len(gap_targets))
    log.info("Output: %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()
