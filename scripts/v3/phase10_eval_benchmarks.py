"""Phase 10 — Evaluate model on all 4 clean benchmarks.

Runs MAP@10 evaluation on fashion200k, atlas, polyvore, KAGL using a
3K stratified subsample (fast screener) or full corpus.

Supports loading either:
  - P4B model (base SigLIP + Phase 4b + Phase 10 checkpoint)
  - FSL model (FashionSigLIP + Phase 10 checkpoint)

Usage:
  python3 -u scripts/v3/phase10_eval_benchmarks.py --model-source phase4b --checkpoint path/to/best.pt
  python3 -u scripts/v3/phase10_eval_benchmarks.py --model-source fsl --checkpoint path/to/best.pt
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("phase10-eval")

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "results"

BENCHMARKS = ["fashion200k", "atlas", "polyvore", "KAGL"]
HF_DATASETS = {
    "fashion200k": "Marqo/fashion200k",
    "atlas": "Marqo/atlas",
    "polyvore": "Marqo/polyvore",
    "KAGL": "Marqo/KAGL",
    "iMaterialist": "Marqo/iMaterialist",
}

DATASET_SPLITS = {
    "fashion200k": "data",
    "atlas": "data",
    "polyvore": "data",
    "KAGL": "data",
    "iMaterialist": "data",
}

# FSL baseline (3K screener, from Phase 0 audit)
FSL_BASELINE = {
    "fashion200k": 0.3859,
    "atlas": 0.6919,
    "polyvore": 0.5783,
    "KAGL": 0.6779,
}

# Returned when a task cannot be evaluated (missing columns, etc.)
EMPTY_METRICS = {
    "map10": 0.0,
    "recall_1": 0.0,
    "recall_10": 0.0,
    "avg_recall": 0.0,
    "precision_1": 0.0,
    "precision_10": 0.0,
    "mrr": 0.0,
}


MODEL_REGISTRY = {
    "fsl": ("hf-hub:Marqo/marqo-fashionSigLIP", None),
    "phase4b": ("ViT-B-16-SigLIP", "webli"),
    "b16-256": ("ViT-B-16-SigLIP2-256", "webli"),
    "b16-384": ("ViT-B-16-SigLIP2-384", "webli"),
    "l16": ("ViT-L-16-SigLIP2-384", "webli"),
    "so400m": ("ViT-SO400M-14-SigLIP2-378", "webli"),
    "hybrid": ("hf-hub:Marqo/marqo-fashionSigLIP", None),
    "moda-distilled": ("hf-hub:Marqo/marqo-fashionSigLIP", None),
}

MODA_DISTILLED_CKPT = REPO_ROOT / "models/moda-siglip-distilled/best/model_state_dict.pt"


def load_model(model_source: str, checkpoint_path: str | None, device: torch.device):
    """Load model based on source type."""
    import open_clip

    if model_source not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model source: {model_source}. Choose from: {list(MODEL_REGISTRY.keys())}")

    model_name, pretrained = MODEL_REGISTRY[model_source]
    log.info("Loading %s (pretrained=%s)...", model_name, pretrained)

    if pretrained:
        model, _, preprocess_val = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
    else:
        model, _, preprocess_val = open_clip.create_model_and_transforms(model_name)
    tokenizer = open_clip.get_tokenizer(model_name)

    model = model.to(device)

    if model_source == "hybrid":
        log.info("HYBRID mode: FSL text tower + MODA-Distilled vision tower")
        ckpt = torch.load(MODA_DISTILLED_CKPT, map_location=device, weights_only=True)
        visual_sd = {k: v for k, v in ckpt.items() if "visual" in k}
        missing, unexpected = model.load_state_dict(visual_sd, strict=False)
        log.info("  Swapped %d visual keys (kept FSL text tower intact)", len(visual_sd))
    elif model_source == "moda-distilled":
        log.info("MODA-Distilled: loading full checkpoint (vision + text)")
        ckpt = torch.load(MODA_DISTILLED_CKPT, map_location=device, weights_only=True)
        missing, unexpected = model.load_state_dict(ckpt, strict=False)
        log.info("  Loaded %d keys", len(ckpt))
    elif checkpoint_path and Path(checkpoint_path).exists():
        log.info("Loading checkpoint: %s", checkpoint_path)
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state_dict = ckpt.get("model_state_dict", ckpt)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        log.info("  Loaded %d keys (%d missing, %d unexpected)",
                 len(state_dict) - len(unexpected), len(missing), len(unexpected))
    elif checkpoint_path:
        log.warning("Checkpoint not found: %s — using base weights", checkpoint_path)

    model.eval()
    return model, tokenizer, preprocess_val


def load_benchmark(dataset_name: str, corpus_size: int = 3000):
    """Load a benchmark dataset from HuggingFace."""
    from datasets import load_dataset

    split = DATASET_SPLITS.get(dataset_name, "data")
    hf_name = HF_DATASETS[dataset_name]

    log.info("Loading %s (split=%s)...", hf_name, split)

    # iMaterialist is 721K images (~70GB); full download often fails at ~72%.
    # Stream with stratified sampling by category to get a representative subset.
    if dataset_name == "iMaterialist":
        from itertools import islice
        from collections import defaultdict

        target_n = corpus_size
        stream_ds = load_dataset(hf_name, split=split, streaming=True)
        log.info("  Streaming iMaterialist with stratified sampling (target=%d)...", target_n)

        # First pass: collect items grouped by category (stream up to 100K)
        cat_buckets = defaultdict(list)
        for i, item in enumerate(stream_ds):
            cat = item.get("category", "unknown") or "unknown"
            cat_buckets[cat].append(item)
            if i >= 100000:
                break

        # Stratified sample: equal items per category, fill remainder randomly
        n_cats = len(cat_buckets)
        per_cat = max(1, target_n // n_cats)
        rng = np.random.RandomState(42)
        sampled = []
        for cat, items in cat_buckets.items():
            if len(items) <= per_cat:
                sampled.extend(items)
            else:
                idx = rng.choice(len(items), size=per_cat, replace=False)
                sampled.extend([items[i] for i in idx])

        # If we overshot, trim; if under, that's fine
        if len(sampled) > target_n:
            idx = rng.choice(len(sampled), size=target_n, replace=False)
            sampled = [sampled[i] for i in idx]

        from datasets import Dataset as HFDataset
        ds = HFDataset.from_list(sampled)
        log.info("  Stratified sample: %d items from %d categories", len(ds), n_cats)
    else:
        ds = load_dataset(hf_name, split=split)

    log.info("  Full size: %d", len(ds))

    if corpus_size > 0 and corpus_size < len(ds):
        rng = np.random.RandomState(42)
        indices = rng.choice(len(ds), size=corpus_size, replace=False)
        indices = sorted(indices.tolist())
        ds = ds.select(indices)
        log.info("  Subsampled to %d", len(ds))

    # Marqo/iMaterialist has no free-text query field; build a caption from attributes
    # (same general idea as attribute-based queries in fashion benchmarks).
    if dataset_name == "iMaterialist":
        attr_keys = (
            "gender",
            "color",
            "material",
            "pattern",
            "neckline",
            "style",
            "sleeve",
            "category",
        )

        def add_caption(ex):
            parts = []
            for k in attr_keys:
                if k not in ex or ex[k] is None:
                    continue
                s = str(ex[k]).strip()
                if s:
                    parts.append(s)
            return {"caption": ", ".join(parts)}

        ds = ds.map(add_caption)
        log.info("  iMaterialist: synthetic caption column from %s", attr_keys)

    return ds


def encode_images(model, preprocess, ds, device, batch_size=128):
    """Encode all images in the dataset."""
    import io
    from torch.utils.data import DataLoader, Dataset

    class ImgDataset(Dataset):
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

    loader = DataLoader(ImgDataset(ds, preprocess), batch_size=batch_size,
                        num_workers=0, shuffle=False)

    all_embs = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            emb = model.encode_image(batch)
            emb = F.normalize(emb, dim=-1)
            all_embs.append(emb.cpu())
            if device.type == "mps":
                torch.mps.empty_cache()

    return torch.cat(all_embs, dim=0)


def encode_texts(model, tokenizer, texts, device, batch_size=128):
    """Encode a list of text strings."""
    all_embs = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            tok = tokenizer(batch).to(device)
            emb = model.encode_text(tok)
            emb = F.normalize(emb, dim=-1)
            all_embs.append(emb.cpu())
            if device.type == "mps":
                torch.mps.empty_cache()
    return torch.cat(all_embs, dim=0)


def compute_all_metrics(query_embs, doc_embs, gt, k=10):
    """Compute MAP@k, Recall@1, Recall@10, Precision@1, Precision@10, MRR."""
    aps, recalls_1, recalls_10, prec_1, prec_10, mrrs = [], [], [], [], [], []

    for i in range(query_embs.shape[0]):
        sims = (query_embs[i:i+1] @ doc_embs.T).squeeze(0)
        _, top_indices = torch.topk(sims, min(k, sims.shape[0]))
        top_indices = top_indices.tolist()

        relevant = gt[i]
        n_rel = len(relevant)
        if n_rel == 0:
            continue

        hits = 0
        precision_sum = 0.0
        first_hit_rank = None
        for rank, idx in enumerate(top_indices[:k], 1):
            if idx in relevant:
                hits += 1
                precision_sum += hits / rank
                if first_hit_rank is None:
                    first_hit_rank = rank

        ap = precision_sum / min(k, n_rel) if n_rel > 0 else 0.0
        aps.append(ap)

        # Recall@1: did we find at least 1 relevant in top-1?
        hit_at_1 = 1.0 if top_indices[0] in relevant else 0.0
        recalls_1.append(hit_at_1)

        # Recall@10: fraction of relevant docs found in top-10
        hits_at_10 = sum(1 for idx in top_indices[:10] if idx in relevant)
        recalls_10.append(hits_at_10 / n_rel)

        # Precision@1, Precision@10
        prec_1.append(hit_at_1)
        prec_10.append(hits_at_10 / min(10, len(top_indices)))

        # MRR
        mrrs.append(1.0 / first_hit_rank if first_hit_rank else 0.0)

    return {
        "map10": float(np.mean(aps)) if aps else 0.0,
        "recall_1": float(np.mean(recalls_1)) if recalls_1 else 0.0,
        "recall_10": float(np.mean(recalls_10)) if recalls_10 else 0.0,
        "avg_recall": float(np.mean([(r1 + r10) / 2 for r1, r10 in zip(recalls_1, recalls_10)])) if recalls_1 else 0.0,
        "precision_1": float(np.mean(prec_1)) if prec_1 else 0.0,
        "precision_10": float(np.mean(prec_10)) if prec_10 else 0.0,
        "mrr": float(np.mean(mrrs)) if mrrs else 0.0,
    }


def eval_text_to_image(model, tokenizer, preprocess, ds, device, k=10):
    """Evaluate text-to-image retrieval (primary task)."""
    # Build ground truth: each unique text query maps to its image index
    text_col = "text" if "text" in ds.column_names else "caption"
    if text_col not in ds.column_names:
        for col in ds.column_names:
            if "text" in col.lower() or "caption" in col.lower() or "description" in col.lower():
                text_col = col
                break

    if text_col not in ds.column_names:
        log.warning("  No text column found. Columns: %s", ds.column_names)
        return dict(EMPTY_METRICS)

    # Group by unique query text
    query_to_indices = defaultdict(set)
    for idx in range(len(ds)):
        text = ds[idx].get(text_col, "")
        if text:
            query_to_indices[text].add(idx)

    queries = list(query_to_indices.keys())
    if len(queries) > 2000:
        rng = np.random.RandomState(42)
        selected = rng.choice(len(queries), size=2000, replace=False)
        queries = [queries[i] for i in selected]

    log.info("  text-to-image: %d queries", len(queries))

    # Encode
    img_embs = encode_images(model, preprocess, ds, device)
    query_embs = encode_texts(model, tokenizer, queries, device)

    # Build GT for selected queries
    gt = [query_to_indices[q] for q in queries]

    return compute_all_metrics(query_embs, img_embs, gt, k=k)


def eval_category_to_product(model, tokenizer, preprocess, ds, device, cat_col="category3", k=10):
    """Evaluate category-to-product retrieval."""
    if cat_col not in ds.column_names:
        # Try alternatives
        for alt in ["category3", "category2", "category1", "subcategory", "category"]:
            if alt in ds.column_names:
                cat_col = alt
                break
        else:
            log.warning("  No category column found. Columns: %s", ds.column_names)
            return dict(EMPTY_METRICS)

    # Group by category
    cat_to_indices = defaultdict(set)
    for idx in range(len(ds)):
        cat = ds[idx].get(cat_col, "")
        if cat:
            cat_to_indices[cat].add(idx)

    # Filter to categories with >= 2 items
    categories = [c for c, indices in cat_to_indices.items() if len(indices) >= 2]
    if len(categories) > 2000:
        rng = np.random.RandomState(42)
        selected = rng.choice(len(categories), size=2000, replace=False)
        categories = [categories[i] for i in selected]

    log.info("  category-to-product (%s): %d queries", cat_col, len(categories))

    if not categories:
        return dict(EMPTY_METRICS)

    # Encode images (with text fusion: 0.9*image + 0.1*text)
    img_embs = encode_images(model, preprocess, ds, device)

    # Encode product text for fusion
    text_col = None
    for col in ["text", "caption", "title", "description"]:
        if col in ds.column_names:
            text_col = col
            break

    if text_col:
        texts = [ds[idx].get(text_col, "") for idx in range(len(ds))]
        text_embs = encode_texts(model, tokenizer, texts, device)
        doc_embs = F.normalize(0.9 * img_embs + 0.1 * text_embs, dim=-1)
    else:
        doc_embs = img_embs

    # Encode category queries
    query_embs = encode_texts(model, tokenizer, categories, device)

    # Build GT
    gt = [cat_to_indices[c] for c in categories]

    return compute_all_metrics(query_embs, doc_embs, gt, k=k)


def evaluate_benchmark(model, tokenizer, preprocess, dataset_name, device, corpus_size=3000):
    """Run full evaluation on a single benchmark."""
    log.info("Evaluating: %s (corpus=%d)", dataset_name, corpus_size)

    ds = load_benchmark(dataset_name, corpus_size)

    results = {}

    # Text-to-image (primary metric)
    t2i = eval_text_to_image(model, tokenizer, preprocess, ds, device, k=10)
    results["text_to_image"] = t2i
    results["text_to_image_map10"] = t2i["map10"]
    log.info("  text-to-image MAP@10=%.4f R@1=%.3f R@10=%.3f MRR=%.3f",
             t2i["map10"], t2i["recall_1"], t2i["recall_10"], t2i["mrr"])

    # Category-to-product (try different category columns)
    for cat_col in ["category3", "category2", "category1", "subcategory", "category"]:
        if cat_col in ds.column_names:
            cat_metrics = eval_category_to_product(
                model, tokenizer, preprocess, ds, device, cat_col=cat_col, k=10
            )
            results[f"{cat_col}_to_product"] = cat_metrics
            results[f"{cat_col}_to_product_map10"] = cat_metrics["map10"]
            log.info("  %s MAP@10=%.4f P@1=%.3f P@10=%.3f MRR=%.3f",
                     cat_col, cat_metrics["map10"], cat_metrics["precision_1"],
                     cat_metrics["precision_10"], cat_metrics["mrr"])

    results["primary_map10"] = t2i["map10"]

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-source", type=str, required=True,
                        choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--corpus-size", type=int, default=3000)
    parser.add_argument("--output-tag", type=str, default="eval")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--benchmarks", nargs="+", default=BENCHMARKS)
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

    # Load model
    model, tokenizer, preprocess = load_model(args.model_source, args.checkpoint, device)

    # Evaluate on each benchmark
    all_results = {}
    for bm in args.benchmarks:
        if bm not in HF_DATASETS:
            log.warning("Unknown benchmark: %s — skipping", bm)
            continue
        try:
            results = evaluate_benchmark(model, tokenizer, preprocess, bm, device, args.corpus_size)
            all_results[bm] = results
        except Exception as e:
            log.error("Failed on %s: %s", bm, e)
            all_results[bm] = {"error": str(e)}

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / f"phase10_{args.output_tag}.json"
    with open(output_path, "w") as f:
        json.dump({
            "model_source": args.model_source,
            "checkpoint": args.checkpoint,
            "corpus_size": args.corpus_size,
            "results": all_results,
            "fsl_baseline": FSL_BASELINE,
        }, f, indent=2)

    # Print comparison
    elapsed = time.time() - t0
    log.info("")
    log.info("=" * 60)
    log.info("Phase 10 Evaluation — %s (%s)", args.output_tag, args.model_source)
    log.info("=" * 60)
    log.info("")
    log.info("| Benchmark | Ours (MAP@10) | FSL Baseline | Delta | Beats +10%? |")
    log.info("|---|---:|---:|---:|---|")

    all_beat = True
    for bm in args.benchmarks:
        if bm in all_results and "primary_map10" in all_results[bm]:
            ours = all_results[bm]["primary_map10"]
            fsl = FSL_BASELINE.get(bm, 0)
            delta = (ours - fsl) / fsl * 100 if fsl > 0 else 0
            target = fsl * 1.10
            beats = "YES" if ours >= target else "no"
            if ours < target:
                all_beat = False
            log.info("| %s | %.4f | %.4f | %+.1f%% | %s |",
                     bm, ours, fsl, delta, beats)
        else:
            log.info("| %s | ERROR | %.4f | — | — |", bm, FSL_BASELINE.get(bm, 0))
            all_beat = False

    log.info("")
    if all_beat:
        log.info("*** GOAL MET: Beats FSL by >=10%% on ALL benchmarks! ***")
    else:
        log.info("Goal NOT met yet. See results for details.")
    log.info("")
    log.info("Results saved: %s", output_path)
    log.info("Time: %.1f min", elapsed / 60)


if __name__ == "__main__":
    main()
