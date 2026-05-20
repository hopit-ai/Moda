"""Phase 7 — Category-Level Embedding Alignment Analysis.

Measures how well each model (FSL vs Base SigLIP) aligns category-label text embeddings
to actual images in those categories. This directly measures the "category retrieval" 
capability that benchmarks test.

For each L1 category label:
  1. Encode the category name as text with both models
  2. Encode all images belonging to that category  
  3. Compute mean cosine similarity (category_text → category_images)
  4. Compare FSL vs Base alignment per category

Also measures overall embedding drift between FSL and Base on the actual benchmark queries.

Usage:
  python3 scripts/v3/phase7_embedding_alignment.py
  python3 scripts/v3/phase7_embedding_alignment.py --corpus-size 5000
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import random
import re
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
log = logging.getLogger("phase7-align")

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "results" / "v3_phases" / "phase7_error_analysis"

L1_PATTERNS = [
    (re.compile(r"\b(jackets?|blazers?|coats?|parkas?|anorak|windbreaker|trench|overcoat|capes?|ponchos?|bombers?|vests?|waistcoats?|outerwear|sherwani)\b", re.I), "outerwear"),
    (re.compile(r"(dress|gowns?|rompers?|jumpsuits?|playsuits?|sarees?|kurtas?|kurtis?|kurta.?sets?|galabiyyas?|dhoti)", re.I), "dresses"),
    (re.compile(r"\b(tops?|blouses?|shirts?|tees?|t-?shirts?|tshirts?|tanks?|camisoles?|tunics?|polos?|henleys?|crop.?tops?|halters?|bustiers?|corsets?|hoodies?|sweatshirts?|sweaters?|pullovers?|cardigans?|knits?|chemises?)\b", re.I), "tops"),
    (re.compile(r"\b(pants?|trousers?|jeans?|denim|leggings?|chinos?|shorts?|skirts?|skorts?|culottes?|joggers?|sweatpants?|cargos?|capris?|churidars?|tracksuits?|tights?|hosiery)\b", re.I), "bottoms"),
    (re.compile(r"\b(shoes?|boots?|sneakers?|sandals?|heels?|pumps?|flats?|loafers?|slippers?|mules?|clogs?|espadrilles?|oxfords?|derby|brogues?|stilettos?|wedges?|platforms?|flip.?flops?|booties?|moccasins?)\b", re.I), "shoes"),
    (re.compile(r"\b(bags?|handbags?|purses?|clutches?|totes?|backpacks?|satchels?|crossbody|messenger|wallets?|pouches?|duffles?|weekenders?|rucksacks?|luggage|shoulder.?bags?)\b", re.I), "bags"),
    (re.compile(r"\b(accessor|jewelry|jewellery|necklaces?|bracelets?|bangles?|earrings?|rings?|watches?|sunglasses|eyeglasses|eyewear|scarves?|scarfs?|belts?|hats?|caps?|beanies?|gloves?|ties?|bow.?ties?|cufflinks?|brooches?|pins?|charms?|pendants?|stoles?|dupattas?|umbrellas?|socks?|earmuffs?)\b", re.I), "accessories"),
    (re.compile(r"\b(swimsuits?|bikinis?|swimwear|bathing|swim|one.?piece.?swim)\b", re.I), "swimwear"),
    (re.compile(r"\b(lingerie|bras?|underwear|panty|panties|briefs?|boxers?|nightgowns?|pajamas?|pyjamas?|robes?|sleepwear|lounge|intimates?|shapewear)\b", re.I), "intimates"),
    (re.compile(r"\b(activewear|sportswear|athletic|yoga|gym|workout|running|cycling|fitness|track)\b", re.I), "activewear"),
    (re.compile(r"\b(makeup|mascara|lipstick|foundation|concealer|eyeliner|eyeshadow|blush|fragrance|perfume|cologne|skincare|moisturizer|cleanser|serum|toner|sunscreen|lotion|shampoo|conditioner|nail.?polish|beauty|cosmetic|palette|highlighter)\b", re.I), "beauty"),
    (re.compile(r"\b(furniture|chairs?|tables?|lamps?|rugs?|pillows?|curtains?|beds?|sofas?|mirrors?|vases?|candles?|decor|lighting|shelves?|storage|kitchen|dining|drinkware|flatware|frames?|clocks?|ottoman|dresser|nightstand)\b", re.I), "home"),
]

L1_CATEGORIES = [
    "dresses", "tops", "bottoms", "shoes", "bags", "accessories",
    "outerwear", "swimwear", "intimates", "activewear", "beauty", "home", "other"
]


def classify_l1(text: str) -> str:
    if not text:
        return "other"
    for pattern, label in L1_PATTERNS:
        if pattern.search(text):
            return label
    return "other"


BENCHMARKS = {
    "fashion200k": {"hf": "Marqo/fashion200k", "split": "data", "query_col": "category3"},
    "atlas":       {"hf": "Marqo/atlas",       "split": "data", "query_col": "sub-category"},
    "polyvore":    {"hf": "Marqo/polyvore",    "split": "data", "query_col": "category"},
    "KAGL":        {"hf": "Marqo/KAGL",        "split": "data", "query_col": "category3"},
}


def load_fsl_model(device: torch.device):
    import open_clip
    model, _, preprocess = open_clip.create_model_and_transforms(
        "hf-hub:Marqo/marqo-fashionSigLIP"
    )
    tokenizer = open_clip.get_tokenizer("hf-hub:Marqo/marqo-fashionSigLIP")
    model = model.to(device).eval()
    log.info("Loaded FashionSigLIP")
    return model, preprocess, tokenizer


def load_base_model(device: torch.device):
    import open_clip
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-16-SigLIP", pretrained="webli"
    )
    tokenizer = open_clip.get_tokenizer("ViT-B-16-SigLIP")
    model = model.to(device).eval()
    log.info("Loaded base SigLIP")
    return model, preprocess, tokenizer


@torch.no_grad()
def compute_category_alignment(
    model, preprocess, tokenizer, benchmark_name: str, device: torch.device,
    corpus_size: int = 3000, seed: int = 42, batch_size: int = 32,
) -> dict:
    """For each L1 category, compute mean(cos_sim(category_label_embedding, category_image_embeddings))."""
    from datasets import load_dataset

    cfg = BENCHMARKS[benchmark_name]
    log.info("  Loading %s...", benchmark_name)

    ds = load_dataset(cfg["hf"], split=cfg["split"], streaming=True,
                      cache_dir=str(REPO_ROOT / "data" / "hf_cache"))

    categories = []
    images_pil = []
    for row in ds:
        if len(categories) >= corpus_size:
            break
        query = (row.get(cfg["query_col"]) or "").strip()
        if not query or not row.get("image"):
            continue
        categories.append(query)
        images_pil.append(row["image"])

    rng = random.Random(seed)
    indices = list(range(len(categories)))
    rng.shuffle(indices)
    indices = indices[:corpus_size]
    categories = [categories[i] for i in indices]
    images_pil = [images_pil[i] for i in indices]

    # Classify each item into L1
    l1_labels = [classify_l1(cat) for cat in categories]

    # Group image indices by L1 category
    l1_to_indices = defaultdict(list)
    for idx, l1 in enumerate(l1_labels):
        l1_to_indices[l1].append(idx)

    log.info("    %d items, L1 distribution: %s",
             len(categories),
             {k: len(v) for k, v in sorted(l1_to_indices.items())})

    # Encode all images
    img_feats = []
    for i in range(0, len(images_pil), batch_size):
        batch_pil = images_pil[i:i + batch_size]
        tensors = torch.stack([preprocess(im.convert("RGB")) for im in batch_pil]).to(device)
        feat = model.encode_image(tensors)
        feat = F.normalize(feat, dim=-1)
        img_feats.append(feat.cpu())
        del tensors, feat
        for j in range(i, min(i + batch_size, len(images_pil))):
            images_pil[j] = None

    img_feats = torch.cat(img_feats, dim=0).numpy()
    del images_pil
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()

    # Encode L1 category labels as text
    present_l1s = sorted(l1_to_indices.keys())
    l1_texts = present_l1s  # Use the raw category name as query
    tokens = tokenizer(l1_texts).to(device)
    l1_text_feats = model.encode_text(tokens)
    l1_text_feats = F.normalize(l1_text_feats, dim=-1).cpu().numpy()
    del tokens
    gc.collect()

    # Also encode the actual benchmark query texts (not just L1 labels)
    # to compare how well the model aligns specific queries
    cat_to_indices = defaultdict(list)
    for idx, cat in enumerate(categories):
        cat_to_indices[cat].append(idx)
    valid_cats = [cat for cat, idxs in cat_to_indices.items() if len(idxs) >= 2]
    rng2 = random.Random(seed)
    rng2.shuffle(valid_cats)
    query_cats = valid_cats[:min(500, len(valid_cats))]

    query_text_feats = []
    for i in range(0, len(query_cats), 64):
        tokens = tokenizer(query_cats[i:i + 64]).to(device)
        feat = model.encode_text(tokens)
        feat = F.normalize(feat, dim=-1)
        query_text_feats.append(feat.cpu())
        del tokens, feat
    query_text_feats = torch.cat(query_text_feats, dim=0).numpy()

    # Compute per-L1-category alignment scores
    alignment_scores = {}
    for li, l1 in enumerate(present_l1s):
        cat_indices = l1_to_indices[l1]
        if len(cat_indices) < 2:
            continue
        cat_img_feats = img_feats[cat_indices]
        l1_emb = l1_text_feats[li]

        # Cosine similarity between category label and all images in that category
        sims = cat_img_feats @ l1_emb
        alignment_scores[l1] = {
            "mean_sim": float(np.mean(sims)),
            "std_sim": float(np.std(sims)),
            "median_sim": float(np.median(sims)),
            "n_images": len(cat_indices),
        }

    # Compute per-query alignment (how well does specific query text match its own images?)
    per_query_alignment = []
    for qi, cat in enumerate(query_cats):
        cat_indices = cat_to_indices[cat]
        cat_img_feats_sub = img_feats[cat_indices]
        query_emb = query_text_feats[qi]
        sims = cat_img_feats_sub @ query_emb
        per_query_alignment.append({
            "query": cat,
            "l1": classify_l1(cat),
            "mean_sim_to_own_images": float(np.mean(sims)),
            "n_own_images": len(cat_indices),
        })

    del img_feats, l1_text_feats, query_text_feats
    gc.collect()

    return {
        "benchmark": benchmark_name,
        "l1_alignment": alignment_scores,
        "per_query_alignment": per_query_alignment,
        "query_embeddings_for_drift": None,  # placeholder
    }


@torch.no_grad()
def compute_query_drift_between_models(
    model_a, tokenizer_a, model_b, tokenizer_b,
    benchmark_name: str, device: torch.device,
    corpus_size: int = 3000, seed: int = 42,
) -> dict:
    """Compute cosine distance between model A and model B text embeddings for same queries."""
    from datasets import load_dataset

    cfg = BENCHMARKS[benchmark_name]
    ds = load_dataset(cfg["hf"], split=cfg["split"], streaming=True,
                      cache_dir=str(REPO_ROOT / "data" / "hf_cache"))

    categories = []
    for row in ds:
        if len(categories) >= corpus_size:
            break
        query = (row.get(cfg["query_col"]) or "").strip()
        if not query or not row.get("image"):
            continue
        categories.append(query)

    rng = random.Random(seed)
    indices = list(range(len(categories)))
    rng.shuffle(indices)
    indices = indices[:corpus_size]
    categories = [categories[i] for i in indices]

    # Get unique queries
    cat_to_count = defaultdict(int)
    for cat in categories:
        cat_to_count[cat] += 1
    valid_cats = [cat for cat, c in cat_to_count.items() if c >= 2]
    rng2 = random.Random(seed)
    rng2.shuffle(valid_cats)
    query_cats = valid_cats[:min(500, len(valid_cats))]

    # Encode with model A
    feats_a = []
    for i in range(0, len(query_cats), 64):
        tokens = tokenizer_a(query_cats[i:i + 64]).to(device)
        feat = model_a.encode_text(tokens)
        feat = F.normalize(feat, dim=-1)
        feats_a.append(feat.cpu())
        del tokens, feat
    feats_a = torch.cat(feats_a, dim=0).numpy()

    # Encode with model B
    feats_b = []
    for i in range(0, len(query_cats), 64):
        tokens = tokenizer_b(query_cats[i:i + 64]).to(device)
        feat = model_b.encode_text(tokens)
        feat = F.normalize(feat, dim=-1)
        feats_b.append(feat.cpu())
        del tokens, feat
    feats_b = torch.cat(feats_b, dim=0).numpy()

    # Per-query cosine distance
    cos_sims = np.sum(feats_a * feats_b, axis=1)
    cos_dists = 1.0 - cos_sims

    # Per-L1 drift
    l1_drifts = defaultdict(list)
    for qi, cat in enumerate(query_cats):
        l1 = classify_l1(cat)
        l1_drifts[l1].append(cos_dists[qi])

    per_l1 = {}
    for l1, dists in sorted(l1_drifts.items()):
        per_l1[l1] = {
            "mean_drift": float(np.mean(dists)),
            "n_queries": len(dists),
        }

    return {
        "benchmark": benchmark_name,
        "mean_cosine_distance": float(np.mean(cos_dists)),
        "median_cosine_distance": float(np.median(cos_dists)),
        "std_cosine_distance": float(np.std(cos_dists)),
        "max_cosine_distance": float(np.max(cos_dists)),
        "per_l1_drift": per_l1,
    }


def generate_alignment_report(
    fsl_alignment: dict, base_alignment: dict, drift_results: dict, output_dir: Path
):
    """Generate comparison report of category alignment between FSL and base."""
    lines = [
        "# Phase 7 — Category Embedding Alignment Report",
        "",
        f"_Generated: {time.strftime('%Y-%m-%d %H:%M IST')}_",
        "",
        "## 1. Category-Label-to-Image Alignment (higher = better category retrieval)",
        "",
        "For each L1 category, we encode the category name and compute mean cosine similarity",
        "to all images in that category. FSL should have higher alignment on categories it wins.",
        "",
    ]

    for bench in ["fashion200k", "atlas", "polyvore", "KAGL"]:
        fsl_a = fsl_alignment[bench]["l1_alignment"]
        base_a = base_alignment[bench]["l1_alignment"]

        all_l1s = sorted(set(list(fsl_a.keys()) + list(base_a.keys())))
        if not all_l1s:
            continue

        lines.extend([
            f"### {bench}",
            "",
            "| Category | FSL Align | Base Align | Delta | FSL Advantage? | N images |",
            "|---|---:|---:|---:|---|---:|",
        ])

        for l1 in all_l1s:
            fsl_sim = fsl_a.get(l1, {}).get("mean_sim", 0)
            base_sim = base_a.get(l1, {}).get("mean_sim", 0)
            delta = fsl_sim - base_sim
            n_imgs = fsl_a.get(l1, {}).get("n_images", base_a.get(l1, {}).get("n_images", 0))
            advantage = "YES" if delta > 0.005 else ("NO" if delta < -0.005 else "~tie")
            lines.append(
                f"| {l1} | {fsl_sim:.4f} | {base_sim:.4f} | {delta:+.4f} | {advantage} | {n_imgs} |"
            )
        lines.append("")

    # Per-query alignment comparison
    lines.extend([
        "## 2. Per-Query Alignment (specific benchmark queries, not just L1 labels)",
        "",
        "Mean cosine similarity between each query and its own positive images.",
        "",
        "| Benchmark | FSL Mean Align | Base Mean Align | Delta | FSL Better? |",
        "|---|---:|---:|---:|---|",
    ])

    for bench in ["fashion200k", "atlas", "polyvore", "KAGL"]:
        fsl_pqa = fsl_alignment[bench]["per_query_alignment"]
        base_pqa = base_alignment[bench]["per_query_alignment"]
        fsl_mean = np.mean([q["mean_sim_to_own_images"] for q in fsl_pqa])
        base_mean = np.mean([q["mean_sim_to_own_images"] for q in base_pqa])
        delta = fsl_mean - base_mean
        better = "YES" if delta > 0.005 else ("NO" if delta < -0.005 else "~tie")
        lines.append(
            f"| {bench} | {fsl_mean:.4f} | {base_mean:.4f} | {delta:+.4f} | {better} |"
        )

    # Per-L1 per-query alignment for key failing strata
    lines.extend([
        "",
        "### Per-L1 Query Alignment (for key failing strata)",
        "",
    ])

    failing_strata = ["bottoms", "tops", "dresses", "bags", "beauty", "home"]
    for bench in ["polyvore", "KAGL", "fashion200k"]:
        fsl_pqa = fsl_alignment[bench]["per_query_alignment"]
        base_pqa = base_alignment[bench]["per_query_alignment"]

        fsl_by_l1 = defaultdict(list)
        base_by_l1 = defaultdict(list)
        for q in fsl_pqa:
            fsl_by_l1[q["l1"]].append(q["mean_sim_to_own_images"])
        for q in base_pqa:
            base_by_l1[q["l1"]].append(q["mean_sim_to_own_images"])

        lines.extend([
            f"**{bench}** (key strata):",
            "",
            "| Stratum | FSL Align | Base Align | Delta | N queries |",
            "|---|---:|---:|---:|---:|",
        ])
        for l1 in failing_strata:
            if l1 in fsl_by_l1 and l1 in base_by_l1:
                fsl_m = np.mean(fsl_by_l1[l1])
                base_m = np.mean(base_by_l1[l1])
                n = len(fsl_by_l1[l1])
                lines.append(f"| {l1} | {fsl_m:.4f} | {base_m:.4f} | {fsl_m - base_m:+.4f} | {n} |")
        lines.append("")

    # Drift analysis
    lines.extend([
        "## 3. Text Embedding Drift (FSL vs Base)",
        "",
        "How far FSL moved text embeddings from the base model (cosine distance).",
        "Higher = more change from GCL training.",
        "",
        "| Benchmark | Mean Drift | Median Drift | Max Drift |",
        "|---|---:|---:|---:|",
    ])

    for bench in ["fashion200k", "atlas", "polyvore", "KAGL"]:
        d = drift_results[bench]
        lines.append(
            f"| {bench} | {d['mean_cosine_distance']:.4f} | "
            f"{d['median_cosine_distance']:.4f} | {d['max_cosine_distance']:.4f} |"
        )

    # Per-L1 drift
    lines.extend([
        "",
        "### Per-L1 Category Drift",
        "",
        "Categories where FSL moved embeddings the most (these are where GCL had most effect):",
        "",
    ])

    for bench in ["polyvore", "KAGL"]:
        d = drift_results[bench]
        per_l1 = d.get("per_l1_drift", {})
        if not per_l1:
            continue
        lines.extend([
            f"**{bench}:**",
            "",
            "| Category | Mean Drift | N queries |",
            "|---|---:|---:|",
        ])
        for l1, info in sorted(per_l1.items(), key=lambda x: -x[1]["mean_drift"]):
            lines.append(f"| {l1} | {info['mean_drift']:.4f} | {info['n_queries']} |")
        lines.append("")

    # Interpretation
    lines.extend([
        "## 4. Interpretation",
        "",
        "### What this tells us about the gap:",
        "",
        "- If FSL has much higher category alignment on the strata it wins, "
        "the gap is **text→category grounding** (our category-loss approach is correct but insufficient).",
        "- If alignment is similar but FSL still wins on retrieval, "
        "the gap is **image-side discrimination** (frozen image tower is limiting).",
        "- If drift is large on the winning strata, "
        "FSL's GCL moved text embeddings significantly — "
        "we need either more training signal or different scope to match.",
        "",
    ])

    report_text = "\n".join(lines)
    report_path = output_dir / "embedding_alignment_report.md"
    report_path.write_text(report_text)
    log.info("Alignment report saved: %s", report_path)
    print("\n" + report_text)


def parse_args():
    p = argparse.ArgumentParser(description="Phase 7 Embedding Alignment Analysis")
    p.add_argument("--corpus-size", type=int, default=3000)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    t0 = time.time()

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    log.info("Device: %s", device)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load both models
    log.info("Loading FSL model...")
    fsl_model, fsl_preprocess, fsl_tokenizer = load_fsl_model(device)

    log.info("Loading base SigLIP model...")
    base_model, base_preprocess, base_tokenizer = load_base_model(device)

    # Compute category alignment for each model on each benchmark
    fsl_alignment = {}
    base_alignment = {}
    drift_results = {}

    for bench in BENCHMARKS:
        log.info("=" * 40)
        log.info("Processing %s...", bench)

        log.info("  FSL alignment...")
        fsl_alignment[bench] = compute_category_alignment(
            fsl_model, fsl_preprocess, fsl_tokenizer, bench, device,
            corpus_size=args.corpus_size, seed=args.seed,
        )

        log.info("  Base alignment...")
        base_alignment[bench] = compute_category_alignment(
            base_model, base_preprocess, base_tokenizer, bench, device,
            corpus_size=args.corpus_size, seed=args.seed,
        )

        log.info("  Computing drift...")
        drift_results[bench] = compute_query_drift_between_models(
            fsl_model, fsl_tokenizer, base_model, base_tokenizer,
            bench, device, corpus_size=args.corpus_size, seed=args.seed,
        )

    # Free models
    del fsl_model, fsl_preprocess, fsl_tokenizer
    del base_model, base_preprocess, base_tokenizer
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()

    # Save raw data
    save_data = {
        "fsl_alignment": {},
        "base_alignment": {},
        "drift": drift_results,
    }
    for bench in BENCHMARKS:
        # Strip per_query_alignment for size (keep l1_alignment)
        save_data["fsl_alignment"][bench] = {
            "l1_alignment": fsl_alignment[bench]["l1_alignment"],
            "per_query_summary": {
                "mean_align": float(np.mean([
                    q["mean_sim_to_own_images"]
                    for q in fsl_alignment[bench]["per_query_alignment"]
                ])),
                "n_queries": len(fsl_alignment[bench]["per_query_alignment"]),
            },
        }
        save_data["base_alignment"][bench] = {
            "l1_alignment": base_alignment[bench]["l1_alignment"],
            "per_query_summary": {
                "mean_align": float(np.mean([
                    q["mean_sim_to_own_images"]
                    for q in base_alignment[bench]["per_query_alignment"]
                ])),
                "n_queries": len(base_alignment[bench]["per_query_alignment"]),
            },
        }

    with open(RESULTS_DIR / "embedding_alignment_data.json", "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    log.info("Raw alignment data saved.")

    # Generate report
    generate_alignment_report(fsl_alignment, base_alignment, drift_results, RESULTS_DIR)

    elapsed = time.time() - t0
    log.info("Alignment analysis complete in %.1f minutes", elapsed / 60)
    print(f"\nTotal time: {elapsed/60:.1f} min")


if __name__ == "__main__":
    main()
