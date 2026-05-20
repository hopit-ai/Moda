"""Phase 7 — Deep Error Analysis: FSL vs Base SigLIP per-query comparison.

Diagnoses WHY FashionSigLIP beats base SigLIP (and by extension, our trained model)
on polyvore, KAGL, and fashion200k. Categorizes every query into 4 buckets:
  - both_win:    Both models get gold in top-10
  - both_fail:   Neither model gets gold in top-10
  - fsl_wins:    FSL gets gold, base doesn't (FSL's training advantage)
  - base_wins:   Base gets gold, FSL doesn't (FSL's regression)

For the 'fsl_wins' bucket, performs deeper analysis:
  - Cosine similarity between query and gold image (near-miss vs total miss)
  - What base retrieves instead (error type classification)
  - L1 category distribution of failures
  - Query vocabulary patterns

Usage:
  python3 scripts/v3/phase7_error_analysis.py
  python3 scripts/v3/phase7_error_analysis.py --corpus-size 5000
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import random
import re
import time
from collections import Counter, defaultdict
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
log = logging.getLogger("phase7-error")

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "results" / "v3_phases" / "phase7_error_analysis"

# L1 taxonomy (same as phase6)
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
def encode_and_retrieve(
    model, preprocess, tokenizer, benchmark_name: str, device: torch.device,
    corpus_size: int = 3000, seed: int = 42, batch_size: int = 32,
):
    """Encode one benchmark. Returns per-query retrieval data including scores and ranks."""
    from datasets import load_dataset

    cfg = BENCHMARKS[benchmark_name]
    log.info("  Loading %s (corpus_size=%d)...", benchmark_name, corpus_size)

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

    # Build query set (categories with >=2 items)
    cat_to_indices = defaultdict(list)
    for idx, cat in enumerate(categories):
        cat_to_indices[cat].append(idx)
    valid_cats = [cat for cat, idxs in cat_to_indices.items() if len(idxs) >= 2]
    rng2 = random.Random(seed)
    rng2.shuffle(valid_cats)
    query_cats = valid_cats[:min(500, len(valid_cats))]

    log.info("    %d items, %d unique queries", len(categories), len(query_cats))

    # Encode images
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

    img_feats = torch.cat(img_feats, dim=0)
    del images_pil
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()

    # Encode queries
    txt_feats = []
    for i in range(0, len(query_cats), 64):
        tokens = tokenizer(query_cats[i:i + 64]).to(device)
        feat = model.encode_text(tokens)
        feat = F.normalize(feat, dim=-1)
        txt_feats.append(feat.cpu())
        del tokens, feat
    txt_feats = torch.cat(txt_feats, dim=0)

    # Compute full similarity matrix
    scores = (txt_feats @ img_feats.T).numpy()

    per_query = []
    for qi, cat in enumerate(query_cats):
        relevant_indices = set(cat_to_indices[cat])
        sims = scores[qi]
        ranked_all = np.argsort(-sims)
        ranked_top10 = ranked_all[:10]

        hits = 0
        precision_sum = 0.0
        for rank, idx in enumerate(ranked_top10, 1):
            if idx in relevant_indices:
                hits += 1
                precision_sum += hits / rank

        n_relevant = len(relevant_indices)
        ap = precision_sum / min(10, n_relevant) if hits > 0 else 0.0
        hit_in_top10 = any(idx in relevant_indices for idx in ranked_top10)

        # Find rank of first relevant item
        first_relevant_rank = None
        for rank, idx in enumerate(ranked_all, 1):
            if idx in relevant_indices:
                first_relevant_rank = rank
                break

        # Get similarity to nearest gold item
        gold_sims = [float(sims[gidx]) for gidx in relevant_indices]
        max_gold_sim = max(gold_sims) if gold_sims else 0.0

        # Get top-5 retrieved items' categories
        top5_cats = [categories[int(ranked_top10[k])] for k in range(min(5, len(ranked_top10)))]
        top5_sims = [float(sims[int(ranked_top10[k])]) for k in range(min(5, len(ranked_top10)))]

        l1 = classify_l1(cat)

        per_query.append({
            "query": cat,
            "l1_category": l1,
            "ap10": ap,
            "hit_in_top10": hit_in_top10,
            "n_relevant": n_relevant,
            "first_relevant_rank": first_relevant_rank,
            "max_gold_sim": max_gold_sim,
            "top5_retrieved_cats": top5_cats,
            "top5_sims": top5_sims,
            "query_embedding": txt_feats[qi].numpy(),
        })

    del scores, img_feats, txt_feats
    gc.collect()

    return {
        "benchmark": benchmark_name,
        "per_query": per_query,
        "categories": categories,
        "cat_to_indices": dict(cat_to_indices),
    }


def analyze_buckets(fsl_results: list, base_results: list, benchmark_name: str) -> dict:
    """Categorize queries into 4 buckets and analyze failure patterns."""
    both_win = []
    both_fail = []
    fsl_wins = []
    base_wins = []

    for fsl_q, base_q in zip(fsl_results, base_results):
        assert fsl_q["query"] == base_q["query"]
        fsl_hit = fsl_q["hit_in_top10"]
        base_hit = base_q["hit_in_top10"]

        entry = {
            "query": fsl_q["query"],
            "l1_category": fsl_q["l1_category"],
            "fsl_ap10": fsl_q["ap10"],
            "base_ap10": base_q["ap10"],
            "fsl_first_rank": fsl_q["first_relevant_rank"],
            "base_first_rank": base_q["first_relevant_rank"],
            "fsl_max_gold_sim": fsl_q["max_gold_sim"],
            "base_max_gold_sim": base_q["max_gold_sim"],
            "base_top5_cats": base_q["top5_retrieved_cats"],
            "fsl_top5_cats": fsl_q["top5_retrieved_cats"],
        }

        if fsl_hit and base_hit:
            both_win.append(entry)
        elif not fsl_hit and not base_hit:
            both_fail.append(entry)
        elif fsl_hit and not base_hit:
            fsl_wins.append(entry)
        else:
            base_wins.append(entry)

    # Analyze the fsl_wins bucket (our gap)
    fsl_wins_analysis = {}
    if fsl_wins:
        # L1 distribution
        l1_counts = Counter(e["l1_category"] for e in fsl_wins)
        fsl_wins_analysis["l1_distribution"] = dict(l1_counts.most_common())

        # Near-miss analysis: how close was base to getting it right?
        gold_sims = [e["base_max_gold_sim"] for e in fsl_wins]
        fsl_wins_analysis["base_gold_sim_mean"] = float(np.mean(gold_sims))
        fsl_wins_analysis["base_gold_sim_median"] = float(np.median(gold_sims))
        fsl_wins_analysis["near_miss_count"] = sum(1 for s in gold_sims if s > 0.2)
        fsl_wins_analysis["total_miss_count"] = sum(1 for s in gold_sims if s <= 0.1)

        # Error type: does base retrieve same category or wrong category?
        same_cat_errors = 0
        wrong_cat_errors = 0
        for e in fsl_wins:
            query_cat = e["query"]
            top5 = e["base_top5_cats"]
            if any(c == query_cat for c in top5):
                same_cat_errors += 1
            else:
                wrong_cat_errors += 1
        fsl_wins_analysis["same_cat_but_wrong_item"] = same_cat_errors
        fsl_wins_analysis["wrong_category_entirely"] = wrong_cat_errors

        # Where does FSL rank the gold vs where base ranks it?
        fsl_ranks = [e["fsl_first_rank"] for e in fsl_wins if e["fsl_first_rank"]]
        base_ranks = [e["base_first_rank"] for e in fsl_wins if e["base_first_rank"]]
        fsl_wins_analysis["fsl_median_rank"] = float(np.median(fsl_ranks)) if fsl_ranks else None
        fsl_wins_analysis["base_median_rank"] = float(np.median(base_ranks)) if base_ranks else None

        # Vocabulary patterns in failing queries
        all_words = []
        for e in fsl_wins:
            words = re.findall(r'\b[a-zA-Z]{3,}\b', e["query"].lower())
            all_words.extend(words)
        word_freq = Counter(all_words)
        fsl_wins_analysis["top_failure_words"] = dict(word_freq.most_common(20))

    # Also analyze base_wins (our strengths to preserve)
    base_wins_analysis = {}
    if base_wins:
        l1_counts = Counter(e["l1_category"] for e in base_wins)
        base_wins_analysis["l1_distribution"] = dict(l1_counts.most_common())

    # Compute AP differences for fsl_wins bucket
    ap_diffs = [e["fsl_ap10"] - e["base_ap10"] for e in fsl_wins]

    return {
        "benchmark": benchmark_name,
        "total_queries": len(fsl_results),
        "bucket_counts": {
            "both_win": len(both_win),
            "both_fail": len(both_fail),
            "fsl_wins": len(fsl_wins),
            "base_wins": len(base_wins),
        },
        "bucket_pcts": {
            "both_win": len(both_win) / len(fsl_results) * 100,
            "both_fail": len(both_fail) / len(fsl_results) * 100,
            "fsl_wins": len(fsl_wins) / len(fsl_results) * 100,
            "base_wins": len(base_wins) / len(fsl_results) * 100,
        },
        "fsl_wins_analysis": fsl_wins_analysis,
        "base_wins_analysis": base_wins_analysis,
        "fsl_wins_entries": fsl_wins[:50],
        "ap_diff_fsl_wins_mean": float(np.mean(ap_diffs)) if ap_diffs else 0.0,
    }


def compute_embedding_drift(fsl_results: list, base_results: list) -> dict:
    """Measure how far FSL text embeddings are from base text embeddings."""
    cos_dists = []
    for fsl_q, base_q in zip(fsl_results, base_results):
        fsl_emb = fsl_q["query_embedding"]
        base_emb = base_q["query_embedding"]
        cos_sim = float(np.dot(fsl_emb, base_emb) / (np.linalg.norm(fsl_emb) * np.linalg.norm(base_emb)))
        cos_dists.append(1.0 - cos_sim)

    return {
        "mean_cosine_distance": float(np.mean(cos_dists)),
        "median_cosine_distance": float(np.median(cos_dists)),
        "max_cosine_distance": float(np.max(cos_dists)),
        "min_cosine_distance": float(np.min(cos_dists)),
        "std_cosine_distance": float(np.std(cos_dists)),
    }


def generate_diagnosis_report(all_analyses: dict, drift_data: dict, output_dir: Path):
    """Generate the final gap diagnosis markdown report."""
    lines = [
        "# Phase 7 — Gap Diagnosis Report",
        "",
        f"_Generated: {time.strftime('%Y-%m-%d %H:%M IST')}_",
        "",
        "## 1. Bucket Summary (per benchmark)",
        "",
        "| Benchmark | Both Win | Both Fail | FSL Wins | Base Wins | Net FSL Advantage |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    for bench in ["fashion200k", "atlas", "polyvore", "KAGL"]:
        a = all_analyses[bench]
        bc = a["bucket_counts"]
        bp = a["bucket_pcts"]
        net = bc["fsl_wins"] - bc["base_wins"]
        lines.append(
            f"| {bench} | {bc['both_win']} ({bp['both_win']:.1f}%) | "
            f"{bc['both_fail']} ({bp['both_fail']:.1f}%) | "
            f"{bc['fsl_wins']} ({bp['fsl_wins']:.1f}%) | "
            f"{bc['base_wins']} ({bp['base_wins']:.1f}%) | "
            f"{net:+d} queries |"
        )

    # FSL wins deep dive
    lines.extend([
        "",
        "## 2. Where FSL Wins (Our Gap) — Deep Dive",
        "",
    ])

    for bench in ["fashion200k", "atlas", "polyvore", "KAGL"]:
        a = all_analyses[bench]
        fwa = a["fsl_wins_analysis"]
        if not fwa:
            lines.extend([f"### {bench}", "", "_No FSL-exclusive wins._", ""])
            continue

        lines.extend([
            f"### {bench} ({a['bucket_counts']['fsl_wins']} queries where FSL wins, we fail)",
            "",
            "**L1 Category Distribution of Failures:**",
            "",
            "| Category | Count | % of FSL-wins |",
            "|---|---:|---:|",
        ])
        total_fsl_wins = a["bucket_counts"]["fsl_wins"]
        for cat, count in sorted(fwa.get("l1_distribution", {}).items(), key=lambda x: -x[1]):
            lines.append(f"| {cat} | {count} | {count/total_fsl_wins*100:.1f}% |")

        lines.extend([
            "",
            "**Near-Miss Analysis (base model's similarity to gold):**",
            "",
            f"- Mean gold similarity (base): {fwa.get('base_gold_sim_mean', 0):.4f}",
            f"- Median gold similarity (base): {fwa.get('base_gold_sim_median', 0):.4f}",
            f"- Near-miss (sim > 0.2): {fwa.get('near_miss_count', 0)} queries",
            f"- Total miss (sim <= 0.1): {fwa.get('total_miss_count', 0)} queries",
            "",
            "**Error Type:**",
            "",
            f"- Same category, wrong item: {fwa.get('same_cat_but_wrong_item', 0)}",
            f"- Wrong category entirely: {fwa.get('wrong_category_entirely', 0)}",
            "",
            "**Rank Comparison (where is gold in the full ranking?):**",
            "",
            f"- FSL median rank of gold: {fwa.get('fsl_median_rank', 'N/A')}",
            f"- Base median rank of gold: {fwa.get('base_median_rank', 'N/A')}",
            "",
            "**Top vocabulary in failing queries:**",
            "",
        ])
        top_words = fwa.get("top_failure_words", {})
        word_items = list(top_words.items())[:15]
        if word_items:
            lines.append("| Word | Frequency |")
            lines.append("|---|---:|")
            for word, freq in word_items:
                lines.append(f"| {word} | {freq} |")
        lines.append("")

    # Embedding drift
    lines.extend([
        "## 3. Embedding Drift (FSL vs Base text embeddings)",
        "",
        "How far did FSL's GCL training move text embeddings from the base?",
        "",
        "| Benchmark | Mean Cos Distance | Median | Max | Min |",
        "|---|---:|---:|---:|---:|",
    ])
    for bench in ["fashion200k", "atlas", "polyvore", "KAGL"]:
        d = drift_data[bench]
        lines.append(
            f"| {bench} | {d['mean_cosine_distance']:.4f} | "
            f"{d['median_cosine_distance']:.4f} | "
            f"{d['max_cosine_distance']:.4f} | "
            f"{d['min_cosine_distance']:.4f} |"
        )

    # Diagnosis summary
    lines.extend([
        "",
        "## 4. Diagnosis Summary",
        "",
        "### Key Findings:",
        "",
    ])

    # Compute aggregate stats for diagnosis
    total_fsl_exclusive = sum(all_analyses[b]["bucket_counts"]["fsl_wins"] for b in BENCHMARKS)
    total_base_exclusive = sum(all_analyses[b]["bucket_counts"]["base_wins"] for b in BENCHMARKS)
    total_both_fail = sum(all_analyses[b]["bucket_counts"]["both_fail"] for b in BENCHMARKS)
    total_queries = sum(all_analyses[b]["total_queries"] for b in BENCHMARKS)

    lines.extend([
        f"- **Total queries across all benchmarks:** {total_queries}",
        f"- **FSL-exclusive wins (our gap):** {total_fsl_exclusive} ({total_fsl_exclusive/total_queries*100:.1f}%)",
        f"- **Base-exclusive wins (FSL regressions):** {total_base_exclusive} ({total_base_exclusive/total_queries*100:.1f}%)",
        f"- **Both fail (unsolvable at this capacity):** {total_both_fail} ({total_both_fail/total_queries*100:.1f}%)",
        "",
        "### Root Cause Hypotheses:",
        "",
        "Based on the analysis above, the gap is likely due to:",
        "",
    ])

    # Determine dominant error types
    all_near_miss = sum(all_analyses[b]["fsl_wins_analysis"].get("near_miss_count", 0)
                        for b in BENCHMARKS if all_analyses[b]["fsl_wins_analysis"])
    all_total_miss = sum(all_analyses[b]["fsl_wins_analysis"].get("total_miss_count", 0)
                         for b in BENCHMARKS if all_analyses[b]["fsl_wins_analysis"])
    all_wrong_cat = sum(all_analyses[b]["fsl_wins_analysis"].get("wrong_category_entirely", 0)
                        for b in BENCHMARKS if all_analyses[b]["fsl_wins_analysis"])
    all_same_cat = sum(all_analyses[b]["fsl_wins_analysis"].get("same_cat_but_wrong_item", 0)
                       for b in BENCHMARKS if all_analyses[b]["fsl_wins_analysis"])

    if all_near_miss > all_total_miss:
        lines.append("1. **NEAR-MISS DOMINANT**: Most failures are near-misses (gold sim > 0.2). "
                     "The base model *almost* retrieves correctly — a small training signal could fix this. "
                     "Implication: **Data quality matters more than data volume.**")
    else:
        lines.append("1. **TOTAL-MISS DOMINANT**: Most failures are total misses (gold sim <= 0.1). "
                     "The base model doesn't even see the gold as relevant. "
                     "Implication: **Fundamental representation gap — needs more training data or unfreezing image tower.**")

    if all_wrong_cat > all_same_cat:
        lines.append("2. **WRONG-CATEGORY ERRORS DOMINATE**: Base retrieves items from wrong categories. "
                     "FSL's GCL training taught category-level discrimination. "
                     "Implication: **Category-level contrastive training is the key gap.**")
    else:
        lines.append("2. **SAME-CATEGORY ERRORS DOMINATE**: Base retrieves correct category but wrong item. "
                     "The issue is fine-grained within-category discrimination. "
                     "Implication: **Need hard-negative mining within categories, not just category labels.**")

    lines.extend([
        "",
        "### Recommended Next Steps:",
        "",
        "(To be filled after reviewing the data above)",
    ])

    report_text = "\n".join(lines)
    report_path = output_dir / "gap_diagnosis.md"
    report_path.write_text(report_text)
    log.info("Diagnosis report saved: %s", report_path)
    print("\n" + report_text)


def parse_args():
    p = argparse.ArgumentParser(description="Phase 7 Deep Error Analysis")
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

    all_analyses = {}
    drift_data = {}

    # Load and evaluate FSL
    log.info("=" * 60)
    log.info("Loading FSL model...")
    fsl_model, fsl_preprocess, fsl_tokenizer = load_fsl_model(device)

    fsl_per_bench = {}
    for bench in BENCHMARKS:
        log.info("Evaluating FSL on %s...", bench)
        fsl_per_bench[bench] = encode_and_retrieve(
            fsl_model, fsl_preprocess, fsl_tokenizer, bench, device,
            corpus_size=args.corpus_size, seed=args.seed,
        )

    del fsl_model, fsl_preprocess, fsl_tokenizer
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()

    # Load and evaluate base SigLIP
    log.info("=" * 60)
    log.info("Loading base SigLIP model...")
    base_model, base_preprocess, base_tokenizer = load_base_model(device)

    base_per_bench = {}
    for bench in BENCHMARKS:
        log.info("Evaluating base on %s...", bench)
        base_per_bench[bench] = encode_and_retrieve(
            base_model, base_preprocess, base_tokenizer, bench, device,
            corpus_size=args.corpus_size, seed=args.seed,
        )

    del base_model, base_preprocess, base_tokenizer
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()

    # Per-benchmark analysis
    for bench in BENCHMARKS:
        log.info("Analyzing %s...", bench)
        fsl_pq = fsl_per_bench[bench]["per_query"]
        base_pq = base_per_bench[bench]["per_query"]

        all_analyses[bench] = analyze_buckets(fsl_pq, base_pq, bench)
        drift_data[bench] = compute_embedding_drift(fsl_pq, base_pq)

    # Save raw results (without embeddings)
    save_data = {}
    for bench in BENCHMARKS:
        a = all_analyses[bench]
        save_data[bench] = {
            k: v for k, v in a.items()
            if k != "fsl_wins_entries"
        }
        # Save top-50 entries separately
        save_data[bench]["fsl_wins_sample"] = a["fsl_wins_entries"]

    save_data["drift"] = drift_data

    with open(RESULTS_DIR / "per_query_comparison.json", "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    log.info("Raw results saved: %s", RESULTS_DIR / "per_query_comparison.json")

    # Generate diagnosis report
    generate_diagnosis_report(all_analyses, drift_data, RESULTS_DIR)

    elapsed = time.time() - t0
    log.info("Error analysis complete in %.1f minutes", elapsed / 60)
    print(f"\nTotal time: {elapsed/60:.1f} min")


if __name__ == "__main__":
    main()
