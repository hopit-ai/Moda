"""Phase 6 — Stratified evaluation on all 4 Marqo benchmarks.

Compares our GCL-trained model vs FashionSigLIP with:
  - Per-benchmark MAP@10
  - Per-stratum (L1 category) MAP@10
  - Bootstrap confidence intervals (95% CI)
  - Statistical significance (p-value)
  - Regression gate: reject if any benchmark drops >2% vs base

Models evaluated:
  1. Ours: ViT-B-16-SigLIP/webli + trained weights (from Phase 4 checkpoint)
  2. FSL:  Marqo-FashionSigLIP (ViT-B-16-SigLIP, fine-tuned)
  3. Base: ViT-B-16-SigLIP/webli (untrained, reference)

Usage:
  python3 scripts/v3/phase6_eval_stratified.py --checkpoint checkpoints/v3_gcl/phase4_text_only_full/best.pt
  python3 scripts/v3/phase6_eval_stratified.py --checkpoint checkpoints/v3_gcl/phase4_text_only_full/best.pt --corpus-size 5000
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
log = logging.getLogger("eval-stratified")

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "results" / "v3_phases" / "phase6_eval"

# ── L1 taxonomy (same as phase0/phase1) ──────────────────────────────────────

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


# ── Benchmark definitions ────────────────────────────────────────────────────

BENCHMARKS = {
    "fashion200k": {"hf": "Marqo/fashion200k", "split": "data", "query_col": "category3"},
    "atlas":       {"hf": "Marqo/atlas",       "split": "data", "query_col": "sub-category"},
    "polyvore":    {"hf": "Marqo/polyvore",    "split": "data", "query_col": "category"},
    "KAGL":        {"hf": "Marqo/KAGL",        "split": "data", "query_col": "category3"},
}


# ── Model loading ────────────────────────────────────────────────────────────

def load_our_model(checkpoint_path: str, device: torch.device):
    """Load base SigLIP + apply our trained checkpoint."""
    import open_clip

    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-16-SigLIP", pretrained="webli"
    )
    tokenizer = open_clip.get_tokenizer("ViT-B-16-SigLIP")

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = ckpt["model_state_dict"]

    # Merge trained params into the model
    model_state = model.state_dict()
    model_state.update(state)
    model.load_state_dict(model_state)

    model = model.to(device).eval()
    log.info("Loaded our model from %s (%d trained keys, scope=%s)",
             checkpoint_path, len(state), ckpt.get("scope", "unknown"))
    return model, preprocess, tokenizer


def load_fsl_model(device: torch.device):
    """Load Marqo-FashionSigLIP."""
    import open_clip

    model, _, preprocess = open_clip.create_model_and_transforms(
        "hf-hub:Marqo/marqo-fashionSigLIP"
    )
    tokenizer = open_clip.get_tokenizer("hf-hub:Marqo/marqo-fashionSigLIP")
    model = model.to(device).eval()
    log.info("Loaded FashionSigLIP (Marqo)")
    return model, preprocess, tokenizer


def load_base_model(device: torch.device):
    """Load vanilla ViT-B-16-SigLIP/webli (untrained baseline)."""
    import open_clip

    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-16-SigLIP", pretrained="webli"
    )
    tokenizer = open_clip.get_tokenizer("ViT-B-16-SigLIP")
    model = model.to(device).eval()
    log.info("Loaded base SigLIP (untrained)")
    return model, preprocess, tokenizer


# ── Evaluation core ──────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_benchmark(
    model, preprocess, tokenizer, benchmark_name: str, device: torch.device,
    corpus_size: int = 3000, seed: int = 42, batch_size: int = 32,
) -> dict:
    """Evaluate one model on one benchmark. Returns per-query AP and stratum info."""
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

    # Deterministic shuffle
    rng = random.Random(seed)
    indices = list(range(len(categories)))
    rng.shuffle(indices)
    indices = indices[:corpus_size]
    categories = [categories[i] for i in indices]
    images_pil = [images_pil[i] for i in indices]

    # Build query set (categories with ≥2 items)
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

    # Compute similarity and per-query AP@10
    scores = txt_feats @ img_feats.T
    del img_feats, txt_feats
    gc.collect()

    per_query_results = []
    for qi, cat in enumerate(query_cats):
        relevant_indices = set(cat_to_indices[cat])
        sims = scores[qi].numpy()
        ranked = np.argsort(-sims)[:10]

        hits = 0
        precision_sum = 0.0
        for rank, idx in enumerate(ranked, 1):
            if idx in relevant_indices:
                hits += 1
                precision_sum += hits / rank

        n_relevant = len(relevant_indices)
        ap = precision_sum / min(10, n_relevant) if hits > 0 else 0.0
        l1 = classify_l1(cat)

        per_query_results.append({
            "query": cat,
            "l1_category": l1,
            "ap10": ap,
            "n_relevant": n_relevant,
        })

    del scores
    gc.collect()

    return {
        "benchmark": benchmark_name,
        "n_queries": len(per_query_results),
        "per_query": per_query_results,
    }


# ── Statistical analysis ─────────────────────────────────────────────────────

def compute_stats(per_query: list[dict]) -> dict:
    """Compute aggregate and per-stratum MAP@10 with bootstrap CI."""
    aps = [q["ap10"] for q in per_query]
    map10 = float(np.mean(aps)) if aps else 0.0

    # Bootstrap 95% CI
    n_bootstrap = 1000
    rng = np.random.RandomState(42)
    boot_means = []
    for _ in range(n_bootstrap):
        sample = rng.choice(aps, size=len(aps), replace=True)
        boot_means.append(np.mean(sample))
    ci_low = float(np.percentile(boot_means, 2.5))
    ci_high = float(np.percentile(boot_means, 97.5))

    # Per-stratum
    strata = defaultdict(list)
    for q in per_query:
        strata[q["l1_category"]].append(q["ap10"])

    per_stratum = {}
    for l1, stratum_aps in sorted(strata.items()):
        per_stratum[l1] = {
            "map10": float(np.mean(stratum_aps)),
            "n_queries": len(stratum_aps),
            "std": float(np.std(stratum_aps)),
        }

    return {
        "map10": map10,
        "ci_95": [ci_low, ci_high],
        "n_queries": len(aps),
        "per_stratum": per_stratum,
    }


def compute_significance(ours_aps: list[float], fsl_aps: list[float]) -> dict:
    """Bootstrap test for significance of MAP@10 difference."""
    n = min(len(ours_aps), len(fsl_aps))
    ours_aps = ours_aps[:n]
    fsl_aps = fsl_aps[:n]

    observed_diff = np.mean(ours_aps) - np.mean(fsl_aps)

    rng = np.random.RandomState(42)
    n_bootstrap = 2000
    boot_diffs = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        diff = np.mean([ours_aps[i] for i in idx]) - np.mean([fsl_aps[i] for i in idx])
        boot_diffs.append(diff)

    # P-value: fraction of bootstraps where diff <= 0 (one-tailed)
    p_value = float(np.mean([d <= 0 for d in boot_diffs]))
    ci_low = float(np.percentile(boot_diffs, 2.5))
    ci_high = float(np.percentile(boot_diffs, 97.5))

    return {
        "observed_diff": float(observed_diff),
        "p_value": p_value,
        "ci_95_diff": [ci_low, ci_high],
        "significant": p_value < 0.05,
    }


# ── Report generation ────────────────────────────────────────────────────────

def generate_report(all_results: dict, output_dir: Path):
    """Generate markdown report with tables."""
    lines = [
        "# Phase 6 — Stratified Evaluation Report",
        "",
        f"_Generated: {time.strftime('%Y-%m-%d %H:%M IST')}_",
        "",
        "## Aggregate Results (MAP@10)",
        "",
        "| Benchmark | Ours | FSL | Base | Ours vs FSL | Significant? |",
        "|---|---:|---:|---:|---|---|",
    ]

    for bench in ["fashion200k", "atlas", "polyvore", "KAGL"]:
        ours = all_results["ours"][bench]["map10"]
        fsl = all_results["fsl"][bench]["map10"]
        base = all_results["base"][bench]["map10"]
        diff = ours - fsl
        sig = all_results["significance"][bench]
        sig_str = f"p={sig['p_value']:.3f} {'YES' if sig['significant'] else 'NO'}"
        lines.append(
            f"| {bench} | {ours:.4f} | {fsl:.4f} | {base:.4f} | "
            f"{diff:+.4f} ({diff/fsl*100:+.1f}%) | {sig_str} |"
        )

    # Per-stratum breakdown for each benchmark
    for bench in ["fashion200k", "atlas", "polyvore", "KAGL"]:
        lines.extend([
            "",
            f"## Per-Stratum: {bench}",
            "",
            "| Stratum | Ours | FSL | Base | Ours vs FSL | n_queries |",
            "|---|---:|---:|---:|---|---:|",
        ])
        ours_strata = all_results["ours"][bench].get("per_stratum", {})
        fsl_strata = all_results["fsl"][bench].get("per_stratum", {})
        base_strata = all_results["base"][bench].get("per_stratum", {})

        all_l1s = sorted(set(list(ours_strata.keys()) + list(fsl_strata.keys())))
        for l1 in all_l1s:
            o = ours_strata.get(l1, {}).get("map10", 0)
            f = fsl_strata.get(l1, {}).get("map10", 0)
            b = base_strata.get(l1, {}).get("map10", 0)
            n = ours_strata.get(l1, {}).get("n_queries", 0)
            diff = o - f
            lines.append(f"| {l1} | {o:.4f} | {f:.4f} | {b:.4f} | {diff:+.4f} | {n} |")

    # Regression gate
    lines.extend([
        "",
        "## Regression Gate Check",
        "",
        "| Benchmark | Ours vs Base | Pass (>-2%)? |",
        "|---|---|---|",
    ])
    all_pass = True
    for bench in ["fashion200k", "atlas", "polyvore", "KAGL"]:
        ours = all_results["ours"][bench]["map10"]
        base = all_results["base"][bench]["map10"]
        if base > 0:
            diff_pct = (ours - base) / base * 100
            passed = diff_pct > -2.0
            all_pass = all_pass and passed
            lines.append(f"| {bench} | {diff_pct:+.1f}% | {'PASS' if passed else 'FAIL'} |")
        else:
            lines.append(f"| {bench} | N/A (base skipped) | — |")

    lines.extend([
        "",
        f"**Overall regression gate: {'PASSED' if all_pass else 'FAILED'}**",
        "",
        "## Confidence Intervals",
        "",
        "| Benchmark | Ours 95% CI | FSL 95% CI | Diff 95% CI |",
        "|---|---|---|---|",
    ])
    for bench in ["fashion200k", "atlas", "polyvore", "KAGL"]:
        ours_ci = all_results["ours"][bench]["ci_95"]
        fsl_ci = all_results["fsl"][bench]["ci_95"]
        diff_ci = all_results["significance"][bench]["ci_95_diff"]
        lines.append(
            f"| {bench} | [{ours_ci[0]:.4f}, {ours_ci[1]:.4f}] | "
            f"[{fsl_ci[0]:.4f}, {fsl_ci[1]:.4f}] | "
            f"[{diff_ci[0]:.4f}, {diff_ci[1]:.4f}] |"
        )

    report = "\n".join(lines)
    report_path = output_dir / "evaluation_report.md"
    report_path.write_text(report)
    log.info("Report saved: %s", report_path)
    print("\n" + report)


# ── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Phase 6 Stratified Evaluation")
    p.add_argument("--checkpoint", required=True, help="Path to our trained model checkpoint")
    p.add_argument("--corpus-size", type=int, default=3000, help="Items per benchmark")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--skip-base", action="store_true", help="Skip base model (faster)")
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

    all_results = {"ours": {}, "fsl": {}, "base": {}, "significance": {}}

    # ── Evaluate each model ──────────────────────────────────────────────────
    models_to_eval = [
        ("ours", lambda: load_our_model(args.checkpoint, device)),
        ("fsl", lambda: load_fsl_model(device)),
    ]
    if not args.skip_base:
        models_to_eval.append(("base", lambda: load_base_model(device)))

    for model_name, loader in models_to_eval:
        log.info("\n{'='*60}")
        log.info("Evaluating: %s", model_name)
        log.info("{'='*60}")

        model, preprocess, tokenizer = loader()

        for bench in ["fashion200k", "atlas", "polyvore", "KAGL"]:
            raw = evaluate_benchmark(
                model, preprocess, tokenizer, bench, device,
                corpus_size=args.corpus_size, seed=args.seed,
            )
            stats = compute_stats(raw["per_query"])
            all_results[model_name][bench] = stats
            all_results[model_name][bench]["per_query_raw"] = raw["per_query"]
            log.info("  %s %s: MAP@10=%.4f [%.4f, %.4f]",
                     model_name, bench, stats["map10"], stats["ci_95"][0], stats["ci_95"][1])

        # Free model memory
        del model, preprocess, tokenizer
        gc.collect()
        if device.type == "mps":
            torch.mps.empty_cache()

    # If base was skipped, fill with zeros
    if args.skip_base:
        for bench in BENCHMARKS:
            all_results["base"][bench] = {"map10": 0.0, "ci_95": [0.0, 0.0], "per_stratum": {}}

    # ── Significance tests ───────────────────────────────────────────────────
    for bench in BENCHMARKS:
        ours_aps = [q["ap10"] for q in all_results["ours"][bench]["per_query_raw"]]
        fsl_aps = [q["ap10"] for q in all_results["fsl"][bench]["per_query_raw"]]
        all_results["significance"][bench] = compute_significance(ours_aps, fsl_aps)

    # ── Generate report ──────────────────────────────────────────────────────
    generate_report(all_results, RESULTS_DIR)

    # Save raw results (without per_query_raw for size)
    save_results = {}
    for model_name in ["ours", "fsl", "base"]:
        save_results[model_name] = {}
        for bench in BENCHMARKS:
            r = {k: v for k, v in all_results[model_name][bench].items() if k != "per_query_raw"}
            save_results[model_name][bench] = r
    save_results["significance"] = all_results["significance"]

    with open(RESULTS_DIR / "results.json", "w") as f:
        json.dump(save_results, f, indent=2)

    elapsed = time.time() - t0
    log.info("\nEvaluation complete in %.1f minutes", elapsed / 60)
    print(f"\nTotal time: {elapsed/60:.1f} min")
    print(f"Results: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
