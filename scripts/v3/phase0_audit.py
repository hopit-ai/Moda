"""Phase 0 — Category taxonomy audit across all 4 Marqo benchmarks.

Memory-efficient: processes one benchmark at a time, encodes images in
batches, and releases PIL images immediately after encoding.

Outputs:
  results/v3_phases/phase0_audit/per_stratum_fsl_performance.csv
  results/v3_phases/phase0_audit/per_stratum_base_siglip_performance.csv
  results/v3_phases/phase0_audit/taxonomy_map.json
  results/v3_phases/phase0_audit/gap_analysis.md
"""

import json, random, time, gc, csv, re
from collections import defaultdict, Counter
from pathlib import Path

import torch
import torch.nn.functional as F

REPO = Path(__file__).parent.parent.parent
HF_CACHE = str(REPO / "data" / "hf_cache")
OUT_DIR = REPO / "results" / "v3_phases" / "phase0_audit"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
CORPUS_SIZE = 3000  # reduced to avoid OOM

# ── L1 garment taxonomy ─────────────────────────────────────────────────────

L1_KEYWORDS = [
    # Outerwear — must come before tops (hoodie is outerwear-ish but we keep it in tops)
    (r"\b(jackets?|blazers?|coats?|parkas?|anorak|windbreaker|trench|overcoat|capes?|ponchos?|bombers?|vests?|waistcoats?|outerwear|sherwani)\b", "outerwear"),
    # Dresses — catches "dress", "dresses", "shirtdress", "gowns" etc
    (r"(dress|gowns?|rompers?|jumpsuits?|playsuits?|sarees?|kurtas?|kurtis?|kurta.?sets?|galabiyyas?|dhoti)", "dresses"),
    # Tops
    (r"\b(tops?|blouses?|shirts?|tees?|t-?shirts?|tshirts?|tanks?|camisoles?|tunics?|polos?|henleys?|crop.?tops?|halters?|bustiers?|corsets?|hoodies?|sweatshirts?|sweaters?|pullovers?|cardigans?|knits?|chemises?)\b", "tops"),
    # Bottoms
    (r"\b(pants?|trousers?|jeans?|denim|leggings?|chinos?|shorts?|skirts?|skorts?|culottes?|joggers?|sweatpants?|cargos?|capris?|churidars?|tracksuits?|tights?|hosiery)\b", "bottoms"),
    # Shoes
    (r"\b(shoes?|boots?|sneakers?|sandals?|heels?|pumps?|flats?|loafers?|slippers?|mules?|clogs?|espadrilles?|oxfords?|derby|brogues?|stilettos?|wedges?|platforms?|flip.?flops?|booties?|moccasins?|sports?.?shoes?|casual.?shoes?|formal.?shoes?)\b", "shoes"),
    # Bags
    (r"\b(bags?|handbags?|purses?|clutches?|totes?|backpacks?|satchels?|crossbody|messenger|wallets?|pouches?|duffles?|weekenders?|rucksacks?|luggage|shoulder.?bags?)\b", "bags"),
    # Accessories
    (r"\b(accessor|jewelry|jewellery|necklaces?|bracelets?|bangles?|earrings?|rings?|watches?|sunglasses|eyeglasses|eyewear|scarves?|scarfs?|belts?|hats?|caps?|beanies?|gloves?|ties?|bow.?ties?|cufflinks?|brooches?|pins?|charms?|pendants?|stoles?|dupattas?|umbrellas?|socks?)\b", "accessories"),
    # Swimwear
    (r"\b(swimsuits?|bikinis?|swimwear|bathing|swim|one.?piece.?swim)\b", "swimwear"),
    # Intimates
    (r"\b(lingerie|bras?|underwear|panty|panties|briefs?|boxers?|nightgowns?|pajamas?|pyjamas?|robes?|sleepwear|lounge|intimates?|shapewear)\b", "intimates"),
    # Activewear
    (r"\b(activewear|sportswear|athletic|yoga|gym|workout|running|cycling|fitness|track|sports?.?sandals?)\b", "activewear"),
    # Beauty/Home (polyvore has these — not fashion, but need a label)
    (r"\b(makeup|mascara|lipstick|foundation|concealer|eyeliner|eyeshadow|blush|fragrance|skincare|haircare|nail|beauty|cosmetics?)\b", "beauty"),
    (r"\b(furniture|chairs?|tables?|lamps?|rugs?|pillows?|curtains?|beds?|sofas?|mirrors?|vases?|candles?|decor|lighting|shelves|storage|kitchen|dining|drinkware|flatware|serveware|frames?|wallpaper|clocks?)\b", "home"),
]

L1_COMPILED = [(re.compile(pat, re.IGNORECASE), label) for pat, label in L1_KEYWORDS]


def classify_l1(text: str) -> str:
    if not text:
        return "other"
    for pattern, label in L1_COMPILED:
        if pattern.search(text):
            return label
    return "other"


# ── L2 attribute extraction ──────────────────────────────────────────────────

COLOR_RE = re.compile(
    r"\b(red|blue|green|yellow|orange|purple|pink|black|white|grey|gray|brown|beige|navy|teal|turquoise|coral|maroon|burgundy|ivory|cream|gold|silver|olive|tan|rust|lavender|mint|peach|magenta|fuchsia|khaki|charcoal|indigo|cyan|multicolor|multi.?color)\b",
    re.IGNORECASE,
)
MATERIAL_RE = re.compile(
    r"\b(cotton|silk|satin|linen|wool|cashmere|polyester|nylon|spandex|velvet|chiffon|lace|leather|suede|denim|tweed|corduroy|fleece|jersey|mesh|organza|tulle|sequin|knit|crochet|woven|embroidered)\b",
    re.IGNORECASE,
)
STYLE_RE = re.compile(
    r"\b(casual|formal|elegant|modern|classic|vintage|retro|bohemian|boho|minimal|sporty|preppy|chic|trendy|romantic|edgy|slim|fitted|relaxed|oversized|loose|tailored|structured|flowy|wrap|a.?line|pencil|maxi|midi|mini|cropped|high.?waisted|ribbed|quilted|ruffle|pleated|tiered|asymmetric|draped|fringe)\b",
    re.IGNORECASE,
)


def extract_attributes(text: str) -> dict:
    return {
        "has_color": bool(COLOR_RE.search(text)) if text else False,
        "has_material": bool(MATERIAL_RE.search(text)) if text else False,
        "has_style": bool(STYLE_RE.search(text)) if text else False,
    }


# ── Benchmark config ─────────────────────────────────────────────────────────

BENCHMARKS = {
    "fashion200k": {"hf": "Marqo/fashion200k", "split": "data", "query_col": "category3"},
    "atlas":       {"hf": "Marqo/atlas",       "split": "data", "query_col": "sub-category"},
    "polyvore":    {"hf": "Marqo/polyvore",     "split": "data", "query_col": "category"},
    "KAGL":        {"hf": "Marqo/KAGL",         "split": "data", "query_col": "category3"},
}


# ── Evaluation (one benchmark at a time, minimal memory) ─────────────────────

def evaluate_one_benchmark(model, preprocess, tokenizer, benchmark_name: str, model_label: str):
    """Load one benchmark, encode, evaluate per-stratum, release all images."""
    from datasets import load_dataset

    cfg = BENCHMARKS[benchmark_name]
    print(f"\n  [{model_label}] Loading {benchmark_name}...")
    ds = load_dataset(cfg["hf"], split=cfg["split"], streaming=True)

    categories = []
    images_pil = []
    for row in ds:
        if len(categories) >= CORPUS_SIZE:
            break
        query = (row.get(cfg["query_col"]) or "").strip()
        if not query or not row.get("image"):
            continue
        categories.append(query)
        images_pil.append(row["image"])

    rng = random.Random(SEED)
    indices = list(range(len(categories)))
    rng.shuffle(indices)
    indices = indices[:CORPUS_SIZE]
    categories = [categories[i] for i in indices]
    images_pil = [images_pil[i] for i in indices]
    print(f"    {len(categories)} items loaded")

    # Build query set
    cat_to_indices = defaultdict(list)
    for idx, cat in enumerate(categories):
        cat_to_indices[cat].append(idx)
    valid_cats = [cat for cat, idxs in cat_to_indices.items() if len(idxs) >= 2]
    rng2 = random.Random(SEED)
    rng2.shuffle(valid_cats)
    query_cats = valid_cats[:min(500, len(valid_cats))]
    print(f"    {len(query_cats)} unique queries (categories with ≥2 items)")

    # Encode images in batches, releasing PIL images as we go
    print(f"    Encoding images...")
    img_feats = []
    model.eval()
    batch_size = 16
    with torch.no_grad():
        for i in range(0, len(images_pil), batch_size):
            batch_pil = images_pil[i : i + batch_size]
            tensors = torch.stack([preprocess(im.convert("RGB")) for im in batch_pil]).to(DEVICE)
            feat = model.encode_image(tensors)
            feat = F.normalize(feat, dim=-1)
            img_feats.append(feat.cpu())
            del tensors, feat
            # Release processed PIL images
            for j in range(i, min(i + batch_size, len(images_pil))):
                images_pil[j] = None

    img_feats = torch.cat(img_feats, dim=0)
    del images_pil
    gc.collect()

    # Encode queries
    print(f"    Encoding queries...")
    txt_feats = []
    with torch.no_grad():
        for i in range(0, len(query_cats), 64):
            tokens = tokenizer(query_cats[i : i + 64]).to(DEVICE)
            feat = model.encode_text(tokens)
            feat = F.normalize(feat, dim=-1)
            txt_feats.append(feat.cpu())
            del tokens, feat
    txt_feats = torch.cat(txt_feats, dim=0)

    # Compute scores and per-query AP
    scores = txt_feats @ img_feats.T
    del img_feats, txt_feats
    gc.collect()

    results = []
    for qi, cat in enumerate(query_cats):
        positives = set(cat_to_indices[cat])
        topk = scores[qi].topk(min(10, scores.shape[1])).indices.tolist()
        ap, n_rel = 0.0, 0
        for rank, idx in enumerate(topk, 1):
            if idx in positives:
                n_rel += 1
                ap += n_rel / rank
        n_pos = min(len(positives), 10)
        if n_pos > 0:
            ap /= n_pos

        l1 = classify_l1(cat)
        attrs = extract_attributes(cat)
        results.append({
            "dataset": benchmark_name,
            "query": cat,
            "l1_category": l1,
            "has_color": attrs["has_color"],
            "has_material": attrs["has_material"],
            "has_style": attrs["has_style"],
            "n_positives": len(positives),
            "ap10": ap,
        })

    del scores
    gc.collect()

    map10 = sum(r["ap10"] for r in results) / max(len(results), 1)
    print(f"    {model_label} {benchmark_name} MAP@10 = {map10:.4f} ({len(results)} queries)")
    return results, categories  # return categories for taxonomy map


# ── Model loading ────────────────────────────────────────────────────────────

def load_fsl():
    import open_clip
    from huggingface_hub import hf_hub_download

    print("\nLoading FashionSigLIP...")
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-16-SigLIP", pretrained="webli")
    ckpt = hf_hub_download("Marqo/marqo-fashionSigLIP", filename="open_clip_pytorch_model.bin",
                           cache_dir=HF_CACHE)
    state = torch.load(ckpt, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval().to(DEVICE)
    tokenizer = open_clip.get_tokenizer("ViT-B-16-SigLIP")
    return model, preprocess, tokenizer


def load_base_siglip():
    import open_clip

    print("\nLoading base SigLIP (ViT-B-16-SigLIP/webli)...")
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-16-SigLIP", pretrained="webli")
    model.eval().to(DEVICE)
    tokenizer = open_clip.get_tokenizer("ViT-B-16-SigLIP")
    return model, preprocess, tokenizer


# ── Aggregation & output ─────────────────────────────────────────────────────

def aggregate_results(results: list[dict]) -> dict:
    groups = defaultdict(list)
    for r in results:
        groups[(r["dataset"], r["l1_category"])].append(r["ap10"])
    return {k: {"mean_ap10": sum(v) / len(v), "n_queries": len(v)} for k, v in groups.items()}


def write_csv(results: list[dict], path: Path, model_name: str):
    fieldnames = ["model", "dataset", "query", "l1_category", "has_color",
                  "has_material", "has_style", "n_positives", "ap10"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({"model": model_name, **r})
    print(f"  Wrote {len(results)} rows to {path}")


def write_gap_analysis(fsl_agg, base_agg, fsl_results, base_results, path: Path):
    fsl_by_ds = defaultdict(list)
    base_by_ds = defaultdict(list)
    for r in fsl_results:
        fsl_by_ds[r["dataset"]].append(r["ap10"])
    for r in base_results:
        base_by_ds[r["dataset"]].append(r["ap10"])

    lines = [
        "# Phase 0 — Gap Analysis",
        "",
        f"_Generated: {time.strftime('%Y-%m-%d %H:%M %Z')}_",
        f"_Corpus size: {CORPUS_SIZE} per benchmark, seed={SEED}_",
        "",
        "## 1. Aggregate MAP@10 by dataset",
        "",
        "| Dataset | FSL MAP@10 | Base SigLIP MAP@10 | Delta% | FSL wins? |",
        "|---|---:|---:|---:|---|",
    ]

    for ds in ["fashion200k", "atlas", "polyvore", "KAGL"]:
        fsl_map = sum(fsl_by_ds[ds]) / max(len(fsl_by_ds[ds]), 1)
        base_map = sum(base_by_ds[ds]) / max(len(base_by_ds[ds]), 1)
        delta_pct = (fsl_map - base_map) / max(base_map, 1e-9) * 100
        wins = "**Yes**" if fsl_map > base_map else "No"
        lines.append(f"| {ds} | {fsl_map:.4f} | {base_map:.4f} | {delta_pct:+.1f}% | {wins} |")

    # Section 2: FSL beats base SigLIP
    lines += [
        "",
        "## 2. Strata where FSL beats base SigLIP by >5% (must close these gaps)",
        "",
        "| Dataset | L1 category | FSL MAP@10 | Base MAP@10 | Delta% | N queries |",
        "|---|---|---:|---:|---:|---:|",
    ]

    fsl_wins = []
    all_keys = sorted(set(list(fsl_agg.keys()) + list(base_agg.keys())))
    for (ds, l1) in all_keys:
        fsl_v = fsl_agg.get((ds, l1), {}).get("mean_ap10", 0)
        base_v = base_agg.get((ds, l1), {}).get("mean_ap10", 0)
        n_q = fsl_agg.get((ds, l1), {}).get("n_queries", 0)
        if base_v > 0 and (fsl_v - base_v) / base_v > 0.05 and n_q >= 5:
            delta = (fsl_v - base_v) / base_v * 100
            fsl_wins.append((ds, l1, fsl_v, base_v, delta, n_q))

    fsl_wins.sort(key=lambda x: -x[4])
    for ds, l1, fsl_v, base_v, delta, n_q in fsl_wins:
        lines.append(f"| {ds} | {l1} | {fsl_v:.4f} | {base_v:.4f} | {delta:+.1f}% | {n_q} |")
    if not fsl_wins:
        lines.append("| _(none)_ | | | | | |")

    # Section 3: Base SigLIP beats FSL
    lines += [
        "",
        "## 3. Strata where base SigLIP beats FSL (must preserve these)",
        "",
        "| Dataset | L1 category | Base MAP@10 | FSL MAP@10 | Delta% | N queries |",
        "|---|---|---:|---:|---:|---:|",
    ]

    base_wins = []
    for (ds, l1) in all_keys:
        fsl_v = fsl_agg.get((ds, l1), {}).get("mean_ap10", 0)
        base_v = base_agg.get((ds, l1), {}).get("mean_ap10", 0)
        n_q = base_agg.get((ds, l1), {}).get("n_queries", 0)
        if fsl_v > 0 and (base_v - fsl_v) / fsl_v > 0.05 and n_q >= 5:
            delta = (base_v - fsl_v) / fsl_v * 100
            base_wins.append((ds, l1, base_v, fsl_v, delta, n_q))

    base_wins.sort(key=lambda x: -x[4])
    for ds, l1, base_v, fsl_v, delta, n_q in base_wins:
        lines.append(f"| {ds} | {l1} | {base_v:.4f} | {fsl_v:.4f} | {delta:+.1f}% | {n_q} |")
    if not base_wins:
        lines.append("| _(none)_ | | | | | |")

    # Section 4: Both weak
    lines += [
        "",
        "## 4. Strata where both models are weak (MAP@10 < 0.3)",
        "",
        "| Dataset | L1 category | FSL MAP@10 | Base MAP@10 | N queries |",
        "|---|---|---:|---:|---:|",
    ]

    both_weak = []
    for (ds, l1) in all_keys:
        fsl_v = fsl_agg.get((ds, l1), {}).get("mean_ap10", 0)
        base_v = base_agg.get((ds, l1), {}).get("mean_ap10", 0)
        n_q = max(
            fsl_agg.get((ds, l1), {}).get("n_queries", 0),
            base_agg.get((ds, l1), {}).get("n_queries", 0),
        )
        if fsl_v < 0.3 and base_v < 0.3 and n_q >= 5:
            both_weak.append((ds, l1, fsl_v, base_v, n_q))

    both_weak.sort(key=lambda x: x[2])
    for ds, l1, fsl_v, base_v, n_q in both_weak:
        lines.append(f"| {ds} | {l1} | {fsl_v:.4f} | {base_v:.4f} | {n_q} |")
    if not both_weak:
        lines.append("| _(none)_ | | | | |")

    # Section 5: Taxonomy distribution
    lines += [
        "",
        "## 5. L1 category distribution across benchmarks (query count)",
        "",
        "| L1 category | fashion200k | atlas | polyvore | KAGL | Total |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    l1_counts = defaultdict(lambda: defaultdict(int))
    for r in fsl_results:
        l1_counts[r["l1_category"]][r["dataset"]] += 1
    for l1 in sorted(l1_counts.keys()):
        f200k = l1_counts[l1].get("fashion200k", 0)
        atlas = l1_counts[l1].get("atlas", 0)
        poly = l1_counts[l1].get("polyvore", 0)
        kagl = l1_counts[l1].get("KAGL", 0)
        total = f200k + atlas + poly + kagl
        lines.append(f"| {l1} | {f200k} | {atlas} | {poly} | {kagl} | {total} |")

    # Section 6: Recommended oversampling
    lines += [
        "",
        "## 6. Recommended oversampling weights for Phase 1",
        "",
        "| L1 category | Avg FSL advantage over base | Recommended weight | Rationale |",
        "|---|---:|---:|---|",
    ]

    l1_gaps = defaultdict(list)
    for (ds, l1) in all_keys:
        fsl_v = fsl_agg.get((ds, l1), {}).get("mean_ap10", 0)
        base_v = base_agg.get((ds, l1), {}).get("mean_ap10", 0)
        if base_v > 0:
            l1_gaps[l1].append((fsl_v - base_v) / base_v)

    for l1 in sorted(l1_gaps.keys()):
        avg_gap = sum(l1_gaps[l1]) / max(len(l1_gaps[l1]), 1) * 100
        if avg_gap > 10:
            weight, rationale = "2.5x", "Large FSL advantage — priority target"
        elif avg_gap > 5:
            weight, rationale = "2.0x", "Moderate FSL advantage — oversample"
        elif avg_gap > 0:
            weight, rationale = "1.5x", "Slight FSL advantage"
        elif avg_gap > -5:
            weight, rationale = "1.0x", "Parity — maintain"
        else:
            weight, rationale = "0.8x", "Base SigLIP already better — light touch"
        lines.append(f"| {l1} | {avg_gap:+.1f}% | {weight} | {rationale} |")

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n  Wrote gap analysis to {path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("Phase 0 — Category Taxonomy Audit")
    print("=" * 70)
    t0 = time.time()

    # Process benchmarks one at a time with two models
    taxonomy_map = {}
    fsl_all_results = []
    base_all_results = []

    # Load FSL first
    model_fsl, preprocess_fsl, tokenizer_fsl = load_fsl()

    for name in BENCHMARKS:
        results, categories = evaluate_one_benchmark(
            model_fsl, preprocess_fsl, tokenizer_fsl, name, "FSL"
        )
        fsl_all_results.extend(results)
        # Save taxonomy map for this benchmark
        cats = set(categories)
        taxonomy_map[name] = {cat: classify_l1(cat) for cat in cats}
        gc.collect()

    del model_fsl
    gc.collect()
    if DEVICE == "mps":
        torch.mps.empty_cache()
    print("\n  FSL evaluation complete. Releasing model.")

    # Load base SigLIP
    model_base, preprocess_base, tokenizer_base = load_base_siglip()

    for name in BENCHMARKS:
        results, _ = evaluate_one_benchmark(
            model_base, preprocess_base, tokenizer_base, name, "Base"
        )
        base_all_results.extend(results)
        gc.collect()

    del model_base
    gc.collect()
    print("\n  Base SigLIP evaluation complete. Releasing model.")

    # Write outputs
    print("\n── Writing outputs ──")

    with open(OUT_DIR / "taxonomy_map.json", "w") as f:
        json.dump(taxonomy_map, f, indent=2)
    print(f"  Saved taxonomy_map.json ({sum(len(v) for v in taxonomy_map.values())} categories)")

    write_csv(fsl_all_results, OUT_DIR / "per_stratum_fsl_performance.csv", "FashionSigLIP")
    write_csv(base_all_results, OUT_DIR / "per_stratum_base_siglip_performance.csv", "BaseSigLIP")

    fsl_agg = aggregate_results(fsl_all_results)
    base_agg = aggregate_results(base_all_results)
    write_gap_analysis(fsl_agg, base_agg, fsl_all_results, base_all_results, OUT_DIR / "gap_analysis.md")

    elapsed = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"Phase 0 complete in {elapsed / 60:.1f} minutes")
    print(f"Results in: {OUT_DIR}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
