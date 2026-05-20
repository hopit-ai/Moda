"""Phase 1 — Build stratified multi-field training set.

Streams GS-10M + DeepFashion, classifies each row by L1 garment type,
applies oversampling weights from Phase 0 gap analysis, extracts multi-field
labels from titles, and writes to disk incrementally.

Key improvements over build_stratified_fashion_training_set_stream.py:
  - L1-stratified oversampling with per-stratum quotas
  - Multi-field label extraction (color, material, garment_type, style, gender)
  - Per-stratum progress tracking and early termination
  - Quality validation pass at the end

No fashion200k / atlas / polyvore / KAGL data used.

Usage:
  python3 -u scripts/v3/phase1_build_dataset.py
  python3 -u scripts/v3/phase1_build_dataset.py --target-total 200000
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("v3-dataset-builder")

REPO_ROOT = Path(__file__).resolve().parents[2]
HF_CACHE = str(REPO_ROOT / "data" / "hf_cache")
DEFAULT_OUT = REPO_ROOT / "data" / "processed" / "v3_stratified_multifield"

# ── L1 taxonomy (same as phase0_audit.py) ────────────────────────────────────

L1_PATTERNS = [
    (re.compile(r"\b(jackets?|blazers?|coats?|parkas?|anorak|windbreaker|trench|overcoat|capes?|ponchos?|bombers?|vests?|waistcoats?|outerwear|sherwani)\b", re.I), "outerwear"),
    (re.compile(r"(dress|gowns?|rompers?|jumpsuits?|playsuits?|sarees?|kurtas?|kurtis?|kurta.?sets?|galabiyyas?|dhoti)", re.I), "dresses"),
    (re.compile(r"\b(tops?|blouses?|shirts?|tees?|t-?shirts?|tshirts?|tanks?|camisoles?|tunics?|polos?|henleys?|crop.?tops?|halters?|bustiers?|corsets?|hoodies?|sweatshirts?|sweaters?|pullovers?|cardigans?|knits?|chemises?)\b", re.I), "tops"),
    (re.compile(r"\b(pants?|trousers?|jeans?|denim|leggings?|chinos?|shorts?|skirts?|skorts?|culottes?|joggers?|sweatpants?|cargos?|capris?|churidars?|tracksuits?|tights?|hosiery)\b", re.I), "bottoms"),
    (re.compile(r"\b(shoes?|boots?|sneakers?|sandals?|heels?|pumps?|flats?|loafers?|slippers?|mules?|clogs?|espadrilles?|oxfords?|derby|brogues?|stilettos?|wedges?|platforms?|flip.?flops?|booties?|moccasins?|sports?.?shoes?|casual.?shoes?|formal.?shoes?)\b", re.I), "shoes"),
    (re.compile(r"\b(bags?|handbags?|purses?|clutches?|totes?|backpacks?|satchels?|crossbody|messenger|wallets?|pouches?|duffles?|weekenders?|rucksacks?|luggage|shoulder.?bags?)\b", re.I), "bags"),
    (re.compile(r"\b(accessor|jewelry|jewellery|necklaces?|bracelets?|bangles?|earrings?|rings?|watches?|sunglasses|eyeglasses|eyewear|scarves?|scarfs?|belts?|hats?|caps?|beanies?|gloves?|ties?|bow.?ties?|cufflinks?|brooches?|pins?|charms?|pendants?|stoles?|dupattas?|umbrellas?|socks?|earmuffs?)\b", re.I), "accessories"),
    (re.compile(r"\b(swimsuits?|bikinis?|swimwear|bathing|swim|one.?piece.?swim)\b", re.I), "swimwear"),
    (re.compile(r"\b(lingerie|bras?|underwear|panty|panties|briefs?|boxers?|nightgowns?|pajamas?|pyjamas?|robes?|sleepwear|lounge|intimates?|shapewear)\b", re.I), "intimates"),
    (re.compile(r"\b(activewear|sportswear|athletic|yoga|gym|workout|running|cycling|fitness|track|sports?.?sandals?)\b", re.I), "activewear"),
]

# ── Multi-field extractors ───────────────────────────────────────────────────

COLOR_RE = re.compile(
    r"\b(red|blue|green|yellow|orange|purple|pink|black|white|grey|gray|brown|beige|navy|teal|turquoise|coral|maroon|burgundy|ivory|cream|gold|silver|olive|tan|rust|lavender|mint|peach|magenta|fuchsia|khaki|charcoal|indigo|cyan|camel|wine|aqua|rose|nude|blush|taupe|mauve|plum|sage|terracotta|mustard|cobalt|cerulean|emerald|ruby|sapphire|amber|copper|bronze|champagne|lemon|lime|scarlet|crimson|slate|pewter|titanium)\b",
    re.I,
)
MATERIAL_RE = re.compile(
    r"\b(cotton|silk|satin|linen|wool|cashmere|polyester|nylon|spandex|rayon|velvet|chiffon|lace|leather|suede|denim|tweed|corduroy|fleece|jersey|mesh|organza|tulle|sequin|knit|crochet|woven|embroidered|faux.?fur|faux.?leather|shearling|canvas|chambray|poplin|crepe|georgette|taffeta|brocade|jacquard|ponte|scuba|lyocell|tencel|bamboo|modal|microfiber|acrylic|mohair|angora|alpaca|hemp|jute|vinyl|latex|rubber|patent|neoprene|terrycloth|terry|seersucker|batiste|voile|lawn|percale|sateen|gabardine|herringbone|flannel|chenille|damask|muslin|oxford)\b",
    re.I,
)
STYLE_RE = re.compile(
    r"\b(casual|formal|elegant|modern|classic|vintage|retro|bohemian|boho|minimal|sporty|preppy|chic|trendy|romantic|edgy|slim|fitted|relaxed|oversized|loose|tailored|structured|flowy|wrap|a.?line|pencil|maxi|midi|mini|cropped|high.?waisted|ribbed|quilted|ruffle|pleated|tiered|asymmetric|draped|fringe|button.?down|zip.?up|pull.?over|off.?shoulder|strapless|backless|halter|v.?neck|crew.?neck|turtleneck|mock.?neck|scoop|cowl|boat.?neck|square.?neck|sleeveless|short.?sleeve|long.?sleeve|three.?quarter|cap.?sleeve|bell.?sleeve|puff.?sleeve|bishop.?sleeve|dolman|raglan|belted|drawstring|elastic|smocked|gathered|shirred|pintuck|embellished|studded|beaded|rhinestone|sequined|metallic|sheer|opaque|matte|glossy|distressed|raw.?hem|frayed|patchwork|color.?block|ombre|tie.?dye|acid.?wash|stone.?wash)\b",
    re.I,
)
GENDER_RE = re.compile(r"\b(women|woman|men|man|girls?|boys?|kids?|children|unisex|ladies|gents|mens|womens)\b", re.I)

GARMENT_RE = re.compile(
    r"\b(dress|gown|romper|jumpsuit|playsuit|top|blouse|shirt|tee|t-?shirt|tshirt|tank|camisole|tunic|polo|henley|hoodie|sweatshirt|sweater|pullover|cardigan|pants|trousers|jeans|leggings|chinos|shorts|skirt|culottes|joggers|jacket|blazer|coat|parka|vest|cape|shoe|boot|sneaker|sandal|heel|pump|flat|loafer|slipper|mule|bag|handbag|clutch|tote|backpack|wallet|bikini|swimsuit|bra|underwear|pajama|robe)s?\b",
    re.I,
)


def classify_l1(text: str) -> str:
    if not text:
        return "other"
    for pat, label in L1_PATTERNS:
        if pat.search(text):
            return label
    return "other"


def extract_fields(query: str, title: str) -> dict[str, Any]:
    """Extract multi-field labels from query + title text."""
    combined = f"{query} {title}"

    colors = list(set(m.lower() for m in COLOR_RE.findall(combined)))
    materials = list(set(m.lower() for m in MATERIAL_RE.findall(combined)))
    styles = list(set(m.lower() for m in STYLE_RE.findall(combined)))
    garments = list(set(m.lower() for m in GARMENT_RE.findall(combined)))
    genders = list(set(m.lower() for m in GENDER_RE.findall(combined)))

    # Canonical gender
    gender = "unknown"
    for g in genders:
        if g in ("women", "woman", "ladies", "womens", "girls", "girl"):
            gender = "women"
            break
        if g in ("men", "man", "gents", "mens", "boys", "boy"):
            gender = "men"
            break

    l1 = classify_l1(combined)

    # Build composite text fields for multi-field training
    color_str = " ".join(colors[:2]) if colors else ""
    garment_str = garments[0] if garments else ""
    material_str = materials[0] if materials else ""
    style_str = " ".join(styles[:2]) if styles else ""

    # Composite: "navy blue cotton dress" — the kind of query benchmarks test
    composite_parts = [p for p in [color_str, material_str, garment_str] if p]
    composite = " ".join(composite_parts)

    n_fields = sum([
        len(colors) > 0,
        len(materials) > 0,
        len(styles) > 0,
        len(garments) > 0,
        gender != "unknown",
    ])

    return {
        "l1_category": l1,
        "colors": colors,
        "materials": materials,
        "styles": styles,
        "garment_types": garments,
        "gender": gender,
        "color_str": color_str,
        "garment_str": garment_str,
        "material_str": material_str,
        "style_str": style_str,
        "composite": composite,
        "n_fields_extracted": n_fields,
    }


# ── Oversampling config (from Phase 0 gap analysis) ─────────────────────────

OVERSAMPLE_WEIGHTS = {
    "bags": 2.5,
    "tops": 2.5,
    "dresses": 2.5,
    "shoes": 2.0,
    "bottoms": 2.0,
    "swimwear": 1.5,
    "intimates": 1.0,
    "accessories": 1.0,
    "outerwear": 0.8,
    "activewear": 1.5,
    "other": 1.5,
    "beauty": 2.5,
    "home": 2.5,
}

RANK_BUCKETS = {
    "top": (1, 10),
    "mid": (11, 40),
    "tail": (41, 100),
}


def rank_bucket(position: int) -> str | None:
    for name, (lo, hi) in RANK_BUCKETS.items():
        if lo <= position <= hi:
            return name
    return None


# ── Streaming writer ─────────────────────────────────────────────────────────

class MultiFieldWriter:
    """Writes multi-field training pairs incrementally to disk."""

    def __init__(self, out_dir: Path) -> None:
        self.out_dir = out_dir
        self.img_dir = out_dir / "images"
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.img_dir.mkdir(parents=True, exist_ok=True)

        self.jsonl_path = out_dir / "pairs.jsonl"
        self.tmp_path = self.jsonl_path.with_suffix(".jsonl.tmp")
        self.handle = self.tmp_path.open("w")

        self.written = 0
        self.source_counts: Counter[str] = Counter()
        self.l1_counts: Counter[str] = Counter()
        self.bucket_counts: Counter[str] = Counter()
        self.field_coverage: Counter[str] = Counter()

    def save_image(self, image: Any, key: str) -> str | None:
        try:
            rgb = image.convert("RGB")
        except Exception:
            return None
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:24]
        rel_path = f"images/{digest}.jpg"
        abs_path = self.out_dir / rel_path
        if not abs_path.exists():
            rgb.save(abs_path, "JPEG", quality=90)
        return rel_path

    def write(self, rec: dict[str, Any]) -> None:
        key = "|".join([
            rec.get("source", ""),
            rec.get("query", ""),
            str(rec.get("product_id", "")),
            str(rec.get("position", "")),
        ])
        image_path = self.save_image(rec["image"], key)
        if image_path is None:
            return

        fields = extract_fields(rec["query"], rec.get("title", ""))

        out = {
            "source": rec["source"],
            "query": rec["query"],
            "title": rec.get("title", ""),
            "image_path": image_path,
            "position": rec["position"],
            "score_linear": rec["score_linear"],
            "rank_bucket": rec["rank_bucket"],
            "weight": rec["weight"],
            # Multi-field labels
            "l1_category": fields["l1_category"],
            "colors": fields["colors"],
            "materials": fields["materials"],
            "styles": fields["styles"],
            "garment_types": fields["garment_types"],
            "gender": fields["gender"],
            # Pre-built text fields for training
            "color_str": fields["color_str"],
            "garment_str": fields["garment_str"],
            "material_str": fields["material_str"],
            "style_str": fields["style_str"],
            "composite": fields["composite"],
            "n_fields_extracted": fields["n_fields_extracted"],
        }

        self.handle.write(json.dumps(out) + "\n")
        self.written += 1
        self.source_counts[rec["source"]] += 1
        self.l1_counts[fields["l1_category"]] += 1
        self.bucket_counts[rec["rank_bucket"]] += 1
        if fields["n_fields_extracted"] >= 2:
            self.field_coverage[">=2_fields"] += 1
        if fields["n_fields_extracted"] >= 3:
            self.field_coverage[">=3_fields"] += 1

    def close(self) -> dict:
        self.handle.close()
        os.replace(self.tmp_path, self.jsonl_path)

        imgs = list(self.img_dir.glob("*.jpg"))
        images_mb = sum(p.stat().st_size for p in imgs) / 1e6

        stats = {
            "total_pairs": self.written,
            "unique_images": len(imgs),
            "images_mb": round(images_mb, 1),
            "sources": dict(self.source_counts),
            "l1_categories": dict(self.l1_counts),
            "rank_buckets": dict(self.bucket_counts),
            "field_coverage": dict(self.field_coverage),
            "field_coverage_pct": {
                ">=2_fields": round(self.field_coverage[">=2_fields"] / max(self.written, 1) * 100, 1),
                ">=3_fields": round(self.field_coverage[">=3_fields"] / max(self.written, 1) * 100, 1),
            },
        }

        with open(self.out_dir / "stats.json", "w") as f:
            json.dump(stats, f, indent=2)

        return stats


# ── GS-10M streaming with L1-stratified quotas ──────────────────────────────

def compute_l1_quotas(target_total: int, gs_fraction: float = 0.85) -> dict[str, int]:
    """Compute per-L1 target counts using oversampling weights."""
    gs_target = int(target_total * gs_fraction)

    raw_weights = {}
    for l1, w in OVERSAMPLE_WEIGHTS.items():
        raw_weights[l1] = w

    total_weight = sum(raw_weights.values())
    quotas = {}
    for l1, w in raw_weights.items():
        quotas[l1] = max(200, int(gs_target * w / total_weight))

    # Scale to exactly hit gs_target
    current = sum(quotas.values())
    scale = gs_target / max(current, 1)
    quotas = {l1: max(200, int(v * scale)) for l1, v in quotas.items()}

    return quotas


def stream_gs10m(
    writer: MultiFieldWriter,
    quotas: dict[str, int],
    max_examined: int = 1_500_000,
    seed: int = 42,
) -> None:
    """Stream GS-10M in_domain, applying L1-stratified sampling with quotas."""
    from datasets import load_dataset

    rng = random.Random(seed)
    log.info("Streaming GS-10M in_domain (max_examined=%d)", max_examined)
    log.info("L1 quotas: %s", {k: v for k, v in sorted(quotas.items(), key=lambda x: -x[1])})

    ds = load_dataset("Marqo/marqo-GS-10M", split="in_domain", streaming=True, cache_dir=HF_CACHE)

    l1_written = Counter()
    l1_full = set()
    examined = 0
    skipped_non_fashion = 0
    skipped_full_stratum = 0
    skipped_no_image = 0
    t0 = time.time()

    # Buffer for per-query grouping (GS-10M rows are grouped by query)
    current_query = ""
    current_rows: list[dict] = []

    def flush_query_buffer():
        nonlocal skipped_full_stratum
        if not current_rows:
            return

        # Classify the query by L1
        sample_title = current_rows[0].get("title", "")
        l1 = classify_l1(f"{current_query} {sample_title}")

        # Check if this stratum is full
        if l1 in l1_full:
            skipped_full_stratum += len(current_rows)
            return

        quota = quotas.get(l1, quotas.get("other", 500))
        remaining = quota - l1_written[l1]
        if remaining <= 0:
            l1_full.add(l1)
            skipped_full_stratum += len(current_rows)
            return

        # Stratified sampling within the query: keep top/mid/tail balanced
        by_bucket: dict[str, list] = defaultdict(list)
        for row in current_rows:
            by_bucket[row["rank_bucket"]].append(row)

        # Take up to per_bucket from each, but respect remaining quota
        per_bucket = max(1, remaining // 3)
        selected = []
        for bucket in ("top", "mid", "tail"):
            rows = by_bucket.get(bucket, [])
            if len(rows) > per_bucket:
                rows = rng.sample(rows, per_bucket)
            selected.extend(rows)

        if len(selected) > remaining:
            selected = rng.sample(selected, remaining)

        for row in selected:
            writer.write(row)
            l1_written[l1] += 1

        if l1_written[l1] >= quota:
            l1_full.add(l1)

    for row in ds:
        examined += 1

        query = (row.get("query") or "").strip()
        title = (row.get("title") or "").strip()
        image = row.get("image")
        position = int(row.get("position") or 0)
        bucket = rank_bucket(position)

        # Query boundary — flush previous
        if query != current_query:
            flush_query_buffer()
            current_query = query
            current_rows = []

        if not query or image is None or bucket is None:
            skipped_no_image += 1
            continue

        score_linear = row.get("score_linear")
        score_linear = int(score_linear) if score_linear is not None else None
        weight = max(0.01, min(1.0, score_linear / 100.0)) if score_linear else max(0.01, 1.0 / max(position, 1))

        current_rows.append({
            "source": "gs10m",
            "query": query,
            "title": title,
            "position": position,
            "score_linear": score_linear,
            "rank_bucket": bucket,
            "weight": weight,
            "image": image,
            "product_id": str(row.get("product_id") or ""),
        })

        # Progress logging
        if examined % 50_000 == 0:
            elapsed = time.time() - t0
            total_written = writer.written
            pct_full = len(l1_full) / max(len(quotas), 1) * 100
            log.info(
                "examined=%d written=%d l1_full=%d/%d (%.0f%%) skip_noimg=%d skip_full=%d elapsed=%.0fs",
                examined, total_written, len(l1_full), len(quotas), pct_full,
                skipped_no_image, skipped_full_stratum, elapsed,
            )
            # Print per-L1 progress
            for l1 in sorted(quotas.keys()):
                q = quotas.get(l1, 0)
                w = l1_written[l1]
                bar = "█" * int(w / max(q, 1) * 20) + "░" * (20 - int(w / max(q, 1) * 20))
                status = "FULL" if l1 in l1_full else f"{w}/{q}"
                log.info("  %-14s %s %s", l1, bar, status)

        # Early termination: all strata full
        if len(l1_full) >= len(quotas):
            log.info("All L1 strata full — stopping early at examined=%d", examined)
            break

        if examined >= max_examined:
            break

    # Flush last query
    flush_query_buffer()

    elapsed = time.time() - t0
    log.info(
        "GS-10M done: examined=%d written=%d in %.1fs",
        examined, writer.written, elapsed,
    )
    log.info("L1 counts: %s", dict(l1_written))
    log.info("Strata not fully filled: %s", {
        l1: f"{l1_written[l1]}/{quotas[l1]}"
        for l1 in quotas if l1 not in l1_full
    })


# ── DeepFashion streaming ───────────────────────────────────────────────────

def stream_deepfashion(
    writer: MultiFieldWriter,
    repo_id: str,
    source_label: str,
    limit: int,
) -> None:
    """Stream DeepFashion dataset with structured attribute extraction."""
    if limit <= 0:
        return
    from datasets import load_dataset

    log.info("Streaming %s (limit=%d)", repo_id, limit)
    ds = load_dataset(repo_id, split="data", streaming=True, cache_dir=HF_CACHE)

    loaded = 0
    for row in ds:
        if loaded >= limit:
            break
        image = row.get("image")
        if image is None:
            continue

        # Build query from structured fields
        parts = []
        color = (row.get("color") or "").strip()
        cat1 = (row.get("category1") or "").strip()
        cat2 = (row.get("category2") or "").strip()
        cat3 = (row.get("category3") or "").strip()
        text = (row.get("text") or "").strip()

        if color and color.lower() != "none":
            parts.append(color)
        if cat3 and cat3.lower() != "none":
            parts.append(cat3)
        elif cat2 and cat2.lower() != "none":
            parts.append(cat2)
        elif cat1 and cat1.lower() != "none":
            parts.append(cat1)

        query = " ".join(parts) if parts else text[:120]
        if not query:
            continue

        title = text[:200] if text else query

        writer.write({
            "source": source_label,
            "query": query,
            "title": title,
            "position": 1,
            "score_linear": 100,
            "rank_bucket": "exact_match",
            "weight": 1.0,
            "image": image,
            "product_id": str(row.get("item_ID") or ""),
        })

        loaded += 1
        if loaded % 5_000 == 0:
            log.info("  %s: loaded=%d written=%d", source_label, loaded, writer.written)

    log.info("%s done: loaded=%d", source_label, loaded)


# ── Quality validation ───────────────────────────────────────────────────────

def validate_dataset(out_dir: Path) -> dict:
    """Post-build quality checks."""
    jsonl_path = out_dir / "pairs.jsonl"
    log.info("Validating %s ...", jsonl_path)

    n_total = 0
    n_with_color = 0
    n_with_material = 0
    n_with_garment = 0
    n_with_2plus = 0
    n_with_composite = 0
    l1_counts = Counter()
    source_counts = Counter()
    unique_queries = set()
    unique_images = set()
    score_sum = 0.0
    score_n = 0

    with open(jsonl_path) as f:
        for line in f:
            row = json.loads(line)
            n_total += 1
            if row.get("colors"):
                n_with_color += 1
            if row.get("materials"):
                n_with_material += 1
            if row.get("garment_types"):
                n_with_garment += 1
            if row.get("n_fields_extracted", 0) >= 2:
                n_with_2plus += 1
            if row.get("composite"):
                n_with_composite += 1
            l1_counts[row.get("l1_category", "unknown")] += 1
            source_counts[row.get("source", "unknown")] += 1
            unique_queries.add(row.get("query", ""))
            unique_images.add(row.get("image_path", ""))
            if row.get("score_linear") is not None:
                score_sum += row["score_linear"]
                score_n += 1

    report = {
        "total_pairs": n_total,
        "unique_queries": len(unique_queries),
        "unique_images": len(unique_images),
        "avg_score_linear": round(score_sum / max(score_n, 1), 1),
        "field_coverage": {
            "has_color": round(n_with_color / max(n_total, 1) * 100, 1),
            "has_material": round(n_with_material / max(n_total, 1) * 100, 1),
            "has_garment_type": round(n_with_garment / max(n_total, 1) * 100, 1),
            ">=2_fields": round(n_with_2plus / max(n_total, 1) * 100, 1),
            "has_composite": round(n_with_composite / max(n_total, 1) * 100, 1),
        },
        "l1_distribution": dict(l1_counts.most_common()),
        "source_distribution": dict(source_counts),
    }

    # Gate checks
    gates = []
    if n_total < 50_000:
        gates.append(f"FAIL: total_pairs={n_total} < 50K minimum")
    if len(unique_queries) < 1_000:
        gates.append(f"FAIL: unique_queries={len(unique_queries)} < 1K minimum")
    if n_with_2plus / max(n_total, 1) < 0.5:
        gates.append(f"FAIL: multi-field coverage {n_with_2plus / max(n_total, 1) * 100:.1f}% < 50% minimum")
    if len(l1_counts) < 5:
        gates.append(f"FAIL: only {len(l1_counts)} L1 categories < 5 minimum")

    report["gate_checks"] = gates if gates else ["ALL PASSED"]

    with open(out_dir / "quality_report.json", "w") as f:
        json.dump(report, f, indent=2)

    # Markdown report
    lines = [
        "# Phase 1 — Dataset Quality Report",
        "",
        f"_Generated: {time.strftime('%Y-%m-%d %H:%M %Z')}_",
        "",
        f"## Summary",
        "",
        f"| Metric | Value |",
        f"|---|---:|",
        f"| Total pairs | {n_total:,} |",
        f"| Unique queries | {len(unique_queries):,} |",
        f"| Unique images | {len(unique_images):,} |",
        f"| Avg score_linear | {report['avg_score_linear']} |",
        "",
        "## Multi-field coverage",
        "",
        "| Field | Coverage % |",
        "|---|---:|",
    ]
    for field, pct in report["field_coverage"].items():
        lines.append(f"| {field} | {pct}% |")

    lines += [
        "",
        "## L1 category distribution",
        "",
        "| L1 category | Count | % |",
        "|---|---:|---:|",
    ]
    for l1, count in l1_counts.most_common():
        lines.append(f"| {l1} | {count:,} | {count / max(n_total, 1) * 100:.1f}% |")

    lines += [
        "",
        "## Source distribution",
        "",
        "| Source | Count | % |",
        "|---|---:|---:|",
    ]
    for src, count in source_counts.most_common():
        lines.append(f"| {src} | {count:,} | {count / max(n_total, 1) * 100:.1f}% |")

    lines += ["", "## Gate checks", ""]
    for g in report["gate_checks"]:
        lines.append(f"- {g}")

    with open(out_dir / "quality_report.md", "w") as f:
        f.write("\n".join(lines) + "\n")

    log.info("Quality report: %s", json.dumps(report, indent=2))
    return report


# ── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--target-total", type=int, default=150_000)
    p.add_argument("--max-examined", type=int, default=2_000_000)
    p.add_argument("--df-inshop", type=int, default=12_000)
    p.add_argument("--df-multimodal", type=int, default=10_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    return p.parse_args()


def main():
    args = parse_args()
    t0 = time.time()

    print("=" * 70)
    print("Phase 1 — Build Stratified Multi-Field Training Set")
    print(f"  Target: ~{args.target_total:,} pairs")
    print(f"  Output: {args.out_dir}")
    print("=" * 70)

    # Compute per-L1 quotas
    quotas = compute_l1_quotas(args.target_total, gs_fraction=0.85)
    gs_target = sum(quotas.values())
    df_target = args.df_inshop + args.df_multimodal
    log.info("GS-10M target: %d pairs across %d L1 strata", gs_target, len(quotas))
    log.info("DeepFashion target: %d pairs", df_target)
    log.info("Total target: %d pairs", gs_target + df_target)

    writer = MultiFieldWriter(args.out_dir)

    try:
        # Phase 1a: Stream GS-10M with L1-stratified quotas
        stream_gs10m(writer, quotas, max_examined=args.max_examined, seed=args.seed)

        # Phase 1b: Stream DeepFashion
        stream_deepfashion(writer, "Marqo/deepfashion-inshop", "df-inshop", args.df_inshop)
        stream_deepfashion(writer, "Marqo/deepfashion-multimodal", "df-multimodal", args.df_multimodal)

    finally:
        stats = writer.close()
        log.info("Final stats: %s", json.dumps(stats, indent=2))

    # Phase 1c: Validate
    validate_dataset(args.out_dir)

    elapsed = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"Phase 1 complete in {elapsed / 60:.1f} minutes")
    print(f"  Written: {writer.written:,} pairs")
    print(f"  Output: {args.out_dir}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
