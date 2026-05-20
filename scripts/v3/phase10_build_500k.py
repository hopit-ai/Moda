"""Phase 10a — Build 500K training set for multi-field GCL.

Key changes vs phase1_build_dataset.py:
  - Target 500K pairs (up from 132K)
  - Stream full GS-10M in_domain (3.93M rows, no early cap)
  - Soft L1 quotas instead of hard caps (overflow allowed at 1.5x)
  - Oversample dresses 3x (FSL's main advantage area)
  - Full DeepFashion (52K InShop + 42K Multimodal, no cap)
  - Inverse-sqrt score-to-weight (GCL paper best)
  - Extract subcategory3 field for fashion200k-style queries
  - Beauty/home from novel_document integrated directly

No fashion200k / atlas / polyvore / KAGL data used.

Usage:
  python3 -u scripts/v3/phase10_build_500k.py
  python3 -u scripts/v3/phase10_build_500k.py --target-total 500000
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
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
log = logging.getLogger("phase10-500k")

REPO_ROOT = Path(__file__).resolve().parents[2]
HF_CACHE = str(REPO_ROOT / "data" / "hf_cache")
DEFAULT_OUT = REPO_ROOT / "data" / "processed" / "v3_phase10_500k"

# ── L1 taxonomy ──────────────────────────────────────────────────────────────

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
    (re.compile(r"\b(beauty|skincare|makeup|cosmetic|fragrance|perfume|lipstick|mascara|serum|moisturiz|cleanser|foundation|concealer|blush|eyeshadow|eyeliner|nail.?polish)\b", re.I), "beauty"),
    (re.compile(r"\b(home|kitchen|decor|furniture|bedding|towel|curtain|rug|pillow|lamp|candle|vase|storage|organizer|bath.?mat|throw|blanket|cushion)\b", re.I), "home"),
]

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
GARMENT_RE = re.compile(
    r"\b(dress|gown|romper|jumpsuit|playsuit|top|blouse|shirt|tee|t-?shirt|tshirt|tank|camisole|tunic|polo|henley|hoodie|sweatshirt|sweater|pullover|cardigan|pants|trousers|jeans|leggings|chinos|shorts|skirt|culottes|joggers|jacket|blazer|coat|parka|vest|cape|shoe|boot|sneaker|sandal|heel|pump|flat|loafer|slipper|mule|bag|handbag|clutch|tote|backpack|wallet|bikini|swimsuit|bra|underwear|pajama|robe)s?\b",
    re.I,
)
GENDER_RE = re.compile(r"\b(women|woman|men|man|girls?|boys?|kids?|children|unisex|ladies|gents|mens|womens)\b", re.I)

SUBCATEGORY_RE = re.compile(
    r"\b(cocktail|evening|maxi|midi|mini|wrap|a.?line|sheath|shift|fit.?and.?flare|bodycon|slip|sundress|shirt.?dress|sweater.?dress|t.?shirt.?dress|halter|strapless|off.?shoulder|backless|high.?low|asymmetric|tiered|pleated|ruffle|lace|floral|striped|polka.?dot|solid|printed|embroidered|sequin|velvet|satin|silk|chiffon|cotton|linen|denim|leather|knit|crochet|jersey|tulle|organza)\b",
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
    combined = f"{query} {title}"

    colors = list(set(m.lower() for m in COLOR_RE.findall(combined)))
    materials = list(set(m.lower() for m in MATERIAL_RE.findall(combined)))
    styles = list(set(m.lower() for m in STYLE_RE.findall(combined)))
    garments = list(set(m.lower() for m in GARMENT_RE.findall(combined)))
    genders = list(set(m.lower() for m in GENDER_RE.findall(combined)))
    subcats = list(set(m.lower() for m in SUBCATEGORY_RE.findall(combined)))

    gender = "unknown"
    for g in genders:
        if g in ("women", "woman", "ladies", "womens", "girls", "girl"):
            gender = "women"
            break
        if g in ("men", "man", "gents", "mens", "boys", "boy"):
            gender = "men"
            break

    l1 = classify_l1(combined)

    color_str = " ".join(colors[:3]) if colors else ""
    garment_str = garments[0] if garments else ""
    material_str = " ".join(materials[:2]) if materials else ""
    style_str = " ".join(styles[:2]) if styles else ""

    # subcategory3: fine-grained descriptor like "blue knee-length dress"
    subcat_parts = [p for p in [color_str, " ".join(subcats[:2]), garment_str] if p]
    subcategory3 = " ".join(subcat_parts) if subcat_parts else ""

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
        "subcategory3": subcategory3,
        "composite": composite,
        "n_fields_extracted": n_fields,
    }


def inverse_sqrt_weight(score_linear: int, s_max: int = 100) -> float:
    """GCL paper best score-to-weight: w = s_max / sqrt(s_max - s + 1)"""
    return s_max / math.sqrt(s_max - score_linear + 1)


def rank_bucket(position: int) -> str | None:
    if 1 <= position <= 10:
        return "top"
    elif 11 <= position <= 40:
        return "mid"
    elif 41 <= position <= 100:
        return "tail"
    return None


# ── Writer ───────────────────────────────────────────────────────────────────

class MultiFieldWriter:
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
            "l1_category": fields["l1_category"],
            "colors": fields["colors"],
            "materials": fields["materials"],
            "styles": fields["styles"],
            "garment_types": fields["garment_types"],
            "gender": fields["gender"],
            "color_str": fields["color_str"],
            "garment_str": fields["garment_str"],
            "material_str": fields["material_str"],
            "style_str": fields["style_str"],
            "subcategory3": fields["subcategory3"],
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
        }

        with open(self.out_dir / "stats.json", "w") as f:
            json.dump(stats, f, indent=2)

        return stats


# ── GS-10M streaming (no hard quotas) ───────────────────────────────────────

SOFT_CAPS_MULTIPLIER = {
    "dresses": 3.0,
    "tops": 1.5,
    "bags": 1.5,
    "shoes": 1.5,
    "bottoms": 1.5,
    "beauty": 2.0,
    "home": 2.0,
}


def stream_gs10m(
    writer: MultiFieldWriter,
    target_gs: int,
    max_examined: int = 3_900_000,
    seed: int = 42,
) -> None:
    """Stream GS-10M in_domain with soft quotas (no hard L1 caps)."""
    from datasets import load_dataset

    rng = random.Random(seed)
    log.info("Streaming GS-10M in_domain (target=%d, max_examined=%d)", target_gs, max_examined)

    ds = load_dataset("Marqo/marqo-GS-10M", split="in_domain", streaming=True, cache_dir=HF_CACHE)

    l1_written = Counter()
    examined = 0
    skipped_no_image = 0
    skipped_non_fashion = 0
    t0 = time.time()

    # Compute per-L1 soft caps: allow overflow but guide proportions
    base_per_l1 = target_gs // 13  # ~13 L1 categories
    soft_caps = {}
    for l1 in ["dresses", "tops", "bottoms", "shoes", "bags", "accessories",
                "swimwear", "intimates", "outerwear", "activewear", "other",
                "beauty", "home"]:
        mult = SOFT_CAPS_MULTIPLIER.get(l1, 1.0)
        soft_caps[l1] = int(base_per_l1 * mult)
    log.info("Soft caps: %s", {k: v for k, v in sorted(soft_caps.items(), key=lambda x: -x[1])})

    current_query = ""
    current_rows: list[dict] = []

    def flush_query_buffer():
        nonlocal skipped_non_fashion
        if not current_rows:
            return

        sample_title = current_rows[0].get("title", "")
        l1 = classify_l1(f"{current_query} {sample_title}")

        # Soft cap: keep with decreasing probability above cap
        cap = soft_caps.get(l1, base_per_l1)
        current = l1_written[l1]
        if current >= cap:
            overflow_ratio = current / cap
            keep_prob = 1.0 / (overflow_ratio ** 0.5)
            if rng.random() > keep_prob:
                return

        for row in current_rows:
            writer.write(row)
            l1_written[l1] += 1

    for row in ds:
        examined += 1

        query = (row.get("query") or "").strip()
        title = (row.get("title") or "").strip()
        image = row.get("image")
        position = int(row.get("position") or 0)
        bucket = rank_bucket(position)

        if query != current_query:
            flush_query_buffer()
            current_query = query
            current_rows = []

        if not query or image is None or bucket is None:
            skipped_no_image += 1
            continue

        score_linear = row.get("score_linear")
        score_linear = int(score_linear) if score_linear is not None else max(1, 101 - position)
        weight = inverse_sqrt_weight(score_linear) / 100.0  # normalize to ~[0.1, 1.0]

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

        if examined % 100_000 == 0:
            elapsed = time.time() - t0
            total_written = writer.written
            log.info(
                "examined=%d written=%d skip_noimg=%d elapsed=%.0fs",
                examined, total_written, skipped_no_image, elapsed,
            )
            for l1 in sorted(soft_caps.keys()):
                w = l1_written[l1]
                c = soft_caps.get(l1, base_per_l1)
                pct = w / max(c, 1) * 100
                status = f"{w:,}/{c:,} ({pct:.0f}%)"
                log.info("  %-14s %s", l1, status)

        if writer.written >= target_gs:
            log.info("Target reached at examined=%d, written=%d", examined, writer.written)
            break

        if examined >= max_examined:
            break

    flush_query_buffer()

    elapsed = time.time() - t0
    log.info("GS-10M done: examined=%d written=%d in %.1fs", examined, writer.written, elapsed)
    log.info("L1 counts: %s", dict(l1_written.most_common()))


def stream_gs10m_novel(
    writer: MultiFieldWriter,
    target: int = 20_000,
    seed: int = 42,
) -> None:
    """Stream beauty/home from novel_document and novel_query splits."""
    from datasets import load_dataset

    rng = random.Random(seed)
    beauty_home_re = re.compile(
        r"\b(beauty|skincare|makeup|cosmetic|fragrance|perfume|lipstick|mascara|serum|moisturiz|cleanser|foundation|concealer|"
        r"home|kitchen|decor|furniture|bedding|towel|curtain|rug|pillow|lamp|candle|vase|storage|organizer)\b",
        re.I,
    )

    written_start = writer.written
    for split in ("novel_document", "novel_query"):
        log.info("Streaming GS-10M %s for beauty/home...", split)
        ds = load_dataset("Marqo/marqo-GS-10M", split=split, streaming=True, cache_dir=HF_CACHE)

        loaded = 0
        for row in ds:
            if writer.written - written_start >= target:
                break

            query = (row.get("query") or "").strip()
            title = (row.get("title") or "").strip()
            image = row.get("image")
            position = int(row.get("position") or 0)
            bucket = rank_bucket(position)

            if not query or image is None or bucket is None:
                continue

            combined = f"{query} {title}"
            if not beauty_home_re.search(combined):
                continue

            score_linear = row.get("score_linear")
            score_linear = int(score_linear) if score_linear is not None else max(1, 101 - position)
            weight = inverse_sqrt_weight(score_linear) / 100.0

            writer.write({
                "source": f"gs10m-{split}",
                "query": query,
                "title": title,
                "position": position,
                "score_linear": score_linear,
                "rank_bucket": bucket,
                "weight": weight,
                "image": image,
                "product_id": str(row.get("product_id") or ""),
            })
            loaded += 1

            if loaded % 5_000 == 0:
                log.info("  %s beauty/home: loaded=%d", split, loaded)

        log.info("  %s: loaded=%d", split, loaded)

    log.info("Novel splits done: %d beauty/home pairs added", writer.written - written_start)


def stream_deepfashion(
    writer: MultiFieldWriter,
    repo_id: str,
    source_label: str,
    limit: int,
) -> None:
    """Stream full DeepFashion dataset."""
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
        if loaded % 10_000 == 0:
            log.info("  %s: loaded=%d written=%d", source_label, loaded, writer.written)

    log.info("%s done: loaded=%d", source_label, loaded)


# ── Quality validation ───────────────────────────────────────────────────────

def validate_dataset(out_dir: Path) -> dict:
    jsonl_path = out_dir / "pairs.jsonl"
    log.info("Validating %s ...", jsonl_path)

    n_total = 0
    n_with_color = 0
    n_with_material = 0
    n_with_garment = 0
    n_with_subcat3 = 0
    n_with_2plus = 0
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
            if row.get("subcategory3"):
                n_with_subcat3 += 1
            if row.get("n_fields_extracted", 0) >= 2:
                n_with_2plus += 1
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
            "has_subcategory3": round(n_with_subcat3 / max(n_total, 1) * 100, 1),
            ">=2_fields": round(n_with_2plus / max(n_total, 1) * 100, 1),
        },
        "l1_distribution": dict(l1_counts.most_common()),
        "source_distribution": dict(source_counts),
    }

    gates = []
    if n_total < 200_000:
        gates.append(f"WARN: total_pairs={n_total} < 200K target")
    if len(unique_queries) < 3_000:
        gates.append(f"WARN: unique_queries={len(unique_queries)} < 3K target")

    report["gate_checks"] = gates if gates else ["ALL PASSED"]

    with open(out_dir / "quality_report.json", "w") as f:
        json.dump(report, f, indent=2)

    lines = [
        "# Phase 10 — 500K Dataset Quality Report",
        "",
        f"_Generated: {time.strftime('%Y-%m-%d %H:%M %Z')}_",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|---|---:|",
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

    log.info("Quality report written")
    for g in report["gate_checks"]:
        log.info("  Gate: %s", g)
    return report


# ── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--target-total", type=int, default=500_000)
    p.add_argument("--max-examined", type=int, default=3_900_000)
    p.add_argument("--df-inshop", type=int, default=52_000)
    p.add_argument("--df-multimodal", type=int, default=42_000)
    p.add_argument("--novel-beauty-home", type=int, default=20_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    return p.parse_args()


def main():
    args = parse_args()
    t0 = time.time()

    gs_target = args.target_total - args.df_inshop - args.df_multimodal - args.novel_beauty_home

    print("=" * 70)
    print("Phase 10a — Build 500K Multi-Field Training Set")
    print(f"  Target: ~{args.target_total:,} pairs")
    print(f"    GS-10M in_domain: ~{gs_target:,}")
    print(f"    DeepFashion InShop: {args.df_inshop:,}")
    print(f"    DeepFashion Multimodal: {args.df_multimodal:,}")
    print(f"    GS-10M novel (beauty/home): {args.novel_beauty_home:,}")
    print(f"  Output: {args.out_dir}")
    print("=" * 70)

    writer = MultiFieldWriter(args.out_dir)

    try:
        stream_gs10m(writer, gs_target, max_examined=args.max_examined, seed=args.seed)
        stream_gs10m_novel(writer, target=args.novel_beauty_home, seed=args.seed)
        stream_deepfashion(writer, "Marqo/deepfashion-inshop", "df-inshop", args.df_inshop)
        stream_deepfashion(writer, "Marqo/deepfashion-multimodal", "df-multimodal", args.df_multimodal)
    finally:
        stats = writer.close()
        log.info("Final stats: %s", json.dumps(stats, indent=2))

    validate_dataset(args.out_dir)

    elapsed = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"Phase 10a complete in {elapsed / 60:.1f} minutes")
    print(f"  Written: {writer.written:,} pairs")
    print(f"  Output: {args.out_dir}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
