"""Phase 1 supplement — Add beauty and home pairs from GS-10M novel splits.

GS-10M in_domain has zero beauty/home products. But novel_document and
novel_query splits contain broader Google Shopping categories including
cosmetics, skincare, fragrances, furniture, and home decor.

Appends to the existing pairs.jsonl without overwriting.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("supplement-beauty-home")

REPO_ROOT = Path(__file__).resolve().parents[2]
HF_CACHE = str(REPO_ROOT / "data" / "hf_cache")
OUT_DIR = REPO_ROOT / "data" / "processed" / "v3_stratified_multifield"

BEAUTY_RE = re.compile(
    r"\b(makeup|mascara|lipstick|foundation|concealer|eyeliner|eyeshadow|blush|bronzer|primer|fragrance|perfume|cologne|skincare|moisturizer|cleanser|serum|toner|sunscreen|lotion|cream|shampoo|conditioner|hairspray|nail.?polish|beauty|cosmetic|palette|highlighter|setting.?spray|lip.?gloss|lip.?balm|face.?mask|exfoliant|retinol|vitamin.?c|hyaluronic)\b",
    re.I,
)

HOME_RE = re.compile(
    r"\b(furniture|chair|table|lamp|rug|pillow|curtain|bed|sofa|mirror|vase|candle|decor|lighting|shelf|storage|kitchen|dining|drinkware|flatware|serveware|frame|wallpaper|clock|ottoman|dresser|nightstand|bookcase|bench|throw|blanket|cushion|duvet|comforter|mattress|headboard|chandelier|pendant.?light|floor.?lamp|table.?lamp|wall.?art|planter|basket)\b",
    re.I,
)

RANK_BUCKETS = {"top": (1, 10), "mid": (11, 40), "tail": (41, 100)}


def rank_bucket(position: int) -> str | None:
    for name, (lo, hi) in RANK_BUCKETS.items():
        if lo <= position <= hi:
            return name
    return None


def classify_beauty_or_home(query: str, title: str) -> str | None:
    combined = f"{query} {title}"
    if BEAUTY_RE.search(combined):
        return "beauty"
    if HOME_RE.search(combined):
        return "home"
    return None


def extract_fields_inline(query: str, title: str) -> dict:
    """Minimal field extraction for beauty/home items."""
    from scripts.v3.phase1_build_dataset import extract_fields
    return extract_fields(query, title)


def main():
    from datasets import load_dataset

    img_dir = OUT_DIR / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    target_per_category = 5000
    counts = Counter()
    written = 0

    jsonl_path = OUT_DIR / "pairs.jsonl"
    handle = open(jsonl_path, "a")

    try:
        for split in ["novel_document", "novel_query"]:
            if counts["beauty"] >= target_per_category and counts["home"] >= target_per_category:
                break

            log.info("Streaming GS-10M split=%s for beauty/home...", split)
            ds = load_dataset("Marqo/marqo-GS-10M", split=split, streaming=True, cache_dir=HF_CACHE)

            examined = 0
            for row in ds:
                examined += 1

                if counts["beauty"] >= target_per_category and counts["home"] >= target_per_category:
                    break

                query = (row.get("query") or "").strip()
                title = (row.get("title") or "").strip()
                image = row.get("image")
                position = int(row.get("position") or 0)
                bucket = rank_bucket(position)

                if not query or image is None or bucket is None:
                    continue

                cat = classify_beauty_or_home(query, title)
                if cat is None or counts[cat] >= target_per_category:
                    continue

                score_linear = row.get("score_linear")
                score_linear = int(score_linear) if score_linear is not None else None
                weight = max(0.01, min(1.0, score_linear / 100.0)) if score_linear else max(0.01, 1.0 / max(position, 1))

                # Save image
                key = f"gs10m_novel|{query}|{row.get('product_id', '')}|{position}"
                digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:24]
                rel_path = f"images/{digest}.jpg"
                abs_path = OUT_DIR / rel_path
                if not abs_path.exists():
                    try:
                        image.convert("RGB").save(abs_path, "JPEG", quality=90)
                    except Exception:
                        continue

                # Extract fields
                combined = f"{query} {title}"
                COLOR_RE = re.compile(r"\b(red|blue|green|yellow|orange|purple|pink|black|white|grey|gray|brown|beige|navy|gold|silver|ivory|cream|nude|rose|coral)\b", re.I)
                colors = list(set(m.lower() for m in COLOR_RE.findall(combined)))

                out = {
                    "source": f"gs10m-{split}",
                    "query": query,
                    "title": title,
                    "image_path": rel_path,
                    "position": position,
                    "score_linear": score_linear,
                    "rank_bucket": bucket,
                    "weight": weight,
                    "l1_category": cat,
                    "colors": colors,
                    "materials": [],
                    "styles": [],
                    "garment_types": [],
                    "gender": "unknown",
                    "color_str": " ".join(colors[:2]) if colors else "",
                    "garment_str": "",
                    "material_str": "",
                    "style_str": "",
                    "composite": "",
                    "n_fields_extracted": 1 + (len(colors) > 0),
                }

                handle.write(json.dumps(out) + "\n")
                counts[cat] += 1
                written += 1

                if written % 1000 == 0:
                    log.info("  written=%d beauty=%d home=%d examined=%d", written, counts["beauty"], counts["home"], examined)

            log.info("Split %s done: examined=%d written=%d", split, examined, written)

    finally:
        handle.close()

    log.info("Supplement complete: +%d pairs (beauty=%d, home=%d)", written, counts["beauty"], counts["home"])

    # Update stats
    stats_path = OUT_DIR / "stats.json"
    if stats_path.exists():
        with open(stats_path) as f:
            stats = json.load(f)
        stats["total_pairs"] = stats.get("total_pairs", 0) + written
        stats["supplement_beauty_home"] = {"beauty": counts["beauty"], "home": counts["home"], "total_added": written}
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)

    print(f"\nDone: added {written} beauty/home pairs to {jsonl_path}")
    print(f"  beauty: {counts['beauty']}")
    print(f"  home: {counts['home']}")


if __name__ == "__main__":
    main()
