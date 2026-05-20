"""Phase 10 — Synthetic Data Generation for FSL Gap-Filling.

Generates targeted synthetic training data using Gemini 3 Flash via PaleBlueDot.
Three types:
  1. Query expansions: natural-language variants of category3 queries FSL fails on
  2. Hard-negative contrastive descriptions: distinguish confusing sibling categories  
  3. Long-tail category descriptions: fill underrepresented categories

Uses gap_targets.json from FSL error analysis if available, otherwise generates
for ALL fashion200k category3 queries to maximize coverage.

Usage:
  python3 -u scripts/v3/phase10_generate_synthetic.py
  python3 -u scripts/v3/phase10_generate_synthetic.py --max-queries 100 --dry-run
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("synth-gen")

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = REPO_ROOT / "data" / "synthetic" / "phase10_gap_fill"
GAP_TARGETS_PATH = REPO_ROOT / "results" / "fsl_error_analysis" / "gap_targets.json"
ENV_PATH = REPO_ROOT / ".env"

FASHION200K_CATEGORIES = {
    "dresses": [
        "dresses/casual-and-day/maxi", "dresses/casual-and-day/midi",
        "dresses/casual-and-day/mini", "dresses/casual-and-day/shift",
        "dresses/cocktail-and-party/maxi", "dresses/cocktail-and-party/midi",
        "dresses/cocktail-and-party/mini", "dresses/cocktail-and-party/bodycon",
        "dresses/evening-and-formal/gown", "dresses/evening-and-formal/maxi",
        "dresses/work-and-office/shift", "dresses/work-and-office/sheath",
        "dresses/work-and-office/wrap", "dresses/summer/sundress",
        "dresses/summer/maxi", "dresses/wedding-guest/midi",
    ],
    "tops": [
        "tops/blouses/silk", "tops/blouses/chiffon", "tops/blouses/cotton",
        "tops/t-shirts/graphic", "tops/t-shirts/plain", "tops/t-shirts/crop",
        "tops/sweaters/pullover", "tops/sweaters/cardigan", "tops/sweaters/turtleneck",
        "tops/tanks/camisole", "tops/tanks/halter",
        "tops/bodysuits/long-sleeve", "tops/bodysuits/sleeveless",
    ],
    "bottoms": [
        "bottoms/jeans/skinny", "bottoms/jeans/wide-leg", "bottoms/jeans/bootcut",
        "bottoms/jeans/straight", "bottoms/jeans/mom",
        "bottoms/pants/tailored", "bottoms/pants/cargo", "bottoms/pants/wide-leg",
        "bottoms/skirts/mini", "bottoms/skirts/midi", "bottoms/skirts/maxi",
        "bottoms/skirts/pleated", "bottoms/shorts/denim", "bottoms/shorts/tailored",
    ],
    "outerwear": [
        "outerwear/jackets/denim", "outerwear/jackets/leather",
        "outerwear/jackets/bomber", "outerwear/jackets/blazer",
        "outerwear/coats/trench", "outerwear/coats/wool", "outerwear/coats/puffer",
        "outerwear/coats/peacoat", "outerwear/vests/puffer", "outerwear/vests/quilted",
    ],
    "shoes": [
        "shoes/heels/stiletto", "shoes/heels/block", "shoes/heels/platform",
        "shoes/flats/ballet", "shoes/flats/loafer", "shoes/flats/mule",
        "shoes/boots/ankle", "shoes/boots/knee-high", "shoes/boots/combat",
        "shoes/sneakers/low-top", "shoes/sneakers/high-top",
        "shoes/sandals/flat", "shoes/sandals/heeled", "shoes/sandals/platform",
    ],
    "bags": [
        "bags/handbags/tote", "bags/handbags/crossbody", "bags/handbags/clutch",
        "bags/handbags/shoulder", "bags/handbags/satchel",
        "bags/backpacks/mini", "bags/backpacks/canvas",
    ],
    "accessories": [
        "accessories/jewelry/necklace", "accessories/jewelry/earrings",
        "accessories/jewelry/bracelet", "accessories/jewelry/ring",
        "accessories/scarves/silk", "accessories/scarves/wool",
        "accessories/hats/fedora", "accessories/hats/beanie",
        "accessories/sunglasses/aviator", "accessories/sunglasses/cat-eye",
        "accessories/belts/leather", "accessories/belts/chain",
    ],
    "swimwear": [
        "swimwear/bikini/triangle", "swimwear/bikini/bandeau",
        "swimwear/one-piece/cutout", "swimwear/one-piece/classic",
        "swimwear/coverups/kaftan", "swimwear/coverups/sarong",
    ],
    "activewear": [
        "activewear/leggings/high-waist", "activewear/leggings/capri",
        "activewear/sports-bras/racerback", "activewear/sports-bras/longline",
        "activewear/tops/tank", "activewear/tops/crop",
        "activewear/shorts/biker", "activewear/shorts/running",
    ],
    "intimates": [
        "intimates/bras/push-up", "intimates/bras/bralette",
        "intimates/lingerie/bodysuit", "intimates/lingerie/chemise",
        "intimates/sleepwear/pajamas", "intimates/sleepwear/nightgown",
    ],
}


QUERY_EXPANSION_PROMPT = """You are a fashion search query generator. Given a fashion product category path, generate {n_variants} diverse, realistic search queries that a shopper would type to find products in this exact category.

Category: {category3}

Requirements:
- Each query should be 3-8 words
- Vary: specificity (broad vs detailed), attributes (color, fabric, occasion, style)
- Include natural language (not just the category path)
- Make them discriminative (someone searching THIS wouldn't want items from sibling categories)
- Mix: some with colors, some with materials, some with occasions, some pure style

Output ONLY the queries, one per line, no numbering or bullets."""

CONTRASTIVE_PROMPT = """You are a fashion expert. Given two SIBLING categories that are often confused, generate {n_pairs} descriptions that CLEARLY belong to category A and NOT category B.

Category A: {cat_a}
Category B: {cat_b}

For each description:
- 8-15 words describing a product that is UNAMBIGUOUSLY category A
- Emphasize the distinguishing feature (length, silhouette, structure, etc.)
- These will be used to train a model to tell A from B apart

Output ONLY the descriptions for category A, one per line, no numbering."""

LONGTAIL_PROMPT = """Generate {n_items} detailed fashion product descriptions for items in the category: {category3}

Each description should:
- Be 15-30 words
- Include: silhouette/style, primary material/fabric, color, one distinguishing detail
- Sound like a real product listing title
- Be diverse (different colors, materials, sub-styles)

Output ONLY the descriptions, one per line, no numbering."""


def get_api_client():
    """Get PaleBlueDot OpenAI client."""
    from openai import OpenAI

    api_key = None
    if ENV_PATH.exists():
        for line in ENV_PATH.read_text().strip().split("\n"):
            if line.startswith("PALEBLUEDOT_API_KEY="):
                api_key = line.split("=", 1)[1].strip()

    if not api_key:
        api_key = os.environ.get("PALEBLUEDOT_API_KEY")

    if not api_key:
        raise ValueError("No PALEBLUEDOT_API_KEY found in .env or environment")

    return OpenAI(api_key=api_key, base_url="https://open.palebluedot.ai/v1")


def call_llm(client, prompt: str, max_retries: int = 3) -> str:
    """Call Gemini 3 Flash with retry logic."""
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model="google/gemini-3-flash-preview",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.9,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 2 ** attempt
                log.warning("LLM call failed (attempt %d): %s. Retrying in %ds...", attempt + 1, e, wait)
                time.sleep(wait)
            else:
                log.error("LLM call failed after %d attempts: %s", max_retries, e)
                return ""


def get_sibling_categories(category3: str, all_categories: list[str]) -> list[str]:
    """Find sibling categories (same parent, different leaf)."""
    parts = category3.rsplit("/", 1)
    if len(parts) < 2:
        return []
    parent = parts[0]
    siblings = [c for c in all_categories if c.startswith(parent + "/") and c != category3]
    return siblings


def generate_query_expansions(
    client, categories: list[str], n_variants: int = 8, max_queries: int = None
) -> list[dict]:
    """Generate Type 1: query expansion variants."""
    results = []
    cats_to_process = categories[:max_queries] if max_queries else categories

    log.info("Generating query expansions for %d categories...", len(cats_to_process))

    for i, cat in enumerate(cats_to_process):
        prompt = QUERY_EXPANSION_PROMPT.format(category3=cat, n_variants=n_variants)
        response = call_llm(client, prompt)

        if response:
            queries = [q.strip() for q in response.split("\n") if q.strip()]
            for q in queries:
                results.append({
                    "type": "query_expansion",
                    "original_category3": cat,
                    "synthetic_query": q,
                    "category1": cat.split("/")[0],
                })

        if (i + 1) % 20 == 0:
            log.info("  Expansions: %d/%d categories done, %d queries generated",
                     i + 1, len(cats_to_process), len(results))
        time.sleep(0.1)  # Rate limiting

    return results


def generate_contrastive_pairs(
    client, categories: list[str], n_pairs: int = 5, max_pairs: int = None
) -> list[dict]:
    """Generate Type 2: hard-negative contrastive descriptions."""
    results = []
    pairs_done = 0

    all_sibling_pairs = []
    for cat in categories:
        siblings = get_sibling_categories(cat, categories)
        for sib in siblings:
            if (sib, cat) not in [(p["cat_a"], p["cat_b"]) for p in all_sibling_pairs]:
                all_sibling_pairs.append({"cat_a": cat, "cat_b": sib})

    random.shuffle(all_sibling_pairs)
    pairs_to_process = all_sibling_pairs[:max_pairs] if max_pairs else all_sibling_pairs

    log.info("Generating contrastive pairs for %d sibling pairs...", len(pairs_to_process))

    for i, pair in enumerate(pairs_to_process):
        prompt = CONTRASTIVE_PROMPT.format(
            cat_a=pair["cat_a"], cat_b=pair["cat_b"], n_pairs=n_pairs
        )
        response = call_llm(client, prompt)

        if response:
            descriptions = [d.strip() for d in response.split("\n") if d.strip()]
            for desc in descriptions:
                results.append({
                    "type": "contrastive",
                    "positive_category3": pair["cat_a"],
                    "negative_category3": pair["cat_b"],
                    "synthetic_description": desc,
                    "category1": pair["cat_a"].split("/")[0],
                })

        if (i + 1) % 20 == 0:
            log.info("  Contrastive: %d/%d pairs done, %d descriptions generated",
                     i + 1, len(pairs_to_process), len(results))
        time.sleep(0.1)

    return results


def generate_longtail_descriptions(
    client, categories: list[str], n_items: int = 10, max_categories: int = None
) -> list[dict]:
    """Generate Type 3: long-tail category descriptions."""
    longtail_cats = [c for c in categories
                     if c.split("/")[0] in ("swimwear", "activewear", "intimates", "accessories")]
    random.shuffle(longtail_cats)
    cats_to_process = longtail_cats[:max_categories] if max_categories else longtail_cats

    log.info("Generating long-tail descriptions for %d categories...", len(cats_to_process))
    results = []

    for i, cat in enumerate(cats_to_process):
        prompt = LONGTAIL_PROMPT.format(category3=cat, n_items=n_items)
        response = call_llm(client, prompt)

        if response:
            descriptions = [d.strip() for d in response.split("\n") if d.strip()]
            for desc in descriptions:
                results.append({
                    "type": "longtail",
                    "category3": cat,
                    "synthetic_description": desc,
                    "category1": cat.split("/")[0],
                })

        if (i + 1) % 10 == 0:
            log.info("  Long-tail: %d/%d categories done, %d descriptions",
                     i + 1, len(cats_to_process), len(results))
        time.sleep(0.1)

    return results


def main():
    parser = argparse.ArgumentParser(description="Synthetic data generation for FSL gap-filling")
    parser.add_argument("--max-queries", type=int, default=None,
                        help="Max categories for query expansion (default: all)")
    parser.add_argument("--max-contrastive-pairs", type=int, default=80,
                        help="Max sibling pairs for contrastive generation")
    parser.add_argument("--max-longtail-categories", type=int, default=30,
                        help="Max long-tail categories")
    parser.add_argument("--n-variants", type=int, default=8,
                        help="Query variants per category")
    parser.add_argument("--n-contrastive", type=int, default=5,
                        help="Contrastive descriptions per pair")
    parser.add_argument("--n-longtail", type=int, default=10,
                        help="Descriptions per long-tail category")
    parser.add_argument("--dry-run", action="store_true",
                        help="Only show what would be generated, don't call LLM")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    t0 = time.time()

    # Collect all categories
    all_categories = []
    for cat1, cat3_list in FASHION200K_CATEGORIES.items():
        all_categories.extend(cat3_list)

    log.info("Total fashion200k categories in taxonomy: %d", len(all_categories))

    # If gap targets exist, prioritize those categories
    gap_categories = []
    if GAP_TARGETS_PATH.exists():
        log.info("Loading gap targets from FSL error analysis...")
        with open(GAP_TARGETS_PATH) as f:
            gap_targets = json.load(f)
        gap_categories = [t["category3_full"] for t in gap_targets]
        log.info("  %d gap-target categories loaded", len(gap_categories))
        # Prioritize gap categories, then fill with taxonomy
        priority_cats = gap_categories + [c for c in all_categories if c not in gap_categories]
    else:
        log.info("No gap_targets.json found — using full taxonomy")
        priority_cats = all_categories

    if args.dry_run:
        log.info("DRY RUN — would generate:")
        n_exp = min(len(priority_cats), args.max_queries or len(priority_cats))
        log.info("  Type 1 (query expansion): %d categories × %d variants = ~%d queries",
                 n_exp, args.n_variants, n_exp * args.n_variants)
        n_cont = min(args.max_contrastive_pairs, len(priority_cats) * 2)
        log.info("  Type 2 (contrastive): %d pairs × %d descriptions = ~%d",
                 n_cont, args.n_contrastive, n_cont * args.n_contrastive)
        n_lt = min(args.max_longtail_categories, 30)
        log.info("  Type 3 (long-tail): %d categories × %d descriptions = ~%d",
                 n_lt, args.n_longtail, n_lt * args.n_longtail)
        total = n_exp * args.n_variants + n_cont * args.n_contrastive + n_lt * args.n_longtail
        log.info("  TOTAL: ~%d synthetic texts", total)
        est_calls = n_exp + n_cont + n_lt
        log.info("  Estimated API calls: %d", est_calls)
        log.info("  Estimated cost: $%.2f", est_calls * 0.00015)
        return

    # Get API client
    client = get_api_client()
    log.info("PaleBlueDot API connected (model: google/gemini-3-flash-preview)")

    # Type 1: Query expansions
    query_expansions = generate_query_expansions(
        client, priority_cats, n_variants=args.n_variants, max_queries=args.max_queries
    )

    # Type 2: Contrastive pairs
    contrastive_pairs = generate_contrastive_pairs(
        client, all_categories, n_pairs=args.n_contrastive,
        max_pairs=args.max_contrastive_pairs
    )

    # Type 3: Long-tail descriptions
    longtail_descs = generate_longtail_descriptions(
        client, all_categories, n_items=args.n_longtail,
        max_categories=args.max_longtail_categories
    )

    # Combine and save
    all_synthetic = query_expansions + contrastive_pairs + longtail_descs
    random.shuffle(all_synthetic)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_DIR / "synthetic_all.jsonl", "w") as f:
        for item in all_synthetic:
            f.write(json.dumps(item) + "\n")

    with open(OUTPUT_DIR / "query_expansions.jsonl", "w") as f:
        for item in query_expansions:
            f.write(json.dumps(item) + "\n")

    with open(OUTPUT_DIR / "contrastive_pairs.jsonl", "w") as f:
        for item in contrastive_pairs:
            f.write(json.dumps(item) + "\n")

    with open(OUTPUT_DIR / "longtail_descriptions.jsonl", "w") as f:
        for item in longtail_descs:
            f.write(json.dumps(item) + "\n")

    # Stats
    stats = {
        "total_synthetic": len(all_synthetic),
        "query_expansions": len(query_expansions),
        "contrastive_pairs": len(contrastive_pairs),
        "longtail_descriptions": len(longtail_descs),
        "categories_covered": len(set(
            item.get("original_category3") or item.get("positive_category3") or item.get("category3", "")
            for item in all_synthetic
        )),
        "elapsed_minutes": (time.time() - t0) / 60,
    }
    with open(OUTPUT_DIR / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    elapsed = time.time() - t0
    log.info("=" * 60)
    log.info("Synthetic Data Generation Complete (%.1f min)", elapsed / 60)
    log.info("=" * 60)
    log.info("  Query expansions: %d", len(query_expansions))
    log.info("  Contrastive pairs: %d", len(contrastive_pairs))
    log.info("  Long-tail descriptions: %d", len(longtail_descs))
    log.info("  TOTAL: %d synthetic texts", len(all_synthetic))
    log.info("  Output: %s", OUTPUT_DIR)
    log.info("=" * 60)


if __name__ == "__main__":
    main()
