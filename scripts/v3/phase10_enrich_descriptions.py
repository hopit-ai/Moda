"""Phase 10 — Enrich training data with LLM-generated descriptions.

Takes existing 500K training pairs (which have title + color + material + l1_category)
and generates richer, varied natural-language descriptions using Gemini.

This is NOT data leakage:
  - We never look at any benchmark dataset
  - We only enrich text for images we already own in our training set
  - The goal is to teach the model diverse ways to describe the same product

Approach:
  - Batch 5 items per LLM call (cost efficiency)
  - Sample 50K items stratified by L1 category
  - Generate 1 rich description per item
  - Output: supplementary training pairs (same images, richer text)

Usage:
  python3 -u scripts/v3/phase10_enrich_descriptions.py
  python3 -u scripts/v3/phase10_enrich_descriptions.py --n-items 50000
"""

from __future__ import annotations

import json
import logging
import random
import time
from collections import defaultdict
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("enrich")

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data" / "processed" / "v3_phase10_500k"
OUTPUT_PATH = REPO_ROOT / "data" / "processed" / "v3_phase10_500k" / "enriched_descriptions.jsonl"
ENV_PATH = REPO_ROOT / ".env"

BATCH_PROMPT = """You are a fashion product description writer. For each item below, write ONE natural search query (3-8 words) that a shopper would use to find this exact product. Use natural shopping language, not the exact title.

{items_block}

Rules:
- One query per item, in the same order
- 3-8 words each
- Include the most distinctive attributes (color, material, style, garment type)
- Sound like a real search query, not a product title
- Format: just the queries, one per line, numbered 1-{n}"""


def get_api_client():
    from openai import OpenAI
    api_key = None
    if ENV_PATH.exists():
        for line in ENV_PATH.read_text().strip().split("\n"):
            if line.startswith("PALEBLUEDOT_API_KEY="):
                api_key = line.split("=", 1)[1].strip()
    if not api_key:
        raise ValueError("No PALEBLUEDOT_API_KEY found")
    return OpenAI(api_key=api_key, base_url="https://open.palebluedot.ai/v1")


def call_llm(client, prompt: str, max_retries: int = 3) -> str:
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model="google/gemini-3-flash-preview",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0.8,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 2 ** (attempt + 1)
                log.warning("API error (attempt %d): %s. Retry in %ds", attempt + 1, e, wait)
                time.sleep(wait)
            else:
                log.error("API failed after %d attempts: %s", max_retries, e)
                return ""


def load_and_sample(n_items: int, seed: int = 42) -> list[dict]:
    """Load pairs and sample stratified by L1 category."""
    log.info("Loading pairs.jsonl...")
    groups = defaultdict(list)

    with open(DATA_DIR / "pairs.jsonl") as f:
        for line in f:
            row = json.loads(line)
            # Only items with decent metadata
            if row.get("title") and len(row["title"]) > 10:
                groups[row.get("l1_category", "other")].append(row)

    log.info("Loaded %d items across %d categories",
             sum(len(v) for v in groups.values()), len(groups))

    # Stratified sample — efficient version
    rng = random.Random(seed)
    per_cat = max(1, n_items // len(groups))
    sampled = []

    for cat, cat_items in groups.items():
        n = min(per_cat, len(cat_items))
        sampled.extend(rng.sample(cat_items, n))

    # Fill remaining quota from random categories
    if len(sampled) < n_items:
        all_items = [item for cat_items in groups.values() for item in cat_items]
        rng.shuffle(all_items)
        sampled_set = set(id(x) for x in sampled)
        for item in all_items:
            if len(sampled) >= n_items:
                break
            if id(item) not in sampled_set:
                sampled.append(item)

    rng.shuffle(sampled)
    sampled = sampled[:n_items]
    log.info("Sampled %d items", len(sampled))
    return sampled


def format_item(item: dict) -> str:
    """Format a single item for the LLM prompt."""
    parts = []
    title = item.get("title", "")[:80]
    parts.append(f"Title: {title}")

    attrs = []
    if item.get("color_str"):
        attrs.append(f"color={item['color_str']}")
    if item.get("material_str"):
        attrs.append(f"material={item['material_str']}")
    if item.get("l1_category"):
        attrs.append(f"category={item['l1_category']}")
    if item.get("garment_str"):
        attrs.append(f"type={item['garment_str']}")

    if attrs:
        parts.append(f"Attrs: {', '.join(attrs)}")

    return " | ".join(parts)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-items", type=int, default=50000)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    t0 = time.time()

    # Sample items
    items = load_and_sample(args.n_items, args.seed)

    # Get API client
    client = get_api_client()
    log.info("API connected")

    # Process in batches
    results = []
    n_batches = (len(items) + args.batch_size - 1) // args.batch_size
    log.info("Processing %d items in %d batches of %d", len(items), n_batches, args.batch_size)

    for batch_idx in range(n_batches):
        start = batch_idx * args.batch_size
        end = min(start + args.batch_size, len(items))
        batch = items[start:end]

        # Build prompt
        items_block = "\n".join(
            f"{i+1}. {format_item(item)}" for i, item in enumerate(batch)
        )
        prompt = BATCH_PROMPT.format(items_block=items_block, n=len(batch))

        response = call_llm(client, prompt)

        if response:
            lines = [l.strip() for l in response.split("\n") if l.strip()]
            # Parse numbered responses
            descriptions = []
            for line in lines:
                # Remove numbering (1. or 1)
                cleaned = line.lstrip("0123456789.)- ").strip()
                if cleaned:
                    descriptions.append(cleaned)

            # Match descriptions to items
            for i, item in enumerate(batch):
                desc = descriptions[i] if i < len(descriptions) else ""
                if desc:
                    results.append({
                        "image_path": item["image_path"],
                        "enriched_query": desc,
                        "original_title": item.get("title", ""),
                        "l1_category": item.get("l1_category", ""),
                        "color_str": item.get("color_str", ""),
                        "material_str": item.get("material_str", ""),
                        "weight": item.get("weight", 0.5),
                        "source": "enriched",
                    })

        if (batch_idx + 1) % 100 == 0:
            elapsed = time.time() - t0
            rate = (batch_idx + 1) / elapsed * 60
            log.info("  [%d/%d batches] %d descriptions, %.0f batches/min, ETA %.1f min",
                     batch_idx + 1, n_batches, len(results), rate,
                     (n_batches - batch_idx - 1) / max(rate, 1))

        # Save periodically
        if (batch_idx + 1) % 500 == 0:
            with open(OUTPUT_PATH, "w") as f:
                for r in results:
                    f.write(json.dumps(r) + "\n")
            log.info("  Checkpoint saved: %d descriptions", len(results))

        time.sleep(0.05)

    # Final save
    with open(OUTPUT_PATH, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    elapsed = time.time() - t0
    log.info("=" * 60)
    log.info("Enrichment Complete (%.1f min)", elapsed / 60)
    log.info("=" * 60)
    log.info("  Total descriptions: %d / %d items (%.1f%% success)",
             len(results), len(items), 100 * len(results) / max(len(items), 1))
    log.info("  Output: %s", OUTPUT_PATH)
    log.info("  Estimated cost: $%.2f", n_batches * 0.00015)


if __name__ == "__main__":
    main()
