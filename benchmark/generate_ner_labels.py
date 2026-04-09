"""
MODA Phase 3.7v2 — Generate LLM-labeled NER training data for search queries

The original NER fine-tuning failed because it trained on product descriptions
(150 chars avg) but deployed on search queries (20 chars avg). This script
fixes that with two strategies:

  Strategy A: Annotate real H&M search queries with the LLM
  Strategy B: Synthesize realistic search queries from H&M product attributes

Both produce query-length text with span-level entity labels — exactly what
GLiNER2 needs to learn from.

Entity types: color, garment type, fit style, material, pattern, occasion,
              gender, brand

Usage:
  export PALEBLUEDOT_API_KEY=...
  python benchmark/generate_ner_labels.py
  python benchmark/generate_ner_labels.py --strategy both --n_annotate 5000 --n_synthesize 5000
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import random
import re
import sys
import time
from pathlib import Path

import pandas as pd
from openai import AsyncOpenAI

_REPO = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

HNM_DIR = _REPO / "data" / "raw" / "hnm_real"
SPLITS_PATH = _REPO / "data" / "processed" / "query_splits.json"
OUTPUT_DIR = _REPO / "data" / "processed"
ENV_PATH = _REPO / ".env"

BASE_URL = "https://open.palebluedot.ai/v1"
MODEL = "anthropic/claude-sonnet-4.6"
MAX_CONCURRENT = 30


def _load_api_key() -> str:
    """Load API key from env var or .env file."""
    key = os.environ.get("PALEBLUEDOT_API_KEY", "")
    if key:
        return key
    if ENV_PATH.exists():
        for line in ENV_PATH.read_text().splitlines():
            line = line.strip()
            if line.startswith("PALEBLUEDOT_API_KEY="):
                key = line.split("=", 1)[1].strip()
                if key:
                    return key
    raise ValueError(
        "Set PALEBLUEDOT_API_KEY in environment or in .env file"
    )

ENTITY_TYPES = [
    "color", "garment type", "fit style", "material",
    "pattern", "occasion", "gender", "brand",
]

# ── Strategy A: Annotate real queries ─────────────────────────────────────────

ANNOTATE_PROMPT = """\
You are a fashion search query NER annotator. Extract entities from the search query below.

Entity types:
- color: color terms (e.g. navy, red, black, beige, khaki)
- garment type: clothing item (e.g. dress, jeans, hoodie, t-shirt, blazer)
- fit style: fit/cut (e.g. slim fit, oversized, wide leg, cropped, a-line)
- material: fabric/material (e.g. cotton, denim, leather, linen, jersey)
- pattern: visual pattern (e.g. striped, floral, check, polka dot, camo)
- occasion: use context (e.g. casual, formal, outdoor, sport, party, summer, winter)
- gender: target demographic (e.g. mens, womens, boys, girls, baby, kids, unisex)
- brand: brand name (only if explicitly a brand, NOT generic words)

Rules:
1. Each entity value MUST be an EXACT substring of the query (case-insensitive match)
2. A word can only belong to ONE entity type — pick the most specific one
3. If no entities of a type exist, omit that type
4. Do NOT hallucinate entities not present in the query text
5. "cotton" is material, NOT color. "summer" is occasion, NOT color.

Query: "{query}"

Return ONLY valid JSON: {{"color": ["..."], "garment type": ["..."], ...}}
Omit types with no matches. No explanation."""


SYNTHESIZE_PROMPT = """\
Generate {n} realistic fashion search queries that a customer would type into an online store search bar.

Each query should contain these attributes:
{attributes}

Rules:
1. Queries should be 2-6 words, informal/telegraphic (how real people search)
2. Use natural vocabulary, not H&M's internal taxonomy names
3. Vary phrasing: "navy slim jeans", "dark blue skinny denim", "mens slim fit navy jeans"
4. Do NOT always put attributes in the same order
5. Some queries can omit some attributes naturally

For each query, provide the exact entity spans.

Return a JSON array:
[{{"query": "...", "entities": {{"color": ["navy"], "garment type": ["jeans"], ...}}}}, ...]

Each entity value MUST be an exact substring of the query. No explanation, just the JSON array."""


async def annotate_query(
    client: AsyncOpenAI,
    query: str,
    semaphore: asyncio.Semaphore,
) -> dict | None:
    """Strategy A: ask LLM to extract entities from a real query."""
    async with semaphore:
        try:
            resp = await client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "user", "content": ANNOTATE_PROMPT.format(query=query)},
                ],
                temperature=0.0,
                max_tokens=200,
            )
            text = resp.choices[0].message.content.strip()
            text = re.sub(r"^```json\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
            entities = json.loads(text)

            validated = {}
            query_lower = query.lower()
            for etype, vals in entities.items():
                if etype not in ENTITY_TYPES:
                    continue
                good = [v for v in vals if v.lower() in query_lower]
                if good:
                    validated[etype] = good

            if not validated:
                return None

            return {"input": query, "output": {"entities": validated}}
        except Exception as e:
            log.debug("Annotation failed for %r: %s", query, e)
            return None


async def synthesize_batch(
    client: AsyncOpenAI,
    attributes: dict[str, str],
    n: int,
    semaphore: asyncio.Semaphore,
) -> list[dict]:
    """Strategy B: ask LLM to generate queries from known attributes."""
    attr_lines = "\n".join(f"- {k}: {v}" for k, v in attributes.items())
    async with semaphore:
        try:
            resp = await client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "user", "content": SYNTHESIZE_PROMPT.format(
                        n=n, attributes=attr_lines)},
                ],
                temperature=0.8,
                max_tokens=1000,
            )
            text = resp.choices[0].message.content.strip()
            text = re.sub(r"^```json\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
            items = json.loads(text)

            results = []
            for item in items:
                query = item.get("query", "")
                entities = item.get("entities", {})
                if not query or not entities:
                    continue

                validated = {}
                query_lower = query.lower()
                for etype, vals in entities.items():
                    if etype not in ENTITY_TYPES:
                        continue
                    good = [v for v in vals if v.lower() in query_lower]
                    if good:
                        validated[etype] = good

                if validated:
                    results.append({"input": query, "output": {"entities": validated}})
            return results
        except Exception as e:
            log.debug("Synthesis failed: %s", e)
            return []


def build_attribute_combos(articles_df: pd.DataFrame, n: int) -> list[dict[str, str]]:
    """Build diverse attribute combinations from H&M catalog for synthesis."""
    colors = articles_df["colour_group_name"].dropna().unique().tolist()
    types_ = articles_df["product_type_name"].dropna().unique().tolist()
    patterns = [p for p in articles_df["graphical_appearance_name"].dropna().unique()
                if p not in {"Solid", "Other pattern", "Unknown", ""}]
    groups = articles_df["index_group_name"].dropna().unique().tolist()
    sections = articles_df["section_name"].dropna().unique().tolist()

    combos = []
    rng = random.Random(42)
    for _ in range(n):
        combo: dict[str, str] = {}
        combo["color"] = rng.choice(colors)
        combo["garment type"] = rng.choice(types_)

        if rng.random() < 0.4:
            combo["pattern"] = rng.choice(patterns)
        if rng.random() < 0.5:
            combo["gender"] = rng.choice(groups)
        if rng.random() < 0.3:
            combo["occasion"] = rng.choice(["casual", "formal", "sport",
                                             "outdoor", "party", "summer",
                                             "winter", "work", "everyday"])
        if rng.random() < 0.3:
            combo["material"] = rng.choice(["cotton", "denim", "leather",
                                            "linen", "wool", "silk",
                                            "polyester", "jersey", "velvet",
                                            "fleece", "chiffon", "satin"])
        if rng.random() < 0.2:
            combo["fit style"] = rng.choice(["slim fit", "regular fit",
                                             "oversized", "relaxed",
                                             "skinny", "wide leg",
                                             "cropped", "fitted", "loose"])
        combos.append(combo)
    return combos


async def run_strategy_a(
    queries: list[str],
    n: int,
    output_path: Path,
):
    """Annotate real queries with LLM."""
    api_key = _load_api_key()

    client = AsyncOpenAI(base_url=BASE_URL, api_key=api_key)
    sem = asyncio.Semaphore(MAX_CONCURRENT)

    sample = random.Random(42).sample(queries, min(n, len(queries)))
    log.info("Strategy A: annotating %d real queries...", len(sample))

    existing = set()
    if output_path.exists():
        with open(output_path) as f:
            for line in f:
                rec = json.loads(line)
                existing.add(rec["input"])
        log.info("  %d already annotated, skipping those", len(existing))
        sample = [q for q in sample if q not in existing]

    results = []
    t0 = time.time()
    batch_size = 100
    for i in range(0, len(sample), batch_size):
        batch = sample[i:i + batch_size]
        tasks = [annotate_query(client, q, sem) for q in batch]
        batch_results = await asyncio.gather(*tasks)

        valid = [r for r in batch_results if r is not None]
        results.extend(valid)

        with open(output_path, "a") as f:
            for r in valid:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        done = i + len(batch)
        elapsed = time.time() - t0
        rate = done / elapsed if elapsed > 0 else 0
        log.info("  A: %d/%d done (%d valid, %.1f q/s)",
                 done, len(sample), len(results), rate)

    log.info("Strategy A complete: %d valid annotations from %d queries (%.1f%%)",
             len(results), len(sample), 100 * len(results) / max(1, len(sample)))
    return results


async def run_strategy_b(
    articles_df: pd.DataFrame,
    n: int,
    output_path: Path,
):
    """Synthesize queries from attribute combinations."""
    api_key = _load_api_key()

    client = AsyncOpenAI(base_url=BASE_URL, api_key=api_key)
    sem = asyncio.Semaphore(MAX_CONCURRENT)

    n_batches = n // 3 + 1
    combos = build_attribute_combos(articles_df, n_batches)
    log.info("Strategy B: synthesizing from %d attribute combos...", len(combos))

    results = []
    t0 = time.time()
    batch_size = 50
    for i in range(0, len(combos), batch_size):
        batch = combos[i:i + batch_size]
        tasks = [synthesize_batch(client, combo, 3, sem) for combo in batch]
        batch_results = await asyncio.gather(*tasks)

        for items in batch_results:
            results.extend(items)
            with open(output_path, "a") as f:
                for r in items:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")

        done = i + len(batch)
        elapsed = time.time() - t0
        log.info("  B: %d/%d combos done (%d queries generated, %.1f combo/s)",
                 done, len(combos), len(results), done / elapsed if elapsed > 0 else 0)

    log.info("Strategy B complete: %d synthetic queries from %d combos",
             len(results), len(combos))
    return results


def validate_and_report(data_path: Path):
    """Load generated data and report entity distribution."""
    records = []
    with open(data_path) as f:
        for line in f:
            records.append(json.loads(line))

    log.info("\n=== NER Training Data Report ===")
    log.info("Total examples: %d", len(records))

    from collections import Counter
    type_counts = Counter()
    multi_type = 0
    for rec in records:
        types = list(rec["output"]["entities"].keys())
        for t in types:
            type_counts[t] += 1
        if len(types) >= 2:
            multi_type += 1

    log.info("Entity type distribution:")
    for etype, cnt in type_counts.most_common():
        log.info("  %-20s %5d  (%.1f%%)", etype, cnt, 100 * cnt / len(records))

    log.info("Multi-type examples: %d (%.1f%%)",
             multi_type, 100 * multi_type / len(records))

    lens = [len(r["input"]) for r in records]
    log.info("Query length: mean=%.0f, median=%.0f, min=%d, max=%d",
             sum(lens) / len(lens),
             sorted(lens)[len(lens) // 2],
             min(lens), max(lens))

    log.info("\nSample entries:")
    for rec in random.Random(42).sample(records, min(10, len(records))):
        log.info("  %r → %s", rec["input"], rec["output"]["entities"])


async def main_async(args):
    splits = json.loads(SPLITS_PATH.read_text())
    train_qids = set(splits["train"])

    queries_df = pd.read_csv(HNM_DIR / "queries.csv", dtype=str)
    train_queries = queries_df[queries_df["query_id"].isin(train_qids)]
    unique_texts = list(train_queries["query_text"].unique())
    log.info("Unique train query texts: %d", len(unique_texts))

    articles_df = pd.read_csv(HNM_DIR / "articles.csv", dtype=str).fillna("")

    output_a = OUTPUT_DIR / "ner_labels_real_queries.jsonl"
    output_b = OUTPUT_DIR / "ner_labels_synthetic_queries.jsonl"
    combined = OUTPUT_DIR / "ner_training_data_v2.jsonl"

    if args.strategy in ("annotate", "both"):
        await run_strategy_a(unique_texts, args.n_annotate, output_a)

    if args.strategy in ("synthesize", "both"):
        await run_strategy_b(articles_df, args.n_synthesize, output_b)

    log.info("Merging into combined training file...")
    all_records = []
    for p in [output_a, output_b]:
        if p.exists():
            with open(p) as f:
                for line in f:
                    all_records.append(json.loads(line))

    seen = set()
    deduped = []
    for rec in all_records:
        key = rec["input"].lower().strip()
        if key not in seen:
            seen.add(key)
            deduped.append(rec)

    random.Random(42).shuffle(deduped)
    with open(combined, "w") as f:
        for rec in deduped:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    log.info("Combined: %d unique examples → %s", len(deduped), combined)

    validate_and_report(combined)


def main():
    p = argparse.ArgumentParser(description="Generate LLM-labeled NER training data")
    p.add_argument("--strategy", choices=["annotate", "synthesize", "both"],
                   default="both")
    p.add_argument("--n_annotate", type=int, default=5000,
                   help="Number of real queries to annotate (Strategy A)")
    p.add_argument("--n_synthesize", type=int, default=5000,
                   help="Number of synthetic queries to generate (Strategy B)")
    args = p.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
