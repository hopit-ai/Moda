"""Phase 10 — Round 2 Synthetic Data: Directly targets FSL's actual failure queries.

Round 1 generated generic taxonomy expansions.
Round 2 generates variations of the ACTUAL queries FSL fails on from fashion200k.

For each failure query like "black short dress" (50 relevant, AP=0),
we generate:
  1. Paraphrases that mean the same thing
  2. Attribute-detailed expansions (adding specificity)
  3. Near-miss negatives (queries for adjacent categories)

This teaches FSL to properly embed color+garment+length combinations.

Usage:
  python3 -u scripts/v3/phase10_generate_synthetic_r2.py
"""

from __future__ import annotations

import json
import logging
import random
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("synth-r2")

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = REPO_ROOT / "data" / "synthetic" / "phase10_gap_fill"
GAP_TARGETS_PATH = REPO_ROOT / "results" / "fsl_error_analysis" / "gap_targets.json"
ENV_PATH = REPO_ROOT / ".env"


PARAPHRASE_PROMPT = """You are a fashion search expert. Given a fashion product search query, generate {n} diverse paraphrases and variations that describe the SAME type of product.

Query: "{query}"

Requirements:
- Each variation should retrieve the same products as the original
- Vary word order, synonyms, specificity level
- Some shorter (2-3 words), some longer (5-8 words)  
- Include natural shopping language
- DO NOT change the core product type (keep same garment, same color family, same length)

Output ONLY the variations, one per line, no numbering."""

EXPANSION_PROMPT = """You are a fashion expert. Given a basic fashion query, generate {n} MORE SPECIFIC versions that add attributes while staying within the same product category.

Query: "{query}"

Generate specific variations by adding ONE of these attributes each time:
- Fabric/material (silk, cotton, leather, denim, chiffon, velvet)
- Pattern (floral, striped, solid, plaid, polka dot)
- Occasion (casual, party, work, formal, date night)
- Fit/silhouette (fitted, relaxed, A-line, bodycon, oversized)
- Season (summer, winter, spring)
- Style detail (ruffled, pleated, wrap, button-front, backless)

Output ONLY the expanded queries, one per line, no numbering."""

NEGATIVE_PROMPT = """You are a fashion expert. Given a fashion query, generate {n} SIMILAR BUT DIFFERENT queries that describe products that look similar but are NOT the same category.

Query: "{query}"

Generate near-miss queries by changing ONE key attribute:
- Change the length (short→midi, midi→maxi, knee-length→ankle)
- Change the color to a SIMILAR shade
- Change the garment type to something similar (dress→jumpsuit, skirt→culottes)
- Change a key detail that alters the category

These should be "hard negatives" — things a model might confuse with the original.

Output ONLY the near-miss queries, one per line, no numbering."""


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
                max_tokens=400,
                temperature=0.9,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                log.error("LLM failed: %s", e)
                return ""


def main():
    t0 = time.time()
    random.seed(42)

    # Load gap targets
    with open(GAP_TARGETS_PATH) as f:
        all_targets = json.load(f)

    # Focus on actionable failures: n_relevant >= 2 (can actually improve MAP)
    actionable = [t for t in all_targets if t["n_relevant"] >= 2]
    log.info("Loaded %d actionable gap targets (n_relevant >= 2)", len(actionable))

    # Prioritize by impact: higher n_relevant = more MAP improvement possible
    actionable.sort(key=lambda x: -x["n_relevant"])

    # Top 200 highest-impact queries (covers most MAP potential)
    top_targets = actionable[:200]
    log.info("Targeting top %d queries (n_relevant range: %d to %d)",
             len(top_targets), top_targets[-1]["n_relevant"], top_targets[0]["n_relevant"])

    client = get_api_client()
    log.info("API connected")

    all_results = []
    n_paraphrase = 8
    n_expansion = 6
    n_negative = 5

    for i, target in enumerate(top_targets):
        query = target["query"]

        # Type 1: Paraphrases
        prompt = PARAPHRASE_PROMPT.format(query=query, n=n_paraphrase)
        response = call_llm(client, prompt)
        if response:
            for line in response.split("\n"):
                line = line.strip()
                if line:
                    all_results.append({
                        "type": "paraphrase",
                        "original_query": query,
                        "synthetic_text": line,
                        "n_relevant": target["n_relevant"],
                        "label": "positive",
                    })

        # Type 2: Attribute expansions
        prompt = EXPANSION_PROMPT.format(query=query, n=n_expansion)
        response = call_llm(client, prompt)
        if response:
            for line in response.split("\n"):
                line = line.strip()
                if line:
                    all_results.append({
                        "type": "expansion",
                        "original_query": query,
                        "synthetic_text": line,
                        "n_relevant": target["n_relevant"],
                        "label": "positive",
                    })

        # Type 3: Hard negatives (only for high-impact queries)
        if target["n_relevant"] >= 5:
            prompt = NEGATIVE_PROMPT.format(query=query, n=n_negative)
            response = call_llm(client, prompt)
            if response:
                for line in response.split("\n"):
                    line = line.strip()
                    if line:
                        all_results.append({
                            "type": "hard_negative",
                            "original_query": query,
                            "synthetic_text": line,
                            "n_relevant": target["n_relevant"],
                            "label": "negative",
                        })

        if (i + 1) % 20 == 0:
            log.info("  Progress: %d/%d queries, %d texts generated",
                     i + 1, len(top_targets), len(all_results))

        time.sleep(0.05)

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "round2_targeted.jsonl"
    with open(out_path, "w") as f:
        for item in all_results:
            f.write(json.dumps(item) + "\n")

    # Stats
    from collections import Counter
    type_counts = Counter(r["type"] for r in all_results)
    stats = {
        "total": len(all_results),
        "paraphrases": type_counts.get("paraphrase", 0),
        "expansions": type_counts.get("expansion", 0),
        "hard_negatives": type_counts.get("hard_negative", 0),
        "queries_targeted": len(top_targets),
        "positives": sum(1 for r in all_results if r["label"] == "positive"),
        "negatives": sum(1 for r in all_results if r["label"] == "negative"),
        "elapsed_minutes": (time.time() - t0) / 60,
    }
    with open(OUTPUT_DIR / "round2_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    elapsed = time.time() - t0
    log.info("=" * 60)
    log.info("Round 2 Synthetic Generation Complete (%.1f min)", elapsed / 60)
    log.info("=" * 60)
    log.info("  Queries targeted: %d", len(top_targets))
    log.info("  Paraphrases: %d", type_counts.get("paraphrase", 0))
    log.info("  Expansions: %d", type_counts.get("expansion", 0))
    log.info("  Hard negatives: %d", type_counts.get("hard_negative", 0))
    log.info("  TOTAL: %d synthetic texts", len(all_results))
    log.info("  Output: %s", out_path)


if __name__ == "__main__":
    main()
