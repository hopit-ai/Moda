"""
Phase 1b: Generate synthetic gap-filling data via LLM API.

Generates text descriptions in patterns underrepresented in GS-10M:
- Template-style descriptions (DF-Multimodal style)
- Compound color labels
- Lifestyle / non-fashion items (home, beauty, tech)
- Long rich descriptions (20-60 words)
- Text paraphrases of existing queries for diversity

Uses existing GS-10M images as anchors, generates new text pairings.
"""
import os, sys, json, time, random, argparse
from pathlib import Path
from collections import Counter

import requests

PROJ_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = PROJ_ROOT / "data" / "processed" / "v4_pattern_targeted"
GS10M_PAIRS = OUT_DIR / "pairs.jsonl"
SYNTH_FILE = OUT_DIR / "synthetic_pairs.jsonl"
STATS_FILE = OUT_DIR / "synthetic_stats.json"

API_URL = "https://open.palebluedot.ai/v1/chat/completions"


def call_llm(messages: list[dict], api_key: str, temperature: float = 0.9,
             max_tokens: int = 4096) -> str:
    """Call LLM API and return text response."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "qwen/qwen3.5-flash",
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    for attempt in range(5):
        try:
            resp = requests.post(API_URL, headers=headers, json=payload, timeout=90)
            resp.raise_for_status()
            msg = resp.json()["choices"][0]["message"]
            text = msg.get("content") or msg.get("reasoning_content") or ""
            return text.strip()
        except Exception as e:
            wait = min(2 ** attempt, 30)
            print(f"      API retry {attempt+1}/5: {e} (wait {wait}s)")
            time.sleep(wait)
    raise RuntimeError("API failed after 5 retries")


# --------------- Prompt Templates ---------------

TEMPLATE_DESC_PROMPT = """Generate {n} fashion product descriptions in this EXACT template style:
"The [garment_type] has [sleeve_type] sleeves, [fabric_type] fabric and [pattern_type] patterns"

Vary the garments (dress, shirt, jacket, blouse, sweater, coat, top, skirt), fabrics (cotton, silk, wool, chiffon, denim, linen, polyester, velvet, satin, lace), sleeve types (short, long, three-quarter, cap, puff, bell, raglan, flutter, bishop), and patterns (solid color, floral, striped, plaid, polka dot, geometric, abstract, paisley, checkered, animal print).

Return ONLY a JSON array of strings. No explanations."""

COMPOUND_COLOR_PROMPT = """Generate {n} compound fashion color descriptions. Each should be a realistic color name used in fashion, combining two or more elements. Examples: "Cream-black", "Heather grey", "Navy-white stripe", "Dusty rose", "Burnt sienna", "Midnight blue", "Forest green", "Steel grey melange".

Include both hyphenated combinations (e.g., "Red-black") and descriptive compounds (e.g., "Washed denim blue").

Return ONLY a JSON array of strings. No explanations."""

LIFESTYLE_PROMPT = """Generate {n} product descriptions for non-fashion lifestyle items. Each description should be 15-40 words.

Categories to cover equally:
- Home decor (pillows, vases, candles, rugs, wall art, planters)
- Kitchen/dining (mugs, plates, cutting boards, utensils, storage)
- Beauty/skincare (serums, moisturizers, lipsticks, palettes, brushes)
- Tech accessories (phone cases, laptop sleeves, chargers, speakers)
- Fitness (yoga mats, resistance bands, water bottles, gym bags)

Each description should include color, material, and a key feature.

Return ONLY a JSON array of strings. No explanations."""

LONG_DESC_PROMPT = """Generate {n} rich fashion product descriptions, each 25-50 words long. Cover diverse categories:
- Dresses, tops, jackets, pants, skirts
- Shoes, boots, sandals, sneakers
- Bags, jewelry, watches, scarves

Each description must include:
1. Product type
2. Color/pattern
3. Material/fabric
4. Style/occasion detail
5. A distinguishing feature

Example: "Elegant floor-length emerald green silk evening gown featuring a plunging V-neckline, delicate hand-sewn beadwork along the bodice, and a dramatic side slit. Perfect for black-tie events and formal galas."

Return ONLY a JSON array of strings. No explanations."""

PARAPHRASE_PROMPT = """Paraphrase each of these fashion search queries into 3 different variations. Keep the same intent but change wording, add detail, or shift style (casual/formal/descriptive).

Queries:
{queries}

Return a JSON object where keys are the original queries and values are arrays of 3 paraphrases each. No explanations."""

CATEGORY_QUERY_PROMPT = """Generate {n} fashion search queries that match these specific benchmark patterns:

Pattern 1 - Short categorical (Atlas/KAGL style): single category words or 2-3 word category phrases
Examples: "Dresses", "Casual shoes", "Winter jackets", "Leather bags"

Pattern 2 - Descriptive search (Fashion200K style): natural language queries 8-15 words
Examples: "black cocktail dress with lace overlay and V-neck", "comfortable running shoes for wide feet"

Pattern 3 - Fine-grained attributes (Polyvore style): multi-attribute compound queries
Examples: "vintage floral print midi wrap dress", "men's brown suede chelsea boots"

Generate equal numbers of each pattern.

Return ONLY a JSON array of strings. No explanations."""


def build_local_synthetic_records(args, existing_pairs: list[dict]) -> tuple[list[dict], Counter]:
    """Deterministic synthetic text (no API). Covers DF-multimodal templates, colors, lifestyle, long text."""
    rng = random.Random(42)
    synth_records: list[dict] = []
    stats: Counter = Counter()

    garments = [
        "dress", "shirt", "jacket", "blouse", "sweater", "coat", "top", "skirt",
        "cardigan", "hoodie", "tunic", "polo", "parka", "blazer", "vest",
    ]
    sleeves = ["short", "long", "three-quarter", "cap", "puff", "bell", "bishop", "flutter"]
    fabrics = [
        "cotton", "silk", "wool", "chiffon", "denim", "linen", "polyester",
        "velvet", "satin", "lace", "mesh", "tweed", "jersey",
    ]
    patt = [
        "solid color", "floral", "striped", "plaid", "polka dot", "geometric",
        "abstract", "paisley", "checkered", "animal print",
    ]

    print(f"\n--- Local: {args.template_count} template descriptions ---")
    for _ in range(args.template_count):
        desc = (
            f"The {rng.choice(garments)} has {rng.choice(sleeves)} sleeves, "
            f"{rng.choice(fabrics)} fabric and {rng.choice(patt)} patterns"
        )
        synth_records.append({
            "synth_type": "template_description",
            "text": desc,
            "buckets": ["long_description", "compound_attr"],
        })
    stats["template_description"] = args.template_count

    ca = ["cream", "navy", "burgundy", "heather", "dusty", "midnight", "forest", "steel", "wine", "camel", "slate"]
    cb = ["black", "white", "grey", "blue", "red", "brown", "pink", "green", "tan", "ivory"]
    print(f"\n--- Local: {args.color_count} compound colors ---")
    for _ in range(args.color_count):
        if rng.random() < 0.5:
            text = f"{rng.choice(ca)}-{rng.choice(cb)}"
        else:
            text = f"{rng.choice(ca)} {rng.choice(cb)} melange"
        synth_records.append({
            "synth_type": "compound_color",
            "text": text,
            "buckets": ["color_centric", "compound_attr"],
        })
    stats["compound_color"] = args.color_count

    home = ["ceramic vase", "throw pillow", "scented candle", "wall print", "planter pot", "area rug"]
    kitchen = ["stoneware mug", "bamboo cutting board", "glass storage jar", "stainless fork set"]
    beauty = ["vitamin C serum", "matte lipstick", "hydrating moisturizer", "makeup brush set"]
    tech = ["silicone phone case", "laptop sleeve", "USB-C charging cable", "bluetooth speaker"]
    fitness = ["non-slip yoga mat", "resistance band set", "insulated water bottle", "gym duffel bag"]

    print(f"\n--- Local: {args.lifestyle_count} lifestyle descriptions ---")
    pools = [home, kitchen, beauty, tech, fitness]
    adj = ["minimal", "modern", "rustic", "sleek", "compact", "premium", "eco-friendly"]
    feat = ["easy to clean", "gift-ready packaging", "durable construction", "space-saving design"]
    for i in range(args.lifestyle_count):
        cat = pools[i % len(pools)]
        item = rng.choice(cat)
        desc = (
            f"{rng.choice(adj).capitalize()} {item} in {rng.choice(cb)} and {rng.choice(fabrics)}-look finish; "
            f"{rng.choice(feat)}. Ideal for daily use at home or travel."
        )
        synth_records.append({
            "synth_type": "lifestyle",
            "text": desc,
            "buckets": ["non_fashion", "long_description"],
        })
    stats["lifestyle"] = args.lifestyle_count

    occasions = ["work", "weekend brunch", "evening out", "travel", "gym", "office", "date night"]
    print(f"\n--- Local: {args.long_desc_count} long fashion descriptions ---")
    shoe_types = ["chelsea boots", "running sneakers", "block heel sandals", "loafers", "ankle boots"]
    bag_types = ["crossbody bag", "tote bag", "evening clutch", "backpack"]
    for _ in range(args.long_desc_count):
        kind = rng.choice(["apparel", "footwear", "bag"])
        if kind == "apparel":
            piece = rng.choice(["midi wrap dress", "tailored blazer", "high-rise jeans", "knit sweater"])
            desc = (
                f"{rng.choice(adj).capitalize()} {rng.choice(ca)} {piece} in {rng.choice(fabrics)} with "
                f"{rng.choice(patt)} detailing; tailored silhouette suited for {rng.choice(occasions)}. "
                f"Features refined stitching and comfortable lining."
            )
        elif kind == "footwear":
            desc = (
                f"{rng.choice(adj).capitalize()} {rng.choice(cb)} {rng.choice(shoe_types)} with "
                f"{rng.choice(['leather', 'suede', 'mesh'])} upper and cushioned sole; versatile for "
                f"{rng.choice(occasions)} while staying lightweight."
            )
        else:
            desc = (
                f"{rng.choice(adj).capitalize()} {rng.choice(['black', 'tan', 'burgundy'])} {rng.choice(bag_types)} "
                f"with {rng.choice(['gold', 'silver'])} hardware and structured shape; roomy interior for essentials."
            )
        synth_records.append({
            "synth_type": "long_description",
            "text": desc,
            "buckets": ["long_description"],
        })
    stats["long_description"] = args.long_desc_count

    shorts = ["Dresses", "Casual shoes", "Winter jackets", "Leather bags", "Watches", "Scarves"]
    print(f"\n--- Local: {args.category_query_count} category-style queries ---")
    for i in range(args.category_query_count):
        pattern = i % 3
        if pattern == 0:
            base = rng.choice(shorts)
            prefix = rng.choice(["women", "mens", ""])
            q = f"{prefix} {base}".strip() if prefix else base
        elif pattern == 1:
            q = (
                f"{rng.choice(cb)} {rng.choice(['lace', 'knit', 'silk'])} "
                f"{rng.choice(garments)} with {rng.choice(patt)} pattern for {rng.choice(occasions)}"
            )
        else:
            q = (
                f"vintage {rng.choice(patt)} print {rng.choice(['midi', 'maxi'])} "
                f"{rng.choice(['wrap dress', 'A-line skirt'])} with {rng.choice(['belt', 'pockets', 'pleats'])} detail"
            )
        wc = len(q.split())
        buckets = ["compound_attr"]
        if 5 <= wc <= 9:
            buckets.append("short_title")
        if wc >= 20:
            buckets.append("long_description")
        synth_records.append({"synth_type": "category_query", "text": q, "buckets": buckets})
    stats["category_query"] = args.category_query_count

    para_count = 0
    if existing_pairs and args.paraphrase_count > 0:
        unique_queries = list({p["query"] for p in existing_pairs})
        rng.shuffle(unique_queries)
        sample_q = unique_queries[: min(args.paraphrase_count, len(unique_queries))]
        suffixes = [" for everyday wear", " casual outfit idea", " versatile staple", " comfy daily pick"]
        prefixes = ["stylish ", "classic ", "affordable ", "designer-inspired "]
        print(f"\n--- Local: rule-based paraphrases ({len(sample_q)} queries) ---")
        for q in sample_q:
            variants = [
                q + rng.choice(suffixes),
                rng.choice(prefixes) + q.lower(),
                f"{q} — {rng.choice(['neutral tones', 'bold look', 'minimal aesthetic'])}",
            ]
            for v in variants:
                synth_records.append({
                    "synth_type": "paraphrase",
                    "text": v,
                    "original_query": q,
                    "buckets": ["short_title"],
                })
                para_count += 1
        stats["paraphrase"] = para_count

    return synth_records, stats


def load_existing_pairs() -> list[dict]:
    """Load pairs from phase1a to get images to pair with synthetic text."""
    pairs = []
    if GS10M_PAIRS.exists():
        with open(GS10M_PAIRS) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    pairs.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return pairs


def generate_batch(prompt_template: str, n: int, api_key: str,
                   batch_size: int = 80) -> list[str]:
    """Generate items in batches, return flat list."""
    results = []
    remaining = n
    while remaining > 0:
        batch_n = min(batch_size, remaining)
        prompt = prompt_template.format(n=batch_n)
        messages = [
            {"role": "system", "content": "You are a fashion data generator. Always return valid JSON arrays."},
            {"role": "user", "content": prompt},
        ]
        try:
            raw = call_llm(messages, api_key)
            raw = raw.strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1]
                raw = raw.rsplit("```", 1)[0].strip()
            # Try to find JSON array in response
            if not raw.startswith("["):
                start = raw.find("[")
                if start >= 0:
                    end = raw.rfind("]")
                    if end > start:
                        raw = raw[start:end+1]
            items = json.loads(raw)
            if isinstance(items, list):
                str_items = [str(x) if not isinstance(x, str) else x for x in items]
                results.extend(str_items)
                remaining -= len(str_items)
                print(f"    Generated {len(str_items)} items ({len(results)}/{n})")
            else:
                print(f"    Unexpected response type: {type(items)}, retrying...")
        except (json.JSONDecodeError, Exception) as e:
            print(f"    Error: {e}, retrying...")
            time.sleep(2)
    return results[:n]


def generate_paraphrases(queries: list[str], api_key: str,
                         batch_size: int = 20) -> dict[str, list[str]]:
    """Generate paraphrases for a list of queries."""
    all_paraphrases = {}
    for i in range(0, len(queries), batch_size):
        batch = queries[i:i + batch_size]
        query_list = "\n".join(f'- "{q}"' for q in batch)
        prompt = PARAPHRASE_PROMPT.format(queries=query_list)
        messages = [
            {"role": "system", "content": "You are a fashion search query paraphraser. Always return valid JSON."},
            {"role": "user", "content": prompt},
        ]
        try:
            raw = call_llm(messages, api_key)
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1]
                raw = raw.rsplit("```", 1)[0]
            result = json.loads(raw)
            if isinstance(result, dict):
                all_paraphrases.update(result)
                print(f"    Paraphrased {len(result)} queries ({len(all_paraphrases)}/{len(queries)})")
        except Exception as e:
            print(f"    Paraphrase error: {e}")
            time.sleep(2)
    return all_paraphrases


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-only", action="store_true",
                        help="No LLM: fill gaps with deterministic templates (no API key).")
    parser.add_argument("--api-key", type=str, default=os.environ.get("PALEBLUEDOT_API_KEY", ""),
                        help="PaleBlueDot/OpenAI-compatible API key (or set PALEBLUEDOT_API_KEY).")
    parser.add_argument("--template-count", type=int, default=5000)
    parser.add_argument("--color-count", type=int, default=3000)
    parser.add_argument("--lifestyle-count", type=int, default=8000)
    parser.add_argument("--long-desc-count", type=int, default=10000)
    parser.add_argument("--category-query-count", type=int, default=5000)
    parser.add_argument("--paraphrase-count", type=int, default=2000,
                        help="Number of existing queries to paraphrase")
    args = parser.parse_args()

    existing_pairs = load_existing_pairs()
    print(f"Loaded {len(existing_pairs)} existing pairs from phase1a")

    synth_records = []
    stats = Counter()

    if args.local_only:
        synth_records, stats = build_local_synthetic_records(args, existing_pairs)
    else:
        if not args.api_key.strip():
            print(
                "Error: no API key. Set PALEBLUEDOT_API_KEY or pass --api-key, "
                "or run with --local-only.",
                file=sys.stderr,
            )
            sys.exit(1)

        # 1. Template-style descriptions (DF-Multimodal pattern)
        print(f"\n--- Generating {args.template_count} template descriptions ---")
        templates = generate_batch(TEMPLATE_DESC_PROMPT, args.template_count, args.api_key)
        for desc in templates:
            synth_records.append({
                "synth_type": "template_description",
                "text": desc,
                "buckets": ["long_description", "compound_attr"],
            })
        stats["template_description"] = len(templates)

        # 2. Compound colors
        print(f"\n--- Generating {args.color_count} compound colors ---")
        colors = generate_batch(COMPOUND_COLOR_PROMPT, args.color_count, args.api_key)
        for color in colors:
            synth_records.append({
                "synth_type": "compound_color",
                "text": color,
                "buckets": ["color_centric", "compound_attr"],
            })
        stats["compound_color"] = len(colors)

        # 3. Lifestyle / non-fashion
        print(f"\n--- Generating {args.lifestyle_count} lifestyle descriptions ---")
        lifestyle = generate_batch(LIFESTYLE_PROMPT, args.lifestyle_count, args.api_key)
        for desc in lifestyle:
            synth_records.append({
                "synth_type": "lifestyle",
                "text": desc,
                "buckets": ["non_fashion", "long_description"],
            })
        stats["lifestyle"] = len(lifestyle)

        # 4. Long rich descriptions
        print(f"\n--- Generating {args.long_desc_count} long descriptions ---")
        long_descs = generate_batch(LONG_DESC_PROMPT, args.long_desc_count, args.api_key)
        for desc in long_descs:
            synth_records.append({
                "synth_type": "long_description",
                "text": desc,
                "buckets": ["long_description"],
            })
        stats["long_description"] = len(long_descs)

        # 5. Category/search queries for diverse pattern coverage
        print(f"\n--- Generating {args.category_query_count} category queries ---")
        cat_queries = generate_batch(CATEGORY_QUERY_PROMPT, args.category_query_count, args.api_key)
        for q in cat_queries:
            word_count = len(q.split())
            buckets = []
            if 5 <= word_count <= 9:
                buckets.append("short_title")
            if word_count >= 20:
                buckets.append("long_description")
            buckets.append("compound_attr")
            synth_records.append({
                "synth_type": "category_query",
                "text": q,
                "buckets": buckets,
            })
        stats["category_query"] = len(cat_queries)

        # 6. Paraphrases of existing queries
        if existing_pairs:
            unique_queries = list(set(p["query"] for p in existing_pairs))
            sample_queries = random.sample(unique_queries,
                                           min(args.paraphrase_count, len(unique_queries)))
            print(f"\n--- Paraphrasing {len(sample_queries)} existing queries ---")
            paraphrases = generate_paraphrases(sample_queries, args.api_key)
            para_count = 0
            for orig, variants in paraphrases.items():
                for v in variants:
                    synth_records.append({
                        "synth_type": "paraphrase",
                        "text": v,
                        "original_query": orig,
                        "buckets": ["short_title"],
                    })
                    para_count += 1
            stats["paraphrase"] = para_count
        else:
            print("\nSkipping paraphrases (no existing pairs loaded)")

    # Now pair synthetic texts with GS-10M images
    print(f"\n--- Pairing {len(synth_records)} synthetic texts with images ---")

    if existing_pairs:
        apparel_pairs = [p for p in existing_pairs if p["category1"] == "apparel"]
        accessory_pairs = [p for p in existing_pairs if p["category1"] == "accessories"]
        footwear_pairs = [p for p in existing_pairs if p["category1"] == "footwear"]
        other_pairs = [p for p in existing_pairs if p["category1"] in ("non_fashion", "other")]
        all_pools = {
            "apparel": apparel_pairs or existing_pairs,
            "accessories": accessory_pairs or existing_pairs,
            "footwear": footwear_pairs or existing_pairs,
            "non_fashion": other_pairs or existing_pairs,
            "other": existing_pairs,
        }

        output_records = []
        for rec in synth_records:
            stype = rec["synth_type"]
            if stype == "lifestyle":
                pool = all_pools["non_fashion"]
            elif stype in ("template_description", "long_description"):
                pool = all_pools["apparel"]
            else:
                pool = all_pools["other"]

            anchor = random.choice(pool)
            output_rec = {
                "query": rec["text"],
                "title": anchor["title"],
                "product_id": anchor["product_id"],
                "pair_id": f"synth-{anchor['product_id']}-{hash(rec['text']) % 10**8}",
                "score_linear": 80,
                "image_file": anchor["image_file"],
                "category1": anchor["category1"],
                "category2": anchor["category2"],
                "colors": anchor.get("colors", []),
                "materials": anchor.get("materials", []),
                "brand": anchor.get("brand"),
                "buckets": rec["buckets"],
                "synth_type": rec["synth_type"],
                "is_synthetic": True,
            }
            if "original_query" in rec:
                output_rec["original_query"] = rec["original_query"]
            output_records.append(output_rec)

        with open(SYNTH_FILE, "w") as f:
            for rec in output_records:
                f.write(json.dumps(rec) + "\n")
        print(f"Saved {len(output_records)} synthetic pairs to {SYNTH_FILE}")
    else:
        with open(SYNTH_FILE, "w") as f:
            for rec in synth_records:
                f.write(json.dumps(rec) + "\n")
        print(f"Saved {len(synth_records)} synthetic records (no image pairing yet)")

    final_stats = {
        "total_synthetic": len(synth_records),
        "type_counts": dict(stats),
        "bucket_coverage": dict(Counter(
            b for rec in synth_records for b in rec["buckets"]
        )),
    }
    with open(STATS_FILE.parent / "synthetic_stats.json", "w") as f:
        json.dump(final_stats, f, indent=2)

    print(f"\n{'='*60}")
    print("Synthetic data generation complete!")
    print(f"Total records: {len(synth_records)}")
    for k, v in sorted(stats.items()):
        print(f"  {k}: {v}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
