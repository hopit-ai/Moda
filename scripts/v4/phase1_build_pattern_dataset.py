"""
Phase 1a: Stream Marqo-GS-10M and filter into pattern buckets.

Streams the dataset (no full download), classifies each pair into
pattern buckets, extracts multi-field metadata (category, color, etc.),
saves selected pairs: metadata to JSONL + images to disk at 224px.

Target: ~120K pairs across all pattern buckets.
"""
import os, sys, json, re, hashlib, io, time, argparse
from pathlib import Path
from collections import Counter, defaultdict
from PIL import Image

PROJ_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = PROJ_ROOT / "data" / "processed" / "v4_pattern_targeted"
IMG_DIR = OUT_DIR / "images"
PAIRS_FILE = OUT_DIR / "pairs.jsonl"
STATS_FILE = OUT_DIR / "stats.json"

IMG_DIR.mkdir(parents=True, exist_ok=True)

# --------------- Pattern Bucket Targets ---------------

BUCKET_TARGETS = {
    "short_title": 40_000,
    "long_description": 30_000,
    "apparel": 50_000,
    "accessories": 20_000,
    "footwear": 15_000,
    "non_fashion": 15_000,
    "brand_product": 10_000,
    "color_centric": 15_000,
    "compound_attr": 15_000,
}

# --------------- Keyword Dictionaries ---------------

APPAREL_KW = {
    "dress", "dresses", "gown", "gowns", "top", "tops", "shirt", "shirts",
    "blouse", "blouses", "tee", "t-shirt", "t-shirts", "tank", "camisole",
    "pants", "trousers", "jeans", "leggings", "shorts", "skirt", "skirts",
    "jacket", "jackets", "coat", "coats", "blazer", "blazers", "cardigan",
    "sweater", "sweaters", "hoodie", "hoodies", "sweatshirt", "vest",
    "jumpsuit", "romper", "bodysuit", "lingerie", "bra", "underwear",
    "pajamas", "robe", "kimono", "suit", "tuxedo", "saree", "sari",
    "kurta", "tunic", "polo", "henley", "parka", "anorak", "windbreaker",
    "overalls", "dungarees", "cape", "poncho", "swimsuit", "bikini",
    "swimwear", "boardshorts", "tankini", "coverup", "wetsuit",
    "activewear", "sportswear", "yoga", "leotard", "tracksuit",
    "jersey", "uniform",
}

ACCESSORIES_KW = {
    "bag", "bags", "handbag", "handbags", "purse", "purses", "clutch",
    "backpack", "backpacks", "tote", "satchel", "crossbody", "wallet",
    "wallets", "watch", "watches", "jewelry", "necklace", "bracelet",
    "earrings", "ring", "rings", "pendant", "brooch", "cufflinks",
    "belt", "belts", "scarf", "scarves", "sunglasses", "glasses",
    "hat", "hats", "cap", "caps", "beanie", "headband", "earmuffs",
    "gloves", "mittens", "tie", "ties", "bowtie", "pocket square",
    "hair accessory", "hair clip", "hairband", "scrunchie",
    "keychain", "lanyard", "umbrella", "luggage", "suitcase",
}

FOOTWEAR_KW = {
    "shoe", "shoes", "boot", "boots", "sandal", "sandals", "sneaker",
    "sneakers", "heel", "heels", "flat", "flats", "loafer", "loafers",
    "mule", "mules", "oxford", "oxfords", "pump", "pumps", "slipper",
    "slippers", "clog", "clogs", "espadrille", "wedge", "wedges",
    "stiletto", "platform", "ankle boot", "chelsea boot", "combat boot",
    "hiking boot", "rain boot", "riding boot", "thigh-high", "flip-flop",
    "flip flop", "slide", "slides", "trainer", "trainers",
}

NON_FASHION_KW = {
    "home", "decor", "furniture", "pillow", "cushion", "throw", "blanket",
    "rug", "carpet", "curtain", "lamp", "candle", "vase", "plant",
    "kitchen", "cookware", "tableware", "dinnerware", "mug", "cup",
    "plate", "bowl", "utensil", "appliance", "gadget", "phone", "case",
    "tech", "electronic", "speaker", "headphone", "earphone", "charger",
    "cable", "beauty", "skincare", "makeup", "cosmetic", "lipstick",
    "foundation", "mascara", "serum", "moisturizer", "perfume", "fragrance",
    "cologne", "shampoo", "conditioner", "lotion", "cream", "soap",
    "candle", "diffuser", "stationery", "notebook", "pen", "pencil",
    "toy", "game", "puzzle", "pet", "dog", "cat", "fitness", "gym",
    "weight", "dumbbell", "mat", "band", "bottle", "tumbler",
}

COLORS = {
    "black", "white", "red", "blue", "green", "yellow", "orange", "purple",
    "pink", "grey", "gray", "brown", "beige", "cream", "ivory", "tan",
    "navy", "burgundy", "maroon", "coral", "teal", "turquoise", "aqua",
    "lavender", "lilac", "mauve", "rose", "blush", "gold", "silver",
    "bronze", "copper", "champagne", "charcoal", "olive", "sage", "mint",
    "peach", "rust", "wine", "khaki", "camel", "taupe", "indigo",
    "magenta", "fuchsia", "cobalt", "emerald", "ruby", "sapphire",
    "plum", "mustard", "terracotta", "slate", "steel", "denim",
    "heather", "midnight", "nude", "off-white", "multicolor", "multi",
    "floral", "striped", "plaid", "leopard", "camo", "camouflage",
    "neon", "pastel", "metallic",
}

MATERIALS = {
    "cotton", "silk", "linen", "wool", "cashmere", "polyester", "nylon",
    "rayon", "viscose", "chiffon", "satin", "velvet", "denim", "leather",
    "suede", "faux leather", "faux fur", "fur", "lace", "mesh", "tulle",
    "tweed", "corduroy", "fleece", "jersey", "spandex", "lycra",
    "elastane", "acrylic", "modal", "bamboo", "hemp", "canvas",
    "rubber", "patent", "stainless steel", "sterling silver", "gold",
    "titanium", "ceramic", "porcelain", "glass", "wood", "marble",
}

KNOWN_BRANDS = {
    "nike", "adidas", "gucci", "prada", "chanel", "louis vuitton", "versace",
    "balenciaga", "fendi", "dior", "hermes", "burberry", "valentino",
    "givenchy", "bottega", "loewe", "celine", "saint laurent", "ysl",
    "zara", "h&m", "uniqlo", "mango", "gap", "forever 21",
    "ralph lauren", "tommy hilfiger", "calvin klein", "michael kors",
    "coach", "kate spade", "tory burch", "ugg", "timberland",
    "converse", "vans", "new balance", "puma", "reebok", "asics",
    "skechers", "north face", "patagonia", "columbia", "under armour",
    "lululemon", "anthropologie", "free people", "urban outfitters",
    "asos", "boohoo", "shein", "prettylittlething", "revolve",
    "nordstrom", "amazon", "walmart", "target", "macys",
}


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:[-'][a-z0-9]+)*", text.lower())


def extract_colors(tokens: set[str], text_lower: str) -> list[str]:
    found = []
    for c in COLORS:
        if " " in c:
            if c in text_lower:
                found.append(c)
        elif c in tokens:
            found.append(c)
    return found


def extract_materials(tokens: set[str], text_lower: str) -> list[str]:
    found = []
    for m in MATERIALS:
        if " " in m:
            if m in text_lower:
                found.append(m)
        elif m in tokens:
            found.append(m)
    return found


def detect_brand(text_lower: str) -> str | None:
    for b in KNOWN_BRANDS:
        if b in text_lower:
            return b
    return None


def classify_category(tokens: set[str]) -> tuple[str, str]:
    """Return (category1, category2) based on keyword match priority."""
    for kw in tokens:
        if kw in APPAREL_KW:
            return "apparel", kw
        if kw in FOOTWEAR_KW:
            return "footwear", kw
        if kw in ACCESSORIES_KW:
            return "accessories", kw
        if kw in NON_FASHION_KW:
            return "non_fashion", kw
    return "other", "general"


def classify_buckets(query: str, title: str) -> dict:
    """Classify a pair into pattern buckets and extract metadata."""
    combined = f"{query} {title}"
    tokens_list = tokenize(combined)
    tokens = set(tokens_list)
    text_lower = combined.lower()
    title_words = len(title.split())

    cat1, cat2 = classify_category(tokens)
    colors = extract_colors(tokens, text_lower)
    materials = extract_materials(tokens, text_lower)
    brand = detect_brand(text_lower)

    buckets = []
    if 5 <= title_words <= 9:
        buckets.append("short_title")
    if title_words >= 20:
        buckets.append("long_description")
    if cat1 == "apparel":
        buckets.append("apparel")
    elif cat1 == "accessories":
        buckets.append("accessories")
    elif cat1 == "footwear":
        buckets.append("footwear")
    elif cat1 == "non_fashion":
        buckets.append("non_fashion")
    if brand:
        buckets.append("brand_product")
    if colors:
        buckets.append("color_centric")
    if len(colors) >= 1 and cat2 != "general" and (materials or brand):
        buckets.append("compound_attr")

    return {
        "buckets": buckets,
        "category1": cat1,
        "category2": cat2,
        "colors": colors,
        "materials": materials,
        "brand": brand,
    }


def save_image(img: Image.Image, product_id: str) -> str:
    """Save image as compressed 224x224 JPEG, return filename."""
    fname = f"{product_id}.jpg"
    fpath = IMG_DIR / fname
    if fpath.exists():
        return fname
    img_resized = img.convert("RGB").resize((224, 224), Image.LANCZOS)
    img_resized.save(fpath, "JPEG", quality=80, optimize=True)
    return fname


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-stream", type=int, default=5_000_000,
                        help="Max rows to stream before stopping")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing pairs.jsonl")
    args = parser.parse_args()

    from datasets import load_dataset

    bucket_counts = Counter()
    seen_products = set()
    total_saved = 0
    total_streamed = 0

    if args.resume and PAIRS_FILE.exists():
        print(f"Resuming from {PAIRS_FILE}")
        with open(PAIRS_FILE) as f:
            for line in f:
                rec = json.loads(line)
                seen_products.add(rec["product_id"])
                for b in rec["buckets"]:
                    bucket_counts[b] += 1
                total_saved += 1
        print(f"  Loaded {total_saved} existing pairs")
        for b, t in BUCKET_TARGETS.items():
            print(f"  {b}: {bucket_counts[b]}/{t}")

    # long_description and non_fashion won't fill well from GS-10M;
    # synthetic data covers these in phase1b.
    SKIP_FROM_GS10M = {"long_description"}

    def all_buckets_filled():
        return all(
            bucket_counts[b] >= BUCKET_TARGETS[b]
            for b in BUCKET_TARGETS if b not in SKIP_FROM_GS10M
        )

    out_f = open(PAIRS_FILE, "a" if args.resume else "w")

    splits_to_try = ["in_domain", "novel_document", "novel_query", "zero_shot"]
    start_time = time.time()

    for split in splits_to_try:
        if all_buckets_filled():
            break

        print(f"\n{'='*60}")
        print(f"Streaming split: {split}")
        print(f"{'='*60}")

        ds = load_dataset("Marqo/marqo-GS-10M", split=split, streaming=True)

        for row in ds:
            total_streamed += 1

            if total_streamed > args.max_stream:
                print(f"\nReached max stream limit: {args.max_stream}")
                break

            pid = row["product_id"]
            if pid in seen_products:
                continue

            query = row["query"]
            title = row["title"]
            score = row["score_linear"]

            if score < 50:
                continue

            meta = classify_buckets(query, title)
            buckets = meta["buckets"]

            if not buckets:
                continue

            needed = [b for b in buckets if bucket_counts[b] < BUCKET_TARGETS[b]]
            if not needed:
                continue

            try:
                img = row["image"]
                if img is None:
                    continue
                fname = save_image(img, pid)
            except Exception:
                continue

            record = {
                "query": query,
                "title": title,
                "product_id": pid,
                "pair_id": row["pair_id"],
                "score_linear": score,
                "image_file": fname,
                "category1": meta["category1"],
                "category2": meta["category2"],
                "colors": meta["colors"],
                "materials": meta["materials"],
                "brand": meta["brand"],
                "buckets": needed,
            }

            out_f.write(json.dumps(record) + "\n")
            seen_products.add(pid)
            for b in needed:
                bucket_counts[b] += 1
            total_saved += 1

            if total_streamed % 5000 == 0:
                elapsed = time.time() - start_time
                rate = total_streamed / elapsed if elapsed > 0 else 0
                print(f"  [{split}] Streamed {total_streamed:,} | Saved {total_saved:,} | "
                      f"Rate {rate:.0f}/s | Elapsed {elapsed:.0f}s")
                for b, t in sorted(BUCKET_TARGETS.items()):
                    pct = 100 * bucket_counts[b] / t
                    bar = "#" * int(pct / 5) + "." * (20 - int(pct / 5))
                    print(f"    {b:20s} [{bar}] {bucket_counts[b]:>6,}/{t:>6,} ({pct:.0f}%)")

                if all_buckets_filled():
                    print("\n*** All pattern buckets filled! ***")
                    break

        if total_streamed > args.max_stream:
            break

    out_f.close()

    stats = {
        "total_streamed": total_streamed,
        "total_saved": total_saved,
        "bucket_counts": dict(bucket_counts),
        "bucket_targets": BUCKET_TARGETS,
        "unique_products": len(seen_products),
        "elapsed_seconds": time.time() - start_time,
    }

    with open(STATS_FILE, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\n{'='*60}")
    print(f"DONE: {total_saved:,} pairs saved to {PAIRS_FILE}")
    print(f"Stats written to {STATS_FILE}")
    print(f"{'='*60}")
    for b, t in sorted(BUCKET_TARGETS.items()):
        filled = bucket_counts[b]
        pct = 100 * filled / t
        status = "DONE" if filled >= t else "PARTIAL"
        print(f"  [{status}] {b:20s}: {filled:>6,}/{t:>6,} ({pct:.0f}%)")


if __name__ == "__main__":
    main()
