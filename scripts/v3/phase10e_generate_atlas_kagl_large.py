"""Phase 10e — Large-scale atlas/KAGL-style synthetic query generation.

Generates 8K+ high-quality synthetic queries matching:
1. Atlas style: structured Indian fashion descriptions
   "Printed Georgette Saree in Teal", "Embroidered Cotton Slim Fit Kurta in Navy Blue"
2. KAGL style: brand-abbreviated product titles
   "Nike Men Dri-Fit Training Grey Shorts", "Fossil Men Grant Chrono Brown Analog Watch"

Then pairs each query with real training images by category matching.
Output: data/processed/v3_phase10_500k/synthetic_atlas_kagl_large.jsonl
"""

import json
import random
from pathlib import Path
from itertools import product as iterproduct

REPO_ROOT = Path(__file__).resolve().parents[2]
TRAINING_DATA = REPO_ROOT / "data" / "processed" / "v3_phase10_500k" / "pairs.jsonl"
OUT_PATH = REPO_ROOT / "data" / "processed" / "v3_phase10_500k" / "synthetic_atlas_kagl_large.jsonl"

rng = random.Random(42)

# ─── ATLAS VOCABULARY ────────────────────────────────────────────────────────

ATLAS_PATTERNS = [
    "{pattern} {material} {garment} in {color}",
    "{pattern} {material} {fit} {garment} in {color}",
    "{adjective} {material} {garment} with {detail} in {color}",
    "{pattern} {material} {garment}",
    "{adjective} {fit} {garment} in {material} and {color}",
    "{material} {garment} with {detail}",
]

PATTERNS = [
    "Solid Color", "Printed", "Embroidered", "Plain", "Striped", "Checked",
    "Floral Print", "Abstract Print", "Geometric Print", "Polka Dot",
    "Animal Print", "Self Design", "Woven", "Textured", "Jacquard",
    "Tie & Dye", "Bandhani", "Ikat", "Block Printed", "Kalamkari",
]

MATERIALS = [
    "Cotton", "Silk", "Linen", "Rayon", "Polyester", "Chiffon",
    "Georgette", "Crepe", "Velvet", "Satin", "Art Silk", "Dupion Silk",
    "Viscose", "Wool", "Denim", "Knit", "Jersey", "Fleece", "Nylon",
    "Organza", "Net", "Lycra", "Modal", "Bamboo", "Tencel",
]

FITS = [
    "Slim Fit", "Regular Fit", "Relaxed Fit", "Straight Fit", "Tailored Fit",
    "A-Line", "Flared", "Oversized", "Fitted", "Loose Fit",
]

COLORS = [
    "Black", "White", "Navy Blue", "Grey", "Red", "Green", "Beige", "Brown",
    "Maroon", "Olive", "Teal", "Mustard", "Coral", "Burgundy", "Charcoal",
    "Royal Blue", "Cream", "Turquoise", "Lavender", "Pink", "Peach",
    "Orange", "Yellow", "Purple", "Indigo", "Rose Gold", "Off White",
    "Dark Green", "Sky Blue", "Rust", "Mauve", "Mint Green",
]

DETAILS = [
    "embroidered neckline", "lace trim", "sequin embellishment",
    "mirror work", "zari border", "button detail", "ruffle hem",
    "cut-out detail", "embroidered hem", "printed border",
    "tassel detail", "fringe detail", "contrast piping",
]

ADJECTIVES = [
    "Casual", "Formal", "Party Wear", "Festive", "Ethnic",
    "Bohemian", "Contemporary", "Traditional", "Fusion", "Elegant",
]

# Atlas garments by training category
ATLAS_GARMENTS_BY_CAT = {
    "tops": [
        "Kurta", "Kurti", "Tunic", "Shirt", "Top", "Blouse", "Crop Top",
        "Tank Top", "Polo Shirt", "Henley", "Sweatshirt", "Hoodie",
        "T-Shirt", "Formal Shirt", "Oxford Shirt", "Linen Shirt",
    ],
    "bottoms": [
        "Trousers", "Chinos", "Salwar", "Palazzo Pants", "Leggings",
        "Jeans", "Shorts", "Skirt", "Dhoti Pants", "Harem Pants",
        "Straight Pants", "Slim Fit Jeans", "Bootcut Jeans",
    ],
    "dresses": [
        "Saree", "Lehenga Choli", "Anarkali Suit", "Salwar Kameez",
        "Maxi Dress", "Midi Dress", "Mini Dress", "Wrap Dress",
        "A-Line Dress", "Bodycon Dress", "Shift Dress",
    ],
    "outerwear": [
        "Blazer", "Jacket", "Coat", "Shrug", "Cardigan",
        "Bomber Jacket", "Denim Jacket", "Leather Jacket", "Puffer Jacket",
        "Trench Coat", "Windbreaker", "Overcoat",
    ],
    "activewear": [
        "Track Pants", "Yoga Pants", "Sports Top", "Athletic Shorts",
        "Running Tights", "Sports Bra", "Training Shorts", "Joggers",
    ],
    "swimwear": [
        "Swimsuit", "Bikini", "Swimwear", "Beachwear",
        "Monokini", "Tankini", "Board Shorts",
    ],
    "intimates": [
        "Kurta Pyjama Set", "Night Suit", "Loungewear", "Pyjamas",
        "Night Dress", "Robe", "Camisole",
    ],
}

# ─── KAGL VOCABULARY ─────────────────────────────────────────────────────────

KAGL_BRANDS = [
    "Nike", "Adidas", "Puma", "Reebok", "Under Armour", "New Balance",
    "Levi's", "Wrangler", "Lee", "Pepe Jeans", "Spykar", "Jack & Jones",
    "Zara", "H&M", "Mango", "Forever 21", "GAP", "Uniqlo",
    "Calvin Klein", "Tommy Hilfiger", "Ralph Lauren", "Lacoste",
    "Fossil", "Titan", "Fastrack", "Casio", "Seiko", "Timex",
    "Ray-Ban", "Oakley", "Carrera", "Vogue Eyewear",
    "Woodland", "Bata", "Liberty", "Clarks", "Steve Madden",
    "Hush Puppies", "Crocs", "Skechers", "ASICS",
    "Wildcraft", "Skybags", "American Tourister", "VIP", "Safari",
    "Hidesign", "Baggit", "Caprese", "Lavie", "Diana Korr",
]

KAGL_GENDERS = ["Men", "Women", "Unisex", "Boys", "Girls"]
KAGL_COLORS_ABBREV = [
    "Blk", "Wht", "Nvy", "Gry", "Red", "Grn", "Bgr", "Brn",
    "Blue", "Pink", "Org", "Ylw", "Prp", "Olv", "Mus",
    "Black", "White", "Navy", "Grey", "Brown", "Green", "Beige",
]

KAGL_GARMENTS_BY_CAT = {
    "accessories": [
        ("Watch", ["Analog Wt", "Chrono Wt", "Digi Wt", "Smrt Wt", "Spts Wt",
                   "Mlt Fnc Wt", "Slm Wt", "Clsc Wt"]),
        ("Sunglasses", ["Avtr Sng", "Wyfr Sng", "Sprt Sng", "Rnd Sng", "Rct Sng"]),
        ("Wallet", ["Bfld Wlt", "Crdhdr Wlt", "Zip Wlt", "Trfld Wlt"]),
        ("Belt", ["Lthr Blt", "Rvrs Blt", "Clsp Blt", "Wvn Blt"]),
        ("Cap", ["Bsbll Cp", "Bkt Hp", "Snapbk Cp", "Spts Cp"]),
    ],
    "bags": [
        ("Handbag", ["Tote Hb", "Satchel Hb", "Xbdy Hb", "Shdr Hb", "Clch Hb"]),
        ("Backpack", ["Lptp Bkpk", "Csul Bkpk", "Spts Bkpk", "Hkng Bkpk"]),
        ("Laptop Bag", ["Msngr Bg", "Slvr Bg", "Brf Bg"]),
    ],
    "shoes": [
        ("Casual Shoes", ["Slpon Shs", "Loafer Shs", "Btshr Shs", "Canvas Shs"]),
        ("Sports Shoes", ["Rnnng Shs", "Trnng Shs", "Bskbl Shs", "Tns Shs"]),
        ("Formal Shoes", ["Dby Shs", "Oxford Shs", "Mnk Shs", "Brgn Shs"]),
        ("Heels", ["Stlt Hl", "Blk Hl", "Wdg Hl", "Kttn Hl", "Pltfrm Hl"]),
        ("Sandals", ["Flp Flp", "Sprt Sndl", "Ftbd Sndl", "Gladtr Sndl"]),
        ("Boots", ["Ankl Bt", "Chelsea Bt", "Rngs Bt", "Cbt Bt"]),
    ],
    "tops": [
        ("T-Shirt", ["Crwn Ts", "V-Nck Ts", "Grphc Ts", "Strpd Ts", "Pln Ts"]),
        ("Shirt", ["Slm Shrt", "Rgr Shrt", "Csl Shrt", "Frml Shrt", "Oxfrd Shrt"]),
        ("Polo", ["Slm Polo", "Pq Polo", "Sprt Polo"]),
        ("Sweatshirt", ["Crwnck Swt", "Zip Swt", "Gphc Swt", "Hd Swt"]),
    ],
    "bottoms": [
        ("Jeans", ["Slm Jns", "Strt Jns", "Skny Jns", "Rpd Jns", "Btct Jns"]),
        ("Trousers", ["Chn Trs", "Fml Trs", "Cgo Trs", "Lnn Trs"]),
        ("Shorts", ["Denim Shrt", "Cgo Shrt", "Ath Shrt", "Swim Shrt"]),
    ],
    "outerwear": [
        ("Jacket", ["Bmbr Jkt", "Dnm Jkt", "Lthr Jkt", "Pfr Jkt", "Wdbrkr Jkt"]),
        ("Blazer", ["Slm Blzr", "Sgbl Blzr", "Lnn Blzr", "Tweed Blzr"]),
        ("Hoodie", ["Zip Hd", "Plvr Hd", "Flce Hd", "Tech Hd"]),
    ],
}

KAGL_PATTERNS = [
    "Slim Fit", "Regular Fit", "Straight Fit", "Oversized", "Relaxed Fit",
    "Dry Fit", "Dri Fit", "Quick Dry", "Stretch", "Comfort Fit",
]


def gen_atlas_query(category: str) -> str:
    garments = ATLAS_GARMENTS_BY_CAT.get(category, ["Garment"])
    pattern = rng.choice(PATTERNS)
    material = rng.choice(MATERIALS)
    color = rng.choice(COLORS)
    garment = rng.choice(garments)
    template = rng.choice(ATLAS_PATTERNS)

    result = template.format(
        pattern=pattern,
        material=material,
        garment=garment,
        color=color,
        fit=rng.choice(FITS),
        adjective=rng.choice(ADJECTIVES),
        detail=rng.choice(DETAILS),
    )
    return result


def gen_kagl_query(category: str) -> str:
    garment_options = KAGL_GARMENTS_BY_CAT.get(category)
    if not garment_options:
        return None

    brand = rng.choice(KAGL_BRANDS)
    gender = rng.choice(KAGL_GENDERS)
    color = rng.choice(KAGL_COLORS_ABBREV)

    garment_name, style_codes = rng.choice(garment_options)
    style = rng.choice(style_codes)

    # Vary format to match KAGL diversity
    fmt = rng.randint(0, 3)
    if fmt == 0:
        return f"{brand} {gender} {style} {color} {garment_name}"
    elif fmt == 1:
        return f"{brand} {gender} {color} {style} {garment_name}"
    elif fmt == 2:
        model = f"{''.join(rng.choices('ABCDEFGHJKLMNPQRSTUVWXYZ0123456789', k=rng.randint(2,5)))}"
        return f"{brand} {gender} {model} {color} {garment_name}"
    else:
        return f"{brand} {gender} {garment_name} {style} {color}"


def main():
    # Load training images grouped by l1_category
    print("Loading training pairs by category...")
    cat_images: dict[str, list[str]] = {}
    with open(TRAINING_DATA) as f:
        for line in f:
            d = json.loads(line)
            cat = d.get("l1_category", "other")
            img = d.get("image_path", "")
            if img:
                cat_images.setdefault(cat, [])
                if img not in cat_images[cat]:
                    cat_images[cat].append(img)

    total_by_cat = {k: len(v) for k, v in cat_images.items()}
    print(f"  Categories: {sorted(total_by_cat.items(), key=lambda x: -x[1])}")

    # Generation targets per category
    # Atlas style: top/bottom/dress/outerwear/activewear/swimwear/intimates
    # KAGL style: accessories/bags/shoes/tops/bottoms/outerwear
    ATLAS_CATS = ["tops", "bottoms", "dresses", "outerwear", "activewear", "swimwear", "intimates"]
    KAGL_CATS = ["accessories", "bags", "shoes", "tops", "bottoms", "outerwear"]
    N_PER_CAT_ATLAS = 500   # 7 cats × 500 = 3500 atlas queries
    N_PER_CAT_KAGL = 350    # 6 cats × 350 = 2100 kagl queries
    # Total target: ~5600

    results = []

    print("\nGenerating ATLAS-style queries...")
    for cat in ATLAS_CATS:
        imgs = cat_images.get(cat, [])
        if not imgs:
            print(f"  {cat}: no images — skipping")
            continue
        generated = 0
        for _ in range(N_PER_CAT_ATLAS):
            query = gen_atlas_query(cat)
            img = rng.choice(imgs)
            results.append({
                "query": query,
                "image_path": img,
                "l1_category": cat,
                "style": "atlas",
                "source": "synthetic_atlas_kagl_large",
            })
            generated += 1
        print(f"  {cat}: {generated} atlas queries")

    print("\nGenerating KAGL-style queries...")
    for cat in KAGL_CATS:
        imgs = cat_images.get(cat, [])
        if not imgs:
            print(f"  {cat}: no images — skipping")
            continue
        generated = 0
        for _ in range(N_PER_CAT_KAGL):
            query = gen_kagl_query(cat)
            if query is None:
                continue
            img = rng.choice(imgs)
            results.append({
                "query": query,
                "image_path": img,
                "l1_category": cat,
                "style": "kagl",
                "source": "synthetic_atlas_kagl_large",
            })
            generated += 1
        print(f"  {cat}: {generated} kagl queries")

    rng.shuffle(results)

    print(f"\nTotal generated: {len(results)}")
    print(f"  Atlas-style: {sum(1 for r in results if r['style'] == 'atlas')}")
    print(f"  KAGL-style:  {sum(1 for r in results if r['style'] == 'kagl')}")

    with open(OUT_PATH, "w") as f:
        for item in results:
            f.write(json.dumps(item) + "\n")

    print(f"Saved: {OUT_PATH}")

    # Quick quality check — print 10 random samples
    print("\nSample queries:")
    for item in rng.sample(results, min(10, len(results))):
        print(f"  [{item['style']:5s}] [{item['l1_category']:12s}] {item['query']}")


if __name__ == "__main__":
    main()
