"""Phase 10c — Generate synthetic texts in atlas and KAGL style.

Generates:
1. Atlas-style: "[Adjective] [Material] [Garment] in [Color]" descriptions
2. KAGL-style: "[Brand] [Gender] [Style] [Color] [Garment]" descriptions
3. Ethnic wear descriptions (kurta, saree, lehenga)
4. Accessories descriptions (watches, jewelry, bags)

Uses PaleBlueDot AI (Gemini Flash) for generation.
"""

import json
import os
import time
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "data" / "synthetic" / "phase10c_atlas_kagl"
OUT_DIR.mkdir(parents=True, exist_ok=True)

client = OpenAI(
    api_key=os.getenv("PALEBLUEDOT_API_KEY"),
    base_url="https://api.palebluedot.ai/v1",
)
MODEL = "google/gemini-2.5-flash-preview"

# Atlas categories and their sub-categories
ATLAS_CATEGORIES = {
    "Shorts": ["denim shorts", "cargo shorts", "athletic shorts", "chino shorts", "swim shorts"],
    "Shirts": ["formal shirts", "casual shirts", "linen shirts", "oxford shirts", "flannel shirts"],
    "Tops&Tees": ["crew neck tees", "v-neck tees", "polo shirts", "henley", "tank tops"],
    "Trousers": ["formal trousers", "chinos", "linen pants", "corduroy pants", "pleated trousers"],
    "Sweatshirts&Hoodies": ["pullover hoodies", "zip hoodies", "crewneck sweatshirts", "fleece hoodies"],
    "Blazers&Suits": ["single breasted blazer", "double breasted blazer", "linen suit", "tweed blazer"],
    "Jeans": ["slim fit jeans", "straight fit jeans", "bootcut jeans", "skinny jeans", "ripped jeans"],
    "Jackets": ["bomber jacket", "leather jacket", "denim jacket", "puffer jacket", "windbreaker"],
    "Sarees": ["silk saree", "cotton saree", "chiffon saree", "georgette saree", "banarasi saree"],
    "Kurta": ["cotton kurta", "silk kurta", "linen kurta", "embroidered kurta", "printed kurta"],
    "Lehenga Choli": ["bridal lehenga", "party wear lehenga", "silk lehenga", "net lehenga"],
    "Salwar Kameez": ["anarkali suit", "palazzo suit", "churidar suit", "patiala suit"],
}

# KAGL categories
KAGL_CATEGORIES = {
    "Watches": ["analog watch", "digital watch", "chronograph watch", "smartwatch", "sports watch"],
    "Sunglasses": ["aviator sunglasses", "wayfarer sunglasses", "round sunglasses", "sport sunglasses"],
    "Casual Shoes": ["canvas shoes", "slip-on shoes", "boat shoes", "espadrilles", "loafers"],
    "Sports Shoes": ["running shoes", "training shoes", "basketball shoes", "tennis shoes"],
    "Heels": ["stiletto heels", "block heels", "wedge heels", "kitten heels", "platform heels"],
    "Handbags": ["tote bag", "satchel bag", "crossbody bag", "shoulder bag", "hobo bag"],
    "Wallets": ["bi-fold wallet", "tri-fold wallet", "card holder", "zip-around wallet"],
    "Perfume": ["eau de parfum", "eau de toilette", "body mist", "cologne"],
    "Earrings": ["stud earrings", "hoop earrings", "drop earrings", "chandelier earrings"],
    "Flip Flops": ["rubber flip flops", "leather sandals", "sports sandals", "slides"],
}

BRANDS = ["Nike", "Adidas", "Puma", "Reebok", "Levi's", "Zara", "H&M", "Calvin Klein",
           "Tommy Hilfiger", "Under Armour", "New Balance", "Vans", "Converse", "Fossil",
           "Titan", "Fastrack", "Ray-Ban", "Wildcraft", "Woodland", "Bata"]

COLORS = ["Black", "White", "Navy Blue", "Grey", "Red", "Green", "Beige", "Brown",
           "Maroon", "Olive", "Teal", "Mustard", "Coral", "Burgundy", "Charcoal",
           "Royal Blue", "Fawn", "Cream", "Turquoise", "Lavender"]

MATERIALS = ["Cotton", "Silk", "Linen", "Polyester", "Rayon", "Denim", "Leather",
             "Wool", "Chiffon", "Georgette", "Velvet", "Satin", "Dupion Silk",
             "Art Silk", "Crepe", "Nylon", "Canvas", "Suede", "Knit", "Fleece"]


ATLAS_PROMPT = """Generate {n} product descriptions in this EXACT style for the category "{category}".

Style: "[Adjective/Pattern] [Material] [Sub-style] [Garment] in [Color]"

Examples of the exact format:
- "Plain Dupion Silk Dhoti Kurta in Fawn"
- "Solid Color Rayon Asymmetric Dhoti Kurta in Black"
- "Printed Georgette Saree in Teal"
- "Embroidered Cotton Slim Fit Shirt in Navy Blue"
- "Solid Color Linen Regular Fit Trousers in Beige"

Requirements:
- Each description must follow this structured format precisely
- Vary: materials ({materials}), colors ({colors}), patterns (solid, printed, embroidered, plain, striped, checked)
- Include fit/style qualifiers where appropriate (slim fit, regular fit, relaxed fit)
- Make them diverse and realistic
- One per line, no numbering"""

KAGL_PROMPT = """Generate {n} product titles in this EXACT style for the category "{category}".

Style: "[Brand] [Gender] [Style/Model] [Color] [Product Type]"

Examples of the exact format:
- "Nike Men As 7 Sw Temp Grey Shorts"
- "Puma Women Rs Running White Sports Shoes"
- "Fossil Men Grant Chrono Brown Analog Watch"
- "Ray-Ban Unisex Aviator Gold Sunglasses"
- "Wildcraft Unisex Urbana Black Backpack"

Requirements:
- Use these brands: {brands}
- Use abbreviated style (like real e-commerce product titles)
- Mix genders: Men, Women, Unisex, Boys, Girls
- Include model names/codes (made up but realistic)
- Each on its own line, no numbering
- Make them feel like real product catalog entries"""


def generate_batch(prompt: str, category: str, n: int = 50, style: str = "atlas") -> list[str]:
    if style == "atlas":
        formatted = ATLAS_PROMPT.format(
            n=n, category=category,
            materials=", ".join(MATERIALS[:10]),
            colors=", ".join(COLORS[:10]),
        )
    else:
        formatted = KAGL_PROMPT.format(
            n=n, category=category,
            brands=", ".join(BRANDS[:10]),
        )

    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": formatted}],
            temperature=0.9,
            max_tokens=4000,
        )
        text = resp.choices[0].message.content.strip()
        lines = [l.strip() for l in text.split("\n") if l.strip() and len(l.strip()) > 5]
        # Remove numbering if present
        cleaned = []
        for l in lines:
            if l[0].isdigit() and ('. ' in l[:4] or ') ' in l[:4]):
                l = l.split('. ', 1)[-1] if '. ' in l[:4] else l.split(') ', 1)[-1]
            if l.startswith('- '):
                l = l[2:]
            cleaned.append(l.strip())
        return cleaned
    except Exception as e:
        print(f"  ERROR: {e}")
        return []


def main():
    all_texts = []
    
    # Generate atlas-style texts
    print("Generating ATLAS-style descriptions...")
    for category, subcats in ATLAS_CATEGORIES.items():
        texts = generate_batch(ATLAS_PROMPT, category, n=60, style="atlas")
        for t in texts:
            all_texts.append({"text": t, "style": "atlas", "category": category})
        print(f"  {category}: {len(texts)} texts")
        time.sleep(0.5)

    # Generate KAGL-style texts
    print("\nGenerating KAGL-style descriptions...")
    for category, subcats in KAGL_CATEGORIES.items():
        texts = generate_batch(KAGL_PROMPT, category, n=60, style="kagl")
        for t in texts:
            all_texts.append({"text": t, "style": "kagl", "category": category})
        print(f"  {category}: {len(texts)} texts")
        time.sleep(0.5)

    # Generate extra batches for the most critical categories
    critical_atlas = ["Sarees", "Kurta", "Lehenga Choli", "Blazers&Suits", "Jackets"]
    critical_kagl = ["Watches", "Casual Shoes", "Sports Shoes", "Handbags", "Sunglasses"]
    
    print("\nGenerating extra batches for critical categories...")
    for category in critical_atlas:
        texts = generate_batch(ATLAS_PROMPT, category, n=80, style="atlas")
        for t in texts:
            all_texts.append({"text": t, "style": "atlas", "category": category})
        print(f"  atlas/{category}: +{len(texts)} texts")
        time.sleep(0.5)

    for category in critical_kagl:
        texts = generate_batch(KAGL_PROMPT, category, n=80, style="kagl")
        for t in texts:
            all_texts.append({"text": t, "style": "kagl", "category": category})
        print(f"  kagl/{category}: +{len(texts)} texts")
        time.sleep(0.5)

    # Save
    out_path = OUT_DIR / "atlas_kagl_synthetic.jsonl"
    with open(out_path, "w") as f:
        for item in all_texts:
            f.write(json.dumps(item) + "\n")

    print(f"\nTotal generated: {len(all_texts)} texts")
    print(f"  Atlas-style: {sum(1 for t in all_texts if t['style']=='atlas')}")
    print(f"  KAGL-style: {sum(1 for t in all_texts if t['style']=='kagl')}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
