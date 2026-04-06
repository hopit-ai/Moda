"""
MODA Phase 2 — SOTA Query Understanding Module

Implements two complementary query enhancement techniques:

  1. SynonymExpander  — Client-side query-time synonym expansion
     Industry approach: Whatnot, Zalando, ASOS all use query-time (not index-time)
     expansion to avoid index bloat and false positives.
     Research basis: LESER (2025), LEAPS/Taobao (2026), OpenSearch synonym_graph docs.

  2. FashionNER  — Zero-shot attribute extraction via GLiNER (NAACL 2024)
     Outperforms ChatGPT on NER benchmarks, CPU-friendly, no fine-tuning needed.
     Extracts: COLOR, GARMENT_TYPE, MATERIAL, FIT, OCCASION, GENDER, PATTERN
     Maps extracted entities → H&M taxonomy fields → boosted OpenSearch query.
     Research basis: GLiNER paper, EcomBERT-NER (23 entity labels), QueryNER dataset.

Architecture:
  raw query
    → SynonymExpander  → expanded query string (OR terms)
    → FashionNER       → {color: "dark blue", garment_type: "dress", ...}
    → build_boosted_query() → OpenSearch multi_match + function_score boosts

Usage:
  from benchmark.query_expansion import SynonymExpander, FashionNER, build_boosted_query
"""

from __future__ import annotations
import re
import logging
from functools import lru_cache
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# 1. Synonym Dictionary — grounded in H&M taxonomy
#    Each tuple: (canonical_term, [user_synonyms...])
#    Canonical is H&M's language; synonyms are what users type.
#    Source: H&M articles.csv taxonomy + industry fashion glossary.
# ─────────────────────────────────────────────────────────────────────────────

FASHION_SYNONYMS: list[tuple[str, list[str]]] = [

    # ── Garment types (mapped to H&M product_type_name values) ───────────────
    ("t-shirt",         ["tee", "tshirt", "t shirt", "basic top", "crew neck tee"]),
    ("sweater",         ["hoodie", "hooded sweatshirt", "sweatshirt", "jumper",
                         "pullover", "hooded top", "zip hoodie", "crew neck sweater",
                         "fleece", "half zip", "quarter zip"]),
    ("trousers",        ["pants", "slacks", "bottoms", "chinos", "joggers",
                         "sweatpants", "tracksuit bottoms", "jogger pants",
                         "cargo pants", "wide leg pants", "straight leg pants"]),
    ("trousers denim",  ["jeans", "denim", "denim pants", "denim jeans", "skinny jeans",
                         "slim jeans", "wide jeans", "straight jeans", "mom jeans",
                         "boyfriend jeans", "ripped jeans", "distressed jeans"]),
    ("leggings",        ["tights", "legging", "yoga pants", "gym leggings",
                         "sport leggings", "running tights", "compression tights"]),
    ("dress",           ["frock", "sundress", "mini dress", "midi dress", "maxi dress",
                         "shirt dress", "wrap dress", "slip dress"]),
    ("skirt",           ["mini skirt", "midi skirt", "maxi skirt", "pleated skirt",
                         "a-line skirt", "pencil skirt"]),
    ("shorts",          ["short pants", "bermuda", "cargo shorts", "denim shorts",
                         "gym shorts", "running shorts", "board shorts"]),
    ("coat",            ["overcoat", "winter coat", "parka", "long coat",
                         "trench coat", "puffer coat", "padded coat"]),
    ("jacket",          ["bomber", "denim jacket", "biker jacket", "windbreaker",
                         "rain jacket", "track jacket", "varsity jacket"]),
    ("blazer",          ["suit jacket", "sport coat", "sport jacket", "formal jacket"]),
    ("blouse",          ["shirt", "top", "woven top", "button up", "button down"]),
    ("cardigan",        ["cardi", "open knit", "knit jacket"]),
    ("bra",             ["bralette", "brassiere", "sports bra", "push up bra",
                         "wireless bra", "underwire bra", "bikini bra"]),
    ("bikini top",      ["swim top", "swimwear top", "bandeau top"]),
    ("swimsuit",        ["one piece", "one-piece", "bathing suit", "swimming costume"]),
    ("swimwear bottom", ["swim shorts", "swim trunks", "board shorts swim",
                         "bikini bottom", "swim briefs"]),
    ("hat",             ["cap", "beanie", "bucket hat", "baseball cap", "snapback",
                         "fedora", "sun hat", "winter hat", "knit hat"]),
    ("boots",           ["ankle boots", "knee boots", "combat boots", "chelsea boots",
                         "riding boots", "boot"]),
    ("ballerinas",      ["flats", "ballet flats", "flat shoes", "pumps flat"]),
    ("sandals",         ["sandal", "flip flops", "thongs", "slides", "mules",
                         "strappy sandals"]),
    ("shoes",           ["sneakers", "trainers", "runners", "athletic shoes",
                         "sports shoes", "kicks", "tennis shoes"]),
    ("bag",             ["handbag", "purse", "tote", "tote bag", "shoulder bag",
                         "clutch", "evening bag"]),
    ("backpack",        ["rucksack", "daypack", "school bag"]),
    ("socks",           ["ankle socks", "crew socks", "knee socks", "trainer socks",
                         "no-show socks"]),
    ("underwear",       ["briefs", "boxers", "boxer briefs", "underpants",
                         "knickers", "underwear bottoms", "panties"]),
    ("bodysuit",        ["body suit", "leotard", "body"]),
    ("pyjama",          ["pajama", "pyjamas", "pajamas", "sleepwear", "nightwear",
                         "sleep set", "loungewear set"]),
    ("dungarees",       ["overalls", "dungaree", "bib overalls", "pinafore"]),
    ("jumper",          ["knit dress", "sweater dress"]),

    # ── Colors (mapped to H&M colour_group_name values) ──────────────────────
    ("dark blue",       ["navy", "navy blue", "midnight blue", "indigo", "cobalt"]),
    ("greenish khaki",  ["khaki", "olive", "olive green", "army green", "military green",
                         "moss green", "sage", "sage green"]),
    ("dark red",        ["burgundy", "wine", "maroon", "oxblood", "claret"]),
    ("off white",       ["ivory", "cream", "ecru", "eggshell", "bone"]),
    ("beige",           ["tan", "camel", "nude", "skin", "sand", "oatmeal", "latte"]),
    ("dark pink",       ["fuchsia", "magenta", "hot pink", "coral pink", "raspberry"]),
    ("light orange",    ["coral", "peach", "salmon", "apricot"]),
    ("light purple",    ["lavender", "lilac", "mauve", "violet", "wisteria"]),
    ("dark turquoise",  ["teal", "teal blue", "petrol", "duck egg"]),
    ("dark grey",       ["charcoal", "slate", "anthracite", "graphite"]),
    ("light grey",      ["heather grey", "heather", "marl", "melange grey"]),
    ("dark green",      ["forest green", "hunter green", "bottle green", "emerald"]),
    ("light green",     ["mint", "sage light", "pistachio", "lime"]),
    ("other blue",      ["powder blue", "sky blue", "baby blue", "cornflower",
                         "electric blue", "royal blue", "denim blue"]),
    ("yellowish brown", ["mustard", "ochre", "amber", "golden brown", "rust"]),

    # ── Materials / Fabrics ───────────────────────────────────────────────────
    ("jersey",          ["jersey fabric", "knit fabric", "stretch fabric"]),
    ("denim",           ["denim fabric", "denim material"]),
    ("linen",           ["linen fabric", "linen blend"]),
    ("cotton",          ["100% cotton", "pure cotton", "organic cotton"]),
    ("fleece",          ["polar fleece", "microfleece", "sherpa"]),
    ("velvet",          ["velour", "velvet fabric"]),
    ("satin",           ["silky", "silk-like", "shiny fabric"]),
    ("leather",         ["faux leather", "pu leather", "vegan leather", "pleather"]),

    # ── Fit / Style ───────────────────────────────────────────────────────────
    ("slim",            ["slim fit", "skinny fit", "fitted", "tailored",
                         "form fitting", "body hugging"]),
    ("relaxed",         ["relaxed fit", "oversized", "baggy", "loose fit",
                         "loose fitting", "oversized fit", "boyfriend fit"]),
    ("regular",         ["regular fit", "classic fit", "standard fit", "straight fit"]),
    ("cropped",         ["crop", "crop top", "cropped top", "short length"]),
    ("ribbed",          ["rib knit", "ribbed fabric", "textured knit"]),

    # ── Occasions / Sections ──────────────────────────────────────────────────
    ("sport",           ["gym", "workout", "athletic", "training", "active",
                         "activewear", "sportswear", "running", "fitness",
                         "exercise", "performance"]),
    ("swimwear",        ["beach", "swim", "pool", "holiday wear", "resort wear"]),
    ("lingerie",        ["underwear set", "intimates", "intimate apparel"]),
    ("nightwear",       ["sleepwear", "loungewear", "pyjamas", "night suit"]),
    ("outdoor",         ["outerwear", "winter wear", "cold weather", "layering"]),

    # ── Gender ────────────────────────────────────────────────────────────────
    ("menswear",        ["mens", "men's", "men", "male", "guys", "him", "his"]),
    ("ladieswear",      ["womens", "women's", "women", "ladies", "female",
                         "her", "hers", "womenswear"]),
    ("baby",            ["infant", "newborn", "toddler", "baby clothes"]),
    ("children",        ["kids", "children's", "boys", "girls", "youth"]),

    # ── Colloquial / Regional ─────────────────────────────────────────────────
    ("earring",         ["earrings", "studs", "hoops", "drops", "dangles"]),
    ("necklace",        ["necklaces", "chain", "pendant", "choker"]),
    ("bracelet",        ["bracelets", "bangle", "bangles", "cuff"]),
    ("scarf",           ["scarves", "shawl", "wrap", "neck warmer"]),
    ("gloves",          ["mittens", "hand warmers"]),
    ("headband",        ["hair band", "hair accessory", "ear warmer", "earband",
                         "head band", "alice band"]),
    ("belt",            ["belts", "waist belt", "leather belt"]),
    ("wallet",          ["purse", "card holder", "cardholder", "billfold"]),
    ("sunglasses",      ["sunnies", "shades", "glasses"]),
]

# Build lookup: any synonym → set of expansion terms (all synonyms + canonical)
def _build_synonym_map(synonyms: list[tuple[str, list[str]]]) -> dict[str, list[str]]:
    mapping: dict[str, list[str]] = {}
    for canonical, user_terms in synonyms:
        all_terms = [canonical] + user_terms
        for term in all_terms:
            key = term.lower().strip()
            # All other terms this should expand to
            expansions = [t for t in all_terms if t.lower().strip() != key]
            if key not in mapping:
                mapping[key] = []
            mapping[key].extend(t for t in expansions if t not in mapping[key])
    return mapping


class SynonymExpander:
    """
    Client-side query-time synonym expansion.

    Industry approach (Whatnot, Zalando): expand at query time, not index time.
    This avoids index bloat, false positives, and allows zero-downtime updates.

    Strategy: detect known fashion terms in query, append OR-expanded alternatives.
    Uses longest-match to avoid partial overlaps (e.g., "dark blue" before "blue").
    """

    def __init__(self):
        self._map = _build_synonym_map(FASHION_SYNONYMS)
        # Sort by length descending for longest-match-first
        self._sorted_keys = sorted(self._map.keys(), key=len, reverse=True)

    def expand(self, query: str) -> str:
        """
        Expand a query string with synonym alternatives.

        "tee" → "tee OR t-shirt OR basic top"
        "navy hoodie" → "navy dark blue OR hoodie sweater sweatshirt"
        """
        q = query.lower().strip()
        additions: list[str] = []
        covered: set[int] = set()   # character positions already expanded

        for key in self._sorted_keys:
            # Find all occurrences using word-boundary matching
            pattern = r'\b' + re.escape(key) + r'\b'
            for m in re.finditer(pattern, q):
                start, end = m.start(), m.end()
                # Skip if any character in this span is already expanded
                span = set(range(start, end))
                if span & covered:
                    continue
                covered |= span
                expansions = self._map[key]
                if expansions:
                    additions.extend(expansions)

        if additions:
            unique_additions = list(dict.fromkeys(
                a for a in additions
                if a.lower() not in q  # don't add what's already there
            ))
            if unique_additions:
                return query + " " + " ".join(unique_additions)
        return query

    def get_expansion_terms(self, query: str) -> list[str]:
        """Return only the expansion terms (for debugging)."""
        expanded = self.expand(query)
        original_terms = set(query.lower().split())
        new_terms = [t for t in expanded.split() if t.lower() not in original_terms]
        return new_terms


# ─────────────────────────────────────────────────────────────────────────────
# 2. Fashion NER — GLiNER zero-shot entity extraction
#    Maps extracted entities → H&M taxonomy fields → OpenSearch boost queries
#    Research: GLiNER (NAACL 2024), EcomBERT-NER (23 labels)
# ─────────────────────────────────────────────────────────────────────────────

# H&M taxonomy mappings for NER-detected entities → field values
COLOR_MAP: dict[str, str] = {
    "navy": "Dark Blue", "navy blue": "Dark Blue", "dark blue": "Dark Blue",
    "blue": "Blue", "light blue": "Light Blue", "baby blue": "Light Blue",
    "black": "Black", "white": "White", "off white": "Off White",
    "cream": "Off White", "ivory": "Off White",
    "red": "Red", "dark red": "Dark Red", "burgundy": "Dark Red",
    "wine": "Dark Red", "maroon": "Dark Red",
    "pink": "Pink", "light pink": "Light Pink", "dark pink": "Dark Pink",
    "coral": "Light Orange", "hot pink": "Dark Pink", "fuchsia": "Dark Pink",
    "green": "Green", "dark green": "Dark Green", "light green": "Light Green",
    "olive": "Greenish Khaki", "khaki": "Greenish Khaki",
    "grey": "Grey", "gray": "Grey", "dark grey": "Dark Grey",
    "charcoal": "Dark Grey", "light grey": "Light Grey",
    "beige": "Beige", "tan": "Beige", "camel": "Beige", "nude": "Beige",
    "yellow": "Yellow", "mustard": "Yellowish Brown", "orange": "Orange",
    "purple": "Purple", "lavender": "Light Purple", "lilac": "Light Purple",
    "teal": "Dark Turquoise", "turquoise": "Turquoise",
    "gold": "Gold", "silver": "Silver", "brown": "Yellowish Brown",
}

GARMENT_TYPE_MAP: dict[str, str] = {
    "t-shirt": "T-shirt", "tee": "T-shirt", "tshirt": "T-shirt",
    "dress": "Dress", "skirt": "Skirt", "blouse": "Blouse", "shirt": "Blouse",
    "jeans": "Trousers", "pants": "Trousers", "trousers": "Trousers",
    "shorts": "Shorts", "leggings": "Leggings/Tights", "tights": "Leggings/Tights",
    "hoodie": "Sweater", "sweatshirt": "Sweater", "sweater": "Sweater",
    "jumper": "Sweater", "cardigan": "Cardigan", "blazer": "Blazer",
    "jacket": "Jacket", "coat": "Coat", "parka": "Coat",
    "bikini": "Bikini top", "swimsuit": "Swimsuit", "bra": "Bra",
    "boots": "Boots", "sneakers": "Shoes", "trainers": "Shoes",
    "sandals": "Sandals", "bag": "Bag", "backpack": "Backpack",
    "hat": "Hat", "cap": "Cap/peaked", "beanie": "Beanie",
    "scarf": "Scarf", "gloves": "Gloves", "socks": "Socks",
    "underwear": "Underwear bottom", "briefs": "Underwear bottom",
    "bodysuit": "Bodysuit", "top": "Top",
}

GENDER_MAP: dict[str, str] = {
    "mens": "Menswear", "men": "Menswear", "male": "Menswear", "his": "Menswear",
    "womens": "Ladieswear", "women": "Ladieswear", "ladies": "Ladieswear",
    "female": "Ladieswear", "her": "Ladieswear",
    "kids": "Baby/Children", "children": "Baby/Children",
    "boys": "Baby/Children", "girls": "Baby/Children",
}

NER_LABELS = [
    "color", "garment type", "material", "fit style",
    "occasion", "gender", "pattern", "brand",
]

# Map NER label → (OpenSearch field, boost, value_map)
LABEL_TO_FIELD: dict[str, tuple[str, float, dict]] = {
    "color":        ("colour_group_name", 4.0, COLOR_MAP),
    "garment type": ("product_type_name",  5.0, GARMENT_TYPE_MAP),
    "gender":       ("index_group_name",   2.0, GENDER_MAP),
    "occasion":     ("section_name",       2.0, {}),
    "material":     ("detail_desc",        1.5, {}),
    "fit style":    ("detail_desc",        1.0, {}),
    "pattern":      ("graphical_appearance_name", 1.5, {}),
    "brand":        ("prod_name",          3.0, {}),
}


class FashionNER:
    """
    Zero-shot fashion attribute extraction using GLiNER (NAACL 2024).

    GLiNER uses a bidirectional transformer encoder for parallel entity
    extraction — faster than LLMs, no API costs, CPU-friendly, no fine-tuning.
    Outperforms ChatGPT on standard NER benchmarks.

    Model: urchade/gliner_medium-v2.1 — best accuracy/speed tradeoff for CPU.
    """

    def __init__(self, model_name: str = "urchade/gliner_medium-v2.1",
                 threshold: float = 0.4):
        from gliner import GLiNER
        log.info("Loading GLiNER model: %s", model_name)
        self.model = GLiNER.from_pretrained(model_name)
        self.threshold = threshold
        self.labels = NER_LABELS
        log.info("GLiNER ready ✓")

    def extract(self, query: str) -> dict[str, list[str]]:
        """
        Extract fashion entities from a query.

        Returns: {"color": ["red"], "garment type": ["dress"], ...}
        """
        entities = self.model.predict_entities(
            query, self.labels, threshold=self.threshold
        )
        result: dict[str, list[str]] = {}
        for ent in entities:
            label = ent["label"]
            text  = ent["text"].lower().strip()
            result.setdefault(label, []).append(text)
        return result

    def extract_batch(self, queries: list[str]) -> list[dict[str, list[str]]]:
        """Batch extraction — more efficient than one-by-one."""
        results = []
        for q in queries:
            results.append(self.extract(q))
        return results


class FashionNER2:
    """
    Zero-shot fashion attribute extraction using GLiNER2 (EMNLP 2025).

    GLiNER2 extends the original GLiNER architecture with multi-task support
    (NER + classification + structured extraction + relation extraction)
    in a single 205M-parameter model. CPU-friendly, no fine-tuning needed.

    Model: fastino/gliner2-base-v1 — default base model.
    API is drop-in compatible with FashionNER via the same extract() interface.
    """

    def __init__(self, model_name: str = "fastino/gliner2-base-v1",
                 threshold: float = 0.4):
        from gliner2 import GLiNER2
        log.info("Loading GLiNER2 model: %s", model_name)
        self.model = GLiNER2.from_pretrained(model_name)
        self.threshold = threshold
        self.labels = NER_LABELS
        log.info("GLiNER2 ready")

    def extract(self, query: str) -> dict[str, list[str]]:
        """
        Extract fashion entities from a query.

        Returns same format as FashionNER: {"color": ["red"], "garment type": ["dress"], ...}
        """
        result_raw = self.model.extract_entities(
            query, self.labels, threshold=self.threshold
        )
        entities = result_raw.get("entities", {})
        return {
            label: [t.lower().strip() for t in texts]
            for label, texts in entities.items() if texts
        }

    def extract_batch(self, queries: list[str]) -> list[dict[str, list[str]]]:
        """Batch extraction — more efficient than one-by-one."""
        return [self.extract(q) for q in queries]


# ─────────────────────────────────────────────────────────────────────────────
# 3. Boosted OpenSearch Query Builder
#    Combines synonym expansion + NER attribute boosts into a single query
# ─────────────────────────────────────────────────────────────────────────────

def build_boosted_query(
    query: str,
    expanded_query: Optional[str] = None,
    ner_entities: Optional[dict[str, list[str]]] = None,
    top_k: int = 50,
) -> dict:
    """
    Build an OpenSearch query combining:
      1. Multi-match on expanded query (BM25 core)
      2. Function-score boosts for NER-detected attributes

    Follows industry pattern: base retrieval + attribute boosting.
    """
    search_text = expanded_query or query

    base_query = {
        "multi_match": {
            "query": search_text,
            "fields": [
                "prod_name^5",
                "product_type_name^3",
                "colour_group_name^2",
                "section_name^1.5",
                "garment_group_name^1.5",
                "detail_desc^1",
            ],
            "type": "best_fields",
            "tie_breaker": 0.3,
            "operator": "or",
        }
    }

    # If no NER entities, return simple query
    if not ner_entities:
        return {"query": base_query, "size": top_k, "_source": False}

    # Build function_score boosts from NER entities
    functions = []
    for label, values in ner_entities.items():
        if label not in LABEL_TO_FIELD:
            continue
        field, boost, value_map = LABEL_TO_FIELD[label]
        for raw_value in values:
            # Map to H&M taxonomy value if possible, else use raw
            mapped = value_map.get(raw_value, raw_value.title())
            functions.append({
                "filter": {"match": {field: mapped}},
                "weight": boost,
            })

    if not functions:
        return {"query": base_query, "size": top_k, "_source": False}

    return {
        "query": {
            "function_score": {
                "query": base_query,
                "functions": functions,
                "score_mode": "sum",
                "boost_mode": "multiply",
            }
        },
        "size": top_k,
        "_source": False,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Quick self-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    expander = SynonymExpander()
    test_queries = [
        "navy hoodie",
        "black jeans skinny",
        "red summer dress",
        "warm earband",
        "wireless bra",
        "white sneakers women",
        "khaki cargo pants",
        "burgundy velvet blazer",
        "coral bikini top",
        "charcoal grey sweatpants",
    ]

    print("=" * 65)
    print("SYNONYM EXPANSION TEST")
    print("=" * 65)
    for q in test_queries:
        expanded = expander.expand(q)
        new_terms = expander.get_expansion_terms(q)
        print(f"  IN:  {q!r}")
        print(f"  OUT: {expanded!r}")
        print(f"  NEW: {new_terms}")
        print()
