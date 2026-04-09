"""
MODA Phase 3.7 — Fine-tune GLiNER2 for Fashion NER

Builds NER training data from H&M structured product fields (weak supervision)
and fine-tunes GLiNER2 with LoRA for parameter-efficient adaptation.

Entity types trained:
  color, garment type, gender, pattern, material, fit style

Data construction:
  For each H&M article, the structured fields (colour_group_name,
  product_type_name, etc.) are matched against spans in prod_name
  and detail_desc. Only examples with at least one valid span match
  are kept.

Usage:
  python benchmark/train_fashion_ner.py
  python benchmark/train_fashion_ner.py --epochs 5 --batch-size 4
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import re
import time
from pathlib import Path
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

_REPO = Path(__file__).parent.parent
HNM_DIR = _REPO / "data" / "raw" / "hnm_real"
SPLITS_PATH = _REPO / "data" / "processed" / "query_splits.json"
OUTPUT_DIR = _REPO / "models" / "moda-fashion-ner"
TRAIN_DATA_PATH = _REPO / "data" / "processed" / "ner_training_data.jsonl"
TRAIN_DATA_V2_PATH = _REPO / "data" / "processed" / "ner_training_data_v2.jsonl"
EVAL_DATA_PATH = _REPO / "data" / "processed" / "ner_eval_data.jsonl"

# ── Reverse maps: H&M taxonomy value → common query-side synonyms ────────────
# Used to find entity spans in product text that use different vocabulary
_COLOR_SYNONYMS: dict[str, list[str]] = {
    "Black": ["black"],
    "White": ["white"],
    "Off White": ["off white", "cream", "ivory", "ecru"],
    "Blue": ["blue"],
    "Dark Blue": ["dark blue", "navy", "navy blue", "indigo"],
    "Light Blue": ["light blue", "baby blue", "sky blue", "powder blue"],
    "Red": ["red"],
    "Dark Red": ["dark red", "burgundy", "wine", "maroon"],
    "Pink": ["pink"],
    "Light Pink": ["light pink", "blush", "rose"],
    "Dark Pink": ["dark pink", "fuchsia", "magenta", "hot pink"],
    "Green": ["green"],
    "Dark Green": ["dark green", "forest green", "hunter green", "emerald"],
    "Light Green": ["light green", "mint", "sage", "pistachio"],
    "Greenish Khaki": ["khaki", "olive", "olive green", "army green"],
    "Grey": ["grey", "gray"],
    "Dark Grey": ["dark grey", "dark gray", "charcoal", "anthracite"],
    "Light Grey": ["light grey", "light gray", "heather grey", "marl"],
    "Beige": ["beige", "tan", "camel", "nude", "sand"],
    "Yellow": ["yellow"],
    "Yellowish Brown": ["brown", "mustard", "ochre", "rust", "amber"],
    "Orange": ["orange", "burnt orange"],
    "Purple": ["purple"],
    "Light Purple": ["lavender", "lilac", "mauve", "violet"],
    "Turquoise": ["turquoise"],
    "Dark Turquoise": ["teal", "petrol", "teal blue"],
    "Gold": ["gold", "golden"],
    "Silver": ["silver"],
}

_GARMENT_SYNONYMS: dict[str, list[str]] = {
    "T-shirt": ["t-shirt", "tee", "tshirt"],
    "Vest top": ["vest top", "tank top", "tank", "strap top", "camisole"],
    "Top": ["top"],
    "Sweater": ["sweater", "hoodie", "sweatshirt", "jumper", "pullover"],
    "Cardigan": ["cardigan", "cardi"],
    "Blouse": ["blouse", "shirt"],
    "Dress": ["dress"],
    "Skirt": ["skirt"],
    "Trousers": ["trousers", "pants", "jeans", "chinos", "joggers"],
    "Leggings/Tights": ["leggings", "tights"],
    "Shorts": ["shorts"],
    "Coat": ["coat", "parka", "overcoat"],
    "Jacket": ["jacket", "bomber", "windbreaker"],
    "Blazer": ["blazer"],
    "Bra": ["bra", "bralette"],
    "Bikini top": ["bikini top", "bikini"],
    "Swimsuit": ["swimsuit", "one piece", "bathing suit"],
    "Underwear bottom": ["underwear", "briefs", "boxers", "knickers"],
    "Bodysuit": ["bodysuit", "body"],
    "Socks": ["socks", "ankle socks"],
    "Hat": ["hat"],
    "Cap/peaked": ["cap", "baseball cap", "peaked cap"],
    "Beanie": ["beanie"],
    "Shoes": ["shoes", "sneakers", "trainers"],
    "Boots": ["boots", "ankle boots"],
    "Sandals": ["sandals", "flip flops", "slides"],
    "Bag": ["bag", "handbag", "tote"],
    "Backpack": ["backpack", "rucksack"],
    "Scarf": ["scarf"],
    "Gloves": ["gloves"],
    "Earring": ["earring", "earrings"],
    "Necklace": ["necklace", "pendant", "chain"],
    "Bracelet": ["bracelet", "bangle"],
}

_PATTERN_VALUES = {
    "Solid", "Stripe", "Check", "Dot", "Print", "Floral",
    "Lace", "Melange", "Denim", "Colour blocking",
    "Glittering/Metallic", "Embroidery", "Mesh",
    "Application/3D", "Contrast", "Chambray",
    "Transparent", "Treatment", "Neps", "Jacquard",
    "Mixed solid/pattern", "Argyle", "Front print",
}
_SKIP_PATTERNS = {"Solid", "Other pattern", "Unknown"}

_MATERIAL_KEYWORDS = [
    "cotton", "polyester", "viscose", "elastane", "nylon", "linen",
    "wool", "silk", "denim", "jersey", "fleece", "velvet", "satin",
    "leather", "faux leather", "mesh", "chiffon", "twill", "corduroy",
    "modal", "lyocell", "tencel", "cashmere", "acrylic", "spandex",
    "rayon", "organza", "canvas", "chambray", "crepe",
]

_FIT_KEYWORDS = [
    "slim fit", "regular fit", "relaxed fit", "loose fit", "oversized",
    "skinny", "straight", "wide leg", "tapered", "bootcut", "flared",
    "cropped", "longline", "fitted", "a-line", "wrap", "bodycon",
]

_GENDER_MAP = {
    "Ladieswear": ["women", "womens", "ladies", "ladieswear"],
    "Menswear": ["men", "mens", "menswear"],
    "Baby/Children": ["baby", "kids", "children", "boys", "girls"],
    "Sport": ["sport", "sportswear"],
    "Divided": ["divided"],
}


def _find_span(text_lower: str, candidates: list[str]) -> Optional[str]:
    """Find the first candidate that appears as a word boundary match in text."""
    for c in candidates:
        pattern = r'\b' + re.escape(c.lower()) + r'\b'
        m = re.search(pattern, text_lower)
        if m:
            return text_lower[m.start():m.end()]
    return None


def _find_keyword_spans(text_lower: str, keywords: list[str]) -> list[str]:
    """Find all keyword matches in text."""
    found = []
    for kw in keywords:
        pattern = r'\b' + re.escape(kw.lower()) + r'\b'
        if re.search(pattern, text_lower):
            found.append(kw)
    return found


def build_training_example(row: dict) -> Optional[dict]:
    """
    Convert an H&M article row into a GLiNER2 NER training example.

    Returns None if no entity spans can be found in the text.
    """
    prod_name = str(row.get("prod_name", "")).strip()
    detail_desc = str(row.get("detail_desc", "")).strip()
    if not prod_name:
        return None

    text = f"{prod_name} | {detail_desc[:200]}" if detail_desc else prod_name
    text_lower = text.lower()
    entities: dict[str, list[str]] = {}

    # Color
    color_val = str(row.get("colour_group_name", "")).strip()
    if color_val and color_val in _COLOR_SYNONYMS:
        span = _find_span(text_lower, _COLOR_SYNONYMS[color_val])
        if span:
            entities.setdefault("color", []).append(span)

    # Garment type
    ptype = str(row.get("product_type_name", "")).strip()
    if ptype and ptype in _GARMENT_SYNONYMS:
        span = _find_span(text_lower, _GARMENT_SYNONYMS[ptype])
        if span:
            entities.setdefault("garment type", []).append(span)

    # Pattern
    pattern_val = str(row.get("graphical_appearance_name", "")).strip()
    if pattern_val and pattern_val not in _SKIP_PATTERNS:
        span = _find_span(text_lower, [pattern_val.lower()])
        if span:
            entities.setdefault("pattern", []).append(span)

    # Gender (from index_group_name)
    gender_val = str(row.get("index_group_name", "")).strip()
    if gender_val and gender_val in _GENDER_MAP:
        span = _find_span(text_lower, _GENDER_MAP[gender_val])
        if span:
            entities.setdefault("gender", []).append(span)

    # Material (from detail_desc keywords)
    if detail_desc:
        mat_spans = _find_keyword_spans(text_lower, _MATERIAL_KEYWORDS)
        if mat_spans:
            entities["material"] = mat_spans[:2]

    # Fit style (from detail_desc / prod_name keywords)
    fit_spans = _find_keyword_spans(text_lower, _FIT_KEYWORDS)
    if fit_spans:
        entities["fit style"] = fit_spans[:2]

    if not entities:
        return None

    return {"input": text, "output": {"entities": entities}}


def load_train_article_ids() -> set[str]:
    """Get article IDs that appear in the train split queries."""
    with open(SPLITS_PATH) as f:
        splits = json.load(f)
    train_qids = set(splits["train"])

    qrels_path = HNM_DIR / "qrels.csv"
    train_aids: set[str] = set()
    with open(qrels_path, newline="") as f:
        for row in csv.DictReader(f):
            qid = row["query_id"].strip()
            if qid not in train_qids:
                continue
            for aid in row.get("positive_ids", "").split():
                train_aids.add(aid.strip())
            for aid in row.get("negative_ids", "").split():
                train_aids.add(aid.strip())
    return train_aids


def build_dataset(max_examples: int = 15_000) -> tuple[list[dict], list[dict]]:
    """Build train + val NER datasets from H&M articles."""
    log.info("Loading train-split article IDs...")
    train_aids = load_train_article_ids()
    log.info("  %d articles linked to train queries", len(train_aids))

    log.info("Loading articles and building NER examples...")
    import pandas as pd
    df = pd.read_csv(HNM_DIR / "articles.csv", dtype=str).fillna("")

    train_only = df[df["article_id"].isin(train_aids)]
    log.info("  %d articles in train split", len(train_only))

    examples = []
    entity_counts: dict[str, int] = {}
    for _, row in train_only.iterrows():
        ex = build_training_example(row.to_dict())
        if ex:
            examples.append(ex)
            for etype in ex["output"]["entities"]:
                entity_counts[etype] = entity_counts.get(etype, 0) + 1

    log.info("  %d valid examples from %d articles", len(examples), len(train_only))
    for etype, cnt in sorted(entity_counts.items(), key=lambda x: -x[1]):
        log.info("    %s: %d mentions", etype, cnt)

    random.seed(42)
    random.shuffle(examples)
    if len(examples) > max_examples:
        examples = examples[:max_examples]
        log.info("  Capped to %d examples", max_examples)

    split_idx = int(len(examples) * 0.9)
    train_data = examples[:split_idx]
    val_data = examples[split_idx:]
    log.info("  Train: %d | Val: %d", len(train_data), len(val_data))

    return train_data, val_data


def save_jsonl(data: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for rec in data:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    log.info("Saved %d records → %s", len(data), path)


def train_ner(
    train_data: list[dict],
    val_data: list[dict],
    epochs: int = 5,
    batch_size: int = 4,
    use_lora: bool = True,
    lora_r: int = 16,
):
    """Fine-tune GLiNER2 on fashion NER data."""
    from gliner2 import GLiNER2
    from gliner2.training.trainer import TrainingConfig, GLiNER2Trainer

    log.info("Loading GLiNER2 base model: fastino/gliner2-base-v1")
    model = GLiNER2.from_pretrained("fastino/gliner2-base-v1")

    output_dir = str(OUTPUT_DIR)

    # batch_size=1 works around a GLiNER2 bug in compute_span_rep_batched:
    # when samples in a batch have different token lengths, span_rep is sliced
    # to actual length but span_mask keeps the padded length, causing a
    # tensor shape mismatch in compute_struct_loss. This is especially
    # problematic with short, variable-length query texts (v2 data).
    config = TrainingConfig(
        output_dir=output_dir,
        experiment_name="moda-fashion-ner",
        num_epochs=epochs,
        batch_size=1,
        eval_batch_size=1,
        gradient_accumulation_steps=8,
        encoder_lr=1e-5,
        task_lr=5e-4,
        scheduler_type="cosine",
        warmup_ratio=0.1,
        fp16=False,
        bf16=False,
        eval_strategy="epoch",
        save_best=True,
        metric_for_best="eval_loss",
        greater_is_better=False,
        early_stopping=True,
        early_stopping_patience=2,
        logging_steps=50,
        save_total_limit=2,
        num_workers=0,
        seed=42,
        validate_data=True,
        use_lora=use_lora,
        lora_r=lora_r,
        lora_alpha=lora_r * 2.0,
        lora_dropout=0.05,
        lora_target_modules=["encoder", "span_rep", "classifier"],
        save_adapter_only=True,
    )

    log.info("Training config: epochs=%d, batch=%d, LoRA=%s (r=%d)",
             epochs, batch_size, use_lora, lora_r)

    trainer = GLiNER2Trainer(model=model, config=config)

    t0 = time.time()
    results = trainer.train(train_data=train_data, eval_data=val_data)

    elapsed = (time.time() - t0) / 60
    log.info("Training complete in %.1f min", elapsed)
    log.info("  Total steps: %d", results["total_steps"])
    log.info("  Best eval_loss: %.4f", results["best_metric"])

    return results


def evaluate_ner(model_path: str, test_queries: list[tuple[str, str]], n_samples: int = 200):
    """
    Compare fine-tuned vs off-the-shelf GLiNER2 on real query NER extraction.

    Reports per-entity-type extraction counts and qualitative examples.
    """
    import sys
    sys.path.insert(0, str(_REPO))
    from benchmark.query_expansion import FashionNER2

    log.info("Evaluating fine-tuned model vs off-the-shelf...")

    # Off-the-shelf
    ner_base = FashionNER2(model_name="fastino/gliner2-base-v1", threshold=0.4)

    # Fine-tuned
    ner_ft = FashionNER2(model_name=model_path, threshold=0.4)

    random.seed(42)
    sample = random.sample(test_queries, min(n_samples, len(test_queries)))

    base_counts: dict[str, int] = {}
    ft_counts: dict[str, int] = {}
    diffs = []

    for qid, text in sample:
        base_ents = ner_base.extract(text)
        ft_ents = ner_ft.extract(text)

        for label, vals in base_ents.items():
            base_counts[label] = base_counts.get(label, 0) + len(vals)
        for label, vals in ft_ents.items():
            ft_counts[label] = ft_counts.get(label, 0) + len(vals)

        if base_ents != ft_ents:
            diffs.append((text, base_ents, ft_ents))

    log.info("\nExtraction counts (%d queries):", len(sample))
    all_labels = sorted(set(list(base_counts.keys()) + list(ft_counts.keys())))
    print(f"\n{'Label':<20} {'Off-shelf':>10} {'Fine-tuned':>10} {'Delta':>10}")
    print("-" * 55)
    for label in all_labels:
        b = base_counts.get(label, 0)
        f = ft_counts.get(label, 0)
        delta = f - b
        sign = "+" if delta > 0 else ""
        print(f"{label:<20} {b:>10} {f:>10} {sign}{delta:>9}")

    if diffs:
        print(f"\nExample differences (showing first 10 of {len(diffs)}):")
        for text, base, ft in diffs[:10]:
            print(f"  Query: {text!r}")
            print(f"    Base: {base}")
            print(f"    FT:   {ft}")
            print()

    return {"base_counts": base_counts, "ft_counts": ft_counts, "n_diffs": len(diffs)}


def main():
    parser = argparse.ArgumentParser(description="MODA 3.7: Fine-tune GLiNER2 for fashion NER")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-examples", type=int, default=15000)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--no-lora", action="store_true")
    parser.add_argument("--skip-train", action="store_true", help="Skip training, only evaluate")
    parser.add_argument("--eval-only", action="store_true", help="Alias for --skip-train")
    parser.add_argument("--v2", action="store_true",
                        help="Use LLM-generated query-level NER data (from generate_ner_labels.py)")
    args = parser.parse_args()

    if not args.skip_train and not args.eval_only:
        if args.v2:
            # Use LLM-generated query-level data
            if not TRAIN_DATA_V2_PATH.exists():
                log.error("V2 data not found at %s — run generate_ner_labels.py first", TRAIN_DATA_V2_PATH)
                return
            log.info("Loading LLM-generated NER data (v2)...")
            all_data = []
            with open(TRAIN_DATA_V2_PATH) as f:
                for line in f:
                    all_data.append(json.loads(line))
            random.seed(42)
            random.shuffle(all_data)
            if len(all_data) > args.max_examples:
                all_data = all_data[:args.max_examples]
            split_idx = int(len(all_data) * 0.9)
            train_data = all_data[:split_idx]
            val_data = all_data[split_idx:]
            log.info("  V2 data: %d train, %d val", len(train_data), len(val_data))
        else:
            # Original: build from product descriptions (v1)
            train_data, val_data = build_dataset(max_examples=args.max_examples)

        save_jsonl(train_data, TRAIN_DATA_PATH)
        save_jsonl(val_data, EVAL_DATA_PATH)

        # Train
        train_ner(
            train_data=train_data,
            val_data=val_data,
            epochs=args.epochs,
            batch_size=args.batch_size,
            use_lora=not args.no_lora,
            lora_r=args.lora_r,
        )

    # Evaluate on real queries
    best_model = OUTPUT_DIR / "best"
    if not best_model.exists():
        best_model = OUTPUT_DIR / "final"

    if best_model.exists():
        import pandas as pd
        with open(SPLITS_PATH) as f:
            splits = json.load(f)
        test_qids = set(splits["test"])
        queries_df = pd.read_csv(HNM_DIR / "queries.csv")
        test_queries = [
            (str(r["query_id"]), str(r["query_text"]))
            for _, r in queries_df.iterrows()
            if str(r["query_id"]) in test_qids
        ]
        evaluate_ner(str(best_model), test_queries, n_samples=500)
    else:
        log.warning("No trained model found at %s — skipping evaluation", OUTPUT_DIR)


if __name__ == "__main__":
    main()
