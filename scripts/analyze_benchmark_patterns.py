"""
Deep pattern analysis of all 7 Marqo benchmark datasets.

Goal: Understand exactly what patterns each benchmark tests so we can design
training data that teaches those patterns without data leakage.

Analyzes:
  1. Schema & field coverage per dataset
  2. Text field distributions (length, vocabulary, structure)
  3. Category/attribute distributions (what garment types, colors, etc.)
  4. Query-type analysis (how text-to-image queries differ from category queries)
  5. Cross-dataset pattern comparison (shared vs unique patterns)
  6. Difficulty signals (text specificity, category granularity)

Outputs:
  - results/benchmark_patterns/per_dataset/ — one JSON + markdown per dataset
  - results/benchmark_patterns/cross_dataset_analysis.md — comparative summary
  - results/benchmark_patterns/training_data_implications.md — actionable insights

Usage:
    python scripts/analyze_benchmark_patterns.py [--max-rows 0]  # 0 = all rows
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import datasets

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results" / "benchmark_patterns"
PER_DS_DIR = RESULTS_DIR / "per_dataset"

DATASET_CONFIGS = {
    "atlas": {
        "hf_id": "Marqo/atlas",
        "text_cols": ["text"],
        "category_cols": ["category", "sub-category"],
        "attribute_cols": ["gender"],
        "tasks": ["text-to-image", "sub-category-to-product"],
    },
    "fashion200k": {
        "hf_id": "Marqo/fashion200k",
        "text_cols": ["text"],
        "category_cols": ["category1", "category2", "category3"],
        "attribute_cols": [],
        "tasks": ["text-to-image", "category-to-product", "sub-category-to-product", "fine-category-to-product"],
    },
    "KAGL": {
        "hf_id": "Marqo/KAGL",
        "text_cols": ["text"],
        "category_cols": ["category1", "category2", "category3"],
        "attribute_cols": ["gender", "baseColour", "season", "usage"],
        "tasks": ["text-to-image", "category-to-product", "sub-category-to-product",
                  "fine-category-to-product", "color-to-product", "season-to-product", "usage-to-product"],
    },
    "polyvore": {
        "hf_id": "Marqo/polyvore",
        "text_cols": ["text"],
        "category_cols": ["category"],
        "attribute_cols": [],
        "tasks": ["text-to-image", "category-to-product"],
    },
    "deepfashion_inshop": {
        "hf_id": "Marqo/deepfashion-inshop",
        "text_cols": ["text", "description"],
        "category_cols": ["category1", "category2", "category3"],
        "attribute_cols": ["color"],
        "tasks": ["text-to-image", "category-to-product", "sub-category-to-product", "color-to-product"],
    },
    "deepfashion_multimodal": {
        "hf_id": "Marqo/deepfashion-multimodal",
        "text_cols": ["text"],
        "category_cols": ["category1", "category2"],
        "attribute_cols": [],
        "tasks": ["text-to-image", "category-to-product"],
    },
    "iMaterialist": {
        "hf_id": "Marqo/iMaterialist",
        "text_cols": [],
        "category_cols": [],
        "attribute_cols": [],
        "tasks": ["category-to-image", "style-to-image", "neckline-to-image"],
        "skip_streaming": True,
    },
}


def tokenize_simple(text: str) -> list[str]:
    """Simple whitespace + punctuation tokenizer."""
    return re.findall(r"[a-z0-9]+(?:'[a-z]+)?", text.lower())


def analyze_text_field(values: list[str], field_name: str) -> dict:
    """Deep analysis of a text field."""
    non_null = [v for v in values if v and str(v).strip()]
    if not non_null:
        return {"field": field_name, "count": 0, "null_rate": 1.0}

    lengths_chars = [len(v) for v in non_null]
    lengths_words = [len(tokenize_simple(v)) for v in non_null]

    all_tokens = []
    for v in non_null:
        all_tokens.extend(tokenize_simple(v))
    token_freq = Counter(all_tokens)

    unique_texts = set(non_null)
    text_freq = Counter(non_null)

    has_comma = sum(1 for v in non_null if "," in v)
    has_period = sum(1 for v in non_null if "." in v)
    has_numbers = sum(1 for v in non_null if re.search(r"\d", v))
    starts_upper = sum(1 for v in non_null if v[0].isupper())
    all_lower = sum(1 for v in non_null if v == v.lower())
    all_upper = sum(1 for v in non_null if v == v.upper())

    bigrams = Counter()
    for v in non_null:
        tokens = tokenize_simple(v)
        for i in range(len(tokens) - 1):
            bigrams[(tokens[i], tokens[i + 1])] += 1

    pattern_types = Counter()
    for v in non_null:
        words = tokenize_simple(v)
        n = len(words)
        if n <= 2:
            pattern_types["very_short (1-2 words)"] += 1
        elif n <= 5:
            pattern_types["short (3-5 words)"] += 1
        elif n <= 10:
            pattern_types["medium (6-10 words)"] += 1
        elif n <= 20:
            pattern_types["long (11-20 words)"] += 1
        else:
            pattern_types["very_long (21+ words)"] += 1

    return {
        "field": field_name,
        "total_rows": len(values),
        "non_null_count": len(non_null),
        "null_rate": round(1 - len(non_null) / len(values), 4),
        "unique_count": len(unique_texts),
        "uniqueness_ratio": round(len(unique_texts) / len(non_null), 4),
        "duplication_ratio": round(1 - len(unique_texts) / len(non_null), 4),
        "top_repeated_values": text_freq.most_common(30),
        "char_length": {
            "min": min(lengths_chars),
            "max": max(lengths_chars),
            "mean": round(sum(lengths_chars) / len(lengths_chars), 1),
            "median": sorted(lengths_chars)[len(lengths_chars) // 2],
            "p10": sorted(lengths_chars)[int(len(lengths_chars) * 0.1)],
            "p90": sorted(lengths_chars)[int(len(lengths_chars) * 0.9)],
        },
        "word_length": {
            "min": min(lengths_words),
            "max": max(lengths_words),
            "mean": round(sum(lengths_words) / len(lengths_words), 1),
            "median": sorted(lengths_words)[len(lengths_words) // 2],
            "p10": sorted(lengths_words)[int(len(lengths_words) * 0.1)],
            "p90": sorted(lengths_words)[int(len(lengths_words) * 0.9)],
        },
        "length_distribution": dict(pattern_types.most_common()),
        "vocabulary": {
            "total_tokens": len(all_tokens),
            "unique_tokens": len(token_freq),
            "top_50_tokens": token_freq.most_common(50),
            "top_30_bigrams": [
                (f"{b[0]} {b[1]}", c)
                for b, c in bigrams.most_common(30)
            ],
        },
        "text_style": {
            "has_comma_pct": round(has_comma / len(non_null) * 100, 1),
            "has_period_pct": round(has_period / len(non_null) * 100, 1),
            "has_numbers_pct": round(has_numbers / len(non_null) * 100, 1),
            "starts_uppercase_pct": round(starts_upper / len(non_null) * 100, 1),
            "all_lowercase_pct": round(all_lower / len(non_null) * 100, 1),
            "all_uppercase_pct": round(all_upper / len(non_null) * 100, 1),
        },
        "sample_values": non_null[:20],
    }


def analyze_category_field(values: list, field_name: str) -> dict:
    """Analyze a categorical field."""
    non_null = [str(v).strip() for v in values if v is not None and str(v).strip() and str(v).strip() != "nan"]
    if not non_null:
        return {"field": field_name, "count": 0, "null_rate": 1.0}

    freq = Counter(non_null)
    total = len(non_null)
    n_unique = len(freq)

    sorted_counts = sorted(freq.values(), reverse=True)
    top_5_share = sum(sorted_counts[:5]) / total if total else 0
    top_10_share = sum(sorted_counts[:10]) / total if total else 0

    return {
        "field": field_name,
        "total_rows": len(values),
        "non_null_count": total,
        "null_rate": round(1 - total / len(values), 4),
        "n_unique_values": n_unique,
        "value_distribution": freq.most_common(),
        "top_5_concentration": round(top_5_share * 100, 1),
        "top_10_concentration": round(top_10_share * 100, 1),
        "sample_values": list(freq.keys())[:30],
    }


def analyze_text_category_overlap(text_values: list[str], cat_values: dict[str, list]) -> dict:
    """Analyze how text descriptions relate to category labels."""
    results = {}

    text_non_null = [v.lower() for v in text_values if v and str(v).strip()]
    if not text_non_null:
        return results

    for cat_name, cat_vals in cat_values.items():
        cat_non_null = [str(v).strip().lower() for v in cat_vals
                        if v is not None and str(v).strip() and str(v).strip() != "nan"]
        unique_cats = set(cat_non_null)
        if not unique_cats:
            continue

        contains_count = 0
        partial_count = 0
        for text in text_non_null:
            text_words = set(tokenize_simple(text))
            for cat in unique_cats:
                cat_words = set(tokenize_simple(cat))
                if cat in text:
                    contains_count += 1
                    break
                elif cat_words & text_words:
                    partial_count += 1
                    break

        results[cat_name] = {
            "text_contains_exact_category_pct": round(contains_count / len(text_non_null) * 100, 1),
            "text_shares_words_with_category_pct": round(partial_count / len(text_non_null) * 100, 1),
            "total_overlap_pct": round((contains_count + partial_count) / len(text_non_null) * 100, 1),
        }

    return results


def analyze_cross_field_correlations(data: dict[str, list]) -> dict:
    """Analyze correlations between category fields."""
    results = {}
    cat_fields = [k for k in data if k not in ("text", "description", "image", "item_ID", "year")]

    for i, f1 in enumerate(cat_fields):
        for f2 in cat_fields[i + 1:]:
            vals1 = data[f1]
            vals2 = data[f2]
            co_occurrence = Counter()
            for v1, v2 in zip(vals1, vals2):
                s1 = str(v1).strip() if v1 is not None else ""
                s2 = str(v2).strip() if v2 is not None else ""
                if s1 and s2 and s1 != "nan" and s2 != "nan":
                    co_occurrence[(s1, s2)] += 1

            if co_occurrence:
                results[f"{f1} × {f2}"] = {
                    "n_combinations": len(co_occurrence),
                    "top_20_combinations": [
                        (f"{k[0]} | {k[1]}", c)
                        for k, c in co_occurrence.most_common(20)
                    ],
                }

    return results


def load_dataset_rows(hf_id: str, max_rows: int = 0) -> dict[str, list]:
    """Load dataset into column-oriented dict, skipping image bytes."""
    print(f"  Loading {hf_id}...")
    t0 = time.time()

    ds = datasets.load_dataset(hf_id, split="data", streaming=True)
    features = ds.features
    text_and_cat_cols = [k for k in features if k != "image"]

    data = defaultdict(list)
    count = 0
    for row in ds:
        for col in text_and_cat_cols:
            data[col].append(row.get(col))
        count += 1
        if max_rows > 0 and count >= max_rows:
            break
        if count % 10000 == 0:
            print(f"    ... {count:,} rows loaded")

    elapsed = time.time() - t0
    print(f"    Done: {count:,} rows in {elapsed:.1f}s")
    return dict(data), count


def analyze_dataset(ds_name: str, config: dict, max_rows: int = 0) -> dict:
    """Full analysis of one dataset."""
    print(f"\n{'='*70}")
    print(f"ANALYZING: {ds_name} ({config['hf_id']})")
    print(f"{'='*70}")

    if config.get("skip_streaming"):
        print(f"  Skipping {ds_name} (requires special handling / too large)")
        return {
            "name": ds_name,
            "hf_id": config["hf_id"],
            "skipped": True,
            "reason": "Too large for streaming analysis (721K images, 71.5 GB)",
            "tasks": config["tasks"],
            "notes": "iMaterialist has no text field -- uses attribute-based queries only (category, style, neckline)",
        }

    data, total_rows = load_dataset_rows(config["hf_id"], max_rows)

    result = {
        "name": ds_name,
        "hf_id": config["hf_id"],
        "total_rows": total_rows,
        "tasks": config["tasks"],
        "n_tasks": len(config["tasks"]),
        "columns": list(data.keys()),
    }

    print("  Analyzing text fields...")
    result["text_analysis"] = {}
    for col in config["text_cols"]:
        if col in data:
            result["text_analysis"][col] = analyze_text_field(data[col], col)

    print("  Analyzing category fields...")
    result["category_analysis"] = {}
    for col in config["category_cols"]:
        if col in data:
            result["category_analysis"][col] = analyze_category_field(data[col], col)

    print("  Analyzing attribute fields...")
    result["attribute_analysis"] = {}
    for col in config["attribute_cols"]:
        if col in data:
            result["attribute_analysis"][col] = analyze_category_field(data[col], col)

    print("  Analyzing text-category overlap...")
    cat_data = {}
    for col in config["category_cols"] + config["attribute_cols"]:
        if col in data:
            cat_data[col] = data[col]
    if config["text_cols"] and config["text_cols"][0] in data:
        result["text_category_overlap"] = analyze_text_category_overlap(
            data[config["text_cols"][0]], cat_data
        )

    print("  Analyzing cross-field correlations...")
    result["cross_field_correlations"] = analyze_cross_field_correlations(data)

    return result


def format_dataset_markdown(analysis: dict) -> str:
    """Generate readable markdown report for one dataset."""
    lines = []
    name = analysis["name"]
    lines.append(f"# {name} — Benchmark Pattern Analysis\n")

    if analysis.get("skipped"):
        lines.append(f"**Skipped:** {analysis['reason']}\n")
        lines.append(f"**Tasks:** {', '.join(analysis['tasks'])}\n")
        lines.append(f"**Notes:** {analysis.get('notes', 'N/A')}\n")
        return "\n".join(lines)

    lines.append(f"- **HuggingFace:** `{analysis['hf_id']}`")
    lines.append(f"- **Total rows:** {analysis['total_rows']:,}")
    lines.append(f"- **Columns:** {', '.join(analysis['columns'])}")
    lines.append(f"- **Tasks ({analysis['n_tasks']}):** {', '.join(analysis['tasks'])}")
    lines.append("")

    lines.append("---\n")
    lines.append("## Text Fields\n")
    for col, ta in analysis.get("text_analysis", {}).items():
        lines.append(f"### `{col}`\n")
        if ta.get("count", -1) == 0:
            lines.append("*All null*\n")
            continue

        lines.append(f"- Non-null: {ta['non_null_count']:,} / {ta['total_rows']:,} ({100 - ta['null_rate']*100:.1f}%)")
        lines.append(f"- Unique values: {ta['unique_count']:,} (uniqueness: {ta['uniqueness_ratio']*100:.1f}%)")
        lines.append(f"- **Duplication:** {ta['duplication_ratio']*100:.1f}% of texts are duplicates")
        lines.append("")

        wl = ta["word_length"]
        lines.append(f"**Word count distribution:** min={wl['min']}, p10={wl['p10']}, median={wl['median']}, mean={wl['mean']}, p90={wl['p90']}, max={wl['max']}")
        lines.append("")

        lines.append("**Length buckets:**")
        for bucket, count in ta.get("length_distribution", {}).items():
            pct = count / ta["non_null_count"] * 100
            lines.append(f"  - {bucket}: {count:,} ({pct:.1f}%)")
        lines.append("")

        ts = ta["text_style"]
        lines.append("**Text style:**")
        lines.append(f"  - Has commas: {ts['has_comma_pct']}%")
        lines.append(f"  - Has periods: {ts['has_period_pct']}%")
        lines.append(f"  - Has numbers: {ts['has_numbers_pct']}%")
        lines.append(f"  - Starts uppercase: {ts['starts_uppercase_pct']}%")
        lines.append(f"  - All lowercase: {ts['all_lowercase_pct']}%")
        lines.append("")

        vocab = ta["vocabulary"]
        lines.append(f"**Vocabulary:** {vocab['unique_tokens']:,} unique tokens across {vocab['total_tokens']:,} total")
        lines.append("")
        lines.append("**Top 30 tokens:**")
        for token, count in vocab["top_50_tokens"][:30]:
            lines.append(f"  - `{token}`: {count:,}")
        lines.append("")

        lines.append("**Top 20 bigrams:**")
        for bigram, count in vocab["top_30_bigrams"][:20]:
            lines.append(f"  - `{bigram}`: {count:,}")
        lines.append("")

        lines.append("**Sample texts (first 20):**")
        for i, s in enumerate(ta.get("sample_values", [])[:20]):
            lines.append(f"  {i+1}. \"{s[:150]}{'...' if len(s)>150 else ''}\"")
        lines.append("")

        lines.append("**Most repeated texts (top 15):**")
        for val, count in ta.get("top_repeated_values", [])[:15]:
            lines.append(f"  - ({count:,}×) \"{val[:120]}{'...' if len(val)>120 else ''}\"")
        lines.append("")

    lines.append("---\n")
    lines.append("## Category Fields\n")
    for col, ca in analysis.get("category_analysis", {}).items():
        lines.append(f"### `{col}`\n")
        if ca.get("count", -1) == 0:
            lines.append("*All null*\n")
            continue
        lines.append(f"- Non-null: {ca['non_null_count']:,} / {ca['total_rows']:,}")
        lines.append(f"- Unique values: {ca['n_unique_values']}")
        lines.append(f"- Top-5 concentration: {ca['top_5_concentration']}%")
        lines.append(f"- Top-10 concentration: {ca['top_10_concentration']}%")
        lines.append("")
        lines.append("**Full value distribution:**")
        for val, count in ca["value_distribution"]:
            pct = count / ca["non_null_count"] * 100
            bar = "█" * max(1, int(pct / 2))
            lines.append(f"  - `{val}`: {count:,} ({pct:.1f}%) {bar}")
        lines.append("")

    if analysis.get("attribute_analysis"):
        lines.append("---\n")
        lines.append("## Attribute Fields\n")
        for col, aa in analysis["attribute_analysis"].items():
            lines.append(f"### `{col}`\n")
            if aa.get("count", -1) == 0:
                lines.append("*All null*\n")
                continue
            lines.append(f"- Non-null: {aa['non_null_count']:,} / {aa['total_rows']:,}")
            lines.append(f"- Unique values: {aa['n_unique_values']}")
            lines.append(f"- Top-5 concentration: {aa['top_5_concentration']}%")
            lines.append("")
            lines.append("**Full value distribution:**")
            for val, count in aa["value_distribution"][:50]:
                pct = count / aa["non_null_count"] * 100
                lines.append(f"  - `{val}`: {count:,} ({pct:.1f}%)")
            if len(aa["value_distribution"]) > 50:
                lines.append(f"  - ... and {len(aa['value_distribution']) - 50} more")
            lines.append("")

    if analysis.get("text_category_overlap"):
        lines.append("---\n")
        lines.append("## Text ↔ Category Overlap\n")
        lines.append("How often does the `text` field contain (or share words with) category labels?\n")
        for cat_name, overlap in analysis["text_category_overlap"].items():
            lines.append(f"### text vs `{cat_name}`")
            lines.append(f"  - Text contains exact category string: {overlap['text_contains_exact_category_pct']}%")
            lines.append(f"  - Text shares words with category: {overlap['text_shares_words_with_category_pct']}%")
            lines.append(f"  - **Total overlap: {overlap['total_overlap_pct']}%**")
            lines.append("")

    if analysis.get("cross_field_correlations"):
        lines.append("---\n")
        lines.append("## Cross-Field Correlations\n")
        for pair, corr in analysis["cross_field_correlations"].items():
            lines.append(f"### {pair}")
            lines.append(f"- Unique combinations: {corr['n_combinations']:,}")
            lines.append("")
            lines.append("**Top 20 combinations:**")
            for combo, count in corr["top_20_combinations"]:
                lines.append(f"  - `{combo}`: {count:,}")
            lines.append("")

    return "\n".join(lines)


def generate_cross_dataset_analysis(all_analyses: dict[str, dict]) -> str:
    """Generate cross-dataset comparative analysis."""
    lines = []
    lines.append("# Cross-Dataset Benchmark Pattern Analysis\n")
    lines.append("Comparative analysis across all Marqo benchmark datasets.\n")

    lines.append("## 1. Dataset Overview\n")
    lines.append("| Dataset | Rows | Columns | Tasks | Text Field? | Category Depth |")
    lines.append("|---------|------|---------|-------|-------------|----------------|")
    for name, a in all_analyses.items():
        if a.get("skipped"):
            lines.append(f"| {name} | ~721K | ? | {a['n_tasks'] if 'n_tasks' in a else len(a['tasks'])} | No (attribute only) | N/A |")
            continue
        n_cats = len(a.get("category_analysis", {}))
        has_text = "Yes" if a.get("text_analysis") else "No"
        lines.append(f"| {name} | {a['total_rows']:,} | {len(a['columns'])} | {a['n_tasks']} | {has_text} | {n_cats} levels |")
    lines.append("")

    lines.append("## 2. Task Coverage Matrix\n")
    all_tasks = set()
    for a in all_analyses.values():
        all_tasks.update(a.get("tasks", []))
    all_tasks = sorted(all_tasks)

    header = "| Dataset | " + " | ".join(all_tasks) + " |"
    sep = "|---------|" + "|".join(["---"] * len(all_tasks)) + "|"
    lines.append(header)
    lines.append(sep)
    for name, a in all_analyses.items():
        ds_tasks = set(a.get("tasks", []))
        row = f"| {name} | " + " | ".join(["Y" if t in ds_tasks else "-" for t in all_tasks]) + " |"
        lines.append(row)
    lines.append("")

    lines.append("## 3. Text Description Patterns\n")
    lines.append("| Dataset | Median Words | Mean Words | P10 | P90 | Unique Texts | Vocab Size |")
    lines.append("|---------|-------------|------------|-----|-----|-------------|------------|")
    for name, a in all_analyses.items():
        ta = a.get("text_analysis", {}).get("text", {})
        if not ta or ta.get("count", -1) == 0:
            lines.append(f"| {name} | — | — | — | — | — | — |")
            continue
        wl = ta["word_length"]
        v = ta["vocabulary"]
        lines.append(f"| {name} | {wl['median']} | {wl['mean']} | {wl['p10']} | {wl['p90']} | {ta['unique_count']:,} | {v['unique_tokens']:,} |")
    lines.append("")

    lines.append("## 4. Category Concentration\n")
    lines.append("How concentrated are category distributions? (Higher = more skewed = easier task)\n")
    for name, a in all_analyses.items():
        cats = a.get("category_analysis", {})
        if not cats:
            continue
        lines.append(f"### {name}")
        for col, ca in cats.items():
            if ca.get("count", -1) == 0:
                continue
            lines.append(f"- `{col}`: {ca['n_unique_values']} unique, top-5 = {ca['top_5_concentration']}%, top-10 = {ca['top_10_concentration']}%")
        lines.append("")

    lines.append("## 5. Shared Vocabulary Across Datasets\n")
    vocab_sets = {}
    for name, a in all_analyses.items():
        ta = a.get("text_analysis", {}).get("text", {})
        if ta and ta.get("vocabulary"):
            tokens = set(t for t, _ in ta["vocabulary"]["top_50_tokens"])
            vocab_sets[name] = tokens

    if len(vocab_sets) >= 2:
        all_vocabs = list(vocab_sets.values())
        shared_all = set.intersection(*all_vocabs) if all_vocabs else set()
        lines.append(f"**Tokens in top-50 of ALL datasets:** {sorted(shared_all)}\n")

        for name, tokens in vocab_sets.items():
            unique_to_ds = tokens - set.union(*(v for n, v in vocab_sets.items() if n != name))
            if unique_to_ds:
                lines.append(f"**Unique to {name}:** {sorted(unique_to_ds)}")
        lines.append("")

    lines.append("## 6. Category Taxonomy Comparison\n")
    cat_values_by_level = defaultdict(lambda: defaultdict(set))
    for name, a in all_analyses.items():
        for col, ca in a.get("category_analysis", {}).items():
            if ca.get("count", -1) == 0:
                continue
            for val, _ in ca.get("value_distribution", []):
                cat_values_by_level[name][col].add(val.lower().strip())

    all_cat_values = defaultdict(set)
    for name, levels in cat_values_by_level.items():
        for col, vals in levels.items():
            all_cat_values[name].update(vals)

    if len(all_cat_values) >= 2:
        ds_names = list(all_cat_values.keys())
        lines.append("**Category overlap between datasets:**\n")
        for i, n1 in enumerate(ds_names):
            for n2 in ds_names[i + 1:]:
                shared = all_cat_values[n1] & all_cat_values[n2]
                if shared:
                    lines.append(f"- {n1} ∩ {n2}: {len(shared)} shared ({sorted(list(shared)[:20])}{'...' if len(shared)>20 else ''})")
        lines.append("")

    return "\n".join(lines)


def generate_training_implications(all_analyses: dict[str, dict]) -> str:
    """Generate actionable training data design implications."""
    lines = []
    lines.append("# Training Data Design Implications\n")
    lines.append("Based on deep analysis of all 7 Marqo benchmark datasets.\n")
    lines.append("## Key Principle\n")
    lines.append("We learn the **patterns** (query types, category structures, text styles) not the actual data.\n")
    lines.append("Training data must cover these patterns using **non-benchmark sources** only.\n")

    lines.append("---\n")
    lines.append("## 1. Query Type Coverage Required\n")
    lines.append("The benchmarks test these distinct query types:\n")

    all_tasks = set()
    for a in all_analyses.values():
        all_tasks.update(a.get("tasks", []))

    for task in sorted(all_tasks):
        ds_with_task = [n for n, a in all_analyses.items() if task in a.get("tasks", [])]
        lines.append(f"### {task}")
        lines.append(f"Tested in: {', '.join(ds_with_task)}\n")

    lines.append("---\n")
    lines.append("## 2. Text Description Patterns to Teach\n")
    lines.append("(To be filled after analysis run with actual data patterns)\n")

    lines.append("---\n")
    lines.append("## 3. Category Granularity Levels\n")
    lines.append("(To be filled after analysis run with actual category distributions)\n")

    lines.append("---\n")
    lines.append("## 4. Attribute Coverage Requirements\n")
    lines.append("(To be filled after analysis run with actual attribute distributions)\n")

    lines.append("---\n")
    lines.append("## 5. Recommended Training Data Sources\n")
    lines.append("| Source | What it covers | Benchmark overlap risk |")
    lines.append("|--------|---------------|----------------------|")
    lines.append("| Marqo-GS-10M (fashion 5M) | Real search queries + ranked results | None with eval benchmarks |")
    lines.append("| DeepFashion (In-Shop) | Retail catalog + descriptions | LEAKAGE: used in benchmark |")
    lines.append("| DeepFashion (Multimodal) | Multi-attribute fashion | LEAKAGE: used in benchmark |")
    lines.append("| Open Images (fashion subset) | Diverse real-world images | None |")
    lines.append("| LLM-generated captions | Any pattern we want to teach | None |")
    lines.append("")
    lines.append("**Safe sources:** Marqo-GS-10M, Open Images, LLM-generated\n")
    lines.append("**Must exclude from training:** All 7 benchmark datasets\n")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Analyze Marqo benchmark dataset patterns")
    parser.add_argument("--max-rows", type=int, default=0,
                        help="Max rows per dataset (0 = all rows)")
    parser.add_argument("--datasets", nargs="*", default=None,
                        help="Specific datasets to analyze (default: all)")
    args = parser.parse_args()

    PER_DS_DIR.mkdir(parents=True, exist_ok=True)

    datasets_to_run = args.datasets or list(DATASET_CONFIGS.keys())
    print(f"Will analyze {len(datasets_to_run)} datasets: {', '.join(datasets_to_run)}")
    print(f"Max rows per dataset: {'ALL' if args.max_rows == 0 else args.max_rows}")
    print()

    all_analyses = {}
    for ds_name in datasets_to_run:
        if ds_name not in DATASET_CONFIGS:
            print(f"WARNING: Unknown dataset '{ds_name}', skipping")
            continue

        config = DATASET_CONFIGS[ds_name]
        try:
            analysis = analyze_dataset(ds_name, config, args.max_rows)
            all_analyses[ds_name] = analysis

            json_path = PER_DS_DIR / f"{ds_name}_analysis.json"
            with open(json_path, "w") as f:
                json.dump(analysis, f, indent=2, default=str)
            print(f"  Saved JSON: {json_path}")

            md_path = PER_DS_DIR / f"{ds_name}_analysis.md"
            with open(md_path, "w") as f:
                f.write(format_dataset_markdown(analysis))
            print(f"  Saved MD:   {md_path}")

        except Exception as e:
            print(f"  ERROR analyzing {ds_name}: {e}")
            import traceback
            traceback.print_exc()
            all_analyses[ds_name] = {"name": ds_name, "error": str(e), "tasks": config["tasks"]}

    print(f"\n{'='*70}")
    print("GENERATING CROSS-DATASET ANALYSIS")
    print(f"{'='*70}")

    cross_md = generate_cross_dataset_analysis(all_analyses)
    cross_path = RESULTS_DIR / "cross_dataset_analysis.md"
    with open(cross_path, "w") as f:
        f.write(cross_md)
    print(f"Saved: {cross_path}")

    impl_md = generate_training_implications(all_analyses)
    impl_path = RESULTS_DIR / "training_data_implications.md"
    with open(impl_path, "w") as f:
        f.write(impl_md)
    print(f"Saved: {impl_path}")

    print(f"\n{'='*70}")
    print("ALL DONE")
    print(f"{'='*70}")
    print(f"Results in: {RESULTS_DIR}")
    print(f"  Per-dataset: {PER_DS_DIR}/")
    print(f"  Cross-dataset: {cross_path}")
    print(f"  Training implications: {impl_path}")


if __name__ == "__main__":
    main()
