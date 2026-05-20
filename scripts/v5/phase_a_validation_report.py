"""
Phase A.2 — Generate a markdown validation report for human review of multi-field labels.

Reads a labeled JSONL (output of phase_a_extract_multifield.py) and produces:
  - Distribution tables for each field
  - Suspicious-label section (heuristic flags)
  - Full table of every record's (query, title) -> labels for hand-checking

Usage:
    python scripts/v5/phase_a_validation_report.py \\
        --input data/processed/v5_multifield/validation_200.jsonl \\
        --output data/processed/v5_multifield/validation_200_report.md
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

# Heuristic suspicion rules — flag rows where extraction is likely off
FASHION_QUERY_KEYWORDS = {
    "dress", "top", "blouse", "shirt", "tee", "tshirt", "jacket", "coat",
    "vest", "sweater", "hoodie", "pants", "jeans", "skirt", "shorts",
    "shoes", "boots", "sneakers", "heels", "sandals", "bag", "purse",
    "hat", "cap", "beanie", "scarf", "gloves", "belt", "watch", "earrings",
    "necklace", "ring", "bracelet",
}

HOME_LIFESTYLE_L1 = {"home_lifestyle", "beauty"}


def is_suspicious(rec: dict) -> str | None:
    """Return a one-line reason if the row looks suspicious, else None."""
    q = (rec.get("query") or "").lower()
    title = (rec.get("title") or "").lower()
    l1 = rec.get("category_l1", "")

    # Fashion-keyword query but classified as home/beauty
    if any(kw in q for kw in FASHION_QUERY_KEYWORDS) and l1 in HOME_LIFESTYLE_L1:
        return f"fashion query keyword but L1='{l1}'"

    # All four LLM-only fields are 'unknown' — extraction got nothing
    unk_count = sum(
        1 for k in ("pattern", "fit_style", "gender", "occasion")
        if rec.get(k) == "unknown"
    )
    if unk_count == 4:
        return "all 4 LLM-only fields are 'unknown'"

    # Obvious gender/category mismatch
    if "infant" in q or "baby" in q or "toddler" in q or "kids" in q:
        if rec.get("gender") not in {"kids", "unknown"}:
            return f"kids-keyword query but gender='{rec.get('gender')}'"

    # Title says vest/jacket but L1 is home_lifestyle (regression of the seed bug)
    for ap in ("vest", "jacket", "coat", "sweater", "hoodie"):
        if ap in title and l1 == "home_lifestyle":
            return f"title contains '{ap}' but L1='home_lifestyle'"

    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()

    records = []
    with args.input.open() as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    n = len(records)
    if n == 0:
        print("ERROR: input file is empty")
        return

    # Distribution stats
    fields = ["category_l1", "pattern", "fit_style", "gender", "occasion"]
    free_fields = ["color", "material", "category_l2"]
    distributions = {}
    for f in fields:
        distributions[f] = Counter(r.get(f, "unknown") for r in records)
    free_unknown = {f: sum(1 for r in records if r.get(f) == "unknown") for f in free_fields}

    # Suspicion
    suspects = []
    for i, r in enumerate(records):
        reason = is_suspicious(r)
        if reason:
            suspects.append((i, reason, r))

    # Write report
    with args.output.open("w") as out:
        out.write(f"# v5 Multi-field Label Validation — {n} records\n\n")
        out.write(f"Source: `{args.input.name}`\n\n")
        out.write("## TL;DR\n\n")
        out.write(f"- Records: **{n}**\n")
        out.write(f"- Suspicious rows flagged by heuristics: **{len(suspects)}** ({100*len(suspects)/n:.1f}%)\n")
        out.write(f"- 'unknown' rate by field:\n")
        for f in fields:
            unk = distributions[f].get("unknown", 0)
            out.write(f"  - {f}: {unk}/{n} ({100*unk/n:.1f}%)\n")
        for f in free_fields:
            out.write(f"  - {f}: {free_unknown[f]}/{n} ({100*free_unknown[f]/n:.1f}%)\n")
        out.write("\n## Gate criteria\n\n")
        out.write(
            "Hand-check 50 random records below. The gate from PLAN_V5 §A.2 is:\n"
            "**≥85% accuracy on category_l1 AND color** before scaling to full corpus.\n\n"
        )

        out.write("## Field distributions\n\n")
        for f in fields:
            out.write(f"### `{f}`\n\n")
            out.write("| value | count | % |\n|---|---:|---:|\n")
            for v, c in distributions[f].most_common():
                out.write(f"| {v} | {c} | {100*c/n:.1f}% |\n")
            out.write("\n")

        if suspects:
            out.write(f"## Suspicious rows ({len(suspects)} flagged)\n\n")
            out.write("Heuristic-flagged rows — review these first.\n\n")
            out.write("| # | reason | query | title (truncated) | L1 | gender | flags |\n")
            out.write("|---|---|---|---|---|---|---|\n")
            for i, reason, r in suspects:
                tl = (r.get("title", "") or "")[:60].replace("|", "/")
                q = (r.get("query", "") or "").replace("|", "/")
                flags = f"pat={r.get('pattern')}, fit={r.get('fit_style')}, occ={r.get('occasion')}"
                out.write(f"| {i} | {reason} | {q} | {tl} | {r.get('category_l1')} | {r.get('gender')} | {flags} |\n")
            out.write("\n")

        out.write("## All records\n\n")
        out.write("Full table for hand-validation. Mark each row mentally as ✓ or ✗ on category_l1 and color.\n\n")
        out.write("| # | query | title (truncated) | L1 | L2 | color | material | pattern | fit | gender | occasion |\n")
        out.write("|---|---|---|---|---|---|---|---|---|---|---|\n")
        for i, r in enumerate(records):
            tl = (r.get("title", "") or "")[:50].replace("|", "/")
            q = (r.get("query", "") or "").replace("|", "/")
            out.write(
                f"| {i} | {q} | {tl} | {r.get('category_l1')} | {r.get('category_l2')} | "
                f"{r.get('color')} | {r.get('material')} | {r.get('pattern')} | "
                f"{r.get('fit_style')} | {r.get('gender')} | {r.get('occasion')} |\n"
            )

    print(f"Wrote {args.output}")
    print(f"  {n} records, {len(suspects)} flagged as suspicious")


if __name__ == "__main__":
    main()
