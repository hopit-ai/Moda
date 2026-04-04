"""
MODA Phase 1 — Final Leaderboard Generator

Collects results from:
  Tier 1: repos/marqo-FashionCLIP/results/<dataset>/<run_name>/text-to-image/*.json
  Tier 2: results/hnm_dense_<model>.json

Outputs:
  results/PHASE1_LEADERBOARD.md
  results/PHASE1_LEADERBOARD.json
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
TIER1_RESULTS = REPO_ROOT / "repos" / "marqo-FashionCLIP" / "results"
TIER2_RESULTS = REPO_ROOT / "results"

MODEL_DISPLAY = {
    "Marqo-FashionSigLIP": "marqo-fashionSigLIP",
    "Marqo-FashionCLIP": "marqo-fashionCLIP",
    "OpenAI-CLIP-ViT-B-32": "CLIP ViT-B/32",
    "fashion-siglip": "marqo-fashionSigLIP",
    "fashion-clip": "marqo-fashionCLIP",
    "clip": "CLIP ViT-B/32",
}

DATASET_DISPLAY = {
    "deepfashion_inshop": "DeepFashion InShop",
    "deepfashion_multimodal": "DeepFashion Multimodal",
    "fashion200k": "Fashion200K",
    "KAGL": "KAGL",
    "atlas": "Atlas",
    "polyvore": "Polyvore",
    "iMaterialist": "iMaterialist",
}


def collect_tier1():
    """Collect all Tier 1 results from Marqo eval harness output."""
    records = []
    if not TIER1_RESULTS.exists():
        return records

    for dataset_dir in TIER1_RESULTS.iterdir():
        if not dataset_dir.is_dir():
            continue
        dataset = dataset_dir.name
        for run_dir in dataset_dir.iterdir():
            if not run_dir.is_dir():
                continue
            run_name = run_dir.name
            model = MODEL_DISPLAY.get(run_name, run_name)
            dataset_label = DATASET_DISPLAY.get(dataset, dataset)

            # Look for result JSON files
            for query_type_dir in run_dir.iterdir():
                if not query_type_dir.is_dir():
                    continue
                query_type = query_type_dir.name
                for json_file in query_type_dir.glob("result_*.json"):
                    try:
                        with open(json_file) as f:
                            data = json.load(f)
                        records.append({
                            "dataset": dataset_label,
                            "model": model,
                            "query_type": query_type,
                            "metrics": data,
                            "source_file": str(json_file.relative_to(REPO_ROOT)),
                        })
                    except Exception as e:
                        print(f"  Warning: could not parse {json_file}: {e}", file=sys.stderr)
    return records


def collect_tier2():
    """Collect all Tier 2 H&M results."""
    records = []
    for json_file in TIER2_RESULTS.glob("hnm_dense_*.json"):
        model_key = json_file.stem.replace("hnm_dense_", "")
        model = MODEL_DISPLAY.get(model_key, model_key)
        try:
            with open(json_file) as f:
                data = json.load(f)
            records.append({
                "model": model,
                "aggregated": data.get("aggregated", {}),
                "source_file": str(json_file.relative_to(REPO_ROOT)),
            })
        except Exception as e:
            print(f"  Warning: could not parse {json_file}: {e}", file=sys.stderr)
    return records


def format_val(v, decimals=4):
    if isinstance(v, float):
        return f"{v:.{decimals}f}"
    return str(v)


def generate_markdown(tier1_records, tier2_records):
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "# MODA Phase 1 — Benchmark Leaderboard",
        f"_Generated: {ts}_",
        "",
        "## Overview",
        "",
        "| Tier | Description | Status |",
        "| --- | --- | --- |",
        "| Tier 1 | Marqo 7-dataset Text-to-Image Recall | Partial (deepfashion_inshop done) |",
        "| Tier 2 | H&M Full-Pipeline (MODA's new benchmark) | ✅ All 3 models |",
        "| Tier 3 | FashionIQ Composed Retrieval | 🔜 Phase 2 |",
        "",
    ]

    # ── Tier 2 (MODA's contribution — complete) ──────────────────────────────
    lines += [
        "---",
        "",
        "## Tier 2 — H&M Full-Pipeline Benchmark (MODA Contribution)",
        "",
        "> **Benchmark design:** Category-based relevance over 105,542 H&M articles.",
        "> Queries are `product_type_name` labels. Positive articles share the same type,",
        "> negative articles share garment group but differ in type.",
        "> 4,078 evaluation queries.",
        "",
        "| Model | nDCG@5 | nDCG@10 | nDCG@20 | MRR | Recall@10 | Recall@20 | P@10 | Latency (ms) |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]

    # Sort by nDCG@10 descending
    tier2_sorted = sorted(tier2_records, key=lambda r: r["aggregated"].get("ndcg@10", 0), reverse=True)
    for rec in tier2_sorted:
        m = rec["aggregated"]
        lines.append(
            f"| {rec['model']} "
            f"| {format_val(m.get('ndcg@5', 0))} "
            f"| {format_val(m.get('ndcg@10', 0))} "
            f"| {format_val(m.get('ndcg@20', 0))} "
            f"| {format_val(m.get('mrr', 0))} "
            f"| {format_val(m.get('recall@10', 0))} "
            f"| {format_val(m.get('recall@20', 0))} "
            f"| {format_val(m.get('p@10', 0))} "
            f"| {format_val(m.get('mean_latency_ms', 0), 2)} |"
        )

    lines.append("")

    # ── Tier 1 ────────────────────────────────────────────────────────────────
    lines += [
        "---",
        "",
        "## Tier 1 — Marqo 7-Dataset Text-to-Image Embedding Benchmark",
        "",
        "> Reproduces Marqo FashionCLIP paper evaluation. Metrics: Recall@1, Recall@10, MRR.",
        "",
    ]

    if tier1_records:
        # Group by dataset and query type
        from collections import defaultdict
        by_dataset: dict = defaultdict(list)
        for rec in tier1_records:
            key = f"{rec['dataset']} ({rec['query_type']})"
            by_dataset[key].append(rec)

        for key, recs in sorted(by_dataset.items()):
            lines += [f"### {key}", ""]

            def flat_metrics(m: dict) -> dict:
                """Flatten nested metric dict into {display_name: value}."""
                flat = {}
                for section, vals in m.items():
                    if section == "MRR" and isinstance(vals, float):
                        flat["MRR"] = vals
                    elif isinstance(vals, dict):
                        for k, v in vals.items():
                            flat[k] = v
                return flat

            # Collect all metric keys across models
            all_flat = [flat_metrics(rec["metrics"]) for rec in recs]
            all_metric_keys = sorted(set(k for d in all_flat for k in d.keys()))

            header = "| Model | " + " | ".join(all_metric_keys) + " |"
            sep = "| --- | " + " | ".join(["---"] * len(all_metric_keys)) + " |"
            lines += [header, sep]

            for rec, flat in zip(recs, all_flat):
                vals = [format_val(flat.get(col, "-")) if isinstance(flat.get(col, "-"), float) else str(flat.get(col, "-")) for col in all_metric_keys]
                lines.append(f"| {rec['model']} | " + " | ".join(vals) + " |")
            lines.append("")
    else:
        lines += [
            "_No Tier 1 results available yet. Run `scripts/run_tier1_all_models.sh` to generate._",
            "",
        ]

    # ── Tier 3 placeholder ────────────────────────────────────────────────────
    lines += [
        "---",
        "",
        "## Tier 3 — FashionIQ Composed Retrieval",
        "",
        "_Planned for Phase 2. Metric: Recall@{10,50}._",
        "",
        "---",
        "",
        "## Notes",
        "",
        "- **Tier 2 query design:** synthetic category-based queries from H&M article metadata.",
        "  Phase 2 will extend with purchase-signal-based relevance using transaction data.",
        "- **Tier 1 status:** Only `deepfashion_inshop × marqo-fashionSigLIP` completed in Phase 1",
        "  (due to 1hr+ embedding times per dataset per model). Remaining runs are queued for Phase 2.",
        "- **Device:** Apple MPS (M-series) used for article embedding; CPU used for query encoding.",
        "- All models used zero-shot (no fine-tuning on H&M data).",
        "",
    ]

    return "\n".join(lines)


def main():
    print("Collecting Tier 1 results …")
    tier1 = collect_tier1()
    print(f"  Found {len(tier1)} Tier 1 result sets")

    print("Collecting Tier 2 results …")
    tier2 = collect_tier2()
    print(f"  Found {len(tier2)} Tier 2 model evaluations")

    md = generate_markdown(tier1, tier2)
    out_md = TIER2_RESULTS / "PHASE1_LEADERBOARD.md"
    out_md.write_text(md)
    print(f"Leaderboard written to {out_md}")

    # Also write structured JSON
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "tier1": tier1,
        "tier2": [
            {
                "model": r["model"],
                "aggregated": r["aggregated"],
                "source_file": r["source_file"],
            }
            for r in tier2
        ],
    }
    out_json = TIER2_RESULTS / "PHASE1_LEADERBOARD.json"
    with open(out_json, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Structured JSON written to {out_json}")

    # Print summary
    print("\n" + "=" * 60)
    print("PHASE 1 LEADERBOARD SUMMARY")
    print("=" * 60)
    if tier2:
        print("\nTier 2 — H&M Dense Retrieval:")
        sorted_t2 = sorted(tier2, key=lambda r: r["aggregated"].get("ndcg@10", 0), reverse=True)
        for rank, rec in enumerate(sorted_t2, 1):
            m = rec["aggregated"]
            print(f"  #{rank} {rec['model']:30s}  nDCG@10={m.get('ndcg@10', 0):.4f}  MRR={m.get('mrr', 0):.4f}")

    if tier1:
        print("\nTier 1 — Completed evaluations:")
        for rec in tier1:
            print(f"  {rec['dataset']} × {rec['model']} ({rec['query_type']})")


if __name__ == "__main__":
    main()
