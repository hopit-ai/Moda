"""
Phase 3b: Gap analysis after initial training.

Compares fine-tuned model results against FashionSigLIP baseline,
identifies which benchmarks/tasks still lose, and recommends targeted
data to mine for the next iteration.
"""
import json, sys, argparse
from pathlib import Path

PROJ_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJ_ROOT / "results" / "v4_gcl"

FSL_REFERENCE = {
    "deepfashion_inshop": {
        "text_to_image": {"recall@1": 0.225, "recall@10": 0.447, "mrr": 0.302},
    },
    "deepfashion_multimodal": {
        "text_to_image": {"recall@1": 0.055, "recall@10": 0.170, "mrr": 0.094},
    },
    "fashion200k": {
        "text_to_image": {"recall@1": 0.074, "recall@10": 0.212, "mrr": 0.121},
    },
    "KAGL": {
        "text_to_image": {"recall@1": 0.352, "recall@10": 0.595, "mrr": 0.441},
    },
    "atlas": {
        "text_to_image": {"recall@1": 0.178, "recall@10": 0.381, "mrr": 0.253},
    },
    "polyvore": {
        "text_to_image": {"recall@1": 0.029, "recall@10": 0.114, "mrr": 0.057},
    },
}

GAP_FILL_RECIPES = {
    "deepfashion_inshop": {
        "data_strategy": "More in-shop product photos with matching queries; "
                         "increase brand+color queries",
        "pattern": "Studio photography matching, specific product identifiers",
    },
    "deepfashion_multimodal": {
        "data_strategy": "Template descriptions ('The upper clothing has...'); "
                         "attribute-structured text",
        "pattern": "Template-structured attribute descriptions",
    },
    "fashion200k": {
        "data_strategy": "Natural language product descriptions 10-30 words; "
                         "modifier-based queries",
        "pattern": "Descriptive phrases with modifiers and relative comparisons",
    },
    "KAGL": {
        "data_strategy": "E-commerce short titles with brand, color, material; "
                         "categorical labels",
        "pattern": "Standard e-commerce product titles, Indian fashion terms",
    },
    "atlas": {
        "data_strategy": "Broad product catalog queries including home, beauty, "
                         "lifestyle; simple category searches",
        "pattern": "Short category-based queries across diverse product types",
    },
    "polyvore": {
        "data_strategy": "Outfit/styling queries; multi-item coordination; "
                         "home decor and accessories with style context",
        "pattern": "Style-centric queries, outfit coordination, lifestyle items",
    },
}


def analyze_gaps(our_results: dict, baseline_results: dict | None = None) -> dict:
    """Compare results against FSL reference and identify gaps."""
    gaps = []
    wins = []
    reference = baseline_results or FSL_REFERENCE

    for bname, tasks in our_results.items():
        ref_tasks = reference.get(bname, {})
        for task_name, metrics in tasks.items():
            if not isinstance(metrics, dict) or "mrr" not in metrics:
                continue
            ref_metrics = ref_tasks.get(task_name, {})
            if not ref_metrics:
                continue

            our_mrr = metrics["mrr"]
            ref_mrr = ref_metrics.get("mrr", 0)
            delta = our_mrr - ref_mrr

            entry = {
                "benchmark": bname,
                "task": task_name,
                "our_mrr": our_mrr,
                "ref_mrr": ref_mrr,
                "delta": delta,
                "pct_change": (delta / ref_mrr * 100) if ref_mrr > 0 else 0,
            }

            if delta < 0:
                entry["recipe"] = GAP_FILL_RECIPES.get(bname, {})
                gaps.append(entry)
            else:
                wins.append(entry)

    gaps.sort(key=lambda x: x["delta"])
    wins.sort(key=lambda x: x["delta"], reverse=True)
    return {"gaps": gaps, "wins": wins}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", type=str, default="gcl_v4")
    parser.add_argument("--baseline-run", type=str, default=None)
    args = parser.parse_args()

    results_file = RESULTS_DIR / args.run_name / "full_results.json"
    if not results_file.exists():
        print(f"Results not found: {results_file}")
        print("Run phase3_eval_all_benchmarks.py first.")
        sys.exit(1)

    with open(results_file) as f:
        our_results = json.load(f)

    baseline = None
    if args.baseline_run:
        bf = RESULTS_DIR / args.baseline_run / "full_results.json"
        if bf.exists():
            with open(bf) as f:
                baseline = json.load(f)

    analysis = analyze_gaps(our_results, baseline)

    print(f"\n{'='*70}")
    print("GAP ANALYSIS: Our Model vs FashionSigLIP")
    print(f"{'='*70}")

    if analysis["wins"]:
        print(f"\nWINS ({len(analysis['wins'])} tasks):")
        for w in analysis["wins"]:
            print(f"  + {w['benchmark']}/{w['task']}: "
                  f"{w['our_mrr']:.3f} vs {w['ref_mrr']:.3f} "
                  f"(+{w['delta']:.3f}, +{w['pct_change']:.1f}%)")

    if analysis["gaps"]:
        print(f"\nGAPS ({len(analysis['gaps'])} tasks):")
        for g in analysis["gaps"]:
            print(f"  - {g['benchmark']}/{g['task']}: "
                  f"{g['our_mrr']:.3f} vs {g['ref_mrr']:.3f} "
                  f"({g['delta']:.3f}, {g['pct_change']:.1f}%)")
            if g.get("recipe"):
                print(f"    Rx: {g['recipe'].get('data_strategy', 'N/A')}")

        print(f"\nPRIORITY GAP-FILLING PLAN:")
        for i, g in enumerate(analysis["gaps"], 1):
            recipe = g.get("recipe", {})
            print(f"  {i}. {g['benchmark']} (MRR gap: {g['delta']:.3f})")
            print(f"     Strategy: {recipe.get('data_strategy', 'Mine more relevant pairs')}")
    else:
        print("\nNo gaps! Model beats FashionSigLIP on all evaluated tasks.")

    gap_file = RESULTS_DIR / args.run_name / "gap_analysis.json"
    with open(gap_file, "w") as f:
        json.dump(analysis, f, indent=2)
    print(f"\nAnalysis saved to {gap_file}")


if __name__ == "__main__":
    main()
