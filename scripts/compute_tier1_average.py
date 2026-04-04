"""
Compute averaged Tier 1 metrics across all available datasets per model.
Reads from repos/marqo-FashionCLIP/results/<dataset>/<run_name>/<task>/result_*.json
Outputs results/tier1_averaged.json and prints a summary table.
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict

REPO_ROOT = Path(__file__).parent.parent
RESULTS_DIR = REPO_ROOT / "repos" / "marqo-FashionCLIP" / "results"
OUT_DIR = REPO_ROOT / "results"
OUT_DIR.mkdir(exist_ok=True)

TASK_MAP = {
    "text-to-image": "Text-to-Image",
    "category-to-product": "Category-to-Product",
    "sub-category-to-product": "Sub-Category-to-Product",
    "color-to-product": "Color-to-Product",
}


def load_all_results():
    """Returns {run_name: {task: {dataset: {metric: value}}}}"""
    data = defaultdict(lambda: defaultdict(dict))

    for dataset_dir in RESULTS_DIR.iterdir():
        if not dataset_dir.is_dir():
            continue
        dataset = dataset_dir.name

        for run_dir in dataset_dir.iterdir():
            if not run_dir.is_dir():
                continue
            run_name = run_dir.name

            for task_dir in run_dir.iterdir():
                if not task_dir.is_dir():
                    continue
                task = task_dir.name

                for result_file in task_dir.glob("result_*.json"):
                    try:
                        with open(result_file) as f:
                            raw = json.load(f)
                        # Flatten: {Recall@1: v, Recall@10: v, MRR: v, ...}
                        flat = {}
                        for section, vals in raw.items():
                            if section == "MRR" and isinstance(vals, (float, int)):
                                flat["MRR"] = float(vals)
                            elif isinstance(vals, dict):
                                for k, v in vals.items():
                                    flat[k] = float(v)
                        data[run_name][task][dataset] = flat
                    except Exception as e:
                        print(f"  Warning: {result_file}: {e}")
    return data


def average_across_datasets(task_data: dict) -> dict:
    """Average metrics across all datasets for a given task."""
    all_metrics = defaultdict(list)
    for dataset, metrics in task_data.items():
        for k, v in metrics.items():
            all_metrics[k].append(v)
    return {k: float(np.mean(v)) for k, v in all_metrics.items()}


def main():
    all_data = load_all_results()

    summary = {}
    for run_name, tasks in sorted(all_data.items()):
        summary[run_name] = {}
        print(f"\n{'='*60}")
        print(f"Model: {run_name}")
        print(f"{'='*60}")

        for task, dataset_results in sorted(tasks.items()):
            datasets = list(dataset_results.keys())
            avg = average_across_datasets(dataset_results)
            summary[run_name][task] = {
                "avg": avg,
                "n_datasets": len(datasets),
                "datasets": datasets,
                "per_dataset": dataset_results,
            }

            task_label = TASK_MAP.get(task, task)
            print(f"\n  {task_label} (n={len(datasets)} datasets: {', '.join(datasets)})")

            # Print key metrics
            key_metrics = ["Recall@1", "Recall@10", "MRR", "P@1", "P@10", "MAP@1", "MAP@10"]
            for m in key_metrics:
                if m in avg:
                    print(f"    {m:15s}: {avg[m]:.4f}")

            # Per-dataset breakdown
            print(f"    --- per dataset ---")
            for ds, metrics in sorted(dataset_results.items()):
                vals = "  ".join(f"{k}={v:.4f}" for k, v in sorted(metrics.items())
                                 if k in ["Recall@1", "Recall@10", "MRR", "P@1"])
                print(f"    {ds:35s}: {vals}")

    # Save
    out_path = OUT_DIR / "tier1_averaged.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n\nFull results saved to {out_path}")

    # Marqo published numbers for reference
    print("\n" + "="*60)
    print("REFERENCE — Marqo Published (6-dataset avg, text-to-image):")
    print("  Marqo-FashionSigLIP : Recall@1=0.121  Recall@10=0.340  MRR=0.239")
    print("  Marqo-FashionCLIP   : Recall@1=0.077  Recall@10=0.249  MRR=0.165")
    print("="*60)


if __name__ == "__main__":
    main()
