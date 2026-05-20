"""Phase 10 — Compare all evaluation results."""

import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("compare")

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "results"

FSL_BASELINE = {
    "fashion200k": 0.3859,
    "atlas": 0.6919,
    "polyvore": 0.5783,
    "KAGL": 0.6779,
}

def main():
    results_files = sorted(RESULTS_DIR.glob("phase10_*.json"))
    if not results_files:
        log.info("No phase10 result files found.")
        return

    all_models = {}
    for rf in results_files:
        with open(rf) as f:
            data = json.load(f)
        tag = rf.stem.replace("phase10_", "")
        all_models[tag] = data.get("results", {})

    # Build comparison table
    benchmarks = ["fashion200k", "atlas", "polyvore", "KAGL"]

    lines = [
        "# Phase 10 — Results Comparison",
        "",
        "| Model | fashion200k | atlas | polyvore | KAGL | Avg | Beats FSL +10%? |",
        "|---|---:|---:|---:|---:|---:|---|",
        f"| **FSL Baseline** | {FSL_BASELINE['fashion200k']:.4f} | {FSL_BASELINE['atlas']:.4f} | {FSL_BASELINE['polyvore']:.4f} | {FSL_BASELINE['KAGL']:.4f} | {sum(FSL_BASELINE.values())/4:.4f} | — |",
        f"| **Target (+10%)** | {FSL_BASELINE['fashion200k']*1.1:.4f} | {FSL_BASELINE['atlas']*1.1:.4f} | {FSL_BASELINE['polyvore']*1.1:.4f} | {FSL_BASELINE['KAGL']*1.1:.4f} | {sum(v*1.1 for v in FSL_BASELINE.values())/4:.4f} | — |",
    ]

    for tag, results in all_models.items():
        scores = []
        cells = []
        all_beat = True
        for bm in benchmarks:
            if bm in results and "primary_map10" in results[bm]:
                score = results[bm]["primary_map10"]
                scores.append(score)
                delta = (score - FSL_BASELINE[bm]) / FSL_BASELINE[bm] * 100
                cells.append(f"{score:.4f} ({delta:+.1f}%)")
                if score < FSL_BASELINE[bm] * 1.1:
                    all_beat = False
            else:
                cells.append("ERR")
                all_beat = False
                scores.append(0)

        avg = sum(scores) / len(scores) if scores else 0
        beats = "YES" if all_beat else "no"
        lines.append(f"| **{tag}** | {cells[0]} | {cells[1]} | {cells[2]} | {cells[3]} | {avg:.4f} | {beats} |")

    # ── Error Analysis Section ──
    lines.append("")
    lines.append("## Error Analysis")
    lines.append("")

    for tag in ["p4b", "fsl"]:
        ea_path = RESULTS_DIR / f"phase10_{tag}_error_analysis.json"
        if ea_path.exists():
            with open(ea_path) as f:
                ea = json.load(f)
            model_name = ea.get("model", tag)
            lines.append(f"### {model_name}")
            lines.append("")
            weak = ea.get("weakest_benchmark", "?")
            weak_d = ea.get("weakest_delta", 0)
            strong = ea.get("strongest_benchmark", "?")
            strong_d = ea.get("strongest_delta", 0)
            lines.append(f"- **Strongest benchmark**: {strong} ({strong_d:+.1f}% vs FSL)")
            lines.append(f"- **Weakest benchmark**: {weak} ({weak_d:+.1f}% vs FSL)")
            lines.append("")
            bms = ea.get("benchmarks", {})
            for bm_name, bm_data in bms.items():
                gap = bm_data.get("gap_to_target", 0)
                status = "PASSED" if bm_data.get("beats_10pct") else f"gap = {gap:.4f}"
                lines.append(f"  - {bm_name}: {status}")
            lines.append("")

    # ── Recommendations ──
    lines.append("## Recommendations")
    lines.append("")

    best_model = None
    best_count = 0
    for tag in ["p4b", "fsl"]:
        ea_path = RESULTS_DIR / f"phase10_{tag}_error_analysis.json"
        if ea_path.exists():
            with open(ea_path) as f:
                ea = json.load(f)
            n_beat = sum(1 for bm in ea.get("benchmarks", {}).values() if bm.get("beats_10pct"))
            if n_beat > best_count:
                best_count = n_beat
                best_model = tag

    if best_model:
        lines.append(f"- **Best model**: `{best_model}` (beats +10% on {best_count}/4 benchmarks)")
    else:
        lines.append("- No model met the target yet.")

    lines.append("")
    lines.append("If target not met on all 4:")
    lines.append("1. Scale training data to 1.5M pairs (more GS-10M coverage)")
    lines.append("2. Switch to `text-image-light` scope (tune both encoders)")
    lines.append("3. Run 2nd epoch on weakest category subsets")
    lines.append("4. Generate targeted synthetic data for the weakest benchmark's failure patterns")
    lines.append("")

    # ── Data Provenance Section ──
    lines.append("## Data Used (Zero Benchmark Leakage)")
    lines.append("")
    lines.append("| Source | Size | Description |")
    lines.append("|---|---|---|")
    lines.append("| GS-10M (in-domain) | ~440K pairs | Filtered fashion items from Marqo/marqo-GS-10M |")
    lines.append("| DeepFashion (InShop + MM) | ~60K pairs | Marqo/deepfashion-inshop + deepfashion-multimodal |")
    lines.append("| Enriched Descriptions | 10,000 pairs | LLM-generated queries for our own training images |")
    lines.append("| Synthetic Round 1 | 2,910 texts | Generic taxonomy expansions (no benchmark data used) |")
    lines.append("")
    lines.append("**Important**: No benchmark queries, images, or metadata were used in training data generation.")
    lines.append("Enriched descriptions were generated by asking an LLM to rephrase product titles of images")
    lines.append("already in our training set — this adds query diversity without any leakage risk.")
    lines.append("")
    lines.append("---")
    lines.append("_Generated by phase10_compare_results.py_")

    md = "\n".join(lines)
    output_path = RESULTS_DIR / "phase10_comparison.md"
    output_path.write_text(md)
    print(md)
    log.info("Comparison saved: %s", output_path)


if __name__ == "__main__":
    main()
