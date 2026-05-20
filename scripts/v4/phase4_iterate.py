"""
Phase 4: Turn gap_analysis into an actionable mining checklist.

Reads `results/v4_gcl/<run>/gap_analysis.json` and writes
`data/processed/v4_pattern_targeted/iteration_plan.md`.
"""
import argparse
import json
from pathlib import Path

PROJ_ROOT = Path(__file__).resolve().parents[2]
RESULTS = PROJ_ROOT / "results" / "v4_gcl"
OUT = PROJ_ROOT / "data" / "processed" / "v4_pattern_targeted" / "iteration_plan.md"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", type=str, default="ft_v4")
    args = parser.parse_args()

    gap_path = RESULTS / args.run_name / "gap_analysis.json"
    if not gap_path.exists():
        raise SystemExit(f"Missing {gap_path}; run phase3b_gap_analysis.py first.")

    with open(gap_path) as f:
        analysis = json.load(f)

    lines = [
        "# Iteration plan (Phase 4)",
        "",
        f"Source: `{gap_path}`",
        "",
        "## Mining priorities (by worst MRR gap)",
        "",
    ]

    for i, g in enumerate(analysis.get("gaps", []), 1):
        recipe = g.get("recipe") or {}
        lines.append(f"{i}. **{g['benchmark']}** / `{g['task']}` — "
                       f"MRR {g['our_mrr']:.3f} vs ref {g['ref_mrr']:.3f} (Δ {g['delta']:.3f})")
        lines.append(f"   - Data: {recipe.get('data_strategy', 'Mine more GS-10M / synth in this pattern.')}")
        lines.append("")

    lines.extend([
        "## Retrain loop",
        "",
        "1. Stream additional GS-10M rows biased toward gap buckets (see recipes).",
        "2. Optionally extend `phase1b_synthetic_gapfill.py --local-only` templates for weak datasets.",
        "3. Re-run `phase1c_leakage_check.py` after adding pairs.",
        "4. Fine-tune from **best checkpoint** with LR ~1e-6 for 1 epoch.",
        "5. Re-run `phase3_eval_all_benchmarks.py` (full settings, not `--fast`).",
        "",
    ])

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
