#!/usr/bin/env python3
"""Build a consolidated screener leaderboard from on-disk per-(dataset,model) result JSONs.

The wrapper benchmark/eval_marqo_subsample.py overwrites the per-dataset
leaderboard markdown when called once per model, so the on-disk markdowns are
incomplete. This script reads every preserved result JSON under
repos/marqo-FashionCLIP/results/<dataset>/<run_name>/text-to-image/result_text-image.json
and emits a single consolidated leaderboard.

Usage:
    python scripts/build_screener_consolidated.py [--corpus 10000 --seed 42]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATASETS = ["atlas", "polyvore", "KAGL", "fashion200k"]

# (run_name_in_results, friendly_label)
MODELS = [
    ("Marqo-FashionSigLIP",         "Marqo FashionSigLIP (baseline)",            228, 768, 224),
    ("Google-SigLIP-SO400M-384",    "Google SigLIP SO400M-384 (webli)",          840, 1152, 384),
    ("Google-SigLIP-L16-384",       "Google SigLIP L/16-384 (webli)",            304, 1024, 384),
    ("Google-SigLIP2-B16-384",      "Google SigLIP-2 B/16-384 (webli)",           86, 768, 384),
    ("Google-SigLIP2-SO400M-378",   "Google SigLIP-2 SO400M-378 (webli)",        840, 1152, 378),
]

METRICS = ["MAP@10", "NDCG@10", "Recall@10", "MRR"]


def load_metric(dataset: str, run_name: str, suffix: str) -> dict | None:
    p = (
        ROOT
        / "repos"
        / "marqo-FashionCLIP"
        / "results"
        / dataset
        / f"{run_name}_subsample{suffix}"
        / "text-to-image"
        / "result_text-image.json"
    )
    if not p.exists():
        return None
    with p.open() as f:
        d = json.load(f)
    out = {}
    out["MAP@10"] = d["mAP"]["MAP@10"]
    out["NDCG@10"] = d["ndcg"]["NDCG@10"]
    out["Recall@10"] = d["recall"]["Recall@10"]
    out["MRR"] = d["MRR"] if isinstance(d.get("MRR"), (int, float)) else d["MRR"]["MRR"]
    return out


def fmt_delta(v: float, base: float) -> str:
    if base == 0:
        return "  n/a "
    d = (v - base) / base * 100
    sign = "+" if d >= 0 else ""
    return f"{sign}{d:.1f}%"


def fmt_cell(v: float | None, base: float | None) -> str:
    if v is None:
        return "—"
    if base is None or base == 0:
        return f"{v:.4f}"
    d = (v - base) / base * 100
    arrow = "✓" if d > 0 else " "
    sign = "+" if d >= 0 else ""
    return f"{v:.4f} ({sign}{d:.1f}%){arrow}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--out",
        default="results/screener/screener_consolidated.md",
        help="Output path (relative to repo root) for the consolidated markdown.",
    )
    args = ap.parse_args()

    suffix = f"{args.corpus}_seed{args.seed}"

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    table: dict[str, dict[str, dict[str, float] | None]] = {}
    for run_name, *_ in MODELS:
        table[run_name] = {}
        for ds in DATASETS:
            table[run_name][ds] = load_metric(ds, run_name, suffix)

    baseline = table["Marqo-FashionSigLIP"]

    lines: list[str] = []
    lines.append(f"# MoDA Screener — Consolidated Leaderboard (corpus={args.corpus}, seed={args.seed})")
    lines.append("")
    lines.append(
        "Stratified subsampling: every test query keeps all of its positives, "
        "remainder filled with random non-positives (fixed seed). Absolute scores "
        "are inflated vs full-corpus eval; **relative ordering** between models "
        "is what matters for screening."
    )
    lines.append("")
    lines.append(
        f"Datasets: `{', '.join(DATASETS)}`  ·  Task: text-to-image  ·  "
        f"Models below ranked by win count vs FashionSigLIP."
    )
    lines.append("")

    for metric in METRICS:
        lines.append(f"## {metric} per dataset (Δ vs FashionSigLIP, ✓ = beats baseline)")
        lines.append("")
        header = "| Model | Params (M) | Embed dim | " + " | ".join(DATASETS) + " | wins/4 |"
        sep = "|---|---:|---:|" + "|".join(["---:"] * len(DATASETS)) + "|---:|"
        lines.append(header)
        lines.append(sep)
        for run_name, label, params_m, embed_dim, _ in MODELS:
            row_metrics = table[run_name]
            base_metrics = baseline
            cells: list[str] = []
            wins = 0
            for ds in DATASETS:
                v = row_metrics[ds][metric] if row_metrics[ds] else None
                b = base_metrics[ds][metric] if base_metrics[ds] else None
                if v is not None and b is not None and v > b:
                    wins += 1
                cells.append(fmt_cell(v, b if run_name != "Marqo-FashionSigLIP" else None))
            wins_cell = "—" if run_name == "Marqo-FashionSigLIP" else f"{wins}/4"
            lines.append(
                f"| **{label}** | {params_m} | {embed_dim} | "
                + " | ".join(cells)
                + f" | {wins_cell} |"
            )
        lines.append("")

    lines.append("## Per-dataset summary (text-to-image MAP@10)")
    lines.append("")
    lines.append("| Dataset | FashionSigLIP | Best challenger | Best challenger Δ |")
    lines.append("|---|---:|---|---:|")
    for ds in DATASETS:
        base_v = baseline[ds]["MAP@10"] if baseline[ds] else None
        best_run, best_v = None, -1.0
        for run_name, *_ in MODELS:
            if run_name == "Marqo-FashionSigLIP":
                continue
            r = table[run_name][ds]
            if r is None:
                continue
            if r["MAP@10"] > best_v:
                best_v = r["MAP@10"]
                best_run = run_name
        if base_v is None or best_run is None:
            lines.append(f"| {ds} | n/a | n/a | n/a |")
            continue
        d = (best_v - base_v) / base_v * 100
        sign = "+" if d >= 0 else ""
        lines.append(f"| {ds} | {base_v:.4f} | {best_run} ({best_v:.4f}) | {sign}{d:.1f}% |")
    lines.append("")

    lines.append("## Run-time + cost notes")
    lines.append("")
    lines.append("- All runs on Apple-silicon MPS, batch_size=128.")
    lines.append("- Per-model wall time per dataset (10K stratified subsample):")
    lines.append("  - FashionSigLIP B/16-224 ≈ 4 min · SigLIP-2 B/16-384 ≈ 10 min · "
                 "SigLIP L/16-384 ≈ 50 min · SigLIP SO400M-384 ≈ 50 min · "
                 "SigLIP-2 SO400M-378 ≈ 50 min")
    lines.append("- Phase 1 (atlas + polyvore × 5 models) finished in 3h 47m. "
                 "Phase 2 (KAGL + fashion200k × 5 models) in 3h 39m.")
    lines.append("- One bug fixed mid-run: original `google-siglip2-so400m-384` config "
                 "pointed at non-existent `ViT-SO400M-14-SigLIP2-384`; corrected to "
                 "`ViT-SO400M-14-SigLIP2-378` (open_clip ships SigLIP-2 SO400M only at 378px).")
    lines.append("")
    lines.append(f"_Generated by `scripts/build_screener_consolidated.py`._")

    out_path.write_text("\n".join(lines) + "\n")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
