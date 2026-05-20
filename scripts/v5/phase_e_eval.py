"""
Phase E (minimal) — final eval of the best Phase D checkpoint vs baselines.

Probes 3 models on the same 1K-subsample protocol:
  1. SigLIP-2 base (zero-shot)        — our starting point
  2. Best Phase D checkpoint          — what training produced
  3. (Optional) FashionSigLIP         — Marqo's published model, the target to beat

Writes results/v5/phase_e_summary.{json,md}.

Usage:
    python scripts/v5/phase_e_eval.py
    python scripts/v5/phase_e_eval.py --ckpt checkpoints/v5/phase_d_best.pt --include_fsl
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))

from v5_eval_probe import EvalProbe
from v5_model import build_fsl_student, build_student

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results" / "v5"


def pick_device():
    if torch.backends.mps.is_available(): return "mps"
    if torch.cuda.is_available(): return "cuda"
    return "cpu"


def probe_with_baseline(probe: EvalProbe, device: str, fsl_student: bool = False) -> dict:
    """Zero-shot baseline — SL2-B or FSL depending on mode."""
    if fsl_student:
        print("Probing FSL baseline (zero-shot) ...")
        model, tokenizer = build_fsl_student(device=device)
    else:
        print("Probing SigLIP-2 baseline (zero-shot) ...")
        model, tokenizer = build_student(device=device)
    return probe.probe(model, tokenizer)


def probe_with_checkpoint(probe: EvalProbe, ckpt_path: Path, device: str,
                           fsl_student: bool = False) -> dict:
    """Load Phase D checkpoint and probe."""
    print(f"Probing checkpoint {ckpt_path} ...")
    if fsl_student:
        model, tokenizer = build_fsl_student(device=device)
    else:
        model, tokenizer = build_student(device=device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if "model_trainable" in ckpt:
        sd = {n: p.to(device) for n, p in ckpt["model_trainable"].items()}
        model.load_state_dict(sd, strict=False)
    else:
        model.load_state_dict(ckpt, strict=False)
    return probe.probe(model, tokenizer)


def probe_with_fsl(device: str) -> dict:
    """Marqo-FashionSigLIP zero-shot — the one we're trying to beat.

    Uses the SAME eval cache as our student probe; we just swap the model.
    Note: FSL is a SigLIP-1 architecture so we can't reuse student image
    embeddings. We need to re-encode benchmark images through FSL's image
    tower. For tonight's run we'll skip this path (use --include_fsl=false)
    unless the user explicitly opts in — comparison vs SL2 baseline is the
    primary signal anyway.
    """
    import open_clip
    print("Probing Marqo-FashionSigLIP (zero-shot) ...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        "hf-hub:Marqo/marqo-fashionSigLIP"
    )
    tokenizer = open_clip.get_tokenizer("hf-hub:Marqo/marqo-fashionSigLIP")
    model = model.to(device).eval()

    # Re-build a probe with FSL-encoded image embeddings
    # This requires loading benchmark images and re-encoding them, which is
    # ~3 min on MPS for 4×1200 images. Implemented in v5_eval_probe.build_eval_caches
    # but that function uses SL2 by default. For now we skip FSL probe in the
    # autonomous run — it can be added with a separate script.
    raise NotImplementedError(
        "FSL probe needs FSL-encoded image cache; skipped in v5 phase_e minimal. "
        "Run results/v4_gcl/baseline_v4/full_results.json contains FSL numbers from v4 phase3."
    )


def write_summary_md(results: dict, fsl_baseline: dict | None, out_path: Path):
    """Generate human-readable comparison report."""
    bench_order = ["fashion200k", "atlas", "polyvore", "KAGL"]

    lines = ["# v5 Phase E — Final Evaluation\n"]
    lines.append(f"_Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}_\n")
    lines.append("## Per-benchmark MRR (1K-subsample protocol)\n")
    lines.append("| Benchmark | SL2 base | v5 trained | Δ vs SL2 | FSL (target) | Δ vs FSL |")
    lines.append("|---|---:|---:|---:|---:|---:|")

    base = results.get("baseline", {})
    trained = results.get("trained", {})
    for b in bench_order:
        b_mrr = base.get(b, {}).get("mrr", 0)
        t_mrr = trained.get(b, {}).get("mrr", 0)
        f_mrr = (fsl_baseline or {}).get(b, {}).get("text_to_image", {}).get("mrr", None)
        delta_sl2 = (t_mrr - b_mrr) / max(b_mrr, 1e-9) * 100 if b_mrr else 0
        if f_mrr is not None:
            delta_fsl = (t_mrr - f_mrr) / max(f_mrr, 1e-9) * 100
            lines.append(
                f"| {b} | {b_mrr:.4f} | **{t_mrr:.4f}** | {delta_sl2:+.1f}% | "
                f"{f_mrr:.4f} | {delta_fsl:+.1f}% |"
            )
        else:
            lines.append(
                f"| {b} | {b_mrr:.4f} | **{t_mrr:.4f}** | {delta_sl2:+.1f}% | — | — |"
            )

    # Mean row
    if base and trained:
        b_mean = sum(base.get(b, {}).get("mrr", 0) for b in bench_order) / len(bench_order)
        t_mean = sum(trained.get(b, {}).get("mrr", 0) for b in bench_order) / len(bench_order)
        delta_mean = (t_mean - b_mean) / max(b_mean, 1e-9) * 100
        f_mean = None
        if fsl_baseline:
            f_vals = [fsl_baseline.get(b, {}).get("text_to_image", {}).get("mrr") for b in bench_order]
            f_vals = [v for v in f_vals if v is not None]
            if len(f_vals) == len(bench_order):
                f_mean = sum(f_vals) / len(f_vals)
        if f_mean:
            delta_fsl_mean = (t_mean - f_mean) / max(f_mean, 1e-9) * 100
            lines.append(
                f"| **mean** | {b_mean:.4f} | **{t_mean:.4f}** | {delta_mean:+.1f}% | "
                f"{f_mean:.4f} | {delta_fsl_mean:+.1f}% |"
            )
        else:
            lines.append(
                f"| **mean** | {b_mean:.4f} | **{t_mean:.4f}** | {delta_mean:+.1f}% | — | — |"
            )

    lines.append("\n## Decision\n")
    lines.append("Per PLAN_V5 §6.3 decision matrix:")
    lines.append("- Beat FSL on all 4 by ≥3%, p<0.0125 → **ship**")
    lines.append("- Beat FSL on 3/4 by ≥3% → **Phase F surgical fix**")
    lines.append("- Beat FSL on ≤2/4 → **diagnose**")
    lines.append("- Match or worse → **Plan v6 (need scale or larger backbone)**")

    out_path.write_text("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, default=REPO / "checkpoints/v5/phase_d_best.pt")
    ap.add_argument("--device", default=None)
    ap.add_argument("--include_fsl", action="store_true",
                    help="Also probe Marqo-FashionSigLIP (requires re-encoding bench images)")
    ap.add_argument("--fsl_student", action="store_true",
                    help="Checkpoint uses FSL backbone; load FSL eval caches and FSL baseline.")
    ap.add_argument("--out_dir", type=Path, default=RESULTS)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = args.device or pick_device()
    print(f"Device: {device}")

    print("Loading EvalProbe ...")
    if args.fsl_student:
        fsl_eval_cache = REPO / "data" / "processed" / "v5_eval_cache_fsl"
        probe = EvalProbe(cache_dir=fsl_eval_cache, device=device)
    else:
        probe = EvalProbe(device=device)
    if not probe.benchmarks:
        sys.exit("ERROR: no eval caches found. Run phase_a_cache_fsl_eval.py first.")

    results = {}

    # Try to load existing baseline from earlier run
    baseline_key = "baseline_fsl.json" if args.fsl_student else "baseline_sl2.json"
    baseline_cache = args.out_dir / baseline_key
    if baseline_cache.exists():
        print(f"Reusing baseline from {baseline_cache}")
        results["baseline"] = json.loads(baseline_cache.read_text())
    else:
        results["baseline"] = probe_with_baseline(probe, device, fsl_student=args.fsl_student)

    # Probe trained checkpoint (if exists)
    if args.ckpt.exists():
        results["trained"] = probe_with_checkpoint(probe, args.ckpt, device,
                                                    fsl_student=args.fsl_student)
    else:
        print(f"WARNING: {args.ckpt} not found — skipping trained probe")
        results["trained"] = {}

    # Optional: FSL baseline for direct target comparison
    fsl_baseline = None
    fsl_path = REPO / "results/v4_gcl/baseline_v4/full_results.json"
    if fsl_path.exists():
        fsl_baseline = json.loads(fsl_path.read_text())
        print(f"Loaded FSL baseline from {fsl_path}")
    elif args.include_fsl:
        try:
            fsl_baseline = probe_with_fsl(device)
        except NotImplementedError as e:
            print(f"FSL probe skipped: {e}")

    # Save outputs
    out_json = args.out_dir / "phase_e_summary.json"
    out_md = args.out_dir / "phase_e_summary.md"
    out_json.write_text(json.dumps({
        **results, "fsl_baseline_path": str(fsl_path) if fsl_baseline else None,
    }, indent=2, default=str))
    write_summary_md(results, fsl_baseline, out_md)
    print(f"\nWrote {out_json}")
    print(f"Wrote {out_md}")
    print("\n=== Summary (read the .md for full table) ===")
    print(out_md.read_text())


if __name__ == "__main__":
    main()
