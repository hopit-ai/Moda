#!/usr/bin/env python
"""
Fills in the 'Results' placeholder in the Phase 5.X / Recipe X section of
EXPERIMENT_LOG.md using results/attributes/linear_probe_eval.json.

Safe to run multiple times: it replaces the block between a pair of
sentinel comment markers.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
LOG = REPO / "EXPERIMENT_LOG.md"
PROBE_JSON = REPO / "results/attributes/linear_probe_eval.json"

PLACEHOLDER = ("*(Run is in progress — table will be filled in here once "
               "`results/attributes/linear_probe_eval.json` is written.)*")


def build_table(probe: dict) -> str:
    models = probe["config"]["models"]
    if len(models) != 2:
        a = models[0]
        b = models[-1]
    else:
        a, b = models

    lines = []
    lines.append(f"**Models compared:** `{a}` vs `{b}` (Δ = {b} − {a})")
    lines.append("")
    lines.append(
        f"| Dataset | Attribute | #classes | #eval | "
        f"{a} acc / F1 | {b} acc / F1 | Δacc |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")

    for row in probe["summary_table"]:
        if row.get(f"{a}_acc") is None or row.get(f"{b}_acc") is None:
            continue
        delta = row.get("delta_acc", 0.0) or 0.0
        sign = "+" if delta >= 0 else ""
        lines.append(
            f"| {row['dataset']} | {row['attribute']} "
            f"| {row.get('n_classes', '-')} | {row.get('n_eval', '-')} "
            f"| {row[f'{a}_acc']:.3f} / {row[f'{a}_f1']:.3f} "
            f"| {row[f'{b}_acc']:.3f} / {row[f'{b}_f1']:.3f} "
            f"| {sign}{delta:.3f} |")

    totals_a_acc = [r[f"{a}_acc"] for r in probe["summary_table"] if r.get(f"{a}_acc") is not None]
    totals_b_acc = [r[f"{b}_acc"] for r in probe["summary_table"] if r.get(f"{b}_acc") is not None]
    wins_b = sum(1 for r in probe["summary_table"]
                 if (r.get("delta_acc") or 0) > 0.005)
    losses_b = sum(1 for r in probe["summary_table"]
                   if (r.get("delta_acc") or 0) < -0.005)
    ties = len(totals_a_acc) - wins_b - losses_b

    lines.append("")
    lines.append(f"**Summary (across {len(totals_a_acc)} attribute axes):**")
    lines.append("")
    lines.append(f"- Mean top-1 accuracy, `{a}`: "
                 f"**{sum(totals_a_acc)/len(totals_a_acc):.3f}**")
    lines.append(f"- Mean top-1 accuracy, `{b}`: "
                 f"**{sum(totals_b_acc)/len(totals_b_acc):.3f}**")
    lines.append(
        f"- `{b}` wins on **{wins_b}** axes, loses on **{losses_b}**, "
        f"ties on **{ties}** (|Δ|<0.005 threshold).")
    lines.append("")

    if losses_b > 0:
        lines.append("**⚠️ Attribute regression detected** — the DF2 fine-tune "
                     "gave up some attribute discriminability for retrieval "
                     "discriminability on these axes. Recipe A' distillation "
                     "loss will include an explicit attribute-preservation term.")
    else:
        lines.append("**✅ No attribute regression** — DF2 fine-tuning retained "
                     "or improved every probed attribute. Recipe A' can rely "
                     "purely on ensemble distillation without an extra "
                     "attribute-preservation loss.")

    return "\n".join(lines)


def main():
    if not PROBE_JSON.exists():
        print(f"No probe results at {PROBE_JSON}; nothing to update.", file=sys.stderr)
        return 1
    probe = json.loads(PROBE_JSON.read_text())
    text = LOG.read_text()
    if PLACEHOLDER not in text:
        if "| Dataset | Attribute |" in text:
            print("Results table already present; re-writing with latest numbers.")
            # Best-effort replace: target from '### Results' block start to next '### Source'
            import re
            pat = re.compile(
                r"(### Results\n\n).*?(\n\n### Source files \(Recipe X\))",
                re.DOTALL)
            if not pat.search(text):
                print("Could not locate Results block; skipping.", file=sys.stderr)
                return 2
            new_text = pat.sub(
                lambda m: m.group(1) + build_table(probe) + m.group(2), text)
        else:
            print("Placeholder missing and no existing table found.", file=sys.stderr)
            return 2
    else:
        new_text = text.replace(PLACEHOLDER, build_table(probe))

    LOG.write_text(new_text)
    print(f"Updated {LOG} with Recipe X results.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
