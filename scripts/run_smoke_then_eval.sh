#!/usr/bin/env bash
# Wait for the in-flight LoRA smoke FT to finish, then run the fashion200k 10K
# screener for both the zero-shot baseline and the FT'd checkpoint, and
# summarise the delta. Designed to be launched in the background and left to
# run unattended.
#
# Usage:
#   bash scripts/run_smoke_then_eval.sh
#
# Outputs:
#   logs/smoke_then_eval_<timestamp>.log
#   results/screener/screener_fashion200k_10000.md   (refreshed by the screener)
#   results/screener/smoke_summary_<timestamp>.md    (our delta summary)

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$REPO_ROOT/logs"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/smoke_then_eval_${TS}.log"

CKPT="$REPO_ROOT/models/moda-siglip-l16-lora-smoke/best/model_state_dict.pt"
META="$REPO_ROOT/models/moda-siglip-l16-lora-smoke/run_meta.json"

log() { printf '%s  %s\n' "$(date +'%H:%M:%S')" "$*" | tee -a "$LOG"; }

log "chainer started; log -> $LOG"
log "waiting for training checkpoint: $CKPT"

# Wait up to 4h for the training to finish (ckpt to appear AND run_meta.json
# written, which signals the trainer wrote everything cleanly).
DEADLINE=$(( $(date +%s) + 4 * 3600 ))
while :; do
  if [[ -f "$CKPT" && -f "$META" ]]; then
    # Make sure the file is not still being written: size stable for 30s.
    s1=$(stat -f%z "$CKPT" 2>/dev/null || echo 0)
    sleep 30
    s2=$(stat -f%z "$CKPT" 2>/dev/null || echo 0)
    if [[ "$s1" == "$s2" && "$s1" != "0" ]]; then
      log "checkpoint detected and stable (size=$s1 bytes)"
      break
    fi
  fi
  if (( $(date +%s) > DEADLINE )); then
    log "ERROR: training did not finish within 4h. aborting eval."
    exit 1
  fi
  sleep 60
done

log "==== running fashion200k 10K screener: zero-shot baseline + FT'd ===="
caffeinate -dimsu .venv/bin/python benchmark/eval_marqo_subsample.py \
    --models google-siglip-l16-384 siglip-l16-lora-smoke \
    --datasets fashion200k \
    --corpus-size 10000 \
    --seed 42 \
    --batch-size 64 \
    --device mps 2>&1 | tee -a "$LOG"
EVAL_RC=${PIPESTATUS[0]}
log "screener exit code: $EVAL_RC"

# ---- summarise the delta ---------------------------------------------------
log "==== summarising delta ===="
SUMMARY="$REPO_ROOT/results/screener/smoke_summary_${TS}.md"
mkdir -p "$(dirname "$SUMMARY")"

.venv/bin/python - "$SUMMARY" <<'PY' 2>&1 | tee -a "$LOG"
import json, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1] if False else Path.cwd()
MARQO = REPO / "repos" / "marqo-FashionCLIP" / "results" / "fashion200k"

def find_latest(prefix):
    cands = sorted([d for d in MARQO.iterdir() if d.is_dir() and d.name.startswith(prefix)], key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0] if cands else None

def load_t2i(run_dir):
    if run_dir is None:
        return None
    # screener writes per-task subfolders containing result_*.json
    for sub in run_dir.iterdir():
        if not sub.is_dir():
            continue
        if "text-to-image" not in sub.name.lower():
            continue
        files = list(sub.glob("result_*.json"))
        if not files:
            continue
        with open(files[0]) as f:
            raw = json.load(f)
        flat = {}
        for k, v in raw.items():
            if isinstance(v, dict):
                for sk, sv in v.items():
                    flat[sk] = sv
            else:
                flat[k] = v
        return flat
    return None

zs = find_latest("Google-SigLIP-L16-384_subsample10000")
ft = find_latest("MoDA-SigLIP-L16-LoRA-Smoke_subsample10000")
fsl = find_latest("Marqo-FashionSigLIP_subsample10000")

zs_m = load_t2i(zs) if zs else None
ft_m = load_t2i(ft) if ft else None
fsl_m = load_t2i(fsl) if fsl else None

def fmt(m, key):
    return f"{m[key]:.4f}" if m and key in m else "n/a"

def delta(a, b, key):
    if not (a and b and key in a and key in b):
        return "n/a"
    pct = (a[key] - b[key]) / b[key] * 100.0
    sign = "+" if pct >= 0 else ""
    return f"{sign}{pct:.2f}%"

out = []
out.append("# Smoke FT vs Zero-Shot Baseline — fashion200k 10K screener (text-to-image)")
out.append("")
out.append("| Model | MAP@10 | NDCG@10 | Recall@10 | MRR |")
out.append("| --- | ---: | ---: | ---: | ---: |")
for label, m in [
    ("Marqo-FashionSigLIP (228M, 768d) — reference", fsl_m),
    ("Google SigLIP L/16-384 (zero-shot)", zs_m),
    ("MoDA SigLIP L/16-384 LoRA smoke (DF mix, InfoNCE)", ft_m),
]:
    out.append(f"| {label} | {fmt(m,'MAP@10')} | {fmt(m,'NDCG@10')} | {fmt(m,'Recall@10')} | {fmt(m,'MRR')} |")

out.append("")
out.append("## Deltas")
out.append("")
out.append("| Comparison | MAP@10 | NDCG@10 | Recall@10 | MRR |")
out.append("| --- | ---: | ---: | ---: | ---: |")
out.append(f"| FT vs zero-shot L/16-384 | {delta(ft_m, zs_m, 'MAP@10')} | {delta(ft_m, zs_m, 'NDCG@10')} | {delta(ft_m, zs_m, 'Recall@10')} | {delta(ft_m, zs_m, 'MRR')} |")
if fsl_m:
    out.append(f"| FT vs FashionSigLIP | {delta(ft_m, fsl_m, 'MAP@10')} | {delta(ft_m, fsl_m, 'NDCG@10')} | {delta(ft_m, fsl_m, 'Recall@10')} | {delta(ft_m, fsl_m, 'MRR')} |")
    out.append(f"| zero-shot L/16-384 vs FashionSigLIP | {delta(zs_m, fsl_m, 'MAP@10')} | {delta(zs_m, fsl_m, 'NDCG@10')} | {delta(zs_m, fsl_m, 'Recall@10')} | {delta(zs_m, fsl_m, 'MRR')} |")
else:
    out.append("")
    out.append("_Note: FashionSigLIP screener result not found. Run it once to enable the 'beat the baseline' comparison:_")
    out.append("")
    out.append("```")
    out.append(".venv/bin/python benchmark/eval_marqo_subsample.py --models fashion-siglip --datasets fashion200k --corpus-size 10000")
    out.append("```")

out.append("")
out.append("## Verdict (rough)")
out.append("")
if ft_m and zs_m and "MAP@10" in ft_m and "MAP@10" in zs_m:
    diff = ft_m["MAP@10"] - zs_m["MAP@10"]
    if diff > 0.005:
        out.append(f"- FT helped: MAP@10 improved by {diff:+.4f} absolute over zero-shot L/16-384.")
        out.append("- Next: scale to ~50k pairs / 1.5k steps and add real GCL loss + LookBench regression check.")
    elif diff > -0.005:
        out.append(f"- FT effectively neutral (MAP@10 Δ = {diff:+.4f}). 5k pairs / InfoNCE / 500 steps was too small to move the needle.")
        out.append("- Next: either (a) more data (50k+ pairs) or (b) move to GCL loss before scaling.")
    else:
        out.append(f"- FT REGRESSED: MAP@10 dropped by {diff:+.4f}. Check for over-fitting, learning rate too high, or the DF caption distribution being off-target for fashion200k.")
        out.append("- Next: lower LR (try 5e-5), inspect DF captions vs fashion200k captions, consider freezing text tower.")
else:
    out.append("- Could not compute verdict (eval results missing). Inspect logs.")

Path(sys.argv[1]).write_text("\n".join(out))
print(f"wrote {sys.argv[1]}")
PY

log "==== chainer done. summary at: $SUMMARY ===="
log "full log: $LOG"
