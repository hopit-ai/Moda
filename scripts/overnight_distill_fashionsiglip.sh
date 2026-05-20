#!/usr/bin/env bash
# Unattended overnight pipeline for FashionSigLIP distillation:
#   cache teacher embeddings (Marqo-FashionSigLIP, 5k items) ->
#   PHASE A: 50-step safety run + early eval (abort if MAP@10 below baseline) ->
#   PHASE B: full 500-step run ->
#   final fashion200k 10K eval ->
#   verdict + bonus 3-dataset eval if model survived.
#
# Designed to be launched fully detached:
#   nohup bash scripts/overnight_distill_fashionsiglip.sh > logs/overnight_distill.log 2>&1 &
#   disown
#
# Stops on first hard failure (set -e), logs everything.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$REPO_ROOT/logs"
mkdir -p "$LOG_DIR"
MAIN_LOG="$LOG_DIR/overnight_distill_${TS}.log"

exec > >(tee -a "$MAIN_LOG") 2>&1
log() { printf '%s  [distill-orch] %s\n' "$(date +'%H:%M:%S')" "$*"; }

PY="$REPO_ROOT/.venv/bin/python"

# --- Paths & keys ---
TRIPLETS="$REPO_ROOT/data/processed/marqo_gs_wfash_subset/triplets.jsonl"
TEACHER_CACHE="$REPO_ROOT/data/processed/distillation_cache/teacher_embeddings.pt"
PHASE_A_DIR="$REPO_ROOT/models/moda-siglip-distilled-from-fashionsiglip-smoke"
PHASE_A_KEY="moda-siglip-distilled-fsiglip-smoke"
PHASE_B_DIR="$REPO_ROOT/models/moda-siglip-distilled-from-fashionsiglip-v1"
PHASE_B_KEY="moda-siglip-distilled-fsiglip-v1"
BASELINE_KEY="google-siglip-b16-224"   # student init = same as this baseline
TEACHER_KEY="fashion-siglip"

# --- Hyperparams (validated by smoke test) ---
PHASE_A_STEPS=50
PHASE_B_STEPS=500
BATCH_SIZE=32
LR="1e-5"

log "==== overnight distillation pipeline start ===="
log "main log -> $MAIN_LOG"
log ""
log "Strategy: distill Marqo-FashionSigLIP (teacher) into Google-SigLIP-B16-224 (student init)."
log "Loss: L2 + cosine on cached teacher embeddings (no in-batch InfoNCE => no collapse risk)."
log "Why this != prior LoRA approach: LoRA+InfoNCE collapsed v1/v2/v3 within 50 steps."
log ""

# ============================================================================
# STAGE 1 — Verify triplets exist
# ============================================================================
log "STAGE 1 — sanity check on triplets"
if [[ ! -f "$TRIPLETS" ]]; then
    log "FATAL: triplets file missing: $TRIPLETS"
    log "  re-run scripts/build_marqo_gs_smoke_subset.py first"
    exit 1
fi
N_TRIPLETS=$(wc -l < "$TRIPLETS" | tr -d ' ')
log "  found $N_TRIPLETS triplets"
if (( N_TRIPLETS < 1000 )); then
    log "FATAL: too few triplets ($N_TRIPLETS). Need ≥1000."
    exit 1
fi

# ============================================================================
# STAGE 2 — Cache teacher embeddings
# ============================================================================
log "STAGE 2 — cache Marqo-FashionSigLIP teacher embeddings"
if [[ -f "$TEACHER_CACHE" ]]; then
    log "  cache already exists ($(du -h "$TEACHER_CACHE" | cut -f1)) — skipping"
else
    caffeinate -dimsu "$PY" scripts/cache_teacher_fashionsiglip.py \
        --triplets "$TRIPLETS" \
        --batch-size 32 || {
        log "FATAL: teacher caching failed."
        exit 2
    }
fi

# ============================================================================
# STAGE 3 — PHASE A: 50-step safety run
# ============================================================================
log "STAGE 3 — PHASE A: 50-step safety run (output: $PHASE_A_DIR)"
mkdir -p "$PHASE_A_DIR"

# We always re-run phase A so the safety check uses the latest config; cheap (~2 min).
rm -rf "$PHASE_A_DIR"/step_* "$PHASE_A_DIR"/best 2>/dev/null

caffeinate -dimsu "$PY" benchmark/distill_fashionsiglip_to_student.py \
    --teacher-cache "$TEACHER_CACHE" \
    --output-dir "$PHASE_A_DIR" \
    --max-steps "$PHASE_A_STEPS" \
    --batch-size "$BATCH_SIZE" \
    --lr "$LR" \
    --feat-weight 1.0 --cos-weight 1.0 \
    --save-every 25 --log-every 5 --device mps || {
    log "FATAL: phase A training failed"
    exit 3
}

if [[ ! -f "$PHASE_A_DIR/best/model_state_dict.pt" ]]; then
    log "FATAL: phase A produced no checkpoint"
    exit 3
fi
log "  phase A done"

# ============================================================================
# STAGE 4 — Safety eval: phase A on fashion200k 10K
# ============================================================================
log "STAGE 4 — safety eval: phase A model + baseline + teacher on fashion200k 10K"
caffeinate -dimsu "$PY" benchmark/eval_marqo_subsample.py \
    --models "$PHASE_A_KEY" "$BASELINE_KEY" "$TEACHER_KEY" \
    --datasets fashion200k \
    --corpus-size 10000 --seed 42 --batch-size 64 --device mps \
    --overwrite || log "WARN: safety eval returned non-zero (may be partial)"

# Inspect MAP@10 — abort if phase A regressed below baseline
SAFETY_VERDICT_FILE="$LOG_DIR/distill_safety_verdict_${TS}.txt"
"$PY" - "$SAFETY_VERDICT_FILE" <<'PY'
import json, sys
from pathlib import Path
REPO = Path.cwd()
MARQO = REPO / "repos" / "marqo-FashionCLIP" / "results" / "fashion200k"

def find_latest(prefix):
    cands = sorted([d for d in MARQO.iterdir() if d.is_dir() and d.name.startswith(prefix)],
                   key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0] if cands else None

def load_t2i(run_dir):
    if run_dir is None: return None
    for sub in run_dir.iterdir():
        if not sub.is_dir() or "text-to-image" not in sub.name.lower(): continue
        files = list(sub.glob("result_*.json"))
        if not files: continue
        with open(files[0]) as f: raw = json.load(f)
        flat = {}
        for k, v in raw.items():
            if isinstance(v, dict):
                for sk, sv in v.items(): flat[sk] = sv
            else: flat[k] = v
        return flat
    return None

phase_a = load_t2i(find_latest("MoDA-SigLIP-Distilled-FSiglip-Smoke_subsample10000"))
baseline = load_t2i(find_latest("Google-SigLIP-B16-224_subsample10000"))
teacher  = load_t2i(find_latest("Marqo-FashionSigLIP_subsample10000"))

verdict = "UNKNOWN"
msg_lines = []

if phase_a is None:
    verdict = "ABORT_NO_RESULT"
    msg_lines.append("Phase A produced no eval result.")
elif baseline is None:
    verdict = "PROCEED_NO_BASELINE"
    msg_lines.append(f"Phase A MAP@10 = {phase_a.get('MAP@10', 'n/a')}, baseline missing — proceeding cautiously.")
else:
    a_map = phase_a.get("MAP@10", 0.0)
    b_map = baseline.get("MAP@10", 0.0)
    msg_lines.append(f"Phase A MAP@10 = {a_map:.4f}")
    msg_lines.append(f"Baseline (Google SigLIP B/16/224) MAP@10 = {b_map:.4f}")
    if teacher:
        msg_lines.append(f"Teacher (Marqo-FashionSigLIP) MAP@10 = {teacher.get('MAP@10', 0):.4f}")
    if a_map < b_map - 0.02:
        verdict = "ABORT_REGRESSED"
        msg_lines.append(f"Phase A regressed by {(a_map-b_map):+.4f} vs baseline — ABORT phase B.")
    elif a_map < b_map + 0.005:
        verdict = "PROCEED_FLAT"
        msg_lines.append("Phase A is roughly flat — distillation hasn't kicked in yet, expected for 50 steps. Proceeding.")
    else:
        verdict = "PROCEED_IMPROVING"
        msg_lines.append(f"Phase A already improving by {(a_map-b_map):+.4f} — strong signal, proceeding.")

with open(sys.argv[1], "w") as f:
    f.write(verdict + "\n")
    f.write("\n".join(msg_lines) + "\n")
print("\n".join(["VERDICT: " + verdict] + msg_lines))
PY

SAFETY_VERDICT=$(head -n1 "$SAFETY_VERDICT_FILE" 2>/dev/null | tr -d '[:space:]')
log "  safety verdict: $SAFETY_VERDICT"

case "$SAFETY_VERDICT" in
    ABORT_*)
        log "ABORTING phase B per safety check. Inspect $SAFETY_VERDICT_FILE"
        exit 4
        ;;
    PROCEED_*)
        log "  proceeding to phase B"
        ;;
    *)
        log "WARN: unknown safety verdict; proceeding cautiously"
        ;;
esac

# ============================================================================
# STAGE 5 — PHASE B: full 500-step run
# ============================================================================
log "STAGE 5 — PHASE B: full $PHASE_B_STEPS-step run (output: $PHASE_B_DIR)"
mkdir -p "$PHASE_B_DIR"

train_phase_b_once() {
    local resume_flag=""
    if [[ -n "$(ls -d "$PHASE_B_DIR"/step_* 2>/dev/null | head -n1)" ]]; then
        resume_flag="--resume auto"
        log "  (phase B resuming from latest)"
    fi
    caffeinate -dimsu "$PY" benchmark/distill_fashionsiglip_to_student.py \
        --teacher-cache "$TEACHER_CACHE" \
        --output-dir "$PHASE_B_DIR" \
        --max-steps "$PHASE_B_STEPS" \
        --batch-size "$BATCH_SIZE" \
        --lr "$LR" \
        --feat-weight 1.0 --cos-weight 1.0 \
        --save-every 100 --log-every 20 --device mps \
        $resume_flag
}

PHASE_B_OK=0
for attempt in 1 2 3; do
    log "  phase B training attempt $attempt"
    if train_phase_b_once; then
        log "  phase B training attempt $attempt SUCCEEDED"
        PHASE_B_OK=1
        break
    fi
    log "  phase B training attempt $attempt FAILED, will resume"
    sleep 30
done

if (( PHASE_B_OK == 0 )) || [[ ! -f "$PHASE_B_DIR/best/model_state_dict.pt" ]]; then
    log "FATAL: phase B did not produce a final checkpoint."
    exit 5
fi

# ============================================================================
# STAGE 6 — Final eval: phase B model + baseline + teacher on fashion200k 10K
# ============================================================================
log "STAGE 6 — final eval on fashion200k 10K"
caffeinate -dimsu "$PY" benchmark/eval_marqo_subsample.py \
    --models "$PHASE_B_KEY" "$BASELINE_KEY" "$TEACHER_KEY" \
    --datasets fashion200k \
    --corpus-size 10000 --seed 42 --batch-size 64 --device mps \
    --overwrite || log "WARN: final eval returned non-zero"

# ============================================================================
# STAGE 7 — Verdict + summary report
# ============================================================================
log "STAGE 7 — write summary + verdict"
SUMMARY="$REPO_ROOT/results/screener/distill_summary_${TS}.md"
VERDICT_FILE="$LOG_DIR/distill_verdict_${TS}.txt"
mkdir -p "$(dirname "$SUMMARY")"

"$PY" - "$SUMMARY" "$PHASE_B_DIR" "$VERDICT_FILE" <<'PY'
import json, sys
from pathlib import Path
REPO = Path.cwd()
MARQO = REPO / "repos" / "marqo-FashionCLIP" / "results" / "fashion200k"

def find_latest(prefix):
    cands = sorted([d for d in MARQO.iterdir() if d.is_dir() and d.name.startswith(prefix)],
                   key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0] if cands else None

def load_t2i(run_dir):
    if run_dir is None: return None
    for sub in run_dir.iterdir():
        if not sub.is_dir() or "text-to-image" not in sub.name.lower(): continue
        files = list(sub.glob("result_*.json"))
        if not files: continue
        with open(files[0]) as f: raw = json.load(f)
        flat = {}
        for k, v in raw.items():
            if isinstance(v, dict):
                for sk, sv in v.items(): flat[sk] = sv
            else: flat[k] = v
        return flat
    return None

phase_a   = load_t2i(find_latest("MoDA-SigLIP-Distilled-FSiglip-Smoke_subsample10000"))
phase_b   = load_t2i(find_latest("MoDA-SigLIP-Distilled-FSiglip-V1_subsample10000"))
baseline  = load_t2i(find_latest("Google-SigLIP-B16-224_subsample10000"))
teacher   = load_t2i(find_latest("Marqo-FashionSigLIP_subsample10000"))

def fmt(m, k): return f"{m[k]:.4f}" if m and k in m else "n/a"
def delta(a, b, k):
    if not (a and b and k in a and k in b): return "n/a"
    pct = (a[k]-b[k])/b[k]*100.0
    return f"{'+' if pct>=0 else ''}{pct:.2f}%"
def absd(a, b, k):
    if not (a and b and k in a and k in b): return "n/a"
    d = a[k]-b[k]
    return f"{'+' if d>=0 else ''}{d:.4f}"

meta = {}
try: meta = json.loads((Path(sys.argv[2])/"run_meta.json").read_text())
except Exception: pass

out = []
out += [
    "# Overnight Distillation v1 — fashion200k 10K screener",
    "",
    "Strategy: distill **Marqo-FashionSigLIP** (teacher) into a fresh **Google-SigLIP-B16-224** (student init).",
    "Loss = L2 + cosine on cached teacher embeddings (no in-batch InfoNCE → no collapse failure mode).",
    "",
    f"- Teacher cache: {meta.get('n_pairs','?')} pairs (Marqo-GS wfash, distinct from fashion200k)",
    f"- Student arch: ViT-B/16/224, 768d, 203M params (same as teacher; this run is a knowledge transfer, not compression)",
    f"- LR: {meta.get('lr','?')}, batch: {meta.get('batch_size','?')}, steps: {meta.get('max_steps','?')}",
    f"- Final training loss: " + (f"**{meta.get('final_loss'):.5f}**" if isinstance(meta.get('final_loss'), float) else f"{meta.get('final_loss','?')}"),
    "",
    "## Results — fashion200k 10K (text → image retrieval)",
    "",
    "| Model | MAP@10 | NDCG@10 | Recall@10 | MRR |",
    "| --- | ---: | ---: | ---: | ---: |",
]
for label, m in [
    ("Marqo-FashionSigLIP — teacher (target)", teacher),
    ("Google SigLIP B/16/224 — student init (no fashion training)", baseline),
    ("MoDA distilled — phase A (50 steps, safety check)", phase_a),
    ("**MoDA distilled — phase B (full run) — this experiment**", phase_b),
]:
    out.append(f"| {label} | {fmt(m,'MAP@10')} | {fmt(m,'NDCG@10')} | {fmt(m,'Recall@10')} | {fmt(m,'MRR')} |")

out += [
    "",
    "## Deltas",
    "",
    "| Comparison | Δ MAP@10 (abs) | MAP@10 (%) | Recall@10 (%) |",
    "| --- | ---: | ---: | ---: |",
    f"| phase B vs baseline (student init) | {absd(phase_b,baseline,'MAP@10')} | {delta(phase_b,baseline,'MAP@10')} | {delta(phase_b,baseline,'Recall@10')} |",
    f"| phase B vs teacher (FashionSigLIP) | {absd(phase_b,teacher,'MAP@10')} | {delta(phase_b,teacher,'MAP@10')} | {delta(phase_b,teacher,'Recall@10')} |",
    f"| phase A → phase B progression | {absd(phase_b,phase_a,'MAP@10')} | {delta(phase_b,phase_a,'MAP@10')} | {delta(phase_b,phase_a,'Recall@10')} |",
    "",
    "## Verdict",
    "",
]

verdict = "UNKNOWN"
if phase_b and teacher and "MAP@10" in phase_b and "MAP@10" in teacher:
    pct_of_teacher = phase_b["MAP@10"] / teacher["MAP@10"]
    if pct_of_teacher >= 0.90:
        verdict = "WIN"
        out.append(f"- ✅ **WIN**: phase B reached **{pct_of_teacher*100:.1f}%** of teacher MAP@10. Plan target (≥90%) met.")
        if phase_b["MAP@10"] > teacher["MAP@10"]:
            out.append(f"- 🎉 Even slightly beats the teacher itself (Δ {phase_b['MAP@10']-teacher['MAP@10']:+.4f}).")
        out.append("- Next: bonus 3-dataset eval (atlas/polyvore/KAGL).")
    elif pct_of_teacher >= 0.75:
        verdict = "PARTIAL"
        out.append(f"- 🟡 **PARTIAL**: phase B at **{pct_of_teacher*100:.1f}%** of teacher. Distillation is working but not at the 90% target.")
        out.append("- Likely fixes: more data (5k → 50k), more steps (500 → 1500), enable a small InfoNCE term (0.1).")
    elif pct_of_teacher >= 0.5 and baseline and phase_b["MAP@10"] > baseline["MAP@10"]:
        verdict = "WEAK_HELPFUL"
        out.append(f"- 🟠 **WEAK BUT HELPFUL**: phase B at {pct_of_teacher*100:.1f}% of teacher but does beat the student init baseline.")
        out.append("- Distillation has real signal but is far from optimal. Scale up data, then revisit.")
    else:
        verdict = "FAILED"
        out.append(f"- ❌ **FAILED**: phase B at only {pct_of_teacher*100:.1f}% of teacher.")
        out.append("- Investigate: was the teacher cache aligned correctly to images? Was the dataset preprocessed identically?")
else:
    out.append("- Could not compute verdict (eval missing). Inspect logs.")

Path(sys.argv[1]).write_text("\n".join(out))
Path(sys.argv[3]).write_text(verdict)
print(verdict)
PY

VERDICT=$(cat "$VERDICT_FILE" 2>/dev/null | tr -d '[:space:]')
log "verdict: $VERDICT"
log "summary: $SUMMARY"

# ============================================================================
# STAGE 8 — bonus 3-dataset eval if model survived
# ============================================================================
case "$VERDICT" in
    WIN|PARTIAL|WEAK_HELPFUL)
        log "STAGE 8 — bonus eval on atlas + polyvore + KAGL (10K corpus each)"
        caffeinate -dimsu "$PY" benchmark/eval_marqo_subsample.py \
            --models "$PHASE_B_KEY" "$TEACHER_KEY" \
            --datasets atlas polyvore KAGL \
            --corpus-size 10000 --seed 42 --batch-size 64 --device mps \
            --overwrite || log "WARN: bonus eval returned non-zero"
        log "  bonus eval done"
        ;;
    *)
        log "STAGE 8 — skipped (verdict: $VERDICT)"
        ;;
esac

log "==== overnight distillation pipeline COMPLETE ===="
log "summary: $SUMMARY"
log "verdict file: $VERDICT_FILE"
log "main log: $MAIN_LOG"
