#!/usr/bin/env bash
# Unattended overnight pipeline:
#   wait for Marqo-GS metadata tar -> sample 5k wfash triplets ->
#   train SigLIP L/16-384 LoRA on them (auto-resume on crash) ->
#   eval on fashion200k 10K screener -> write a verdict markdown ->
#   if no big regression, also eval on atlas + polyvore + KAGL.
#
# Designed to be launched fully detached:
#   nohup bash scripts/overnight_v2_smoke.sh > logs/overnight_nohup.log 2>&1 &
#   disown

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$REPO_ROOT/logs"
mkdir -p "$LOG_DIR"
MAIN_LOG="$LOG_DIR/overnight_v2_${TS}.log"

# -- redirect EVERYTHING (stdout+stderr) to the main log AND keep on terminal
exec > >(tee -a "$MAIN_LOG") 2>&1

log() { printf '%s  [orch] %s\n' "$(date +'%H:%M:%S')" "$*"; }

PY="$REPO_ROOT/.venv/bin/python"
TAR_PATH="$REPO_ROOT/data/raw/marqo_gs/marqo-gs-dataset.tar"
TAR_URL="https://marqo-gcl-public.s3.amazonaws.com/v1/marqo-gs-dataset.tar"
TAR_EXPECTED_BYTES=$((7545 * 1024 * 1024))   # 7.36 GiB ≈ 7912 MB; using 7545 MiB lower bound

TRIPLETS="$REPO_ROOT/data/processed/marqo_gs_wfash_subset/triplets.jsonl"
OUTPUT_DIR="$REPO_ROOT/models/moda-siglip-l16-lora-v2-smoke"
MODEL_KEY="siglip-l16-lora-v2-smoke"
BASELINE_KEY="google-siglip-l16-384"
FSL_KEY="fashion-siglip"

NUM_PAIRS=5000
MAX_STEPS=200
BATCH_SIZE=12
LR="1e-5"
SAVE_EVERY=50

log "==== overnight v2 smoke pipeline start ===="
log "main log -> $MAIN_LOG"

# ============================================================================
# STAGE 1 — no-op (HF datasets streaming, no S3 tar needed)
# ============================================================================
log "STAGE 1 — using HuggingFace Marqo/marqo-GS-10M streaming (skip S3 tar)"

# ============================================================================
# STAGE 2 — sample wfash subset (extract just wfash from tar, sample, fetch images)
# ============================================================================
log "STAGE 2 — build $NUM_PAIRS-triplet subset from Marqo-GS wfash"

# Retry the sampler up to 2x in case of a transient image-fetch storm.
for attempt in 1 2; do
    if [[ -f "$TRIPLETS" ]]; then
        n=$(wc -l < "$TRIPLETS" | tr -d ' ')
        if (( n >= NUM_PAIRS / 2 )); then
            log "triplets file already has $n records (>= half target $NUM_PAIRS) — skipping sampler"
            break
        fi
    fi
    log "sampler attempt $attempt ..."
    "$PY" scripts/build_marqo_gs_smoke_subset.py \
        --num-pairs "$NUM_PAIRS" --shards 2 --max-examined 200000 || true
    if [[ -f "$TRIPLETS" ]] && (( $(wc -l < "$TRIPLETS" | tr -d ' ') >= NUM_PAIRS / 2 )); then
        break
    fi
    log "sampler attempt $attempt yielded too few triplets, retrying..."
    sleep 5
done

if [[ ! -f "$TRIPLETS" ]]; then
    log "FATAL: triplets file was never created. Aborting."
    exit 2
fi
N_TRIPLETS=$(wc -l < "$TRIPLETS" | tr -d ' ')
log "have $N_TRIPLETS triplets"
if (( N_TRIPLETS < 500 )); then
    log "FATAL: only $N_TRIPLETS triplets — too few to train. Aborting."
    exit 3
fi

# ============================================================================
# STAGE 3 — train with retry-on-crash
# ============================================================================
log "STAGE 3 — train v2 LoRA with auto-resume"
mkdir -p "$OUTPUT_DIR"

train_once() {
    local resume_flag=""
    if [[ -d "$OUTPUT_DIR/latest" || -n "$(ls -d "$OUTPUT_DIR"/step_* 2>/dev/null | head -n1)" ]]; then
        resume_flag="--resume auto"
        log "  (resuming from latest checkpoint)"
    fi
    caffeinate -dimsu "$PY" benchmark/train_siglip_l16_lora_v2.py \
        --triplets "$TRIPLETS" \
        --output-dir "$OUTPUT_DIR" \
        --max-steps "$MAX_STEPS" \
        --batch-size "$BATCH_SIZE" \
        --lr "$LR" \
        --save-every "$SAVE_EVERY" \
        --device mps \
        $resume_flag
}

TRAIN_OK=0
for attempt in 1 2 3; do
    log "training attempt $attempt"
    if train_once; then
        log "training attempt $attempt SUCCEEDED"
        TRAIN_OK=1
        break
    else
        log "training attempt $attempt FAILED (exit $?), will resume"
        sleep 30
    fi
done

if (( TRAIN_OK == 0 )); then
    log "FATAL: training failed 3x. Aborting."
    exit 4
fi

if [[ ! -f "$OUTPUT_DIR/best/model_state_dict.pt" ]]; then
    log "FATAL: training reported success but model_state_dict.pt missing. Aborting."
    exit 5
fi
log "trained checkpoint: $OUTPUT_DIR/best/model_state_dict.pt ($(du -h "$OUTPUT_DIR/best/model_state_dict.pt" | awk '{print $1}'))"

# ============================================================================
# STAGE 4 — eval on fashion200k 10K screener
# ============================================================================
log "STAGE 4 — fashion200k 10K screener (FT'd model only; baseline + FashionSigLIP cached)"
caffeinate -dimsu "$PY" benchmark/eval_marqo_subsample.py \
    --models "$MODEL_KEY" \
    --datasets fashion200k \
    --corpus-size 10000 --seed 42 --batch-size 64 --device mps \
    --overwrite || log "WARN: fashion200k eval returned non-zero"

# ============================================================================
# STAGE 5 — write the verdict summary
# ============================================================================
log "STAGE 5 — write verdict"
SUMMARY_TS="$(date +%Y%m%d_%H%M%S)"
SUMMARY="$REPO_ROOT/results/screener/v2_smoke_summary_${SUMMARY_TS}.md"
mkdir -p "$(dirname "$SUMMARY")"

VERDICT_FILE="$LOG_DIR/verdict_${TS}.txt"
"$PY" - "$SUMMARY" "$OUTPUT_DIR" "$N_TRIPLETS" "$VERDICT_FILE" <<'PY'
import json, sys
from pathlib import Path

REPO = Path.cwd()
MARQO = REPO / "repos" / "marqo-FashionCLIP" / "results" / "fashion200k"

def find_latest(prefix):
    cands = sorted(
        [d for d in MARQO.iterdir() if d.is_dir() and d.name.startswith(prefix)],
        key=lambda p: p.stat().st_mtime, reverse=True,
    )
    return cands[0] if cands else None

def load_t2i(run_dir):
    if run_dir is None: return None
    for sub in run_dir.iterdir():
        if not sub.is_dir() or "text-to-image" not in sub.name.lower():
            continue
        files = list(sub.glob("result_*.json"))
        if not files: continue
        with open(files[0]) as f:
            raw = json.load(f)
        flat = {}
        for k,v in raw.items():
            if isinstance(v, dict):
                for sk,sv in v.items(): flat[sk] = sv
            else: flat[k] = v
        return flat
    return None

zs   = load_t2i(find_latest("Google-SigLIP-L16-384_subsample10000"))
ft   = load_t2i(find_latest("MoDA-SigLIP-L16-LoRA-V2-Smoke_subsample10000"))
ft_v1= load_t2i(find_latest("MoDA-SigLIP-L16-LoRA-Smoke_subsample10000"))
fsl  = load_t2i(find_latest("Marqo-FashionSigLIP_subsample10000"))

def fmt(m,k): return f"{m[k]:.4f}" if m and k in m else "n/a"
def delta(a,b,k):
    if not (a and b and k in a and k in b): return "n/a"
    pct = (a[k]-b[k])/b[k]*100.0
    return f"{'+' if pct>=0 else ''}{pct:.2f}%"

run_meta = {}
try:
    run_meta = json.loads((Path(sys.argv[2])/"run_meta.json").read_text())
except Exception: pass

out = []
out += [
    "# Overnight V2 Smoke — fashion200k 10K screener (text-to-image)",
    "",
    "Hypothesis being tested: **right-shape data alone (Marqo-GS catalog queries instead of DeepFashion prose) + safer LR (1e-5 vs 1e-4) prevents the catastrophic FT regression we saw in v1.**",
    "",
    f"- Triplets used: **{sys.argv[3]}** sampled from `marqo_gs_wfash_1m`",
    f"- LR: **{run_meta.get('lr','?')}**, batch {run_meta.get('batch_size','?')}, {run_meta.get('max_steps','?')} steps, loss = {run_meta.get('loss','?')}",
    f"- Final training loss: **{run_meta.get('final_loss','?'):.4f}**" if isinstance(run_meta.get('final_loss'), float) else f"- Final training loss: {run_meta.get('final_loss','?')}",
    "",
    "## Results",
    "",
    "| Model | MAP@10 | NDCG@10 | Recall@10 | MRR |",
    "| --- | ---: | ---: | ---: | ---: |",
]
for label, m in [
    ("Marqo-FashionSigLIP — reference (model to beat)", fsl),
    ("Google SigLIP L/16-384 (zero-shot)", zs),
    ("MoDA L/16 LoRA v1 smoke (DF mix, InfoNCE, LR 1e-4) — yesterday's regression", ft_v1),
    ("**MoDA L/16 LoRA v2 smoke (Marqo-GS, InfoNCE, LR 1e-5) — this run**", ft),
]:
    out.append(f"| {label} | {fmt(m,'MAP@10')} | {fmt(m,'NDCG@10')} | {fmt(m,'Recall@10')} | {fmt(m,'MRR')} |")
out += [
    "",
    "## Deltas (this run)",
    "",
    "| Comparison | MAP@10 | NDCG@10 | Recall@10 | MRR |",
    "| --- | ---: | ---: | ---: | ---: |",
    f"| v2 FT vs zero-shot L/16-384 | {delta(ft,zs,'MAP@10')} | {delta(ft,zs,'NDCG@10')} | {delta(ft,zs,'Recall@10')} | {delta(ft,zs,'MRR')} |",
    f"| v2 FT vs Marqo-FashionSigLIP | {delta(ft,fsl,'MAP@10')} | {delta(ft,fsl,'NDCG@10')} | {delta(ft,fsl,'Recall@10')} | {delta(ft,fsl,'MRR')} |",
    f"| v2 FT vs v1 FT (yesterday) | {delta(ft,ft_v1,'MAP@10')} | {delta(ft,ft_v1,'NDCG@10')} | {delta(ft,ft_v1,'Recall@10')} | {delta(ft,ft_v1,'MRR')} |",
    f"| zero-shot L/16-384 vs Marqo-FashionSigLIP | {delta(zs,fsl,'MAP@10')} | {delta(zs,fsl,'NDCG@10')} | {delta(zs,fsl,'Recall@10')} | {delta(zs,fsl,'MRR')} |",
    "",
    "## Verdict",
    "",
]

verdict_code = "UNKNOWN"
if ft and zs and "MAP@10" in ft and "MAP@10" in zs:
    diff = ft["MAP@10"] - zs["MAP@10"]
    if diff > 0.005:
        verdict_code = "HELPED"
        out.append(f"- ✅ **HELPED**: v2 FT improved MAP@10 by **{diff:+.4f}** absolute over zero-shot L/16-384.")
        if fsl and ft["MAP@10"] >= fsl["MAP@10"]:
            out.append(f"- 🎉 **AND BEATS FashionSigLIP** by {ft['MAP@10']-fsl['MAP@10']:+.4f} on fashion200k. Massive win — re-run on all 4 datasets to confirm.")
        out.append("- Next: scale to 50k triplets, add ranking-weighted (poor-man's GCL) loss, eval all 4 datasets + LookBench.")
    elif diff > -0.02:
        verdict_code = "NEUTRAL"
        out.append(f"- ➖ **NEUTRAL** (Δ MAP@10 = {diff:+.4f}). The data-shape fix alone wasn't enough.")
        out.append("- Next: add ranking-weighted loss, scale to 50k+, and/or freeze text tower to stop drift.")
    else:
        verdict_code = "REGRESSED"
        out.append(f"- ❌ **REGRESSED** by {diff:+.4f} MAP@10. Less catastrophic than v1 (-0.39) but still bad.")
        out.append("- Hypothesis: 200 steps × 5k pairs at LR 1e-5 still overfits. Try freezing text tower, or use weighted-loss + many more pairs.")
else:
    out.append("- Could not compute verdict (eval results missing). Inspect logs.")

Path(sys.argv[1]).write_text("\n".join(out))
Path(sys.argv[4]).write_text(verdict_code)
print(verdict_code)
PY

VERDICT=$(cat "$VERDICT_FILE" 2>/dev/null || echo "UNKNOWN")
log "verdict: $VERDICT"
log "summary written: $SUMMARY"

# ============================================================================
# STAGE 6 — if not catastrophic, also eval on atlas + polyvore + KAGL
# ============================================================================
if [[ "$VERDICT" == "HELPED" || "$VERDICT" == "NEUTRAL" ]]; then
    log "STAGE 6 — bonus eval on atlas + polyvore + KAGL (full picture)"
    caffeinate -dimsu "$PY" benchmark/eval_marqo_subsample.py \
        --models "$MODEL_KEY" \
        --datasets atlas polyvore KAGL \
        --corpus-size 10000 --seed 42 --batch-size 64 --device mps \
        --overwrite || log "WARN: bonus eval returned non-zero"
    log "bonus eval done; raw results under repos/marqo-FashionCLIP/results/<dataset>/MoDA-SigLIP-L16-LoRA-V2-Smoke_subsample10000_seed42/"
else
    log "STAGE 6 — skipped (verdict was $VERDICT, not worth burning 2 more hours)"
fi

log "==== overnight v2 smoke pipeline COMPLETE ===="
log "summary: $SUMMARY"
log "main log: $MAIN_LOG"
