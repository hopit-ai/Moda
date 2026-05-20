#!/usr/bin/env bash
# Independent watcher: waits for the v2 verdict file written by
# overnight_v2_smoke.sh. If the verdict is NEUTRAL, runs a hotter v3 follow-up
# (LR 5e-5, 50k pairs, 600 steps) in series with whatever v2 is doing — i.e.
# it waits for v2 to fully finish (including bonus eval) before starting, so
# we never have two heavy training/eval jobs fighting for MPS at once.
#
# Triggered by:
#   nohup bash scripts/v3_followup_watcher.sh > logs/v3_watcher.log 2>&1 &
#   disown

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

LOG_DIR="$REPO_ROOT/logs"
mkdir -p "$LOG_DIR"
TS="$(date +%Y%m%d_%H%M%S)"
WATCHER_LOG="$LOG_DIR/v3_watcher_${TS}.log"
exec > >(tee -a "$WATCHER_LOG") 2>&1

log() { printf '%s  [v3-watch] %s\n' "$(date +'%H:%M:%S')" "$*"; }

PY="$REPO_ROOT/.venv/bin/python"

V2_OUTPUT_DIR="$REPO_ROOT/models/moda-siglip-l16-lora-v2-smoke"
V2_MODEL_KEY="siglip-l16-lora-v2-smoke"

V3_OUTPUT_DIR="$REPO_ROOT/models/moda-siglip-l16-lora-v3-hotter"
V3_MODEL_KEY="siglip-l16-lora-v3-hotter"
V3_NUM_PAIRS=50000
V3_MAX_STEPS=600
V3_LR="5e-5"
V3_BATCH_SIZE=12
V3_TRIPLETS_DIR="$REPO_ROOT/data/processed/marqo_gs_wfash_50k"
V3_TRIPLETS="$V3_TRIPLETS_DIR/triplets.jsonl"

log "==== v3 follow-up watcher start ===="

# ---------------------------------------------------------------------------
# 1. Wait for v2 to finish: we look for the v2 verdict file under logs/.
#    overnight_v2_smoke.sh writes verdict_<orchTS>.txt after the fashion200k
#    eval. We poll for the most recent one whose mtime is newer than ours.
# ---------------------------------------------------------------------------
log "waiting for v2 verdict file in $LOG_DIR/verdict_*.txt ..."
WATCHER_START_EPOCH=$(date +%s)
DEADLINE=$(( WATCHER_START_EPOCH + 4 * 60 * 60 ))   # 4 hr max wait
V2_VERDICT_FILE=""
while :; do
    cand=$(ls -t "$LOG_DIR"/verdict_*.txt 2>/dev/null | head -n1 || true)
    if [[ -n "$cand" ]]; then
        cand_mtime=$(stat -f%m "$cand" 2>/dev/null || echo 0)
        if (( cand_mtime > WATCHER_START_EPOCH )); then
            V2_VERDICT_FILE="$cand"
            break
        fi
    fi
    if (( $(date +%s) > DEADLINE )); then
        log "FATAL: v2 verdict never appeared within 4h. Aborting watcher."
        exit 1
    fi
    sleep 60
done

V2_VERDICT=$(cat "$V2_VERDICT_FILE" 2>/dev/null | tr -d '[:space:]')
log "v2 verdict: $V2_VERDICT  (from $V2_VERDICT_FILE)"

# ---------------------------------------------------------------------------
# 2. Wait for the v2 orchestrator to FULLY finish (bonus eval too) so MPS is
#    free. We do this by watching the orchestrator process tree — when no
#    'overnight_v2_smoke.sh' bash and no 'eval_marqo_subsample.py' / 'train_'
#    python is running, MPS is free.
# ---------------------------------------------------------------------------
log "waiting for v2 pipeline (incl. bonus eval) to fully release MPS ..."
while :; do
    n=$(pgrep -f "overnight_v2_smoke.sh|train_siglip_l16_lora_v2.py|eval_marqo_subsample.py" 2>/dev/null | wc -l | tr -d ' ')
    if (( n == 0 )); then
        log "MPS is free (no v2 process running)"
        break
    fi
    sleep 30
done

# ---------------------------------------------------------------------------
# 3. Decide what to do based on the v2 verdict
# ---------------------------------------------------------------------------
case "$V2_VERDICT" in
    HELPED)
        log "v2 already HELPED — no need for v3 hotter run. Bonus 3-dataset eval was already done by v2 orchestrator. Exiting."
        exit 0
        ;;
    REGRESSED)
        log "v2 REGRESSED — hotter LR would make it worse. Exiting without running v3."
        exit 0
        ;;
    NEUTRAL)
        log "v2 NEUTRAL — proceeding with v3 hotter run (LR $V3_LR, $V3_NUM_PAIRS pairs, $V3_MAX_STEPS steps)"
        log "  estimated wall: 25 min sample + ~65 min train + 25 min eval = ~2 hr"
        ;;
    *)
        log "v2 verdict was '$V2_VERDICT' (UNKNOWN/empty). Aborting."
        exit 1
        ;;
esac

# ---------------------------------------------------------------------------
# 4. Sample 50k triplets to a NEW directory (don't trash v2's 5k subset)
# ---------------------------------------------------------------------------
log "STAGE A — sampling $V3_NUM_PAIRS triplets to $V3_TRIPLETS_DIR"
mkdir -p "$V3_TRIPLETS_DIR"

# The sampler hardcodes its output dir to data/processed/marqo_gs_wfash_subset.
# Trick: rename v2's subset out of the way, symlink the default path to v3 dir,
# run sampler, then restore the symlink situation so we keep both.
DEFAULT_DIR="$REPO_ROOT/data/processed/marqo_gs_wfash_subset"
V2_BACKUP_DIR="$REPO_ROOT/data/processed/marqo_gs_wfash_subset_v2_5k"
if [[ -d "$DEFAULT_DIR" && ! -L "$DEFAULT_DIR" ]]; then
    log "  preserving v2 5k subset to $V2_BACKUP_DIR"
    mv "$DEFAULT_DIR" "$V2_BACKUP_DIR"
fi
ln -snf "$V3_TRIPLETS_DIR" "$DEFAULT_DIR"

if [[ -f "$V3_TRIPLETS" ]] && (( $(wc -l < "$V3_TRIPLETS" | tr -d ' ') >= V3_NUM_PAIRS / 2 )); then
    log "  v3 triplets already exist ($(wc -l < "$V3_TRIPLETS" | tr -d ' ') rows), skipping sampler"
else
    log "  invoking sampler (5 shards, max-examined 600k) ..."
    "$PY" scripts/build_marqo_gs_smoke_subset.py \
        --num-pairs "$V3_NUM_PAIRS" --shards 5 --max-examined 600000 || {
        log "FATAL: v3 sampler failed."; exit 2;
    }
fi

if [[ ! -f "$V3_TRIPLETS" ]] || (( $(wc -l < "$V3_TRIPLETS" | tr -d ' ') < V3_NUM_PAIRS / 4 )); then
    log "FATAL: v3 has too few triplets ($(wc -l < "$V3_TRIPLETS" 2>/dev/null | tr -d ' ' || echo 0)). Stopping."
    exit 3
fi
log "  v3 has $(wc -l < "$V3_TRIPLETS" | tr -d ' ') triplets"

# ---------------------------------------------------------------------------
# 5. Train v3 with retry-on-crash (resume from latest adapter checkpoint)
# ---------------------------------------------------------------------------
log "STAGE B — train v3 LoRA ($V3_MAX_STEPS steps @ LR $V3_LR)"
mkdir -p "$V3_OUTPUT_DIR"

train_v3_once() {
    local resume_flag=""
    if [[ -d "$V3_OUTPUT_DIR/latest" || -n "$(ls -d "$V3_OUTPUT_DIR"/step_* 2>/dev/null | head -n1)" ]]; then
        resume_flag="--resume auto"
        log "  (v3 resuming from latest)"
    fi
    caffeinate -dimsu "$PY" benchmark/train_siglip_l16_lora_v2.py \
        --triplets "$V3_TRIPLETS" \
        --output-dir "$V3_OUTPUT_DIR" \
        --max-steps "$V3_MAX_STEPS" \
        --batch-size "$V3_BATCH_SIZE" \
        --lr "$V3_LR" \
        --save-every 100 \
        --device mps \
        $resume_flag
}

V3_TRAIN_OK=0
for attempt in 1 2 3; do
    log "  v3 training attempt $attempt"
    if train_v3_once; then
        log "  v3 training attempt $attempt SUCCEEDED"
        V3_TRAIN_OK=1
        break
    fi
    log "  v3 training attempt $attempt FAILED, will resume"
    sleep 30
done

if (( V3_TRAIN_OK == 0 )) || [[ ! -f "$V3_OUTPUT_DIR/best/model_state_dict.pt" ]]; then
    log "FATAL: v3 training did not produce a checkpoint."
    exit 4
fi

# ---------------------------------------------------------------------------
# 6. Eval v3 on fashion200k 10K
# ---------------------------------------------------------------------------
log "STAGE C — v3 eval on fashion200k 10K"
caffeinate -dimsu "$PY" benchmark/eval_marqo_subsample.py \
    --models "$V3_MODEL_KEY" \
    --datasets fashion200k \
    --corpus-size 10000 --seed 42 --batch-size 64 --device mps \
    --overwrite || log "WARN: v3 eval returned non-zero"

# ---------------------------------------------------------------------------
# 7. Write the v3 summary (same shape as v2 summary, with v3 row added)
# ---------------------------------------------------------------------------
log "STAGE D — write v3 summary"
V3_SUMMARY="$REPO_ROOT/results/screener/v3_smoke_summary_${TS}.md"
V3_VERDICT_FILE="$LOG_DIR/v3_verdict_${TS}.txt"
mkdir -p "$(dirname "$V3_SUMMARY")"
"$PY" - "$V3_SUMMARY" "$V3_OUTPUT_DIR" "$V3_NUM_PAIRS" "$V3_VERDICT_FILE" <<'PY'
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
        for k,v in raw.items():
            if isinstance(v, dict):
                for sk,sv in v.items(): flat[sk] = sv
            else: flat[k] = v
        return flat
    return None

zs  = load_t2i(find_latest("Google-SigLIP-L16-384_subsample10000"))
v2  = load_t2i(find_latest("MoDA-SigLIP-L16-LoRA-V2-Smoke_subsample10000"))
v3  = load_t2i(find_latest("MoDA-SigLIP-L16-LoRA-V3-Hotter_subsample10000"))
fsl = load_t2i(find_latest("Marqo-FashionSigLIP_subsample10000"))

def fmt(m,k): return f"{m[k]:.4f}" if m and k in m else "n/a"
def delta(a,b,k):
    if not (a and b and k in a and k in b): return "n/a"
    pct = (a[k]-b[k])/b[k]*100.0
    return f"{'+' if pct>=0 else ''}{pct:.2f}%"

v3_meta = {}
try: v3_meta = json.loads((Path(sys.argv[2])/"run_meta.json").read_text())
except Exception: pass

out = []
out += [
    "# Overnight V3 (hotter follow-up) — fashion200k 10K screener",
    "",
    "Auto-triggered because v2 came back NEUTRAL. v3 turns up the heat: LR 1e-5 → 5e-5, pairs 5k → 50k, steps 200 → 600.",
    "",
    f"- Triplets: **{sys.argv[3]}** (vs 5k for v2)",
    f"- LR: **{v3_meta.get('lr','?')}** (vs 1e-5 for v2)",
    f"- Steps: {v3_meta.get('max_steps','?')}, batch {v3_meta.get('batch_size','?')}, loss = {v3_meta.get('loss','?')}",
    f"- Final training loss: " + (f"**{v3_meta.get('final_loss'):.4f}**" if isinstance(v3_meta.get('final_loss'), float) else f"{v3_meta.get('final_loss','?')}"),
    "",
    "## Results",
    "",
    "| Model | MAP@10 | NDCG@10 | Recall@10 | MRR |",
    "| --- | ---: | ---: | ---: | ---: |",
]
for label, m in [
    ("Marqo-FashionSigLIP — reference", fsl),
    ("Google SigLIP L/16-384 (zero-shot)", zs),
    ("MoDA L/16 LoRA v2 (LR 1e-5, 5k pairs, 200 steps)", v2),
    ("**MoDA L/16 LoRA v3 (LR 5e-5, 50k pairs, 600 steps) — this run**", v3),
]:
    out.append(f"| {label} | {fmt(m,'MAP@10')} | {fmt(m,'NDCG@10')} | {fmt(m,'Recall@10')} | {fmt(m,'MRR')} |")
out += [
    "",
    "## Deltas",
    "",
    "| Comparison | MAP@10 | NDCG@10 | Recall@10 | MRR |",
    "| --- | ---: | ---: | ---: | ---: |",
    f"| v3 vs zero-shot L/16-384 | {delta(v3,zs,'MAP@10')} | {delta(v3,zs,'NDCG@10')} | {delta(v3,zs,'Recall@10')} | {delta(v3,zs,'MRR')} |",
    f"| v3 vs Marqo-FashionSigLIP | {delta(v3,fsl,'MAP@10')} | {delta(v3,fsl,'NDCG@10')} | {delta(v3,fsl,'Recall@10')} | {delta(v3,fsl,'MRR')} |",
    f"| v3 vs v2 | {delta(v3,v2,'MAP@10')} | {delta(v3,v2,'NDCG@10')} | {delta(v3,v2,'Recall@10')} | {delta(v3,v2,'MRR')} |",
    "",
    "## Verdict",
    "",
]

verdict = "UNKNOWN"
if v3 and zs and "MAP@10" in v3 and "MAP@10" in zs:
    diff = v3["MAP@10"] - zs["MAP@10"]
    if diff > 0.005:
        verdict = "HELPED"
        out.append(f"- ✅ **HELPED**: v3 improved MAP@10 by **{diff:+.4f}** absolute over zero-shot L/16-384.")
        if fsl and v3["MAP@10"] >= fsl["MAP@10"]:
            out.append(f"- 🎉 **AND BEATS FashionSigLIP** by {v3['MAP@10']-fsl['MAP@10']:+.4f} on fashion200k.")
        out.append("- Bonus 3-dataset eval will run next.")
    elif diff > -0.02:
        verdict = "NEUTRAL"
        out.append(f"- ➖ **STILL NEUTRAL** (Δ MAP@10 = {diff:+.4f}). Even 5e-5 + 50k didn't budge fashion200k.")
        out.append("- Strong signal that plain InfoNCE on Marqo-GS is insufficient. Real next steps: enable --use-weights (poor-man's GCL), or try FashionSigLIP's actual training recipe (multi-field text + ranking loss).")
    else:
        verdict = "REGRESSED"
        out.append(f"- ❌ **REGRESSED** (Δ MAP@10 = {diff:+.4f}). The hotter LR overshot. v2 (LR 1e-5) was the safer bet — use that.")
else:
    out.append("- Could not compute v3 verdict (eval missing). Inspect logs.")

Path(sys.argv[1]).write_text("\n".join(out))
Path(sys.argv[4]).write_text(verdict)
print(verdict)
PY

V3_VERDICT=$(cat "$V3_VERDICT_FILE" 2>/dev/null | tr -d '[:space:]')
log "v3 verdict: $V3_VERDICT"
log "v3 summary: $V3_SUMMARY"

# ---------------------------------------------------------------------------
# 8. If v3 helped, bonus 3-dataset eval on v3
# ---------------------------------------------------------------------------
if [[ "$V3_VERDICT" == "HELPED" ]]; then
    log "STAGE E — v3 HELPED, running bonus eval on atlas + polyvore + KAGL"
    caffeinate -dimsu "$PY" benchmark/eval_marqo_subsample.py \
        --models "$V3_MODEL_KEY" \
        --datasets atlas polyvore KAGL \
        --corpus-size 10000 --seed 42 --batch-size 64 --device mps \
        --overwrite || log "WARN: v3 bonus eval returned non-zero"
    log "v3 bonus eval done"
else
    log "STAGE E — skipped (v3 verdict was $V3_VERDICT)"
fi

log "==== v3 follow-up watcher COMPLETE ===="
log "v3 summary: $V3_SUMMARY"
log "watcher log: $WATCHER_LOG"
