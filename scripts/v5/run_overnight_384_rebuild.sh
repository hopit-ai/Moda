#!/bin/bash
# 384-resolution rebuild + relaunch.
#
# Sequence (all MPS-bound, sequential):
#   1. Wait for student image cache (already running, ~45 min)
#   2. SL2 text teacher cache at 384       (~5 min)
#   3. Eval benchmark cache rebuild at 384 (~10 min)
#   4. Baseline SL2-384 probe              (~30 sec)
#   5. 50-step smoke at 384 (sanity)        (~2 min)
#   6. Wait for label extraction to finish (in progress, ETA ~01:00)
#   7. Phase D full training at 384         (~6h)
#   8. Phase E summary

set -u
cd /Users/rohit.anand/Desktop/Hobby/MODA
mkdir -p logs results/v5

REPO=/Users/rohit.anand/Desktop/Hobby/MODA
DATA=$REPO/data/processed/v5_multifield
LOGS=$REPO/logs

stamp() { date +"%Y-%m-%d %H:%M:%S"; }
log_event() { echo "[$(stamp)] $*" | tee -a "$LOGS/overnight_384.log"; }

source .venv/bin/activate

log_event "384 rebuild orchestrator started"

# ---------------------------------------------------------------------------
# 1. Wait for student image cache at 384
# ---------------------------------------------------------------------------
log_event "waiting for student image cache at 384 (logs/v5_student_image_cache_384.log)"
while [ ! -f "$DATA/student_image_emb.pt" ]; do sleep 60; done
sleep 30  # let writer fully flush
log_event "student image cache @ 384 present"

# ---------------------------------------------------------------------------
# 2. SL2 text teacher cache at 384
# ---------------------------------------------------------------------------
log_event "starting SL2 text teacher cache at 384"
caffeinate -i python scripts/v5/phase_a_cache_sl2_text_only.py \
    > "$LOGS/v5_sl2_text_384.log" 2>&1
status=$?
if [ $status -ne 0 ]; then
    log_event "SL2 text cache FAILED (exit $status)"
    exit 1
fi
log_event "SL2 text teacher cache done"

# ---------------------------------------------------------------------------
# 3. Eval benchmark cache rebuild at 384
# ---------------------------------------------------------------------------
log_event "rebuilding eval benchmark caches at 384"
caffeinate -i python scripts/v5/v5_eval_probe.py --build_caches \
    > "$LOGS/v5_eval_build_caches_384.log" 2>&1
status=$?
if [ $status -ne 0 ]; then
    log_event "eval cache build FAILED (exit $status)"
    exit 1
fi
log_event "eval caches @ 384 ready"

# ---------------------------------------------------------------------------
# 4. Baseline SL2-384 probe
# ---------------------------------------------------------------------------
log_event "running baseline SL2-384 probe"
caffeinate -i python scripts/v5/v5_eval_probe.py --baseline \
    --out "$REPO/results/v5/baseline_sl2_384.json" \
    > "$LOGS/v5_baseline_384.log" 2>&1
log_event "baseline @ 384 done — see results/v5/baseline_sl2_384.json"

# ---------------------------------------------------------------------------
# 5. 50-step smoke at 384 (validate loop runs at new resolution)
# ---------------------------------------------------------------------------
log_event "running 50-step smoke at 384 (sanity)"
caffeinate -i python scripts/v5/phase_c_smoke.py \
    --steps 50 --batch_K 4 --batch_N 8 --checkpoint_every 0 \
    > "$LOGS/v5_smoke_384.log" 2>&1
status=$?
if [ $status -ne 0 ]; then
    log_event "384 smoke FAILED (exit $status) — diagnose before Phase D"
    exit 1
fi
log_event "smoke @ 384 done"

# ---------------------------------------------------------------------------
# 6. Wait for label extraction to finish
# ---------------------------------------------------------------------------
log_event "waiting for label extraction to finish ..."
while ! grep -q "Done in" "$LOGS/v5_label_50k.log" 2>/dev/null; do sleep 120; done
n_labels=$(wc -l < "$DATA/pairs_50k_labeled.jsonl" 2>/dev/null || echo 0)
log_event "label extraction done ($n_labels labeled records)"

# ---------------------------------------------------------------------------
# 7. Phase D full training at 384
# ---------------------------------------------------------------------------
log_event "starting Phase D — full training on $n_labels labeled pairs at 384"
caffeinate -i python scripts/v5/phase_d_train_full.py \
    --epochs 3 --probe_every 500 --ckpt_every 1000 \
    > "$LOGS/v5_phase_d_384.log" 2>&1
status=$?
if [ $status -ne 0 ]; then
    log_event "Phase D FAILED (exit $status). See $LOGS/v5_phase_d_384.log"
    exit 1
fi
log_event "Phase D complete"

# ---------------------------------------------------------------------------
# 8. Phase E summary
# ---------------------------------------------------------------------------
log_event "starting Phase E — final eval + summary"
caffeinate -i python scripts/v5/phase_e_eval.py \
    > "$LOGS/v5_phase_e_384.log" 2>&1
log_event "Phase E complete"

log_event "ALL PHASES COMPLETE @ 384 — see results/v5/phase_e_summary.md"
