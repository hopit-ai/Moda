#!/bin/bash
# Watcher: as soon as overnight_384.sh finishes the smoke step, kill its
# wait-for-labels loop and start Phase D directly on the 50K (unlabeled)
# corpus. After Phase D, run Phase E.

set -u
cd /Users/rohit.anand/Desktop/Hobby/MODA
mkdir -p logs

REPO=/Users/rohit.anand/Desktop/Hobby/MODA
DATA=$REPO/data/processed/v5_multifield
LOGS=$REPO/logs
ORCH_LOG=$LOGS/overnight_384.log
EARLY_LOG=$LOGS/overnight_384_early.log

stamp() { date +"%Y-%m-%d %H:%M:%S"; }
log_event() { echo "[$(stamp)] $*" | tee -a "$EARLY_LOG"; }

source .venv/bin/activate

log_event "early-start watcher armed"

# Wait for smoke completion marker in the orchestrator's log
log_event "waiting for 'smoke @ 384 done' in $ORCH_LOG"
while ! grep -q "smoke @ 384 done" "$ORCH_LOG" 2>/dev/null; do
    sleep 30
done
log_event "smoke @ 384 done — taking over"

# Kill the orchestrator (it would otherwise sit in 'wait for labels' for hours)
ORCH_PID=$(pgrep -f run_overnight_384_rebuild.sh | head -1)
if [ -n "$ORCH_PID" ]; then
    log_event "killing orchestrator PID $ORCH_PID"
    kill $ORCH_PID 2>/dev/null
    sleep 2
fi

# ---------------------------------------------------------------------------
# Phase D — full training on whatever labels exist + unlabeled fallback
# ---------------------------------------------------------------------------
n_labels=$(wc -l < "$DATA/pairs_50k_labeled.jsonl" 2>/dev/null || echo 0)
log_event "starting Phase D early — $n_labels labels available, using full 50K (regex fallback)"
caffeinate -i python scripts/v5/phase_d_train_full.py \
    --pairs "$DATA/pairs_50k.jsonl" \
    --epochs 3 --probe_every 500 --ckpt_every 1000 \
    > "$LOGS/v5_phase_d_384.log" 2>&1
status=$?
if [ $status -ne 0 ]; then
    log_event "Phase D FAILED (exit $status). See $LOGS/v5_phase_d_384.log"
    exit 1
fi
log_event "Phase D complete"

# ---------------------------------------------------------------------------
# Phase E — final eval + summary
# ---------------------------------------------------------------------------
log_event "starting Phase E"
caffeinate -i python scripts/v5/phase_e_eval.py \
    > "$LOGS/v5_phase_e_384.log" 2>&1
log_event "Phase E complete"

log_event "ALL PHASES COMPLETE @ 384 — see results/v5/phase_e_summary.md"
