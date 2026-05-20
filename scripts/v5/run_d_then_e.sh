#!/bin/bash
# Run Phase D (full training, ~6h) then Phase E (final eval + summary).
set -u
cd /Users/rohit.anand/Desktop/Hobby/MODA
mkdir -p logs results/v5

REPO=/Users/rohit.anand/Desktop/Hobby/MODA
DATA=$REPO/data/processed/v5_multifield
LOGS=$REPO/logs

stamp() { date +"%Y-%m-%d %H:%M:%S"; }
log_event() { echo "[$(stamp)] $*" | tee -a "$LOGS/run_d_then_e.log"; }

source .venv/bin/activate

n_labels=$(wc -l < "$DATA/pairs_50k_labeled.jsonl" 2>/dev/null || echo 0)
log_event "starting Phase D — $n_labels labels available, training on full 50K"

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

log_event "starting Phase E"
caffeinate -i python scripts/v5/phase_e_eval.py \
    > "$LOGS/v5_phase_e_384.log" 2>&1
log_event "Phase E complete"

log_event "ALL PHASES COMPLETE @ 384 — see results/v5/phase_e_summary.md"
