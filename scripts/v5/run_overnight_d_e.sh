#!/bin/bash
# Phase D + E orchestrator.
#
# Sequence:
#   1. Wait for label extraction to finish      (in progress, ~5-7h remaining)
#   2. Wait for eval cache build to finish      (in progress, ~5 min remaining)
#   3. Probe smoke checkpoint (informational)   (~30s, optional directional signal)
#   4. Phase D: full training                   (~6h on MPS at 1.86s/step × ~12K steps)
#   5. Phase E: final eval + summary report     (~30s)
#
# Caffeinate wraps the whole thing so the laptop doesn't sleep.

set -u
cd /Users/rohit.anand/Desktop/Hobby/MODA
mkdir -p logs results/v5

REPO=/Users/rohit.anand/Desktop/Hobby/MODA
DATA=$REPO/data/processed/v5_multifield
LOGS=$REPO/logs

stamp() { date +"%Y-%m-%d %H:%M:%S"; }
log_event() { echo "[$(stamp)] $*" | tee -a "$LOGS/overnight_d_e.log"; }

source .venv/bin/activate

log_event "phase D+E orchestrator started"

# ---------------------------------------------------------------------------
# Step 1 — wait for label extraction to finish
# ---------------------------------------------------------------------------
log_event "waiting for label extraction (logs/v5_label_50k.log) ..."
while ! grep -q "Done in" "$LOGS/v5_label_50k.log" 2>/dev/null; do sleep 120; done
n_labels=$(wc -l < "$DATA/pairs_50k_labeled.jsonl" 2>/dev/null || echo 0)
log_event "label extraction done ($n_labels labeled records)"

# ---------------------------------------------------------------------------
# Step 2 — wait for eval cache build to finish
# ---------------------------------------------------------------------------
log_event "waiting for eval caches ..."
while [ ! -f "$DATA/../v5_eval_cache/KAGL_image_emb.pt" ]; do sleep 60; done
sleep 30  # ensure baseline probe also finished
log_event "eval caches ready"

# ---------------------------------------------------------------------------
# Step 3 — probe the smoke checkpoint as a directional signal (optional)
# ---------------------------------------------------------------------------
if [ -f "$REPO/checkpoints/v5/smoke_final.pt" ]; then
    log_event "probing smoke checkpoint (directional signal)"
    caffeinate -i python scripts/v5/v5_eval_probe.py \
        --checkpoint "$REPO/checkpoints/v5/smoke_final.pt" \
        --out "$REPO/results/v5/smoke_probe.json" \
        > "$LOGS/v5_smoke_probe.log" 2>&1
    log_event "smoke probe done — see results/v5/smoke_probe.json"
fi

# ---------------------------------------------------------------------------
# Step 4 — Phase D full training
# ---------------------------------------------------------------------------
log_event "starting Phase D — full training on $n_labels labeled pairs"
caffeinate -i python scripts/v5/phase_d_train_full.py \
    --epochs 3 --probe_every 500 --ckpt_every 1000 \
    > "$LOGS/v5_phase_d.log" 2>&1
status=$?
if [ $status -ne 0 ]; then
    log_event "Phase D FAILED (exit $status). See $LOGS/v5_phase_d.log"
    exit 1
fi
log_event "Phase D complete"

# ---------------------------------------------------------------------------
# Step 5 — Phase E summary
# ---------------------------------------------------------------------------
log_event "starting Phase E — final eval + summary"
caffeinate -i python scripts/v5/phase_e_eval.py \
    > "$LOGS/v5_phase_e.log" 2>&1
status=$?
if [ $status -ne 0 ]; then
    log_event "Phase E FAILED (exit $status). See $LOGS/v5_phase_e.log"
    exit 1
fi
log_event "Phase E complete"

log_event "ALL PHASES COMPLETE — see results/v5/phase_e_summary.md"
