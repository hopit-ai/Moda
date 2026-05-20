#!/bin/bash
# v2 chain: wait for prose-gen → prep dataset → Phase D v2 → Phase E v2.
#
# Recipe changes vs v1 (per iteration-2 plan):
#   --pairs               pairs_v2_combined.jsonl (50K orig + 10K synth prose)
#   --image_index_path    v2_image_index.json (synth IDs mapped to same rows)
#   --lr                  1e-6 (was 5e-6)
#   --use_multifield      0 (query-only, was 1 = 0.6q+0.3t+0.1c)
#   --anchor_lambda_init  1.0 (was 0.5) — preserve baseline harder
#   --anchor_lambda_final 0.3 (was 0.1)
#   --kl_lambda_init      0.7 (was 0.3) — lean more on fusion teacher
#   --kl_lambda_final     0.3 (was 0.1)
#   --run_tag             v2

set -u
cd /Users/rohit.anand/Desktop/Hobby/MODA
mkdir -p logs results/v5/v2

REPO=/Users/rohit.anand/Desktop/Hobby/MODA
DATA=$REPO/data/processed/v5_multifield
LOGS=$REPO/logs

stamp() { date +"%Y-%m-%d %H:%M:%S"; }
log_event() { echo "[$(stamp)] $*" | tee -a "$LOGS/run_v2_chain.log"; }

source .venv/bin/activate

log_event "v2 orchestrator started"

# 1. Wait for prose-gen
log_event "waiting for prose-gen completion (logs/v2_long_desc.log)"
while ! grep -q "Done in" "$LOGS/v2_long_desc.log" 2>/dev/null; do sleep 60; done
n_synth=$(wc -l < "$DATA/pairs_v2_long_desc.jsonl" 2>/dev/null || echo 0)
log_event "prose-gen done — $n_synth synthetic prose pairs"

# 2. Prep combined dataset + augmented image index
log_event "preparing v2 combined dataset"
python scripts/v5/v2_prepare_training.py > "$LOGS/v2_prep.log" 2>&1
status=$?
if [ $status -ne 0 ]; then
    log_event "v2 prep FAILED (exit $status)"
    cat "$LOGS/v2_prep.log"
    exit 1
fi
log_event "v2 prep done"

# 3. Phase D v2
log_event "starting Phase D v2"
caffeinate -i python scripts/v5/phase_d_train_full.py \
    --pairs "$DATA/pairs_v2_combined.jsonl" \
    --image_index_path "$DATA/v2_image_index.json" \
    --lr 1e-6 \
    --use_multifield 0 \
    --anchor_lambda_init 1.0 --anchor_lambda_final 0.3 \
    --kl_lambda_init 0.7 --kl_lambda_final 0.3 \
    --run_tag v2 \
    --epochs 3 --probe_every 500 --ckpt_every 1000 \
    > "$LOGS/v5_phase_d_v2.log" 2>&1
status=$?
if [ $status -ne 0 ]; then
    log_event "Phase D v2 FAILED (exit $status). See $LOGS/v5_phase_d_v2.log"
    exit 1
fi
log_event "Phase D v2 complete"

# 4. Phase E v2
log_event "starting Phase E v2"
caffeinate -i python scripts/v5/phase_e_eval.py \
    --ckpt "$REPO/checkpoints/v5/phase_d_best_v2.pt" \
    --out_dir "$REPO/results/v5/v2" \
    > "$LOGS/v5_phase_e_v2.log" 2>&1
log_event "Phase E v2 complete"

log_event "ALL V2 PHASES COMPLETE — see results/v5/v2/phase_e_summary.md"
