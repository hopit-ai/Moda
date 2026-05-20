#!/bin/bash
# Iteration 3: heads-light scope (last 1 text block ~7M params) + pure
# distillation (gcl_lambda=0), no anchor reg.
#
# Reuses v2's combined dataset and augmented index (60K pairs).

set -u
cd /Users/rohit.anand/Desktop/Hobby/MODA
mkdir -p logs results/v5/v3

REPO=/Users/rohit.anand/Desktop/Hobby/MODA
DATA=$REPO/data/processed/v5_multifield
LOGS=$REPO/logs

stamp() { date +"%Y-%m-%d %H:%M:%S"; }
log_event() { echo "[$(stamp)] $*" | tee -a "$LOGS/run_v3_chain.log"; }

source .venv/bin/activate

log_event "v3 orchestrator started"

# Sanity-check artifacts from v2 are present
for f in pairs_v2_combined.jsonl v2_image_index.json student_image_emb.pt \
         teacher_fsl_img_emb.pt teacher_fsl_text_emb.pt teacher_sl2_text_emb.pt; do
    if [ ! -f "$DATA/$f" ]; then
        log_event "MISSING REQUIRED FILE: $DATA/$f"
        exit 1
    fi
done
log_event "all v2 artifacts present"

# Phase D v3 — pure distillation, last-1-block scope
log_event "starting Phase D v3 (scope=text_1block, gcl_lambda=0, kl 1.0→0.5)"
caffeinate -i python scripts/v5/phase_d_train_full.py \
    --pairs "$DATA/pairs_v2_combined.jsonl" \
    --image_index_path "$DATA/v2_image_index.json" \
    --scope text_1block \
    --gcl_lambda 0.0 \
    --use_multifield 0 \
    --anchor_lambda_init 0.0 --anchor_lambda_final 0.0 \
    --kl_lambda_init 1.0 --kl_lambda_final 0.5 \
    --lr 1e-6 \
    --run_tag v3 \
    --epochs 3 --probe_every 500 --ckpt_every 1000 \
    > "$LOGS/v5_phase_d_v3.log" 2>&1
status=$?
if [ $status -ne 0 ]; then
    log_event "Phase D v3 FAILED (exit $status). See $LOGS/v5_phase_d_v3.log"
    exit 1
fi
log_event "Phase D v3 complete"

# Phase E v3
log_event "starting Phase E v3"
caffeinate -i python scripts/v5/phase_e_eval.py \
    --ckpt "$REPO/checkpoints/v5/phase_d_best_v3.pt" \
    --out_dir "$REPO/results/v5/v3" \
    > "$LOGS/v5_phase_e_v3.log" 2>&1
log_event "Phase E v3 complete"

log_event "ALL V3 PHASES COMPLETE — see results/v5/v3/phase_e_summary.md"
