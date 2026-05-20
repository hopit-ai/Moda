#!/bin/bash
# v5 FSL adapter: start from Marqo-FashionSigLIP, distill toward SL2-B teacher.
#
# Strategy:
#   - Student = FSL (ViT-B-16-SigLIP, 203M) — starts with fashion200k=0.4551, KAGL=0.5805
#   - Teacher = SL2-B only (kl_fsl_weight=0) — pulls atlas/polyvore toward SL2 space
#   - Strong anchor (0.4→0.2) — prevents fashion200k/KAGL regression
#   - Very low LR (5e-7) — conservative drift
#   - text_1block scope (7M trainable) — minimal capacity change
#
# Expected: atlas/polyvore gain +3-5%, fashion200k/KAGL held near FSL level.
# FSL eval caches must exist (run phase_a_cache_fsl_eval.py first).

set -u
cd /Users/rohit.anand/Desktop/Hobby/MODA
mkdir -p logs results/v5/v5_fsl

REPO=/Users/rohit.anand/Desktop/Hobby/MODA
DATA=$REPO/data/processed/v5_multifield
LOGS=$REPO/logs

stamp() { date +"%Y-%m-%d %H:%M:%S"; }
log_event() { echo "[$(stamp)] $*" | tee -a "$LOGS/run_v5_fsl_adapter.log"; }

source .venv/bin/activate

log_event "v5 FSL adapter orchestrator started"

# ─── Step 1: Build FSL eval image caches (if not already done, ~10 min) ────
if [ -f "$REPO/data/processed/v5_eval_cache_fsl/fashion200k_image_emb.pt" ]; then
    log_event "FSL eval caches already exist — skipping"
else
    log_event "Building FSL eval image caches (~10 min) ..."
    caffeinate -i python scripts/v5/phase_a_cache_fsl_eval.py \
        > "$LOGS/v5_fsl_eval_cache.log" 2>&1
    status=$?
    if [ $status -ne 0 ]; then
        log_event "FSL eval cache FAILED (exit $status). See $LOGS/v5_fsl_eval_cache.log"
        exit 1
    fi
    log_event "FSL eval caches ready"
fi

# ─── Step 2: Phase D — FSL adapter training (~2h on MPS) ────────────────────
log_event "Starting Phase D FSL adapter (FSL student, SL2-B teacher, strong anchor)"
caffeinate -i python scripts/v5/phase_d_train_full.py \
    --fsl_student \
    --pairs "$DATA/pairs_v2_combined.jsonl" \
    --image_index_path "$DATA/v2_image_index.json" \
    --scope text_1block \
    --gcl_lambda 0.0 \
    --use_multifield 0 \
    --anchor_lambda_init 0.4 --anchor_lambda_final 0.2 \
    --kl_lambda_init 1.0 --kl_lambda_final 0.5 \
    --lr 5e-7 \
    --run_tag v5_fsl \
    --epochs 3 --probe_every 500 --ckpt_every 1000 \
    > "$LOGS/v5_phase_d_v5_fsl.log" 2>&1
status=$?
if [ $status -ne 0 ]; then
    log_event "Phase D FSL FAILED (exit $status). See $LOGS/v5_phase_d_v5_fsl.log"
    exit 1
fi
log_event "Phase D FSL complete"

# ─── Step 3: Phase E — eval vs FSL baseline ──────────────────────────────────
log_event "Starting Phase E FSL adapter eval"
caffeinate -i python scripts/v5/phase_e_eval.py \
    --fsl_student \
    --ckpt "$REPO/checkpoints/v5/phase_d_best_v5_fsl.pt" \
    --out_dir "$REPO/results/v5/v5_fsl" \
    > "$LOGS/v5_phase_e_v5_fsl.log" 2>&1
log_event "Phase E FSL complete — see results/v5/v5_fsl/phase_e_summary.md"

log_event "ALL FSL ADAPTER PHASES COMPLETE"
