#!/bin/bash
# Path 3 overnight experiment: cross-domain distillation
# 
# Steps:
#   1. Build cross-domain data (DFM + Marqo-GS) with teacher caches + hard negatives
#   2. Run Path 2 trainer on the new mixed data with gradient checkpointing
#   3. Evaluate the best checkpoint on all 4 datasets
#
# Expected wall time: ~4-6 hours total
#   - Data build: ~30-40 min (encoding images through 3 models)
#   - Training: ~2-3 hours (500 steps with grad checkpointing)
#   - Evaluation: ~45 min (4 datasets × full corpus)

set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
PYTHON="$REPO/.venv/bin/python"
LOG_DIR="$REPO/logs"
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
DATA_DIR="$REPO/data/processed/path3"
MODEL_DIR="$REPO/models/path3-crossdomain"

echo "============================================================"
echo "Path 3 overnight experiment — cross-domain distillation"
echo "Started at: $(date)"
echo "============================================================"

# Step 1: Build cross-domain data
echo ""
echo "[Step 1/3] Building cross-domain training data..."
echo "  This encodes ~800+ queries through 3 models. ~30-40 min."
echo ""

$PYTHON -u scripts/build_crossdomain_data.py \
    --n-dfm 400 \
    --K 15 \
    --batch-size 64 \
    --out-dir "$DATA_DIR" \
    2>&1 | tee "$LOG_DIR/path3_data_build_${TIMESTAMP}.log"

echo ""
echo "[Step 1/3] Data build complete."
echo ""

# Step 2: Run distillation
echo "[Step 2/3] Running Path 2 distillation on mixed data..."
echo "  Config: 500 steps, B=8, K=15, grad-checkpoint, unfreeze at step 100"
echo "  Probes at: 100, 200, 300, 400, 500"
echo ""

$PYTHON -u benchmark/distill_path2.py \
    --hardnegs "$DATA_DIR/hardnegs.jsonl" \
    --teacher-cache-dir "$DATA_DIR/teacher_cache" \
    --anchor-cache "$DATA_DIR/init_anchor_cache.pt" \
    --output-dir "$MODEL_DIR" \
    --K 15 \
    --batch-size 8 \
    --grad-accum 1 \
    --lr 1e-5 \
    --warmup-steps 30 \
    --max-steps 500 \
    --unfreeze-image-step 100 \
    --grad-checkpoint \
    --w-mm 1.0 --w-kl 0.5 --w-infonce 0.1 --w-anchor 0.05 \
    --tau-teacher 0.3 --tau-student 0.05 --tau-infonce 0.05 \
    --probe-steps "100,200,300,400,500" \
    --probe-datasets "fashion200k,atlas,polyvore" \
    --probe-corpus-size 5000 \
    --probe-batch-size 64 \
    --abort-step1 200 --abort-step1-min-delta -0.03 \
    --abort-step2 400 --abort-step2-min-delta -0.01 \
    --abort-step2-min-r100-delta -0.03 \
    --abort-step2-min-mean-delta -0.02 \
    --seed 42 \
    --log-every 10 \
    2>&1 | tee "$LOG_DIR/path3_train_${TIMESTAMP}.log"

TRAIN_EXIT=$?
echo ""
echo "[Step 2/3] Training complete (exit=$TRAIN_EXIT)."
echo ""

# Step 3: Evaluate (only if training produced a best checkpoint)
if [ -f "$MODEL_DIR/best/student_state_dict.pt" ]; then
    echo "[Step 3/3] Evaluating best checkpoint on fashion200k..."
    echo ""

    $PYTHON -u benchmark/eval_marqo_7dataset.py \
        --models moda-siglip2-path3 \
        --datasets fashion200k atlas polyvore KAGL \
        --batch_size 64 \
        --device mps \
        --overwrite \
        2>&1 | tee "$LOG_DIR/path3_eval_${TIMESTAMP}.log"

    echo ""
    echo "[Step 3/3] Evaluation complete."
else
    echo "[Step 3/3] SKIPPED — no best checkpoint found (training may have aborted early)."
fi

echo ""
echo "============================================================"
echo "Path 3 overnight experiment FINISHED"
echo "Ended at: $(date)"
echo "Check: $MODEL_DIR/summary.json"
echo "============================================================"
