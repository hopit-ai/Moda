#!/usr/bin/env bash
# MODA Phase 1 — Tier 1: FashionCLIP on all 5 datasets (overnight job)
# Runs after run_tier1_siglip.sh finishes.
set -euo pipefail

REPO_ROOT="/Users/rohit.anand/Desktop/Hobby/MODA"
MARQO_REPO="$REPO_ROOT/repos/marqo-FashionCLIP"
PYTHON="$REPO_ROOT/.venv/bin/python"
LOGS="$REPO_ROOT/logs/tier1"
mkdir -p "$LOGS"

CACHE_DIR="$REPO_ROOT/data/hf_cache"
DEVICE="mps"
BATCH_SIZE=128
KS="1 10"

echo "========================================================"
echo "MODA Tier 1 — FashionCLIP — $(date)"
echo "========================================================"

cd "$MARQO_REPO"

MODEL_NAME="Marqo/marqo-fashionCLIP"
RUN_NAME="Marqo-FashionCLIP"
echo ""
echo "=== Model: $RUN_NAME ==="

for DATASET_CONFIG in "atlas:atlas.json" "deepfashion_multimodal:deepfashion_multimodal.json" "fashion200k:fashion200k.json" "polyvore:polyvore.json" "deepfashion_inshop:deepfashion_inshop.json"; do
    DATASET="${DATASET_CONFIG%%:*}"
    CONFIG="${DATASET_CONFIG##*:}"
        LOG="$LOGS/${DATASET}_fashion-clip.log"
        echo ""
        echo ">>> [$DATASET × $RUN_NAME] Starting at $(date)"

        "$PYTHON" eval.py \
            --dataset-config "./configs/$CONFIG" \
            --model-name "$MODEL_NAME" \
            --run-name "$RUN_NAME" \
            --cache-dir "$CACHE_DIR" \
            --device "$DEVICE" \
            --batch-size "$BATCH_SIZE" \
            --num-workers 0 \
            --Ks $KS \
            --overwrite-embeddings \
            --overwrite-retrieval \
            2>&1 | tee "$LOG"

        echo ">>> [$DATASET × $RUN_NAME] Done at $(date)"
    done

echo ""
echo "========================================================"
echo "FashionCLIP complete. Regenerating full leaderboard..."
echo "========================================================"

"$PYTHON" "$REPO_ROOT/scripts/compute_tier1_average.py"
"$PYTHON" "$REPO_ROOT/scripts/generate_leaderboard.py"

echo "All done! Check results/PHASE1_LEADERBOARD.md"
