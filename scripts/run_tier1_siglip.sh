#!/usr/bin/env bash
# MODA Phase 1 — Tier 1: FashionSigLIP on all 5 datasets (InShop already done)
# Runs sequentially. Logs per dataset. Generates averaged results at the end.
set -euo pipefail

REPO_ROOT="/Users/rohit.anand/Desktop/Hobby/MODA"
MARQO_REPO="$REPO_ROOT/repos/marqo-FashionCLIP"
PYTHON="$REPO_ROOT/.venv/bin/python"
LOGS="$REPO_ROOT/logs/tier1"
mkdir -p "$LOGS"

CACHE_DIR="$REPO_ROOT/data/hf_cache"
MODEL_NAME="Marqo/marqo-fashionSigLIP"
RUN_NAME="Marqo-FashionSigLIP"
DEVICE="mps"       # MPS for fast image embedding
BATCH_SIZE=128
KS="1 10"

echo "========================================================"
echo "MODA Tier 1 — FashionSigLIP — $(date)"
echo "Datasets: atlas, deepfashion_multimodal, fashion200k, polyvore"
echo "========================================================"

cd "$MARQO_REPO"

for DATASET_CONFIG in "atlas:atlas.json" "deepfashion_multimodal:deepfashion_multimodal.json" "fashion200k:fashion200k.json" "polyvore:polyvore.json"; do
    DATASET="${DATASET_CONFIG%%:*}"
    CONFIG="${DATASET_CONFIG##*:}"
    LOG="$LOGS/${DATASET}_siglip.log"
    echo ""
    echo ">>> [$DATASET] Starting at $(date) — log: $LOG"

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

    echo ">>> [$DATASET] Done at $(date)"
done

echo ""
echo "========================================================"
echo "All 4 datasets complete. Generating average results..."
echo "========================================================"

"$PYTHON" "$REPO_ROOT/scripts/compute_tier1_average.py"

echo "Done! Full Tier 1 leaderboard updated."
