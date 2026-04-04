#!/usr/bin/env bash
# MODA — Run KAGL for both models + generate final Phase 1 comparison report
# Waits for Option 2 (FashionCLIP run) to finish first, then runs KAGL.
set -euo pipefail

REPO_ROOT="/Users/rohit.anand/Desktop/Hobby/MODA"
MARQO_REPO="$REPO_ROOT/repos/marqo-FashionCLIP"
PYTHON="$REPO_ROOT/.venv/bin/python"
LOGS="$REPO_ROOT/logs/tier1"
CACHE_DIR="$REPO_ROOT/data/hf_cache"
DEVICE="mps"
BATCH_SIZE=128

echo "========================================================"
echo "MODA — Waiting for Option 2 (FashionCLIP) to finish..."
echo "========================================================"

# Poll until fashionclip_run.log shows completion
while ! tail -5 "$LOGS/fashionclip_run.log" 2>/dev/null | grep -q "leaderboard updated\|All done"; do
    echo "  [$(date +%H:%M:%S)] FashionCLIP still running... checking in 60s"
    sleep 60
done

echo "FashionCLIP done. Starting KAGL runs at $(date)"
echo "========================================================"

cd "$MARQO_REPO"

for MODEL_NAME_AND_RUN in "Marqo/marqo-fashionSigLIP:Marqo-FashionSigLIP" "Marqo/marqo-fashionCLIP:Marqo-FashionCLIP"; do
    MODEL_NAME="${MODEL_NAME_AND_RUN%%:*}"
    RUN_NAME="${MODEL_NAME_AND_RUN##*:}"
    LOG="$LOGS/kagl_${RUN_NAME}.log"

    echo ""
    echo ">>> [KAGL × $RUN_NAME] Starting at $(date)"

    "$PYTHON" eval.py \
        --dataset-config "./configs/KAGL.json" \
        --model-name "$MODEL_NAME" \
        --run-name "$RUN_NAME" \
        --cache-dir "$CACHE_DIR" \
        --device "$DEVICE" \
        --batch-size "$BATCH_SIZE" \
        --num-workers 0 \
        --Ks 1 10 \
        --overwrite-embeddings \
        --overwrite-retrieval \
        2>&1 | tee "$LOG"

    echo ">>> [KAGL × $RUN_NAME] Done at $(date)"
done

echo ""
echo "========================================================"
echo "All KAGL runs done. Computing final averaged results..."
echo "========================================================"

"$PYTHON" "$REPO_ROOT/scripts/compute_tier1_average.py"

echo ""
echo "========================================================"
echo "Generating final benchmark comparison report..."
echo "========================================================"

"$PYTHON" "$REPO_ROOT/scripts/generate_final_comparison.py"
"$PYTHON" "$REPO_ROOT/scripts/generate_leaderboard.py"

echo ""
echo "ALL DONE at $(date)"
echo "Check results/PHASE1_BENCHMARK_COMPARISON.md for final report"
