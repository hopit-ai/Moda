#!/usr/bin/env bash
# MODA Phase 1 — Full sequential pipeline
# Runs H&M embedding + eval for all 3 models, then generates leaderboard
# Safe to run while Tier 1 eval is running in background.

set -euo pipefail

REPO=/Users/rohit.anand/Desktop/Hobby/MODA
PYTHON="$REPO/.venv/bin/python"
LOGS="$REPO/logs"
EMBEDDINGS="$REPO/data/processed/embeddings"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# Wait for a FAISS index to exist (polls every 60s)
wait_for_index() {
    local model_safe="$1"
    local index="$EMBEDDINGS/${model_safe}_faiss.index"
    log "Waiting for FAISS index: $index"
    while [ ! -f "$index" ]; do
        sleep 30
    done
    log "Index ready: $index"
}

# ---- STEP 1: Embed H&M with FashionSigLIP ----
# May already be running — check first
SIGLIP_INDEX="$EMBEDDINGS/fashion-siglip_faiss.index"
if [ -f "$SIGLIP_INDEX" ]; then
    log "FashionSigLIP index already exists — skipping embed"
else
    log "Embedding H&M with FashionSigLIP (running in background) ..."
    # If already started, skip
    if ! pgrep -f "embed_hnm.*fashion-siglip" > /dev/null 2>&1; then
        nohup "$PYTHON" -u benchmark/embed_hnm.py \
            --model fashion-siglip --batch_size 64 --device mps \
            --articles_csv data/raw/hnm/articles.csv \
            --output_dir data/processed/embeddings \
            > "$LOGS/embed_hnm_fashion-siglip.log" 2>&1 &
    fi
    wait_for_index "fashion-siglip"
fi

# ---- STEP 2: Eval H&M with FashionSigLIP ----
log "Running H&M eval: FashionSigLIP ..."
"$PYTHON" -u benchmark/eval_hnm.py \
    --retrieval_method dense \
    --model fashion-siglip \
    --top_k 50 \
    --sample_queries 0 \
    --output_dir results \
    --data_dir data/raw/hnm \
    --embeddings_dir data/processed/embeddings \
    --device mps \
    2>&1 | tee "$LOGS/eval_hnm_fashion-siglip.log"

# ---- STEP 3: Embed H&M with FashionCLIP ----
log "Embedding H&M with FashionCLIP ..."
"$PYTHON" -u benchmark/embed_hnm.py \
    --model fashion-clip --batch_size 64 --device mps \
    --articles_csv data/raw/hnm/articles.csv \
    --output_dir data/processed/embeddings \
    2>&1 | tee "$LOGS/embed_hnm_fashion-clip.log"

# ---- STEP 4: Eval H&M with FashionCLIP ----
log "Running H&M eval: FashionCLIP ..."
"$PYTHON" -u benchmark/eval_hnm.py \
    --retrieval_method dense \
    --model fashion-clip \
    --top_k 50 \
    --sample_queries 0 \
    --output_dir results \
    --data_dir data/raw/hnm \
    --embeddings_dir data/processed/embeddings \
    --device mps \
    2>&1 | tee "$LOGS/eval_hnm_fashion-clip.log"

# ---- STEP 5: Embed H&M with CLIP ----
log "Embedding H&M with CLIP ..."
"$PYTHON" -u benchmark/embed_hnm.py \
    --model clip --batch_size 64 --device mps \
    --articles_csv data/raw/hnm/articles.csv \
    --output_dir data/processed/embeddings \
    2>&1 | tee "$LOGS/embed_hnm_clip.log"

# ---- STEP 6: Eval H&M with CLIP ----
log "Running H&M eval: CLIP ..."
"$PYTHON" -u benchmark/eval_hnm.py \
    --retrieval_method dense \
    --model clip \
    --top_k 50 \
    --sample_queries 0 \
    --output_dir results \
    --data_dir data/raw/hnm \
    --embeddings_dir data/processed/embeddings \
    --device mps \
    2>&1 | tee "$LOGS/eval_hnm_clip.log"

# ---- STEP 7: Build final leaderboard ----
log "Building Phase 1 leaderboard ..."
"$PYTHON" -u benchmark/run_baselines.py --leaderboard_only \
    2>&1 | tee "$LOGS/phase1_leaderboard.log"

log "=== Phase 1 Pipeline Complete ==="
cat results/phase1_leaderboard.md
