#!/usr/bin/env bash
#
# Auto-run full-corpus fusion + Matryoshka once SigLIP-2 B/16/384 image
# embeddings are cached for all 4 Marqo datasets. Designed to be safe to run
# in parallel with the eval — it just polls and waits.
#
# Outputs (after completion):
#   results/fusion/fusion_summary_<dataset>_fullcorpus.md  (per dataset)
#   results/matryoshka_pca_joint_full/pareto_summary.md     (Matryoshka)

set -euo pipefail

REPO="/Users/rohit.anand/Desktop/Hobby/MODA"
cd "$REPO"

LOG="logs/full_corpus_fusion_matryoshka.log"
mkdir -p logs results/fusion

DATASETS=(fashion200k atlas polyvore KAGL)

echo "[$(date)] watcher: waiting for Google-SigLIP2-B16-384 full-corpus embeddings..." | tee -a "$LOG"

# Poll every 5 min until all 4 datasets have embeddings.pt
while true; do
    missing=0
    for ds in "${DATASETS[@]}"; do
        emb="repos/marqo-FashionCLIP/results/$ds/Google-SigLIP2-B16-384/embeddings.pt"
        if [ ! -f "$emb" ]; then
            missing=$((missing + 1))
        fi
    done
    if [ "$missing" -eq 0 ]; then
        echo "[$(date)] watcher: ALL 4 datasets have embeddings — proceeding" | tee -a "$LOG"
        break
    fi
    echo "[$(date)] watcher: $missing/4 still missing, sleeping 5min..." >> "$LOG"
    sleep 300
done

# ---------------------------------------------------------------
# 1. Full-corpus fusion eval per dataset
# ---------------------------------------------------------------
for ds in "${DATASETS[@]}"; do
    echo "[$(date)] full-corpus fusion on $ds" | tee -a "$LOG"
    .venv/bin/python benchmark/fuse_and_eval.py \
        --models fashion-siglip google-siglip2-b16-384 \
        --dataset "$ds" \
        --full-corpus \
        --summary-out "results/fusion/fusion_summary_${ds}_fullcorpus.md" \
        >> "$LOG" 2>&1
done

# ---------------------------------------------------------------
# 2. Full-corpus Matryoshka via PCA (joint basis)
# ---------------------------------------------------------------
echo "[$(date)] full-corpus Matryoshka PCA (joint basis)" | tee -a "$LOG"
.venv/bin/python benchmark/matryoshka_pca.py \
    --models fashion-siglip google-siglip2-b16-384 \
    --datasets "${DATASETS[@]}" \
    --fit-dataset joint \
    --dims 64 128 256 384 512 768 \
    --full-corpus \
    --out-dir results/matryoshka_pca_joint_full \
    >> "$LOG" 2>&1

echo "[$(date)] watcher DONE — results in results/fusion/ and results/matryoshka_pca_joint_full/" | tee -a "$LOG"
