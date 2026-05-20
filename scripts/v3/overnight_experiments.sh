#!/bin/bash
# Overnight experiment runner — Phase 14 + Phase 15
# Runs sequentially since all compete for MPS
set -e

cd /Users/rohit.anand/Desktop/Hobby/MODA

echo "================================================================"
echo "OVERNIGHT EXPERIMENTS — Started at $(date)"
echo "================================================================"
echo ""

# Phase 14: Teacher-Mined Hard Negative GCL
echo ">>> Phase 14: Teacher-Mined GCL (InfoNCE + ListNet, progressive unfreezing)"
echo ">>> Started at $(date)"
python3 -u scripts/v3/phase14_teacher_mined_gcl.py train \
    --lr 5e-6 \
    --temperature 0.07 \
    --listnet-weight 0.5 \
    --batch-size 64 \
    --max-epochs 5 \
    --eval-every 50 \
    --max-drift 0.05 \
    --patience 8 \
    --n-hard-negs 7 \
    --unfreeze-schedule progressive \
    2>&1 | tee cache/phase14_train_log.txt

echo ""
echo ">>> Phase 14 done at $(date)"
echo ""

# Phase 15: Same-Architecture Feature Distillation from B16-256
echo ">>> Phase 15: Cache B16-256 teacher embeddings"
echo ">>> Started at $(date)"
python3 -u scripts/v3/phase15_same_arch_distill.py cache-teacher \
    2>&1 | tee cache/phase15_cache_log.txt

echo ""
echo ">>> Phase 15: Train with MSE + ranking + anchor"
echo ">>> Started at $(date)"
python3 -u scripts/v3/phase15_same_arch_distill.py train \
    --lr 3e-6 \
    --mse-weight 1.0 \
    --rank-weight 0.3 \
    --anchor-weight 0.2 \
    --batch-size 64 \
    --max-epochs 5 \
    --eval-every 50 \
    --max-drift 0.04 \
    --patience 8 \
    2>&1 | tee cache/phase15_train_log.txt

echo ""
echo "================================================================"
echo "ALL OVERNIGHT EXPERIMENTS COMPLETE — $(date)"
echo "================================================================"
echo ""
echo "Results:"
echo "  Phase 14: results/phase14_teacher_mined_gcl.json"
echo "  Phase 15: results/phase15_same_arch_distill.json"
echo ""
