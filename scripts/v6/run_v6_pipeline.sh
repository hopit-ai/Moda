#!/usr/bin/env bash
# v6 Pipeline Orchestrator
#
# Chains the 4 v6 steps in order. Each step checks for prior output before
# running so the script is resumable — re-running skips completed steps.
#
# Usage:
#   bash scripts/v6/run_v6_pipeline.sh                  # full pipeline
#   bash scripts/v6/run_v6_pipeline.sh --start_step 3   # resume from step 3
#   SKIP_DATA=1 bash scripts/v6/run_v6_pipeline.sh      # skip data steps (1+2), training only
#
# Outputs:
#   data/processed/v6/pairs_gs10m_long_query.jsonl
#   data/processed/v6/pairs_fashioniq.jsonl
#   checkpoints/v6/prose_teacher_best_prose_teacher.pt
#   checkpoints/v6/v6_student_best_v6a.pt
#   logs/v6_prose_teacher_prose_teacher.jsonl
#   logs/v6_student_v6a.jsonl
#
# MPS note: all steps run sequentially. Never background on MPS — concurrent
# model loads OOM. Call `caffeinate -i bash run_v6_pipeline.sh` to prevent sleep.

set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
V6_DATA="$REPO/data/processed/v6"
CKPT_DIR="$REPO/checkpoints/v6"
LOGS_DIR="$REPO/logs"
PYTHON="${PYTHON:-$REPO/.venv/bin/python}"

START_STEP="${1:-1}"
# Parse --start_step flag
for arg in "$@"; do
    case $arg in
        --start_step=*) START_STEP="${arg#*=}" ;;
        --start_step)   shift; START_STEP="$1" ;;
    esac
done

SKIP_DATA="${SKIP_DATA:-0}"

echo "============================================================"
echo " MODA v6 Pipeline"
echo " REPO: $REPO"
echo " START_STEP: $START_STEP | SKIP_DATA: $SKIP_DATA"
echo " $(date)"
echo "============================================================"

mkdir -p "$V6_DATA" "$CKPT_DIR" "$LOGS_DIR"

# ──────────────────────────────────────────────────────────────────────────────
# STEP 1 — Mine GS-10M long queries
# ──────────────────────────────────────────────────────────────────────────────
if [[ "$START_STEP" -le 1 && "$SKIP_DATA" == "0" ]]; then
    echo ""
    echo "[ STEP 1 ] Mining GS-10M for long queries (≥10 words) ..."

    GS10M_OUT="$V6_DATA/pairs_gs10m_long_query.jsonl"
    if [[ -f "$GS10M_OUT" ]]; then
        N_EXISTING=$(wc -l < "$GS10M_OUT" | tr -d ' ')
        echo "  Found $N_EXISTING existing pairs in $GS10M_OUT"
        if [[ "$N_EXISTING" -ge 80000 ]]; then
            echo "  Target already met — skipping step 1."
        else
            echo "  Resuming (will append until 80K target) ..."
            caffeinate -i "$PYTHON" scripts/v6/step1_mine_gs10m_long_queries.py \
                --target 80000 --min_words 10 --min_score 3.0
        fi
    else
        caffeinate -i "$PYTHON" scripts/v6/step1_mine_gs10m_long_queries.py \
            --target 80000 --min_words 10 --min_score 3.0
    fi

    echo "  Step 1 done: $(wc -l < "$GS10M_OUT" | tr -d ' ') pairs"
else
    echo "[ STEP 1 ] Skipped (START_STEP=$START_STEP or SKIP_DATA=$SKIP_DATA)"
fi

# ──────────────────────────────────────────────────────────────────────────────
# STEP 2 — Download FashionIQ
# ──────────────────────────────────────────────────────────────────────────────
if [[ "$START_STEP" -le 2 && "$SKIP_DATA" == "0" ]]; then
    echo ""
    echo "[ STEP 2 ] Downloading FashionIQ from HuggingFace ..."

    FIQ_OUT="$V6_DATA/pairs_fashioniq.jsonl"
    if [[ -f "$FIQ_OUT" ]]; then
        N_FIQ=$(wc -l < "$FIQ_OUT" | tr -d ' ')
        echo "  Found $N_FIQ existing pairs in $FIQ_OUT"
        if [[ "$N_FIQ" -ge 50000 ]]; then
            echo "  Sufficient pairs already — skipping step 2."
        else
            echo "  Re-running (only $N_FIQ pairs, need ≥50K) ..."
            caffeinate -i "$PYTHON" scripts/v6/step2_prep_fashioniq.py
        fi
    else
        caffeinate -i "$PYTHON" scripts/v6/step2_prep_fashioniq.py
    fi

    if [[ -f "$FIQ_OUT" ]]; then
        echo "  Step 2 done: $(wc -l < "$FIQ_OUT" | tr -d ' ') FashionIQ pairs"
    else
        echo "  WARNING: FashionIQ output not found — step 3 may have reduced data"
    fi
else
    echo "[ STEP 2 ] Skipped (START_STEP=$START_STEP or SKIP_DATA=$SKIP_DATA)"
fi

# ──────────────────────────────────────────────────────────────────────────────
# STEP 3 — Train prose teacher (SL2-B, image tower unfrozen)
# ──────────────────────────────────────────────────────────────────────────────
if [[ "$START_STEP" -le 3 ]]; then
    echo ""
    echo "[ STEP 3 ] Training prose teacher ..."
    echo "  (SL2-B, last 2 text blocks + last 1 image block unfrozen)"
    echo "  (InfoNCE on FashionIQ + GS-10M long queries)"
    echo "  Expected: ~10h on MPS"

    TEACHER_CKPT="$CKPT_DIR/prose_teacher_best_prose_teacher.pt"
    if [[ -f "$TEACHER_CKPT" ]]; then
        echo "  Checkpoint found: $TEACHER_CKPT"
        echo "  Skipping step 3 (delete checkpoint to retrain)."
    else
        caffeinate -i "$PYTHON" scripts/v6/step3_train_prose_teacher.py \
            --epochs 2 \
            --batch_size 48 \
            --lr_text 2e-6 \
            --lr_image 1e-6 \
            --lr_logit 1e-4 \
            --text_blocks 2 \
            --image_blocks 1 \
            --probe_every 200 \
            --ckpt_every 500 \
            --run_tag prose_teacher
    fi

    if [[ ! -f "$TEACHER_CKPT" ]]; then
        echo "ERROR: prose teacher checkpoint not found after step 3. Aborting."
        exit 1
    fi
    echo "  Step 3 done: $TEACHER_CKPT"
else
    echo "[ STEP 3 ] Skipped (START_STEP=$START_STEP)"
fi

# ──────────────────────────────────────────────────────────────────────────────
# STEP 4 — Train v6 student (SL2-B, image tower unfrozen, KL prose teacher)
# ──────────────────────────────────────────────────────────────────────────────
if [[ "$START_STEP" -le 4 ]]; then
    echo ""
    echo "[ STEP 4 ] Training v6 student ..."
    echo "  (SL2-B, image tower unfrozen)"
    echo "  (Loss = InfoNCE + 0.5*KL(prose teacher) + 0.3*anchor)"
    echo "  Expected: ~10h on MPS"

    STUDENT_CKPT="$CKPT_DIR/v6_student_best_v6a.pt"
    if [[ -f "$STUDENT_CKPT" ]]; then
        echo "  Checkpoint found: $STUDENT_CKPT"
        echo "  Skipping step 4 (delete checkpoint to retrain)."
    else
        caffeinate -i "$PYTHON" scripts/v6/step4_train_v6_student.py \
            --prose_teacher_ckpt "$CKPT_DIR/prose_teacher_best_prose_teacher.pt" \
            --epochs 3 \
            --batch_size 48 \
            --lr_text 1e-6 \
            --lr_image 5e-7 \
            --lr_logit 1e-4 \
            --text_blocks 2 \
            --image_blocks 1 \
            --lam_kl 0.5 \
            --lam_anchor 0.3 \
            --anchor_size 256 \
            --probe_every 200 \
            --ckpt_every 500 \
            --run_tag v6a
    fi

    if [[ ! -f "$STUDENT_CKPT" ]]; then
        echo "ERROR: v6 student checkpoint not found after step 4. Aborting."
        exit 1
    fi
    echo "  Step 4 done: $STUDENT_CKPT"
else
    echo "[ STEP 4 ] Skipped (START_STEP=$START_STEP)"
fi

# ──────────────────────────────────────────────────────────────────────────────
# Summary
# ──────────────────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo " v6 Pipeline Complete"
echo " $(date)"
echo ""
echo " Checkpoints:"
echo "   prose teacher : $CKPT_DIR/prose_teacher_best_prose_teacher.pt"
echo "   v6 student    : $CKPT_DIR/v6_student_best_v6a.pt"
echo ""
echo " Training logs:"
echo "   $LOGS_DIR/v6_prose_teacher_prose_teacher.jsonl"
echo "   $LOGS_DIR/v6_student_v6a.jsonl"
echo ""
echo " Probe logs:"
echo "   $LOGS_DIR/v6_prose_teacher_probes_prose_teacher.jsonl"
echo "   $LOGS_DIR/v6_student_probes_v6a.jsonl"
echo ""
echo " Next: run phase_e_eval.py on the v6 student checkpoint"
echo "   python scripts/v5/phase_e_eval.py \\"
echo "     --ckpt checkpoints/v6/v6_student_best_v6a.pt \\"
echo "     --out_dir results/v6"
echo "============================================================"
