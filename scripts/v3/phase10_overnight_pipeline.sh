#!/usr/bin/env bash
set -euo pipefail

# Phase 10 Overnight Pipeline — 3 experiments targeting +10% on ALL 4 benchmarks
# Estimated total: ~9.5 hours
#
# Exp 1 (Phase 10d): Title-primary GCL — aligns training with atlas/KAGL eval format
# Exp 2 (Phase 10e): Title-primary + strong anchor (0.85) + lower LR
# Exp 3 (Phase 10f): Text-image-light — both encoders with differential LR

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

RESULTS_DIR="$REPO_ROOT/results"
CKPT_DIR="$REPO_ROOT/checkpoints/v3_phase10"
LOG_DIR="$RESULTS_DIR/overnight_logs"
DATA_DIR="$REPO_ROOT/data/processed/v3_phase10_500k"
EVAL_SCRIPT="$REPO_ROOT/scripts/v3/phase10_eval_benchmarks.py"
TRAIN_SCRIPT="$REPO_ROOT/scripts/v3/phase10_train_multifield.py"

mkdir -p "$LOG_DIR" "$RESULTS_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MASTER_LOG="$LOG_DIR/overnight_${TIMESTAMP}.log"

log() {
    echo "[$(date '+%H:%M:%S')] $*" | tee -a "$MASTER_LOG"
}

log "============================================================"
log "Phase 10 Overnight Pipeline — Started"
log "============================================================"

# ─── EXPERIMENT 1: Title-Primary GCL ────────────────────────────────────────
EXP1_NAME="phase10d_title_primary"
EXP1_CKPT_DIR="$CKPT_DIR/$EXP1_NAME"

log ""
log "═══ Experiment 1: Phase 10d — Title-Primary GCL ═══"
log "  Text mode: title-primary (title is LHS primary, query is RHS)"
log "  Scope: text-only"
log "  LR: 8e-7, Anchor: 0.5"
log ""

python3 -u "$TRAIN_SCRIPT" \
    --model-source fsl \
    --text-mode title-primary \
    --training-scope text-only \
    --lr 8e-7 \
    --lambda-anchor 0.5 \
    --epochs 1 \
    --queries-per-batch 8 \
    --products-per-query 16 \
    --save-every 500 \
    --data-dir "$DATA_DIR" \
    --run-name "$EXP1_NAME" \
    2>&1 | tee -a "$LOG_DIR/exp1_train.log"

log "Exp 1 training done. Evaluating..."

python3 -u "$EVAL_SCRIPT" \
    --model-source fsl \
    --checkpoint "$EXP1_CKPT_DIR/best.pt" \
    --corpus-size 3000 \
    --output-tag "$EXP1_NAME" \
    2>&1 | tee -a "$LOG_DIR/exp1_eval.log"

log "Exp 1 evaluation saved to $RESULTS_DIR/phase10_${EXP1_NAME}.json"

# Check if Exp 1 already beats target on all 4
python3 -c "
import json, sys
r = json.load(open('$RESULTS_DIR/phase10_${EXP1_NAME}.json'))
fsl = r['fsl_baseline']
all_beat = True
for bm in ['fashion200k', 'atlas', 'polyvore', 'KAGL']:
    if bm in r['results'] and 'primary_map10' in r['results'][bm]:
        ours = r['results'][bm]['primary_map10']
        target = fsl[bm] * 1.10
        delta = (ours - fsl[bm]) / fsl[bm] * 100
        print(f'{bm}: {ours:.4f} vs {fsl[bm]:.4f} ({delta:+.1f}%) target={target:.4f} {\"PASS\" if ours >= target else \"FAIL\"}')
        if ours < target:
            all_beat = False
if all_beat:
    print('ALL_BENCHMARKS_PASSED')
" 2>&1 | tee -a "$MASTER_LOG"

if grep -q "ALL_BENCHMARKS_PASSED" "$MASTER_LOG"; then
    log "*** Exp 1 beats FSL by >=10% on ALL benchmarks! Stopping early. ***"
    log "OVERNIGHT_RESULT: SUCCESS at Exp 1"
    exit 0
fi

# Skip Exp 2 & 3 — user requested only Exp 1 results due to MPS stall delays
log ""
log "Skipping Exp 2 & 3 (user requested). Proceeding to final comparison."
log "OVERNIGHT_RESULT: Completed Exp 1 only"
exit 0

# ─── EXPERIMENT 2: Title-Primary + Strong Anchor ────────────────────────────
EXP2_NAME="phase10e_title_strong_anchor"
EXP2_CKPT_DIR="$CKPT_DIR/$EXP2_NAME"

log ""
log "═══ Experiment 2: Phase 10e — Title-Primary + Strong Anchor ═══"
log "  Text mode: title-primary"
log "  Scope: text-only"
log "  LR: 5e-7, Anchor: 0.85 (very strong preservation)"
log ""

python3 -u "$TRAIN_SCRIPT" \
    --model-source fsl \
    --text-mode title-primary \
    --training-scope text-only \
    --lr 5e-7 \
    --lambda-anchor 0.85 \
    --epochs 1 \
    --queries-per-batch 8 \
    --products-per-query 16 \
    --save-every 500 \
    --data-dir "$DATA_DIR" \
    --run-name "$EXP2_NAME" \
    2>&1 | tee -a "$LOG_DIR/exp2_train.log"

log "Exp 2 training done. Evaluating..."

python3 -u "$EVAL_SCRIPT" \
    --model-source fsl \
    --checkpoint "$EXP2_CKPT_DIR/best.pt" \
    --corpus-size 3000 \
    --output-tag "$EXP2_NAME" \
    2>&1 | tee -a "$LOG_DIR/exp2_eval.log"

log "Exp 2 evaluation saved to $RESULTS_DIR/phase10_${EXP2_NAME}.json"

python3 -c "
import json, sys
r = json.load(open('$RESULTS_DIR/phase10_${EXP2_NAME}.json'))
fsl = r['fsl_baseline']
all_beat = True
for bm in ['fashion200k', 'atlas', 'polyvore', 'KAGL']:
    if bm in r['results'] and 'primary_map10' in r['results'][bm]:
        ours = r['results'][bm]['primary_map10']
        target = fsl[bm] * 1.10
        delta = (ours - fsl[bm]) / fsl[bm] * 100
        print(f'{bm}: {ours:.4f} vs {fsl[bm]:.4f} ({delta:+.1f}%) target={target:.4f} {\"PASS\" if ours >= target else \"FAIL\"}')
        if ours < target:
            all_beat = False
if all_beat:
    print('ALL_BENCHMARKS_PASSED')
" 2>&1 | tee -a "$MASTER_LOG"

if grep -q "ALL_BENCHMARKS_PASSED" <(tail -20 "$MASTER_LOG"); then
    log "*** Exp 2 beats FSL by >=10% on ALL benchmarks! Stopping early. ***"
    log "OVERNIGHT_RESULT: SUCCESS at Exp 2"
    exit 0
fi

# ─── EXPERIMENT 3: Text-Image-Light (both encoders) ─────────────────────────
EXP3_NAME="phase10f_text_image_light"
EXP3_CKPT_DIR="$CKPT_DIR/$EXP3_NAME"

log ""
log "═══ Experiment 3: Phase 10f — Text-Image-Light ═══"
log "  Text mode: title-primary"
log "  Scope: text-image-light (BOTH encoders, differential LR)"
log "  Text LR: 5e-7, Vision LR: 5e-8, Anchor: 0.7"
log ""

python3 -u "$TRAIN_SCRIPT" \
    --model-source fsl \
    --text-mode title-primary \
    --training-scope text-image-light \
    --lr 5e-7 \
    --vision-lr 5e-8 \
    --lambda-anchor 0.7 \
    --epochs 1 \
    --queries-per-batch 8 \
    --products-per-query 16 \
    --save-every 500 \
    --data-dir "$DATA_DIR" \
    --run-name "$EXP3_NAME" \
    2>&1 | tee -a "$LOG_DIR/exp3_train.log"

log "Exp 3 training done. Evaluating..."

python3 -u "$EVAL_SCRIPT" \
    --model-source fsl \
    --checkpoint "$EXP3_CKPT_DIR/best.pt" \
    --corpus-size 3000 \
    --output-tag "$EXP3_NAME" \
    2>&1 | tee -a "$LOG_DIR/exp3_eval.log"

log "Exp 3 evaluation saved to $RESULTS_DIR/phase10_${EXP3_NAME}.json"

# ─── FINAL COMPARISON ───────────────────────────────────────────────────────
log ""
log "════════════════════════════════════════════════════════════"
log "FINAL COMPARISON — All experiments vs FSL baseline"
log "════════════════════════════════════════════════════════════"
log ""

python3 -c "
import json, glob, os

results_dir = '$RESULTS_DIR'
fsl = {'fashion200k': 0.3859, 'atlas': 0.6919, 'polyvore': 0.5783, 'KAGL': 0.6779}
benchmarks = ['fashion200k', 'atlas', 'polyvore', 'KAGL']

experiments = [
    ('phase10d_title_primary', 'Exp1: Title-Primary'),
    ('phase10e_title_strong_anchor', 'Exp2: Title+StrongAnchor'),
    ('phase10f_text_image_light', 'Exp3: TextImageLight'),
]

# Header
header = '| Benchmark | FSL Baseline | Target (+10%) |'
for _, label in experiments:
    header += f' {label} |'
print(header)
print('|' + '---|' * (3 + len(experiments)))

best_exp = None
best_count = 0

for bm in benchmarks:
    row = f'| {bm} | {fsl[bm]:.4f} | {fsl[bm]*1.1:.4f} |'
    for tag, label in experiments:
        path = os.path.join(results_dir, f'phase10_{tag}.json')
        if os.path.exists(path):
            r = json.load(open(path))
            if bm in r.get('results', {}) and 'primary_map10' in r['results'][bm]:
                v = r['results'][bm]['primary_map10']
                delta = (v - fsl[bm]) / fsl[bm] * 100
                beats = 'YES' if v >= fsl[bm]*1.1 else 'no'
                row += f' {v:.4f} ({delta:+.1f}%) {beats} |'
            else:
                row += ' ERROR |'
        else:
            row += ' — |'
    print(row)

# Find best experiment
for tag, label in experiments:
    path = os.path.join(results_dir, f'phase10_{tag}.json')
    if os.path.exists(path):
        r = json.load(open(path))
        count = 0
        for bm in benchmarks:
            if bm in r.get('results', {}) and 'primary_map10' in r['results'][bm]:
                if r['results'][bm]['primary_map10'] >= fsl[bm] * 1.1:
                    count += 1
        if count > best_count:
            best_count = count
            best_exp = label
        print(f'  {label}: {count}/4 benchmarks passed')

print()
print(f'BEST EXPERIMENT: {best_exp} ({best_count}/4 benchmarks)')
if best_count == 4:
    print('*** GOAL ACHIEVED: +10% on ALL 4 benchmarks! ***')
else:
    print(f'Gap: {4 - best_count} benchmarks still below target')
" 2>&1 | tee -a "$MASTER_LOG"

# Save final summary
python3 -c "
import json, os, datetime

results_dir = '$RESULTS_DIR'
fsl = {'fashion200k': 0.3859, 'atlas': 0.6919, 'polyvore': 0.5783, 'KAGL': 0.6779}
benchmarks = ['fashion200k', 'atlas', 'polyvore', 'KAGL']
experiments = ['phase10d_title_primary', 'phase10e_title_strong_anchor', 'phase10f_text_image_light']

summary = {
    'timestamp': str(datetime.datetime.now()),
    'fsl_baseline': fsl,
    'experiments': {},
}

for tag in experiments:
    path = os.path.join(results_dir, f'phase10_{tag}.json')
    if os.path.exists(path):
        r = json.load(open(path))
        exp_results = {}
        for bm in benchmarks:
            if bm in r.get('results', {}) and 'primary_map10' in r['results'][bm]:
                v = r['results'][bm]['primary_map10']
                exp_results[bm] = {
                    'map10': v,
                    'fsl': fsl[bm],
                    'delta_pct': (v - fsl[bm]) / fsl[bm] * 100,
                    'beats_10pct': v >= fsl[bm] * 1.1,
                }
        summary['experiments'][tag] = exp_results

json.dump(summary, open(os.path.join(results_dir, 'overnight_summary.json'), 'w'), indent=2)
print('Summary saved to results/overnight_summary.json')
" 2>&1 | tee -a "$MASTER_LOG"

log ""
log "============================================================"
log "Overnight Pipeline Complete"
log "============================================================"
log "Master log: $MASTER_LOG"
log "Results: $RESULTS_DIR/overnight_summary.json"
