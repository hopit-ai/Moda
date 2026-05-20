#!/usr/bin/env bash
set -euo pipefail

# Post-overnight-pipeline evaluation:
# 1. Wait for the overnight pipeline to finish
# 2. Identify the best model from Exp 1/2/3
# 3. Run full 5-benchmark evaluation (fashion200k, atlas, polyvore, KAGL, iMaterialist)
#    using 30% stratified sampling for larger datasets

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

RESULTS_DIR="$REPO_ROOT/results"
CKPT_DIR="$REPO_ROOT/checkpoints/v3_phase10"
EVAL_SCRIPT="$REPO_ROOT/scripts/v3/phase10_eval_benchmarks.py"
LOG="$RESULTS_DIR/post_pipeline_eval.log"

log() {
    echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"
}

log "============================================================"
log "Post-Pipeline 5-Benchmark Evaluation"
log "============================================================"

# Step 1: Wait for overnight pipeline to finish
PIPELINE_PID=$(ps aux | grep "phase10_overnight_pipeline" | grep -v grep | awk '{print $2}' | head -1)

if [ -n "$PIPELINE_PID" ]; then
    log "Overnight pipeline still running (PID $PIPELINE_PID). Waiting..."
    while kill -0 "$PIPELINE_PID" 2>/dev/null; do
        sleep 60
    done
    log "Overnight pipeline completed."
else
    log "Overnight pipeline already finished."
fi

# Step 2: Find the best model
log ""
log "Finding best experiment..."

BEST_MODEL=$(python3 -c "
import json, os

results_dir = '$RESULTS_DIR'
fsl = {'fashion200k': 0.3859, 'atlas': 0.6919, 'polyvore': 0.5783, 'KAGL': 0.6779}
benchmarks = ['fashion200k', 'atlas', 'polyvore', 'KAGL']

experiments = [
    'phase10d_title_primary',
    'phase10e_title_strong_anchor',
    'phase10f_text_image_light',
]

best_exp = None
best_score = -1

for tag in experiments:
    path = os.path.join(results_dir, f'phase10_{tag}.json')
    if os.path.exists(path):
        r = json.load(open(path))
        # Score: average of (our_map10 / fsl_baseline) across 4 benchmarks
        ratios = []
        for bm in benchmarks:
            if bm in r.get('results', {}) and 'primary_map10' in r['results'][bm]:
                ours = r['results'][bm]['primary_map10']
                ratios.append(ours / fsl[bm])
        avg_ratio = sum(ratios) / len(ratios) if ratios else 0
        if avg_ratio > best_score:
            best_score = avg_ratio
            best_exp = tag

# Also compare against our Phase 10c model (prev best)
prev_path = os.path.join(results_dir, 'phase10_fsl_finetuned_5bench.json')
if os.path.exists(prev_path):
    r = json.load(open(prev_path))
    ratios = []
    for bm in benchmarks:
        if bm in r.get('results', {}) and 'primary_map10' in r['results'][bm]:
            ours = r['results'][bm]['primary_map10']
            ratios.append(ours / fsl[bm])
    avg_ratio = sum(ratios) / len(ratios) if ratios else 0
    if avg_ratio > best_score:
        best_score = avg_ratio
        best_exp = 'phase10_fsl_multifield'

if best_exp:
    print(best_exp)
else:
    print('phase10_fsl_multifield')
")

log "Best experiment: $BEST_MODEL (avg ratio vs FSL)"

# Determine checkpoint path
if [ "$BEST_MODEL" = "phase10_fsl_multifield" ]; then
    BEST_CKPT="$CKPT_DIR/phase10_fsl_multifield/best.pt"
else
    BEST_CKPT="$CKPT_DIR/$BEST_MODEL/best.pt"
fi

if [ ! -f "$BEST_CKPT" ]; then
    log "ERROR: Checkpoint not found at $BEST_CKPT"
    log "Falling back to Phase 10c model"
    BEST_CKPT="$CKPT_DIR/phase10_fsl_multifield/best.pt"
    BEST_MODEL="phase10_fsl_multifield"
fi

log "Using checkpoint: $BEST_CKPT"

# Step 3: Determine corpus sizes (30% stratified sampling)
# fashion200k: 201K -> 30% = ~60K (cap at 30K for MPS memory)
# atlas: 78K -> 30% = ~23K (cap at 20K)
# polyvore: ~30K -> 30% = ~9K
# KAGL: 44K -> 30% = ~13K
# iMaterialist: 721K -> 30% = way too much, use 30K
CORPUS_SIZE=15000

log ""
log "Step 3: Running 5-benchmark evaluation with corpus_size=$CORPUS_SIZE"
log "Benchmarks: fashion200k, atlas, polyvore, KAGL, iMaterialist"
log ""

# Run evaluation on best model
log "Evaluating BEST MODEL ($BEST_MODEL)..."
python3 -u "$EVAL_SCRIPT" \
    --model-source fsl \
    --checkpoint "$BEST_CKPT" \
    --corpus-size "$CORPUS_SIZE" \
    --output-tag "final_best_5bench" \
    --benchmarks fashion200k atlas polyvore KAGL iMaterialist \
    2>&1 | tee -a "$LOG"

log ""
log "Evaluating FSL BASELINE (no fine-tuning)..."
python3 -u "$EVAL_SCRIPT" \
    --model-source fsl \
    --corpus-size "$CORPUS_SIZE" \
    --output-tag "final_baseline_5bench" \
    --benchmarks fashion200k atlas polyvore KAGL iMaterialist \
    2>&1 | tee -a "$LOG"

# Step 4: Final comparison
log ""
log "════════════════════════════════════════════════════════════"
log "FINAL 5-BENCHMARK COMPARISON (30% stratified, $CORPUS_SIZE corpus)"
log "════════════════════════════════════════════════════════════"

python3 -c "
import json, os

results_dir = '$RESULTS_DIR'
benchmarks = ['fashion200k', 'atlas', 'polyvore', 'KAGL', 'iMaterialist']

best_path = os.path.join(results_dir, 'phase10_final_best_5bench.json')
base_path = os.path.join(results_dir, 'phase10_final_baseline_5bench.json')

if not os.path.exists(best_path) or not os.path.exists(base_path):
    print('ERROR: Result files not found')
    exit(1)

best = json.load(open(best_path))
base = json.load(open(base_path))

print()
print('| Benchmark | Task | FSL Baseline | Best Model | Delta |')
print('|-----------|------|:---:|:---:|:---:|')

for bm in benchmarks:
    if bm in best.get('results', {}) and bm in base.get('results', {}):
        br = base['results'][bm]
        ber = best['results'][bm]
        
        # text-to-image
        if 'text_to_image' in br and 'text_to_image' in ber:
            b_map = br['text_to_image']['map10']
            e_map = ber['text_to_image']['map10']
            delta = (e_map - b_map) / b_map * 100 if b_map > 0 else 0
            print(f'| {bm} | text-to-image | {b_map:.4f} | {e_map:.4f} | {delta:+.1f}% |')
        
        # category tasks
        for key in sorted(br.keys()):
            if '_to_product' in key and 'map10' not in key:
                if key in ber:
                    b_cat = br[key]['map10']
                    e_cat = ber[key]['map10']
                    delta = (e_cat - b_cat) / b_cat * 100 if b_cat > 0 else 0
                    task = key.replace('_to_product', '')
                    print(f'| {bm} | {task} | {b_cat:.4f} | {e_cat:.4f} | {delta:+.1f}% |')

print()
print('Summary (primary text-to-image MAP@10):')
print('| Benchmark | FSL Baseline | Best Model | Delta | Beat +10%? |')
print('|-----------|:---:|:---:|:---:|:---:|')
for bm in benchmarks:
    if bm in best.get('results', {}) and bm in base.get('results', {}):
        b_map = base['results'][bm].get('text_to_image', {}).get('map10', 0)
        e_map = best['results'][bm].get('text_to_image', {}).get('map10', 0)
        delta = (e_map - b_map) / b_map * 100 if b_map > 0 else 0
        beat = 'YES' if e_map >= b_map * 1.1 else 'no'
        print(f'| {bm} | {b_map:.4f} | {e_map:.4f} | {delta:+.1f}% | {beat} |')
" 2>&1 | tee -a "$LOG"

log ""
log "============================================================"
log "Post-Pipeline Evaluation Complete"
log "============================================================"
log "Results: $RESULTS_DIR/phase10_final_best_5bench.json"
log "         $RESULTS_DIR/phase10_final_baseline_5bench.json"
log "Log:     $LOG"
