#!/bin/bash
# Phase 10 Full Pipeline — Runs unattended (~5 hours)
# 
# What this does (in order):
#   1. Waits for P4B training to finish (if still running)
#   2. Evaluates P4B model on all 4 clean benchmarks
#   3. Error analysis on P4B results (per-category breakdown, failure patterns)
#   4. Trains FSL variant (fine-tunes FashionSigLIP with 500K data + 10K enriched descriptions)
#   5. Evaluates FSL variant on all 4 benchmarks
#   6. Error analysis on FSL results
#   7. Writes comparison summary + combined error analysis
#
# Data used:
#   - 500K real pairs (GS-10M + DeepFashion) — no benchmark leakage
#   - 10K enriched descriptions (LLM-rewritten product titles) — no benchmark leakage
#
# Expected runtime: ~5 hours total

set -eo pipefail

REPO_ROOT="/Users/rohit.anand/Desktop/Hobby/MODA"
cd "$REPO_ROOT"

LOG_FILE="$REPO_ROOT/results/phase10_pipeline.log"
mkdir -p "$REPO_ROOT/results"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

log "============================================================"
log "Phase 10 Full Pipeline — Started"
log "============================================================"
log "Goal: Beat FSL (203M) by >=10% MAP@10 on ALL 4 benchmarks"
log "Data: 500K real pairs + 10K enriched descriptions (zero leakage)"
log ""

# ─── Step 1: Wait for P4B training ───────────────────────────────────────────
log "Step 1: Waiting for P4B training to complete..."

P4B_BEST="$REPO_ROOT/checkpoints/v3_phase10/phase10_p4b_multifield/best.pt"
P4B_FINAL="$REPO_ROOT/checkpoints/v3_phase10/phase10_p4b_multifield/epoch_1.pt"

for i in $(seq 1 240); do
    if [ -f "$P4B_FINAL" ]; then
        log "  P4B training DONE. Checkpoint: $P4B_FINAL"
        break
    fi
    if [ $i -eq 240 ]; then
        log "  WARNING: P4B didn't finish in 2hr. Using step_1000 checkpoint."
        P4B_BEST="$REPO_ROOT/checkpoints/v3_phase10/phase10_p4b_multifield/step_1000.pt"
        break
    fi
    sleep 30
done

# ─── Step 2: Evaluate P4B ────────────────────────────────────────────────────
log ""
log "Step 2: Evaluating P4B model on 4 benchmarks (3K screener)..."

python3 -u scripts/v3/phase10_eval_benchmarks.py \
    --model-source phase4b \
    --checkpoint "$P4B_BEST" \
    --corpus-size 3000 \
    --output-tag p4b_multifield \
    2>&1 | tee -a "$LOG_FILE"

log "  P4B evaluation DONE."

# ─── Step 3: Error analysis on P4B ───────────────────────────────────────────
log ""
log "Step 3: Error analysis on P4B results..."

python3 -u -c "
import json
from pathlib import Path

results_path = Path('results/phase10_p4b_multifield.json')
if not results_path.exists():
    print('  No P4B results to analyze.')
    exit(0)

with open(results_path) as f:
    data = json.load(f)

results = data.get('results', {})
fsl = data.get('fsl_baseline', {})

analysis = {'model': 'p4b_multifield', 'benchmarks': {}}
for bm, bm_results in results.items():
    if 'primary_map10' not in bm_results:
        continue
    ours = bm_results['primary_map10']
    baseline = fsl.get(bm, 0)
    delta_pct = (ours - baseline) / baseline * 100 if baseline > 0 else 0
    target = baseline * 1.10
    
    analysis['benchmarks'][bm] = {
        'map10': ours,
        'fsl_baseline': baseline,
        'delta_pct': delta_pct,
        'beats_10pct': ours >= target,
        'gap_to_target': target - ours,
        'all_task_results': bm_results,
    }

# Identify weakest benchmark
if analysis['benchmarks']:
    worst = min(analysis['benchmarks'].items(), key=lambda x: x[1]['delta_pct'])
    analysis['weakest_benchmark'] = worst[0]
    analysis['weakest_delta'] = worst[1]['delta_pct']
    
    best = max(analysis['benchmarks'].items(), key=lambda x: x[1]['delta_pct'])
    analysis['strongest_benchmark'] = best[0]
    analysis['strongest_delta'] = best[1]['delta_pct']

with open('results/phase10_p4b_error_analysis.json', 'w') as f:
    json.dump(analysis, f, indent=2)

print(f'  Weakest: {analysis.get(\"weakest_benchmark\", \"?\")} ({analysis.get(\"weakest_delta\", 0):+.1f}%)')
print(f'  Strongest: {analysis.get(\"strongest_benchmark\", \"?\")} ({analysis.get(\"strongest_delta\", 0):+.1f}%)')
print(f'  Saved: results/phase10_p4b_error_analysis.json')
" 2>&1 | tee -a "$LOG_FILE"

log "  P4B error analysis DONE."

# ─── Step 4: Train FSL variant ────────────────────────────────────────────────
log ""
log "Step 4: Training FSL variant (FashionSigLIP + 500K data + enriched descriptions)..."
log "  Model: Marqo-FashionSigLIP (hf-hub)"
log "  LR: 1e-6, Anchor: 0.3, Epochs: 1"
log "  Enriched descriptions: data/processed/v3_phase10_500k/enriched_descriptions.jsonl"

python3 -u scripts/v3/phase10_train_multifield.py \
    --model-source fsl \
    --epochs 1 \
    --lr 1e-6 \
    --lambda-anchor 0.3 \
    --queries-per-batch 8 \
    --products-per-query 16 \
    --run-name phase10_fsl_multifield \
    2>&1 | tee -a "$LOG_FILE"

log "  FSL training DONE."

# ─── Step 5: Evaluate FSL variant ────────────────────────────────────────────
log ""
log "Step 5: Evaluating FSL variant on 4 benchmarks (3K screener)..."

FSL_BEST="$REPO_ROOT/checkpoints/v3_phase10/phase10_fsl_multifield/best.pt"

python3 -u scripts/v3/phase10_eval_benchmarks.py \
    --model-source fsl \
    --checkpoint "$FSL_BEST" \
    --corpus-size 3000 \
    --output-tag fsl_multifield \
    2>&1 | tee -a "$LOG_FILE"

log "  FSL evaluation DONE."

# ─── Step 6: Error analysis on FSL ───────────────────────────────────────────
log ""
log "Step 6: Error analysis on FSL results..."

python3 -u -c "
import json
from pathlib import Path

results_path = Path('results/phase10_fsl_multifield.json')
if not results_path.exists():
    print('  No FSL results to analyze.')
    exit(0)

with open(results_path) as f:
    data = json.load(f)

results = data.get('results', {})
fsl = data.get('fsl_baseline', {})

analysis = {'model': 'fsl_multifield', 'benchmarks': {}}
for bm, bm_results in results.items():
    if 'primary_map10' not in bm_results:
        continue
    ours = bm_results['primary_map10']
    baseline = fsl.get(bm, 0)
    delta_pct = (ours - baseline) / baseline * 100 if baseline > 0 else 0
    target = baseline * 1.10
    
    analysis['benchmarks'][bm] = {
        'map10': ours,
        'fsl_baseline': baseline,
        'delta_pct': delta_pct,
        'beats_10pct': ours >= target,
        'gap_to_target': target - ours,
        'all_task_results': bm_results,
    }

if analysis['benchmarks']:
    worst = min(analysis['benchmarks'].items(), key=lambda x: x[1]['delta_pct'])
    analysis['weakest_benchmark'] = worst[0]
    analysis['weakest_delta'] = worst[1]['delta_pct']
    
    best = max(analysis['benchmarks'].items(), key=lambda x: x[1]['delta_pct'])
    analysis['strongest_benchmark'] = best[0]
    analysis['strongest_delta'] = best[1]['delta_pct']

with open('results/phase10_fsl_error_analysis.json', 'w') as f:
    json.dump(analysis, f, indent=2)

print(f'  Weakest: {analysis.get(\"weakest_benchmark\", \"?\")} ({analysis.get(\"weakest_delta\", 0):+.1f}%)')
print(f'  Strongest: {analysis.get(\"strongest_benchmark\", \"?\")} ({analysis.get(\"strongest_delta\", 0):+.1f}%)')
print(f'  Saved: results/phase10_fsl_error_analysis.json')
" 2>&1 | tee -a "$LOG_FILE"

log "  FSL error analysis DONE."

# ─── Step 7: Final comparison + combined analysis ─────────────────────────────
log ""
log "Step 7: Generating comparison and combined error analysis..."

python3 -u scripts/v3/phase10_compare_results.py 2>&1 | tee -a "$LOG_FILE"

# Combined error analysis summary
python3 -u -c "
import json
from pathlib import Path

print()
print('=' * 60)
print('COMBINED ERROR ANALYSIS')
print('=' * 60)

models = {}
for tag in ['p4b', 'fsl']:
    path = Path(f'results/phase10_{tag}_error_analysis.json')
    if path.exists():
        with open(path) as f:
            models[tag] = json.load(f)

if not models:
    print('No error analysis files found.')
    exit(0)

# Find best model per benchmark
benchmarks = ['fashion200k', 'atlas', 'polyvore', 'KAGL']
print()
print('Best model per benchmark:')
print('| Benchmark | Best Model | MAP@10 | vs FSL | Beats +10%? |')
print('|---|---|---:|---:|---|')

overall_winner = None
all_beat_count = {'p4b': 0, 'fsl': 0}

for bm in benchmarks:
    best_model = None
    best_score = -1
    for tag, analysis in models.items():
        bm_data = analysis.get('benchmarks', {}).get(bm, {})
        score = bm_data.get('map10', 0)
        if score > best_score:
            best_score = score
            best_model = tag
            best_delta = bm_data.get('delta_pct', 0)
            best_beats = bm_data.get('beats_10pct', False)

    if best_model:
        if best_beats:
            all_beat_count[best_model] += 1
        print(f'| {bm} | {best_model} | {best_score:.4f} | {best_delta:+.1f}% | {\"YES\" if best_beats else \"no\"} |')

print()
# Determine overall winner
winner = max(all_beat_count.items(), key=lambda x: x[1])
print(f'Overall winner: {winner[0]} (beats +10% on {winner[1]}/4 benchmarks)')
print()

# Recommendations
print('RECOMMENDATIONS:')
for tag, analysis in models.items():
    weak_bm = analysis.get('weakest_benchmark', '?')
    weak_delta = analysis.get('weakest_delta', 0)
    if weak_delta < 10:
        print(f'  [{tag}] Weakest on {weak_bm} ({weak_delta:+.1f}%). ')
        if weak_delta < 0:
            print(f'         -> Model REGRESSED on {weak_bm}. Increase anchor lambda or reduce LR.')
        elif weak_delta < 5:
            print(f'         -> Marginal gain. Try: more data, 2nd epoch, or text+vision scope.')
        else:
            print(f'         -> Close to target. Try: lower anchor, slightly higher LR.')

print()
print('Files:')
print('  results/phase10_comparison.md — head-to-head table')
print('  results/phase10_p4b_error_analysis.json — P4B per-benchmark analysis')
print('  results/phase10_fsl_error_analysis.json — FSL per-benchmark analysis')
print('  results/phase10_pipeline.log — full pipeline log')
" 2>&1 | tee -a "$LOG_FILE"

log ""
log "============================================================"
log "Phase 10 Full Pipeline — COMPLETE at $(date '+%Y-%m-%d %H:%M:%S')"
log "============================================================"
log ""
log "Check results/phase10_comparison.md for the final table."
log "Check results/phase10_pipeline.log for full details."
