#!/usr/bin/env bash
set -euo pipefail

# Phase 10f — Overnight Pipeline
# GS-10M data extraction → Data prep → Train (differential LR) → Eval (15K) → Analysis
# Run under caffeinate to prevent sleep

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

LOG="$REPO_ROOT/results/phase10f_overnight.log"
mkdir -p "$REPO_ROOT/results"

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

log "========================================================"
log "Phase 10f — Overnight Pipeline (GS-10M data + differential LR)"
log "========================================================"

# ─── Step 1: Wait for GS-10M extraction ──────────────────────────────────────
GS10M_PID=$(ps aux | grep "gs10m_filtered" | grep python | grep -v grep | awk '{print $2}' | head -1)
if [ -n "$GS10M_PID" ]; then
    log "Step 1: GS-10M extraction running (PID $GS10M_PID). Waiting..."
    while kill -0 "$GS10M_PID" 2>/dev/null; do sleep 30; done
    log "GS-10M extraction completed."
else
    log "Step 1: GS-10M extraction already done."
fi

# Verify files exist
ATLAS_FILE="$REPO_ROOT/data/external/gs10m_filtered/atlas_style_items.jsonl"
KAGL_FILE="$REPO_ROOT/data/external/gs10m_filtered/kagl_style_items.jsonl"

if [ ! -f "$ATLAS_FILE" ] || [ ! -f "$KAGL_FILE" ]; then
    log "ERROR: GS-10M filtered files not found. Aborting."
    exit 1
fi

ATLAS_COUNT=$(wc -l < "$ATLAS_FILE" | tr -d ' ')
KAGL_COUNT=$(wc -l < "$KAGL_FILE" | tr -d ' ')
log "GS-10M data: atlas=$ATLAS_COUNT items, kagl=$KAGL_COUNT items"

# ─── Step 2: Data prep ───────────────────────────────────────────────────────
log ""
log "Step 2: Data preparation..."

python3 -u - << 'PYEOF' 2>&1 | tee -a "$LOG"
import json
from pathlib import Path
from collections import defaultdict
import random

REPO_ROOT = Path('/Users/rohit.anand/Desktop/Hobby/MODA')
DATA_DIR = REPO_ROOT / 'data' / 'processed' / 'v3_phase10_500k'
GS10M_DIR = REPO_ROOT / 'data' / 'external' / 'gs10m_filtered'
OUTPUT = REPO_ROOT / 'data' / 'processed' / 'v3_phase10f_combined'
OUTPUT.mkdir(parents=True, exist_ok=True)

# Load existing training data catalog
existing_jsonl = DATA_DIR / 'catalog.jsonl'
if not existing_jsonl.exists():
    # Try alternative paths
    for alt in [DATA_DIR / 'products.jsonl', DATA_DIR / 'train.jsonl']:
        if alt.exists():
            existing_jsonl = alt
            break

existing_items = []
if existing_jsonl.exists():
    with open(existing_jsonl) as f:
        for line in f:
            existing_items.append(json.loads(line))
    print(f"Existing training data: {len(existing_items):,} items from {existing_jsonl.name}")
else:
    print(f"WARNING: No existing catalog found in {DATA_DIR}")
    # List what's in the data dir
    print(f"Files in {DATA_DIR}: {[f.name for f in DATA_DIR.iterdir() if f.is_file()][:20]}")

# Load GS-10M filtered items
atlas_items = []
with open(GS10M_DIR / 'atlas_style_items.jsonl') as f:
    for line in f:
        atlas_items.append(json.loads(line))

kagl_items = []
with open(GS10M_DIR / 'kagl_style_items.jsonl') as f:
    for line in f:
        kagl_items.append(json.loads(line))

print(f"GS-10M atlas: {len(atlas_items):,}")
print(f"GS-10M kagl: {len(kagl_items):,}")

# Format GS-10M items for our training pipeline
# These don't have local images, so we'll use them as text-only pairs
# The training script will match them to images by category
gs10m_formatted = []
for item in atlas_items:
    gs10m_formatted.append({
        'title': item['title'],
        'query': item['query'],
        'l1_category': 'ethnic_wear',
        'source': 'gs10m_atlas',
        'product_id': item.get('product_id', ''),
    })

for item in kagl_items:
    gs10m_formatted.append({
        'title': item['title'],
        'query': item['query'],
        'l1_category': 'branded_fashion',
        'source': 'gs10m_kagl',
        'product_id': item.get('product_id', ''),
    })

random.seed(42)
random.shuffle(gs10m_formatted)

# Save combined GS-10M data
gs10m_output = OUTPUT / 'gs10m_atlas_kagl.jsonl'
with open(gs10m_output, 'w') as f:
    for item in gs10m_formatted:
        f.write(json.dumps(item) + '\n')

print(f"\nCombined GS-10M output: {len(gs10m_formatted):,} items")
print(f"Saved to: {gs10m_output}")

# Create a symlink or copy of existing data dir
import shutil
for f in DATA_DIR.iterdir():
    if f.is_file() and f.name != 'gs10m_atlas_kagl.jsonl':
        dst = OUTPUT / f.name
        if not dst.exists():
            shutil.copy2(f, dst)

print(f"\nData prep complete. Training dir: {OUTPUT}")
print(f"Total new text pairs: {len(gs10m_formatted):,}")
PYEOF

# ─── Step 3: Training ────────────────────────────────────────────────────────
log ""
log "Step 3: Training (differential LR, title-primary, λ=0.3)..."
log "Config: text blocks 0-3 frozen, 4-7 @ 5e-7, 8-11 @ 1e-6, vision frozen"

python3 -u scripts/v3/phase10_train_multifield.py \
    --model-source fsl \
    --text-mode title-primary \
    --training-scope text-only \
    --lr 1e-6 \
    --lambda-anchor 0.3 \
    --epochs 1 \
    --queries-per-batch 8 \
    --products-per-query 16 \
    --save-every 500 \
    --data-dir data/processed/v3_phase10f_combined \
    --run-name phase10f_gs10m_difflr \
    2>&1 | tee -a "$LOG"

CKPT="$REPO_ROOT/checkpoints/v3_phase10/phase10f_gs10m_difflr/best.pt"
if [ ! -f "$CKPT" ]; then
    CKPT=$(ls -t "$REPO_ROOT/checkpoints/v3_phase10/phase10f_gs10m_difflr/"step_*.pt 2>/dev/null | head -1)
fi

if [ -z "$CKPT" ] || [ ! -f "$CKPT" ]; then
    log "ERROR: No checkpoint found after training. Aborting."
    exit 1
fi
log "Best checkpoint: $CKPT"

# ─── Step 4: Evaluation (15K corpus, apples-to-apples) ───────────────────────
log ""
log "Step 4: Evaluation (15K corpus, 4 benchmarks)..."

python3 -u scripts/v3/phase10_eval_benchmarks.py \
    --model-source fsl \
    --checkpoint "$CKPT" \
    --corpus-size 15000 \
    --output-tag "phase10f_15k" \
    --benchmarks fashion200k atlas polyvore KAGL \
    2>&1 | tee -a "$LOG"

log "Evaluation saved: results/phase10_phase10f_15k.json"

# ─── Step 5: Error analysis + comparison ─────────────────────────────────────
log ""
log "Step 5: Analysis and comparison..."

python3 -u - << 'PYEOF' 2>&1 | tee -a "$LOG"
import json
from pathlib import Path
from datetime import datetime

REPO_ROOT = Path('/Users/rohit.anand/Desktop/Hobby/MODA')
RESULTS = REPO_ROOT / 'results'

def load(tag):
    p = RESULTS / f'phase10_{tag}.json'
    if p.exists(): return json.load(open(p)).get('results', {})
    return {}

# Load all results
fsl_15k = load('final_baseline_5bench')  # FSL baseline at 15K
exp3_15k = load('phase10f_15k')          # Our new model at 15K

BMS = ['fashion200k', 'atlas', 'polyvore', 'KAGL']

def t2i(r, bm):
    return r.get(bm, {}).get('text_to_image', {})

print()
print('=' * 72)
print('PHASE 10f — RESULTS (15K corpus, apples-to-apples vs FSL)')
print('=' * 72)
print()
print(f'{"Benchmark":<14} {"Metric":<10} {"FSL Base":<10} {"Exp3":<10} {"Delta":<10} {"Beat+10%?":<10}')
print('-' * 65)

beats_10 = 0
for bm in BMS:
    fsl_m = t2i(fsl_15k, bm)
    exp_m = t2i(exp3_15k, bm)
    if not fsl_m or not exp_m:
        print(f'{bm:<14} — No data —')
        continue
    
    for metric in ['recall_1', 'recall_10', 'mrr']:
        f_val = fsl_m.get(metric, 0)
        e_val = exp_m.get(metric, 0)
        label = {'recall_1':'R@1','recall_10':'R@10','mrr':'MRR'}[metric]
        delta = (e_val - f_val) / f_val * 100 if f_val else 0
        beat = '✓' if delta >= 10 else ''
        print(f'{bm:<14} {label:<10} {f_val:<10.3f} {e_val:<10.3f} {delta:+.1f}% {beat}')
    
    # AvgRecall
    f_avg = (fsl_m.get('recall_1',0) + fsl_m.get('recall_10',0)) / 2
    e_avg = (exp_m.get('recall_1',0) + exp_m.get('recall_10',0)) / 2
    delta_avg = (e_avg - f_avg) / f_avg * 100 if f_avg else 0
    beat = '✓✓ YES' if delta_avg >= 10 else ''
    if delta_avg >= 10: beats_10 += 1
    print(f'{bm:<14} {"AvgRecall":<10} {f_avg:<10.3f} {e_avg:<10.3f} {delta_avg:+.1f}% {beat}')
    print()

print(f'Benchmarks beating FSL by ≥10% (AvgRecall): {beats_10}/4')
if beats_10 == 4:
    print('*** GOAL ACHIEVED! ***')
elif beats_10 >= 2:
    print('Partial success — some benchmarks improved.')
else:
    print('Target not met. Further iteration needed.')

# Update EXPERIMENT_MASTER_LOG.md
now = datetime.now().strftime('%Y-%m-%d %H:%M IST')
entry = f"""
---

## Phase 10f — GS-10M Data + Differential LR (Overnight Run)

**Date**: {now}
**Model**: FSL + title-primary, differential LR (text blocks 8-11 @ 1e-6, 4-7 @ 5e-7, 0-3 frozen), λ_anchor=0.3
**New data**: GS-10M filtered atlas/KAGL items added to training
**Evaluation**: 15K stratified corpus, apples-to-apples vs FSL baseline

### Text-to-Image (15K corpus)

| Benchmark | Metric | FSL Baseline | Exp3 | Delta | Beat +10%? |
|-----------|--------|:---:|:---:|:---:|:---:|
"""

for bm in BMS:
    fsl_m = t2i(fsl_15k, bm)
    exp_m = t2i(exp3_15k, bm)
    if not fsl_m or not exp_m: continue
    f_avg = (fsl_m.get('recall_1',0) + fsl_m.get('recall_10',0)) / 2
    e_avg = (exp_m.get('recall_1',0) + exp_m.get('recall_10',0)) / 2
    delta = (e_avg - f_avg) / f_avg * 100 if f_avg else 0
    beat = '**YES**' if delta >= 10 else 'no'
    entry += f'| {bm} | AvgRecall | {f_avg:.4f} | {e_avg:.4f} | {delta:+.1f}% | {beat} |\n'
    entry += f'| {bm} | R@1 | {fsl_m.get("recall_1",0):.3f} | {exp_m.get("recall_1",0):.3f} | — | — |\n'
    entry += f'| {bm} | R@10 | {fsl_m.get("recall_10",0):.3f} | {exp_m.get("recall_10",0):.3f} | — | — |\n'
    entry += f'| {bm} | MRR | {fsl_m.get("mrr",0):.3f} | {exp_m.get("mrr",0):.3f} | — | — |\n'

entry += f'\n**Benchmarks ≥10% improvement**: {beats_10}/4\n'

LOG_PATH = REPO_ROOT / 'EXPERIMENT_MASTER_LOG.md'
with open(LOG_PATH, 'a') as f:
    f.write(entry)
print(f"\nEXPERIMENT_MASTER_LOG.md updated.")
PYEOF

log ""
log "========================================================"
log "Phase 10f Overnight Pipeline COMPLETE"
log "========================================================"
log "Results: results/phase10_phase10f_15k.json"
log "Log: $LOG"
