#!/usr/bin/env bash
set -euo pipefail

# Phase 10e Post-Training Pipeline — runs unattended after Exp 2 training finishes.
# 1. Wait for training process to exit
# 2. Run 3K quick eval on all 4 benchmarks
# 3. Run 15K full eval on all 4 benchmarks
# 4. Deep error analysis vs FSL baseline, Exp 1, Phase 10c
# 5. Update EXPERIMENT_MASTER_LOG.md

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

RESULTS_DIR="$REPO_ROOT/results"
CKPT="$REPO_ROOT/checkpoints/v3_phase10/phase10e_title_strong_anchor/best.pt"
EVAL_SCRIPT="$REPO_ROOT/scripts/v3/phase10_eval_benchmarks.py"
LOG="$RESULTS_DIR/phase10e_post_training.log"

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

mkdir -p "$RESULTS_DIR"

log "========================================================"
log "Phase 10e Post-Training Pipeline"
log "========================================================"

# ─── Step 1: Wait for training ───────────────────────────────────────────────
TRAIN_PID=$(ps aux | grep "phase10_train_multifield.*phase10e_title_strong_anchor" | grep -v grep | awk '{print $2}' | head -1)
if [ -n "$TRAIN_PID" ]; then
    log "Training still running (PID $TRAIN_PID). Waiting..."
    while kill -0 "$TRAIN_PID" 2>/dev/null; do sleep 30; done
    log "Training completed."
else
    log "Training already finished."
fi

# Verify checkpoint exists
if [ ! -f "$CKPT" ]; then
    log "ERROR: Checkpoint not found at $CKPT"
    # Try step_1500 as fallback
    CKPT_FALLBACK="$REPO_ROOT/checkpoints/v3_phase10/phase10e_title_strong_anchor/step_1500.pt"
    if [ -f "$CKPT_FALLBACK" ]; then
        log "Using fallback: $CKPT_FALLBACK"
        CKPT="$CKPT_FALLBACK"
    else
        log "No checkpoint found. Aborting."
        exit 1
    fi
fi
log "Checkpoint: $CKPT"

# ─── Step 2: Quick eval (3K corpus) ─────────────────────────────────────────
log ""
log "Step 2: Quick eval (3K corpus, 4 benchmarks)..."
python3 -u "$EVAL_SCRIPT" \
    --model-source fsl \
    --checkpoint "$CKPT" \
    --corpus-size 3000 \
    --output-tag "phase10e_3k" \
    --benchmarks fashion200k atlas polyvore KAGL \
    2>&1 | tee -a "$LOG"

log "Quick eval saved: $RESULTS_DIR/phase10_phase10e_3k.json"

# ─── Step 3: Full eval (15K corpus) ─────────────────────────────────────────
log ""
log "Step 3: Full eval (15K corpus, 4 benchmarks)..."
python3 -u "$EVAL_SCRIPT" \
    --model-source fsl \
    --checkpoint "$CKPT" \
    --corpus-size 15000 \
    --output-tag "phase10e_15k" \
    --benchmarks fashion200k atlas polyvore KAGL \
    2>&1 | tee -a "$LOG"

log "Full eval saved: $RESULTS_DIR/phase10_phase10e_15k.json"

# ─── Step 4: Deep error analysis ─────────────────────────────────────────────
log ""
log "Step 4: Deep error analysis..."

python3 - << 'PYEOF' 2>&1 | tee -a "$LOG"
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2] if '__file__' in dir() else Path('/Users/rohit.anand/Desktop/Hobby/MODA')
RESULTS_DIR = REPO_ROOT / 'results'

# ── Load all results ──────────────────────────────────────────────────────────
def load(tag):
    p = RESULTS_DIR / f'phase10_{tag}.json'
    if p.exists():
        return json.load(open(p)).get('results', {})
    return {}

fsl_live      = load('final_baseline_5bench')     # FSL at 15K
phase10c      = load('fsl_finetuned_5bench')       # Phase 10c (best prev) at 3K
phase10d      = load('phase10d_title_primary')     # Exp 1 at 3K
phase10e_3k   = load('phase10e_3k')               # Exp 2 at 3K
phase10e_15k  = load('phase10e_15k')              # Exp 2 at 15K

BMS = ['fashion200k', 'atlas', 'polyvore', 'KAGL']

MARQO_PUBLISHED = {
    'fashion200k': 0.3859,
    'atlas': 0.6919,
    'polyvore': 0.5783,
    'KAGL': 0.6779,
}

def t2i(r, bm): return r.get(bm, {}).get('text_to_image', {}).get('map10', None)
def cat(r, bm):
    rb = r.get(bm, {})
    for k in ['category3_to_product', 'category_to_product', 'category2_to_product']:
        if k in rb: return rb[k].get('map10', None)
    return None

print()
print('=' * 72)
print('PHASE 10e — DEEP ERROR ANALYSIS')
print('=' * 72)

# ── Table 1: Text-to-image MAP@10 comparison ─────────────────────────────────
print()
print('── Table 1: Text-to-Image MAP@10 (primary task) ──')
print(f'{"Benchmark":<14} {"FSL-pub":>8} {"FSL-15K":>8} {"10c-3K":>8} {"Exp1-3K":>8} {"Exp2-3K":>8} {"Exp2-15K":>9} {"vs pub":>7}')
print('-' * 75)
for bm in BMS:
    pub  = MARQO_PUBLISHED[bm]
    fl   = t2i(fsl_live, bm)
    c    = t2i(phase10c, bm)
    d    = t2i(phase10d, bm)
    e3   = t2i(phase10e_3k, bm)
    e15  = t2i(phase10e_15k, bm)
    vs = f'{(e15/fl-1)*100:+.1f}%' if (e15 and fl) else '—'
    def fmt(v): return f'{v:.4f}' if v else '  —  '
    print(f'{bm:<14} {pub:>8.4f} {fmt(fl):>8} {fmt(c):>8} {fmt(d):>8} {fmt(e3):>8} {fmt(e15):>9} {vs:>7}')

# ── Table 2: Category MAP@10 comparison ──────────────────────────────────────
print()
print('── Table 2: Category→Product MAP@10 ──')
print(f'{"Benchmark":<14} {"FSL-15K":>8} {"Exp2-15K":>9} {"delta":>7}')
print('-' * 45)
for bm in BMS:
    fl  = cat(fsl_live, bm)
    e15 = cat(phase10e_15k, bm)
    if fl and e15:
        delta = f'{(e15/fl-1)*100:+.1f}%'
    else:
        delta = '—'
    def fmt(v): return f'{v:.4f}' if v else '  —  '
    print(f'{bm:<14} {fmt(fl):>8} {fmt(e15):>9} {delta:>7}')

# ── Table 3: Did Exp 2 fix the atlas/KAGL degradation? ───────────────────────
print()
print('── Table 3: Atlas/KAGL degradation analysis ──')
print(f'{"Benchmark":<14} {"Phase10c":>10} {"Exp1":>8} {"Exp2":>8} {"FSL-live":>10} {"Fixed?":>8}')
print('-' * 62)
for bm in ['atlas', 'KAGL']:
    c3  = t2i(phase10c, bm)
    d3  = t2i(phase10d, bm)
    e3  = t2i(phase10e_3k, bm)
    fl  = t2i(fsl_live, bm)
    fixed = '✓ YES' if (e3 and fl and e3 >= fl * 0.98) else ('~ CLOSE' if (e3 and fl and e3 >= fl * 0.90) else '✗ NO')
    def fmt(v): return f'{v:.4f}' if v else '  —  '
    print(f'{bm:<14} {fmt(c3):>10} {fmt(d3):>8} {fmt(e3):>8} {fmt(fl):>10} {fixed:>8}')

# ── Table 4: Full benchmark scorecard vs Marqo published ────────────────────
print()
print('── Table 4: Final scorecard vs Marqo published ──')
print(f'{"Benchmark":<14} {"Marqo-pub":>10} {"Exp2-3K":>8} {"delta":>7} {"Beat+10%?":>10}')
print('-' * 55)
beats = 0
for bm in BMS:
    pub = MARQO_PUBLISHED[bm]
    e3  = t2i(phase10e_3k, bm)
    if e3:
        delta = (e3 - pub) / pub * 100
        beat = '✓ YES' if delta >= 10 else ('~ CLOSE' if delta >= 0 else '✗ NO')
        if delta >= 10: beats += 1
    else:
        delta, beat = 0, '—'
    def fmt(v): return f'{v:.4f}' if v else '  —  '
    print(f'{bm:<14} {pub:>10.4f} {fmt(e3):>8} {f"{delta:+.1f}%":>7} {beat:>10}')
print()
print(f'Benchmarks beating FSL by ≥10%: {beats}/4')
if beats == 4:
    print('*** GOAL ACHIEVED! Model beats Marqo-FashionSigLIP by ≥10% on all benchmarks! ***')
elif beats >= 2:
    print('Partial success. Remaining gap analysis follows.')
else:
    print('Goal not yet met. See gap analysis.')

# ── Gap Analysis ─────────────────────────────────────────────────────────────
print()
print('── Gap Analysis: What still needs to improve ──')
for bm in BMS:
    pub = MARQO_PUBLISHED[bm]
    e3  = t2i(phase10e_3k, bm)
    if not e3: continue
    delta = (e3 - pub) / pub * 100
    needed = pub * 1.10
    gap    = needed - e3
    if delta < 10:
        pct_gap = (needed - e3) / pub * 100
        print(f'  {bm}: at {e3:.4f}, need {needed:.4f} (+{pct_gap:.1f}pp more to hit target)')

print()
print('── Anchor effect on forgetting ──')
print('  High anchor (0.85) should strongly preserve FSL text repr.')
print('  If atlas/KAGL still degrade: text-style mismatch is irrecoverable via anchor alone.')
print('  If atlas/KAGL recover: anchor is the key lever, increase further or try frozen text encoder.')
print()
PYEOF

# ─── Step 5: Update EXPERIMENT_MASTER_LOG.md ─────────────────────────────────
log ""
log "Step 5: Updating EXPERIMENT_MASTER_LOG.md..."

python3 - << 'PYEOF' 2>&1 | tee -a "$LOG"
import json
from pathlib import Path
from datetime import datetime

REPO_ROOT = Path('/Users/rohit.anand/Desktop/Hobby/MODA')
RESULTS_DIR = REPO_ROOT / 'results'
LOG_PATH = REPO_ROOT / 'EXPERIMENT_MASTER_LOG.md'

def load(tag):
    p = RESULTS_DIR / f'phase10_{tag}.json'
    if p.exists(): return json.load(open(p)).get('results', {})
    return {}

fsl_live    = load('final_baseline_5bench')
phase10e_3k = load('phase10e_3k')
phase10e_15k = load('phase10e_15k')

BMS = ['fashion200k', 'atlas', 'polyvore', 'KAGL']
MARQO_PUB = {'fashion200k': 0.3859, 'atlas': 0.6919, 'polyvore': 0.5783, 'KAGL': 0.6779}

def t2i(r, bm): return r.get(bm, {}).get('text_to_image', {}).get('map10', None)
def get_cat(r, bm):
    rb = r.get(bm, {})
    for k in ['category3_to_product', 'category_to_product']:
        if k in rb: return rb[k].get('map10', None)
    return None
def get_all_cats(r, bm):
    rb = r.get(bm, {})
    rows = []
    for k in ['category3_to_product','category2_to_product','category1_to_product','category_to_product']:
        if k in rb:
            task = k.replace('_to_product','')
            rows.append((task, rb[k].get('map10',0), rb[k].get('recall_1',0), rb[k].get('mrr',0)))
    return rows

now = datetime.now().strftime('%Y-%m-%d %H:%M IST')

entry = f"""
---

## Phase 10e — Title-Primary + Strong Anchor (λ=0.85) Results

**Date**: {now}
**Model**: FSL + title-primary GCL, λ_anchor=0.85, LR=5e-7, 1570 steps
**New data**: +5,600 synthetic atlas/KAGL-style queries injected into training

### Text-to-Image MAP@10

| Benchmark | Marqo Published | FSL Baseline (15K) | Exp2 (3K) | Exp2 (15K) | vs Published |
|-----------|:---:|:---:|:---:|:---:|:---:|
"""

beats = 0
for bm in BMS:
    pub  = MARQO_PUB[bm]
    fl   = t2i(fsl_live, bm)
    e3   = t2i(phase10e_3k, bm)
    e15  = t2i(phase10e_15k, bm)
    vs   = f'{(e3-pub)/pub*100:+.1f}%' if e3 else '—'
    beat = '**YES**' if (e3 and (e3-pub)/pub*100 >= 10) else 'no'
    if e3 and (e3-pub)/pub*100 >= 10: beats += 1
    def fmt(v): return f'{v:.4f}' if v else '—'
    entry += f'| {bm} | {pub:.4f} | {fmt(fl)} | {fmt(e3)} | {fmt(e15)} | {vs} |\n'

entry += f"""
### Category-to-Product MAP@10 (15K corpus)

| Benchmark | Task | FSL Baseline | Exp2 | Delta |
|-----------|------|:---:|:---:|:---:|
"""
for bm in BMS:
    for task, v, r1, mrr in get_all_cats(phase10e_15k, bm):
        fl_cats = {k:v2.get('map10',0) for k,v2 in fsl_live.get(bm,{}).items() if '_to_product' in k and 'map10' not in k}
        fl_v = fl_cats.get(f'{task}_to_product', None)
        if fl_v and v:
            delta = f'**{(v-fl_v)/fl_v*100:+.1f}%**' if abs((v-fl_v)/fl_v*100) >= 5 else f'{(v-fl_v)/fl_v*100:+.1f}%'
            entry += f'| {bm} | {task} | {fl_v:.4f} | {v:.4f} | {delta} |\n'

entry += f"""
### Verdict

**Benchmarks beating Marqo-published FSL by ≥10% on T2I**: {beats}/4

### Error Analysis

**Key question**: Did λ=0.85 anchor fix atlas/KAGL degradation?

"""

for bm in ['atlas','KAGL']:
    e3 = t2i(phase10e_3k, bm)
    fl = t2i(fsl_live, bm)
    pub = MARQO_PUB[bm]
    if e3 and fl:
        vs_fsl = (e3-fl)/fl*100
        vs_pub = (e3-pub)/pub*100
        status = '**FIXED** ✓' if vs_fsl >= -2 else ('**PARTIALLY FIXED** ~' if vs_fsl >= -10 else '**STILL DEGRADED** ✗')
        entry += f'- **{bm}**: {e3:.4f} vs FSL-live {fl:.4f} ({vs_fsl:+.1f}%) — {status}\n'

entry += """
**Root cause confirmed**: 
- If atlas/KAGL now on par → λ=0.85 was the fix, training drift was the problem
- If still degraded → text-style mismatch is fundamental, need domain-specific training data
"""

# Append to log
with open(LOG_PATH, 'a') as f:
    f.write(entry)

print(f"EXPERIMENT_MASTER_LOG.md updated. Beats target: {beats}/4")
PYEOF

log ""
log "========================================================"
log "Phase 10e Post-Training Pipeline Complete"
log "========================================================"
log "Results: $RESULTS_DIR/phase10_phase10e_3k.json"
log "         $RESULTS_DIR/phase10_phase10e_15k.json"
log "Log:     $LOG"
