#!/bin/bash
# MODA Phase 3 — Overnight Runner
# Waits for in-flight jobs, then runs remaining tasks sequentially
set -e

REPO="/Users/rohit.anand/Desktop/Hobby/MODA"
cd "$REPO"

LOG_DIR="$REPO/results/real"
TIER1_LOG="$REPO/results/tier1"
mkdir -p "$LOG_DIR" "$TIER1_LOG"

timestamp() { date '+%Y-%m-%d %H:%M:%S'; }

echo "============================================================"
echo "MODA Phase 3 — Overnight Runner"
echo "Started: $(timestamp)"
echo "============================================================"

# ── Step 1: Wait for Phase 3.9 eval to finish ─────────────────────────
PHASE39_PID=93295
if kill -0 $PHASE39_PID 2>/dev/null; then
    echo "[$(timestamp)] Step 1: Waiting for Phase 3.9 eval (PID $PHASE39_PID)..."
    while kill -0 $PHASE39_PID 2>/dev/null; do
        sleep 30
    done
    echo "[$(timestamp)] Phase 3.9 eval completed."
else
    echo "[$(timestamp)] Phase 3.9 eval already finished."
fi

# ── Step 2: Wait for Tier 1 deepfashion_inshop to finish ──────────────
TIER1_PID=93908
if kill -0 $TIER1_PID 2>/dev/null; then
    echo "[$(timestamp)] Step 2: Waiting for Tier 1 deepfashion_inshop (PID $TIER1_PID)..."
    while kill -0 $TIER1_PID 2>/dev/null; do
        sleep 30
    done
    echo "[$(timestamp)] Tier 1 deepfashion_inshop completed."
else
    echo "[$(timestamp)] Tier 1 deepfashion_inshop already finished."
fi

# ── Step 3: Run Tier 1 on remaining datasets ──────────────────────────
echo ""
echo "[$(timestamp)] Step 3: Running Tier 1 cross-check on remaining datasets..."
for ds in deepfashion_multimodal fashion200k atlas polyvore; do
    echo "[$(timestamp)] → Tier 1: fashion-clip-ft × $ds"
    python3 benchmark/eval_marqo_7dataset.py \
        --models fashion-clip-ft \
        --datasets "$ds" \
        --device mps \
        --overwrite \
        2>&1 | tee -a "$TIER1_LOG/tier1_finetuned_${ds}.log"
    echo "[$(timestamp)] ✓ Done: $ds"
done

# ── Step 4: Collect all Tier 1 results and build leaderboard ──────────
echo ""
echo "[$(timestamp)] Step 4: Collecting Tier 1 results..."
python3 benchmark/eval_marqo_7dataset.py \
    --collect_only \
    --models fashion-clip fashion-siglip fashion-clip-ft \
    2>&1 | tee "$TIER1_LOG/tier1_final_leaderboard.log"

# ── Step 5: Compile Phase 3 summary ──────────────────────────────────
echo ""
echo "[$(timestamp)] Step 5: Compiling Phase 3 summary..."

python3 -c "
import json
from pathlib import Path

repo = Path('$REPO')
summary = {'phase': 3, 'status': 'complete'}

# Phase 3.9 results
p39 = repo / 'results' / 'real' / 'phase3_9_comprehensive_eval.json'
if p39.exists():
    with open(p39) as f:
        summary['phase3_9'] = json.load(f)
    print('✓ Phase 3.9 comprehensive eval loaded')
else:
    print('✗ Phase 3.9 results not found')

# Phase 3 combined eval (full pipeline with BM25+NER)
pc = repo / 'results' / 'real' / 'phase3_combined_eval.json'
if pc.exists():
    with open(pc) as f:
        summary['phase3_full_pipeline'] = json.load(f)
    print('✓ Phase 3 full pipeline eval loaded')

# Tier 1 results
t1 = repo / 'results' / 'tier1' / 'tier1_raw_results.json'
if t1.exists():
    with open(t1) as f:
        summary['tier1'] = json.load(f)
    print('✓ Tier 1 results loaded')

# Fused item tower eval
ft = repo / 'results' / 'real' / 'phase3_fused_item_tower_eval.json'
if ft.exists():
    with open(ft) as f:
        summary['fused_item_tower'] = json.load(f)
    print('✓ Fused item tower eval loaded')

# Save summary
out = repo / 'results' / 'real' / 'phase3_final_summary.json'
with open(out, 'w') as f:
    json.dump(summary, f, indent=2)
print(f'Summary saved → {out}')

# Print highlights
print()
print('=' * 70)
print('PHASE 3 FINAL RESULTS SUMMARY')
print('=' * 70)

if 'phase3_9' in summary:
    print()
    print('Phase 3.9 — New Components (Path A + Path B):')
    for name, res in summary['phase3_9'].items():
        if isinstance(res, dict) and 'metrics' in res:
            m = res['metrics']
            print(f'  {name:<35} nDCG@10={m.get(\"ndcg@10\",0):.4f}  MRR={m.get(\"mrr\",0):.4f}  R@10={m.get(\"recall@10\",0):.4f}')

if 'phase3_full_pipeline' in summary:
    print()
    print('Phase 3 — Full Pipeline (BM25+NER+Dense+CE):')
    configs = summary['phase3_full_pipeline'].get('configs', {})
    for name, res in configs.items():
        m = res.get('metrics', {})
        print(f'  {name:<35} nDCG@10={m.get(\"ndcg@10\",0):.4f}  MRR={m.get(\"mrr\",0):.4f}')

if 'tier1' in summary:
    print()
    print('Tier 1 — Marqo 7-Dataset Cross-Check:')
    for r in summary['tier1']:
        model = r.get('model', '?')
        tasks = r.get('tasks', {})
        t2i = tasks.get('text-to-image', {})
        if t2i:
            print(f'  {model:<30} R@1={t2i.get(\"Recall@1\",0):.3f}  R@10={t2i.get(\"Recall@10\",0):.3f}  MRR={t2i.get(\"MRR\",0):.3f}  [{r.get(\"dataset\",\"?\")}]')

print()
print('=' * 70)
"

echo ""
echo "============================================================"
echo "MODA Phase 3 — All tasks complete!"
echo "Finished: $(timestamp)"
echo "============================================================"
