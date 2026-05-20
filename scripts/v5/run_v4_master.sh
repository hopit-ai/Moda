#!/bin/bash
# v4 + v4-alt master orchestrator.
#
# Parallel start (no MPS conflict):
#   - v4 prose-gen (API only, ~3-5h)
#   - v4-alt SL2-L teacher caching (MPS, ~75 min)
#
# Then serial MPS-bound:
#   - v4-alt Phase D + E (using SL2-L teacher)        ~1h
#   - wait for v4 prose-gen to finish if not done
#   - v4 prep (CPU, instant)
#   - v4 Phase D + E (using more prose data, KL=1.0 constant) ~1.5h

set -u
cd /Users/rohit.anand/Desktop/Hobby/MODA
mkdir -p logs results/v5/v4 results/v5/v4_alt

REPO=/Users/rohit.anand/Desktop/Hobby/MODA
DATA=$REPO/data/processed/v5_multifield
LOGS=$REPO/logs

stamp() { date +"%Y-%m-%d %H:%M:%S"; }
log_event() { echo "[$(stamp)] $*" | tee -a "$LOGS/run_v4_master.log"; }

source .venv/bin/activate

log_event "v4 master orchestrator started"

# ---------------------------------------------------------------------------
# 1. Kick off v4 prose-gen (API, no MPS) in background
# ---------------------------------------------------------------------------
log_event "starting v4 prose-gen (additional 25K with seed 1338, output to pairs_v4_long_desc.jsonl)"
caffeinate -i python scripts/v5/v2_generate_long_desc.py \
    --output "$DATA/pairs_v4_long_desc.jsonl" \
    --n 25000 --workers 8 --batch_size 25 --timeout 240 --seed 1338 \
    > "$LOGS/v4_long_desc.log" 2>&1 &
PROSE_PID=$!
log_event "  prose-gen PID $PROSE_PID"

# ---------------------------------------------------------------------------
# 2. Build SL2-L teacher caches (MPS, ~75 min)
# ---------------------------------------------------------------------------
log_event "starting SL2-L teacher caching"
caffeinate -i python scripts/v5/phase_a_cache_sl2l_teacher.py \
    > "$LOGS/v4_alt_sl2l_cache.log" 2>&1
status=$?
if [ $status -ne 0 ]; then
    log_event "SL2-L caching FAILED (exit $status). See $LOGS/v4_alt_sl2l_cache.log"
    kill $PROSE_PID 2>/dev/null
    exit 1
fi
log_event "SL2-L teacher caches ready"

# ---------------------------------------------------------------------------
# 3. v4-alt Phase D — same v3 recipe but with SL2-L teacher
# ---------------------------------------------------------------------------
log_event "starting Phase D v4-alt (SL2-L fusion teacher)"
caffeinate -i python scripts/v5/phase_d_train_full.py \
    --pairs "$DATA/pairs_v2_combined.jsonl" \
    --image_index_path "$DATA/v2_image_index.json" \
    --scope text_1block \
    --gcl_lambda 0.0 \
    --use_multifield 0 \
    --anchor_lambda_init 0.0 --anchor_lambda_final 0.0 \
    --kl_lambda_init 1.0 --kl_lambda_final 0.5 \
    --lr 1e-6 \
    --sl2_text_cache_override "$DATA/teacher_sl2l_text_emb.pt" \
    --sl2_img_cache_override "$DATA/teacher_sl2l_img_emb.pt" \
    --run_tag v4_alt \
    --epochs 3 --probe_every 500 --ckpt_every 1000 \
    > "$LOGS/v5_phase_d_v4_alt.log" 2>&1
status=$?
if [ $status -ne 0 ]; then
    log_event "Phase D v4-alt FAILED (exit $status)"
    kill $PROSE_PID 2>/dev/null
    exit 1
fi
log_event "Phase D v4-alt complete"

# ---------------------------------------------------------------------------
# 4. v4-alt Phase E
# ---------------------------------------------------------------------------
log_event "starting Phase E v4-alt"
caffeinate -i python scripts/v5/phase_e_eval.py \
    --ckpt "$REPO/checkpoints/v5/phase_d_best_v4_alt.pt" \
    --out_dir "$REPO/results/v5/v4_alt" \
    > "$LOGS/v5_phase_e_v4_alt.log" 2>&1
log_event "Phase E v4-alt complete"

# ---------------------------------------------------------------------------
# 5. Wait for prose-gen if still running
# ---------------------------------------------------------------------------
log_event "waiting for v4 prose-gen if still running ..."
while ! grep -q "Done in" "$LOGS/v4_long_desc.log" 2>/dev/null; do sleep 60; done
n_synth=$(wc -l < "$DATA/pairs_v4_long_desc.jsonl" 2>/dev/null || echo 0)
log_event "v4 prose-gen done — $n_synth additional pairs"

# ---------------------------------------------------------------------------
# 6. v4 prep — combine 50K labeled + v2 prose (10K) + v4 prose (25K)
# ---------------------------------------------------------------------------
log_event "preparing v4 combined dataset"
python -c "
import json
from pathlib import Path
DATA = Path('$DATA')
out_pairs = DATA / 'pairs_v4_combined.jsonl'
out_index = DATA / 'v4_image_index.json'

base_index = json.loads((DATA / 'student_image_index.json').read_text())

n_base = n_v2 = n_v4 = 0
with out_pairs.open('w') as out:
    # Base 50K labeled
    base_path = DATA / 'pairs_50k_labeled.jsonl'
    if not base_path.exists(): base_path = DATA / 'pairs_50k.jsonl'
    for line in base_path.open():
        line = line.strip()
        if line: out.write(line + '\n'); n_base += 1
    # v2 prose
    for line in (DATA / 'pairs_v2_long_desc.jsonl').open():
        line = line.strip()
        if not line: continue
        r = json.loads(line)
        pid = r['pair_id']
        if '__synth_long' in pid:
            orig = pid.replace('__synth_long', '')
            if orig in base_index:
                base_index[pid] = base_index[orig]
                out.write(line + '\n'); n_v2 += 1
    # v4 prose
    for line in (DATA / 'pairs_v4_long_desc.jsonl').open():
        line = line.strip()
        if not line: continue
        r = json.loads(line)
        pid = r['pair_id']
        if '__synth_long' in pid:
            orig = pid.replace('__synth_long', '')
            if orig in base_index:
                base_index[pid] = base_index[orig]
                out.write(line + '\n'); n_v4 += 1

out_index.write_text(json.dumps(base_index))
print(f'Wrote {out_pairs}: {n_base:,} base + {n_v2:,} v2 + {n_v4:,} v4 = {n_base+n_v2+n_v4:,}')
print(f'Wrote index {out_index}: {len(base_index):,} entries')
" > "$LOGS/v4_prep.log" 2>&1
status=$?
if [ $status -ne 0 ]; then
    log_event "v4 prep FAILED. See $LOGS/v4_prep.log"
    cat "$LOGS/v4_prep.log"
    exit 1
fi
log_event "v4 prep done"

# ---------------------------------------------------------------------------
# 7. v4 Phase D — v3 recipe + extra prose data, KL=1.0 constant, batch_N=24
# ---------------------------------------------------------------------------
# Note: v3 recipe with --kl_lambda_init 1.0 --kl_lambda_final 1.0 (constant)
# and --batch_N 24 for richer in-batch KL signal.
log_event "starting Phase D v4 (extra prose, KL=1.0 constant, batch_N=24)"
caffeinate -i python scripts/v5/phase_d_train_full.py \
    --pairs "$DATA/pairs_v4_combined.jsonl" \
    --image_index_path "$DATA/v4_image_index.json" \
    --scope text_1block \
    --gcl_lambda 0.0 \
    --use_multifield 0 \
    --anchor_lambda_init 0.0 --anchor_lambda_final 0.0 \
    --kl_lambda_init 1.0 --kl_lambda_final 1.0 \
    --lr 1e-6 \
    --batch_K 8 --batch_N 24 \
    --run_tag v4 \
    --epochs 2 --probe_every 500 --ckpt_every 1000 \
    > "$LOGS/v5_phase_d_v4.log" 2>&1
status=$?
if [ $status -ne 0 ]; then
    log_event "Phase D v4 FAILED (exit $status)"
    exit 1
fi
log_event "Phase D v4 complete"

# ---------------------------------------------------------------------------
# 8. v4 Phase E
# ---------------------------------------------------------------------------
log_event "starting Phase E v4"
caffeinate -i python scripts/v5/phase_e_eval.py \
    --ckpt "$REPO/checkpoints/v5/phase_d_best_v4.pt" \
    --out_dir "$REPO/results/v5/v4" \
    > "$LOGS/v5_phase_e_v4.log" 2>&1
log_event "Phase E v4 complete"

log_event "ALL V4 + V4-ALT COMPLETE — see results/v5/v4/phase_e_summary.md and results/v5/v4_alt/phase_e_summary.md"
