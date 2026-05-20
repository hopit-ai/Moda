#!/bin/bash
# Recovery orchestrator: v4-alt + v4 using existing SL2-L caches and 12,979 v4 prose pairs.
# Skips re-doing prose-gen (API auth had 401s) and re-caching SL2-L (already done).

set -u
cd /Users/rohit.anand/Desktop/Hobby/MODA
mkdir -p logs results/v5/v4 results/v5/v4_alt

REPO=/Users/rohit.anand/Desktop/Hobby/MODA
DATA=$REPO/data/processed/v5_multifield
LOGS=$REPO/logs

stamp() { date +"%Y-%m-%d %H:%M:%S"; }
log_event() { echo "[$(stamp)] $*" | tee -a "$LOGS/run_v4_recovery.log"; }

source .venv/bin/activate

log_event "v4 recovery orchestrator started"

# Sanity-check artifacts
for f in pairs_v2_combined.jsonl v2_image_index.json \
         teacher_sl2l_img_emb.pt teacher_sl2l_text_emb.pt \
         pairs_v4_long_desc.jsonl; do
    if [ ! -f "$DATA/$f" ]; then
        log_event "MISSING: $DATA/$f"
        exit 1
    fi
done
log_event "all artifacts present"

# ---------------------------------------------------------------------------
# 1. v4-alt Phase D — same v3 recipe, SL2-L teacher
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
    log_event "Phase D v4-alt FAILED (exit $status). See $LOGS/v5_phase_d_v4_alt.log"
    exit 1
fi
log_event "Phase D v4-alt complete"

# v4-alt Phase E
log_event "starting Phase E v4-alt"
caffeinate -i python scripts/v5/phase_e_eval.py \
    --ckpt "$REPO/checkpoints/v5/phase_d_best_v4_alt.pt" \
    --out_dir "$REPO/results/v5/v4_alt" \
    > "$LOGS/v5_phase_e_v4_alt.log" 2>&1
log_event "Phase E v4-alt complete — see results/v5/v4_alt/phase_e_summary.md"

# ---------------------------------------------------------------------------
# 2. v4 prep — combine 50K labeled + v2 prose + v4 prose (whatever we got)
# ---------------------------------------------------------------------------
log_event "preparing v4 combined dataset (50K + v2_prose + v4_prose)"
python -c "
import json
from pathlib import Path
DATA = Path('$DATA')
out_pairs = DATA / 'pairs_v4_combined.jsonl'
out_index = DATA / 'v4_image_index.json'

base_index = json.loads((DATA / 'student_image_index.json').read_text())

n_base = n_v2 = n_v4 = 0
with out_pairs.open('w') as out:
    base_path = DATA / 'pairs_50k_labeled.jsonl'
    if not base_path.exists(): base_path = DATA / 'pairs_50k.jsonl'
    for line in base_path.open():
        line = line.strip()
        if line: out.write(line + '\n'); n_base += 1
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
    exit 1
fi
log_event "v4 prep done"

# ---------------------------------------------------------------------------
# 3. v4 Phase D + E
# ---------------------------------------------------------------------------
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

log_event "starting Phase E v4"
caffeinate -i python scripts/v5/phase_e_eval.py \
    --ckpt "$REPO/checkpoints/v5/phase_d_best_v4.pt" \
    --out_dir "$REPO/results/v5/v4" \
    > "$LOGS/v5_phase_e_v4.log" 2>&1
log_event "Phase E v4 complete"

log_event "ALL V4 + V4-ALT COMPLETE — see results/v5/v4/phase_e_summary.md and results/v5/v4_alt/phase_e_summary.md"
