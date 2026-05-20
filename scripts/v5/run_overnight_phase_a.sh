#!/bin/bash
# Phase A overnight orchestrator.
#
# Sequencing (assumes phase_a_extract_multifield.py and
# phase_a_cache_student_image_emb.py are already running in the background):
#
#   1. Wait for student image cache  (~30 min on MPS)
#   2. Run teacher cache              (~50 min on MPS)
#   3. Run anchor indices             (instant)
#   4. Run leakage check              (~5 min, network-bound)
#   5. Wait for label extraction      (≤ 9 hr)
#   6. Print final status
#
# Caffeinate is wrapped around the whole thing so the laptop doesn't sleep.

set -u
cd /Users/rohit.anand/Desktop/Hobby/MODA
mkdir -p logs

REPO=/Users/rohit.anand/Desktop/Hobby/MODA
DATA=$REPO/data/processed/v5_multifield
LOGS=$REPO/logs

stamp() { date +"%Y-%m-%d %H:%M:%S"; }
log_event() { echo "[$(stamp)] $*" | tee -a "$LOGS/overnight_phase_a.log"; }

source .venv/bin/activate

log_event "overnight orchestrator started"

# ---------------------------------------------------------------------------
# Step 1 — wait for student image cache
# ---------------------------------------------------------------------------
log_event "waiting for student_image_emb.pt to appear ..."
while [ ! -f "$DATA/student_image_emb.pt" ]; do sleep 60; done
log_event "student image cache present"
# Sanity wait — make sure the file isn't still being written
sleep 30

# ---------------------------------------------------------------------------
# Step 2 — teacher cache (FSL image+text, SL2 text)
# ---------------------------------------------------------------------------
log_event "starting teacher cache"
caffeinate -i python scripts/v5/phase_a_cache_teacher_emb.py \
    > "$LOGS/v5_teacher_cache.log" 2>&1
status=$?
if [ $status -ne 0 ]; then
    log_event "teacher cache FAILED (exit $status). See $LOGS/v5_teacher_cache.log"
    exit 1
fi
log_event "teacher cache done"

# ---------------------------------------------------------------------------
# Step 3 — anchor indices (depends on student_image_index.json from step 1)
# ---------------------------------------------------------------------------
log_event "starting anchor indices"
python scripts/v5/phase_a_anchor_indices.py > "$LOGS/v5_anchor.log" 2>&1
log_event "anchor indices done"

# ---------------------------------------------------------------------------
# Step 4 — leakage check (independent of all the above; only needs pairs_50k.jsonl)
# ---------------------------------------------------------------------------
log_event "starting leakage check"
python scripts/v5/phase_a_leakage_check.py > "$LOGS/v5_leakage.log" 2>&1
status=$?
if [ $status -ne 0 ]; then
    log_event "leakage check FLAGGED OVERLAPS — review $LOGS/v5_leakage.log"
else
    log_event "leakage check clean"
fi

# ---------------------------------------------------------------------------
# Step 5 — wait for label extraction to finish
# ---------------------------------------------------------------------------
log_event "waiting for label extraction to finish ..."
while ! grep -q "Done in" "$LOGS/v5_label_50k.log" 2>/dev/null; do
    sleep 120
done
log_event "label extraction done"

# ---------------------------------------------------------------------------
# Final report
# ---------------------------------------------------------------------------
log_event "PHASE A COMPLETE — final inventory:"
ls -lh "$DATA"/*.pt "$DATA"/*.json "$DATA"/*.jsonl 2>&1 | tee -a "$LOGS/overnight_phase_a.log"

label_count=$(wc -l < "$DATA/pairs_50k_labeled.jsonl" 2>/dev/null || echo 0)
log_event "labeled records: $label_count / 50000"
