#!/bin/bash
# MODA Phase 4 — Overnight Orchestration Script
#
# Runs the complete Phase 4 pipeline sequentially:
#   4B: Image embedding (skip if already done)
#   4D: Zero-shot multimodal evaluation (baseline + finetuned text)
#   4E: LLM labels for image hard negatives
#   4F: Joint text+image fine-tuning
#   4G: Fine-tuned multimodal evaluation
#
# Usage:
#   nohup bash scripts/run_phase4_overnight.sh &> results/real/phase4_overnight.log &

set -eo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

LOG_DIR="$REPO_ROOT/results/real"
EMBED_DIR="$REPO_ROOT/data/processed/embeddings"

log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S')  [PHASE4]  $*"
}

log "============================================================"
log "MODA Phase 4 — Overnight Pipeline"
log "Started at $(date)"
log "============================================================"

# ─── 4B: Image Embedding (skip if FAISS index exists) ─────────────────────
if [ -f "$EMBED_DIR/fashion-clip-visual_faiss.index" ]; then
    log "4B SKIP: Image FAISS index already exists"
else
    log "4B START: Embedding 105K images with FashionCLIP vision encoder..."
    python benchmark/embed_hnm_images.py --batch_size 64 --device mps \
        2>&1 | tee "$LOG_DIR/phase4b_image_embed.log"
    log "4B DONE: Image embedding complete"
fi

# Verify FAISS index exists before proceeding
if [ ! -f "$EMBED_DIR/fashion-clip-visual_faiss.index" ]; then
    log "4B FAILED: Image FAISS index not found. Aborting."
    exit 1
fi

# ─── 4D: Zero-Shot Multimodal Evaluation ──────────────────────────────────
log "4D START: Zero-shot multimodal evaluation (baseline text encoder)..."
python benchmark/eval_multimodal_pipeline.py \
    --image_weight 0.3 \
    2>&1 | tee "$LOG_DIR/phase4d_multimodal_baseline.log"
log "4D-baseline DONE"

log "4D START: Zero-shot multimodal evaluation (fine-tuned text encoder)..."
python benchmark/eval_multimodal_pipeline.py \
    --image_weight 0.3 \
    --use_finetuned_text \
    2>&1 | tee "$LOG_DIR/phase4d_multimodal_finetuned.log"
log "4D-finetuned DONE"

# ─── 4E: LLM Labels for Image Hard Negatives ─────────────────────────────
if [ -z "$PALEBLUEDOT_API_KEY" ]; then
    log "4E SKIP: PALEBLUEDOT_API_KEY not set"
else
    if [ -f "$REPO_ROOT/data/processed/image_retriever_labels.jsonl" ]; then
        EXISTING=$(wc -l < "$REPO_ROOT/data/processed/image_retriever_labels.jsonl")
        log "4E: Found $EXISTING existing labels, will resume..."
    fi
    log "4E START: Generating LLM labels for image hard negatives..."
    python benchmark/generate_image_labels.py \
        --max_queries 5000 \
        --top_k 20 \
        --concurrency 30 \
        2>&1 | tee "$LOG_DIR/phase4e_image_labels.log"
    log "4E DONE: Image hard negative labels generated"

    # Show distribution
    python benchmark/generate_image_labels.py --report \
        2>&1 | tee -a "$LOG_DIR/phase4e_image_labels.log"
fi

# ─── 4F: Joint Text+Image Fine-Tuning ────────────────────────────────────
LABELS_FILE="$REPO_ROOT/data/processed/image_retriever_labels.jsonl"
if [ ! -f "$LABELS_FILE" ]; then
    log "4F SKIP: No image labels found (4E may have been skipped)"
else
    log "4F START: Joint text+image fine-tuning..."
    python benchmark/train_multimodal.py \
        --epochs 5 \
        --batch_size 32 \
        --grad_accum 4 \
        --lr 5e-7 \
        --align_weight 0.1 \
        2>&1 | tee "$LOG_DIR/phase4f_multimodal_train.log"
    log "4F DONE: Joint fine-tuning complete"
fi

# ─── 4G: Fine-Tuned Multimodal Evaluation ────────────────────────────────
MULTIMODAL_MODEL="$REPO_ROOT/models/moda-fashionclip-multimodal/best"
if [ ! -d "$MULTIMODAL_MODEL" ]; then
    log "4G SKIP: No multimodal model found (4F may have been skipped)"
else
    log "4G START: Fine-tuned multimodal evaluation..."

    # Re-embed images with fine-tuned vision encoder
    log "4G: Re-embedding images with fine-tuned model..."
    python -c "
import sys, json, time, torch, numpy as np
from pathlib import Path
sys.path.insert(0, '.')
import open_clip
from benchmark.models import encode_images_clip
from benchmark.embed_hnm import build_faiss_index, save_faiss_index
from benchmark.embed_hnm_images import collect_image_paths

EMBED_DIR = Path('data/processed/embeddings')
MODEL_PATH = Path('models/moda-fashionclip-multimodal/best')

print('Loading fine-tuned multimodal model...')
model, preprocess, _ = open_clip.create_model_and_transforms('hf-hub:Marqo/marqo-fashionCLIP')
state = torch.load(MODEL_PATH / 'model_state_dict.pt', map_location='cpu')
model.load_state_dict(state, strict=False)
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
model = model.to(device).eval()

article_ids, image_paths = collect_image_paths()
print(f'Encoding {len(image_paths)} images...')
t0 = time.time()
embs = encode_images_clip(image_paths, model, preprocess, device, batch_size=64)
elapsed = time.time() - t0
print(f'Done in {elapsed/60:.1f} min')

np.save(str(EMBED_DIR / 'fashion-clip-multimodal-visual_embeddings.npy'), embs)
with open(EMBED_DIR / 'fashion-clip-multimodal-visual_article_ids.json', 'w') as f:
    json.dump(article_ids, f)
idx = build_faiss_index(embs)
import faiss
faiss.write_index(idx, str(EMBED_DIR / 'fashion-clip-multimodal-visual_faiss.index'))
print('Fine-tuned image FAISS index saved')
" 2>&1 | tee "$LOG_DIR/phase4g_reembed.log"

    # Run evaluation with fine-tuned multimodal model
    # (We need a variant of eval_multimodal_pipeline that uses the new index)
    log "4G: Running multimodal eval with fine-tuned model..."
    python -c "
import sys, json
from pathlib import Path
sys.path.insert(0, '.')

# Temporarily swap the image index paths to use fine-tuned model's index
import benchmark.eval_multimodal_pipeline as emp
emp.IMAGE_FAISS_PATH = Path('data/processed/embeddings/fashion-clip-multimodal-visual_faiss.index')
emp.IMAGE_IDS_PATH = Path('data/processed/embeddings/fashion-clip-multimodal-visual_article_ids.json')

# Also set text encoder to fine-tuned multimodal
sys.argv = ['eval', '--use_finetuned_text', '--image_weight', '0.3']

# Patch finetuned path to multimodal model
emp.FINETUNED_BIENC = Path('models/moda-fashionclip-multimodal/best')
emp.main()
" 2>&1 | tee "$LOG_DIR/phase4g_multimodal_eval.log"

    log "4G DONE: Fine-tuned multimodal evaluation complete"
fi

# ─── Summary ─────────────────────────────────────────────────────────────
log "============================================================"
log "MODA Phase 4 — Overnight Pipeline Complete"
log "Finished at $(date)"
log "============================================================"
log ""
log "Results saved in $LOG_DIR/phase4*.log"
log "Check:"
log "  phase4b_image_embed.log      — Image embedding"
log "  phase4d_multimodal_*.log     — Zero-shot multimodal eval"
log "  phase4e_image_labels.log     — LLM label generation"
log "  phase4f_multimodal_train.log — Joint fine-tuning"
log "  phase4g_multimodal_eval.log  — Fine-tuned eval"
