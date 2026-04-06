#!/usr/bin/env bash
# Phase 4G (overnight naming): multimodal re-embed + full pipeline eval using the
# Phase 4F checkpoint at models/moda-fashionclip-multimodal/best/
#
# Requires: 4F must have saved model_state_dict.pt (waits until it exists).
#
# Usage:
#   bash scripts/run_phase4g_multimodal_eval.sh
#   nohup bash scripts/run_phase4g_multimodal_eval.sh >> results/real/phase4g_runner.log 2>&1 &

set -eo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

LOG_DIR="$REPO_ROOT/results/real"
EMBED_DIR="$REPO_ROOT/data/processed/embeddings"
CKPT="$REPO_ROOT/models/moda-fashionclip-multimodal/best/model_state_dict.pt"

log() {
  echo "$(date '+%Y-%m-%d %H:%M:%S')  [PHASE4G]  $*"
}

log "Waiting for 4F checkpoint: $CKPT"
while [ ! -f "$CKPT" ]; do
  sleep 60
  log "Still waiting for 4F (best checkpoint)..."
done
log "Found 4F checkpoint; starting 4G re-embed + eval."

mkdir -p "$LOG_DIR" "$EMBED_DIR"

log "4G: Re-embedding images with fine-tuned multimodal model..."
python -c "
import sys, json, time, torch, numpy as np
from pathlib import Path
sys.path.insert(0, '.')
import open_clip
from benchmark.models import encode_images_clip
from benchmark.embed_hnm import build_faiss_index
from benchmark.embed_hnm_images import collect_image_paths
import faiss

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
faiss.write_index(idx, str(EMBED_DIR / 'fashion-clip-multimodal-visual_faiss.index'))
print('Fine-tuned image FAISS index saved')
" 2>&1 | tee "$LOG_DIR/phase4g_reembed.log"

log "4G: Running multimodal pipeline eval (fine-tuned text + fine-tuned image index)..."
python -c "
import sys
from pathlib import Path
sys.path.insert(0, '.')

import benchmark.eval_multimodal_pipeline as emp
emp.IMAGE_FAISS_PATH = Path('data/processed/embeddings/fashion-clip-multimodal-visual_faiss.index')
emp.IMAGE_IDS_PATH = Path('data/processed/embeddings/fashion-clip-multimodal-visual_article_ids.json')
sys.argv = ['eval', '--use_finetuned_text', '--image_weight', '0.3']
emp.FINETUNED_BIENC = Path('models/moda-fashionclip-multimodal/best')
emp.main()
" 2>&1 | tee "$LOG_DIR/phase4g_multimodal_eval.log"

log "4G DONE. Logs: $LOG_DIR/phase4g_reembed.log , $LOG_DIR/phase4g_multimodal_eval.log"
