"""
MODA Phase 4B — H&M Article Image Embedding

Encodes all H&M product images with FashionCLIP's vision encoder and
builds a FAISS index for text-to-image retrieval.

The resulting index lives alongside the text embeddings:
  data/processed/embeddings/fashion-clip-visual_faiss.index
  data/processed/embeddings/fashion-clip-visual_article_ids.json

Text queries encoded with FashionCLIP's text encoder can be searched
against this image index because CLIP aligns text and image in the
same embedding space.

Usage:
  python benchmark/embed_hnm_images.py
  python benchmark/embed_hnm_images.py --batch_size 128
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from benchmark.models import load_clip_model, encode_images_clip
from benchmark.embed_hnm import build_faiss_index, save_faiss_index

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

IMAGE_DIR = _REPO_ROOT / "data" / "raw" / "hnm_images"
ARTICLES_CSV = _REPO_ROOT / "data" / "raw" / "hnm_real" / "articles.csv"
OUTPUT_DIR = _REPO_ROOT / "data" / "processed" / "embeddings"
MODEL_NAME = "fashion-clip"
SAFE_NAME = "fashion-clip-visual"


def collect_image_paths() -> tuple[list[str], list[str]]:
    """Map article_ids to image paths, returning only those with images."""
    import csv

    article_ids = []
    image_paths = []

    with open(ARTICLES_CSV, newline="") as f:
        for row in csv.DictReader(f):
            aid = row.get("article_id", "").strip()
            if not aid:
                continue
            aid_padded = aid.zfill(10)
            prefix = aid_padded[:3]
            img_path = IMAGE_DIR / prefix / f"{aid_padded}.jpg"
            if img_path.exists():
                article_ids.append(aid)
                image_paths.append(str(img_path))

    log.info("Found %d images out of articles.csv entries", len(article_ids))
    return article_ids, image_paths


def main():
    p = argparse.ArgumentParser(description="Embed H&M product images with FashionCLIP")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--device", default="mps")
    args = p.parse_args()

    import torch
    if args.device == "mps" and not torch.backends.mps.is_available():
        args.device = "cpu"

    log.info("=" * 60)
    log.info("Phase 4B — Image Embedding with FashionCLIP Vision Encoder")
    log.info("=" * 60)

    article_ids, image_paths = collect_image_paths()
    if not image_paths:
        log.error("No images found in %s. Run scripts/download_hnm_images.py first.", IMAGE_DIR)
        return

    log.info("Loading FashionCLIP model...")
    model, preprocess, tokenizer = load_clip_model(MODEL_NAME, device=args.device)

    log.info("Encoding %d images (batch=%d, device=%s)...",
             len(image_paths), args.batch_size, args.device)
    t0 = time.time()
    embeddings = encode_images_clip(
        image_paths, model, preprocess, args.device,
        batch_size=args.batch_size, normalize=True,
    )
    elapsed = time.time() - t0
    log.info("Encoding complete: %s in %.1f min (%.0f img/s)",
             embeddings.shape, elapsed / 60, len(image_paths) / elapsed)

    del model
    if args.device == "mps":
        torch.mps.empty_cache()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    emb_path = OUTPUT_DIR / f"{SAFE_NAME}_embeddings.npy"
    np.save(emb_path, embeddings)
    log.info("Embeddings saved → %s", emb_path)

    ids_path = OUTPUT_DIR / f"{SAFE_NAME}_article_ids.json"
    with open(ids_path, "w") as f:
        json.dump(article_ids, f)
    log.info("Article IDs saved → %s", ids_path)

    log.info("Building FAISS index...")
    index = build_faiss_index(embeddings)
    faiss_path = OUTPUT_DIR / f"{SAFE_NAME}_faiss.index"
    save_faiss_index(index, faiss_path)

    meta = {
        "model": MODEL_NAME,
        "encoder": "vision",
        "n_images": len(image_paths),
        "embed_dim": int(embeddings.shape[1]),
        "batch_size": args.batch_size,
        "device": args.device,
        "elapsed_seconds": round(elapsed, 1),
        "throughput_img_per_sec": round(len(image_paths) / elapsed, 1),
    }
    meta_path = OUTPUT_DIR / f"{SAFE_NAME}_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n{'=' * 60}")
    print("Phase 4B — Image Embedding Complete")
    print(f"{'=' * 60}")
    print(f"  Images encoded: {len(image_paths):,}")
    print(f"  Embed dim:      {embeddings.shape[1]}")
    print(f"  Time:           {elapsed / 60:.1f} min ({len(image_paths) / elapsed:.0f} img/s)")
    print(f"  FAISS index:    {faiss_path}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
