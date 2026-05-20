"""
Build FSL eval image caches for the 4 Marqo benchmarks.

Reuses the same 200 queries / 1200 doc_ids / gt from the existing SL2-B caches
(queries and corpus are model-agnostic). Only re-encodes the 1200 images per
benchmark using Marqo-FashionSigLIP's image tower, so FSL-student probing
stays in FSL's image embedding space.

Output: data/processed/v5_eval_cache_fsl/
  {fashion200k,atlas,polyvore,KAGL}_image_emb.pt  — (1200, 768) fp16
  {fashion200k,atlas,polyvore,KAGL}_queries.json   — symlinked/copied from SL2 cache
  {fashion200k,atlas,polyvore,KAGL}_doc_ids.json
  {fashion200k,atlas,polyvore,KAGL}_gt.json

Usage:
    python scripts/v5/phase_a_cache_fsl_eval.py
"""
from __future__ import annotations

import json
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

REPO = Path(__file__).resolve().parents[2]
SL2_CACHE_DIR = REPO / "data" / "processed" / "v5_eval_cache"
FSL_CACHE_DIR = REPO / "data" / "processed" / "v5_eval_cache_fsl"
BENCHMARKS = ["fashion200k", "atlas", "polyvore", "KAGL"]


def pick_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _load_doc_images(hf_id: str, doc_ids: list[str]) -> dict[str, object]:
    """Stream HF dataset and collect PIL images for the doc_ids we need."""
    from datasets import load_dataset
    needed = set(doc_ids)
    images = {}
    ds = None
    for split in ("data", "test"):
        try:
            ds = load_dataset(hf_id, split=split, streaming=True)
            break
        except Exception:
            pass
    if ds is None:
        raise RuntimeError(f"Could not load {hf_id}")

    for i, row in enumerate(ds):
        row_id = str(row.get("product_id", row.get("id", i)))
        if row_id in needed:
            img = row.get("image")
            if img is not None:
                images[row_id] = img
            needed.discard(row_id)
        if not needed:
            break

    return images


BENCHMARKS_HF = {
    "fashion200k": "Marqo/fashion200k",
    "atlas": "Marqo/atlas",
    "polyvore": "Marqo/polyvore",
    "KAGL": "Marqo/KAGL",
}


def main():
    device = pick_device()
    print(f"Device: {device}")

    FSL_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    import open_clip
    print("Loading Marqo-FashionSigLIP ...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        "hf-hub:Marqo/marqo-fashionSigLIP"
    )
    model = model.to(device).eval()
    print("  FSL model loaded")

    for bname in BENCHMARKS:
        out_emb = FSL_CACHE_DIR / f"{bname}_image_emb.pt"
        if out_emb.exists():
            print(f"\n{bname}: image cache already exists — skipping re-encode")
        else:
            print(f"\n{bname}: encoding images with FSL ...")
            doc_ids = json.loads((SL2_CACHE_DIR / f"{bname}_doc_ids.json").read_text())
            hf_id = BENCHMARKS_HF[bname]

            print(f"  streaming {hf_id} to collect {len(doc_ids)} images ...")
            t0 = time.time()
            doc_images = _load_doc_images(hf_id, doc_ids)
            print(f"  collected {len(doc_images)}/{len(doc_ids)} images in {time.time()-t0:.0f}s")

            D = len(doc_ids)
            emb = torch.zeros((D, 768), dtype=torch.float16)
            BATCH = 32
            with torch.inference_mode():
                for j in tqdm(range(0, D, BATCH), desc=f"{bname} encode"):
                    batch_ids = doc_ids[j:j + BATCH]
                    tensors = []
                    for did in batch_ids:
                        img = doc_images.get(did)
                        if img is not None:
                            try:
                                tensors.append(preprocess(img.convert("RGB")))
                            except Exception:
                                tensors.append(torch.zeros(3, 224, 224))
                        else:
                            tensors.append(torch.zeros(3, 224, 224))
                    stack = torch.stack(tensors).to(device)
                    e = F.normalize(model.encode_image(stack), dim=-1)
                    emb[j:j + len(batch_ids)] = e.detach().cpu().to(torch.float16)

            torch.save(emb, out_emb)
            print(f"  saved {tuple(emb.shape)} → {out_emb}")

        # Copy model-agnostic files (queries, doc_ids, gt) from SL2 cache
        for suffix in ("queries.json", "doc_ids.json", "gt.json"):
            dst = FSL_CACHE_DIR / f"{bname}_{suffix}"
            src = SL2_CACHE_DIR / f"{bname}_{suffix}"
            if not dst.exists() and src.exists():
                shutil.copy2(src, dst)
                print(f"  copied {suffix}")

    print("\nDone. FSL eval caches ready at:", FSL_CACHE_DIR)


if __name__ == "__main__":
    main()
