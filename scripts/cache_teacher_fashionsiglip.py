"""
Cache Marqo-FashionSigLIP teacher embeddings (image + text) on a Marqo-GS-10M
subset, so the distillation loop can compute the loss without a per-step
teacher forward pass.

Why this matters:
  - Per-step teacher forward = ~25% of total wall time on MPS.
  - Caching once = deterministic & reproducible distillation.
  - Cache size: 10k items * (768d image + 768d text) * fp16 = ~31 MB. Tiny.

Inputs:
  - data/processed/marqo_gs_wfash_subset/triplets.jsonl  (already built)

Outputs:
  - data/processed/distillation_cache/teacher_embeddings.pt
      dict { "image": Tensor[N, 768], "text": Tensor[N, 768],
             "queries": list[str], "image_paths": list[str] }
  - data/processed/distillation_cache/meta.json

Usage:
  .venv/bin/python scripts/cache_teacher_fashionsiglip.py \
      --triplets data/processed/marqo_gs_wfash_subset/triplets.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("cache-teacher")

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
HF_CACHE = REPO / "data" / "hf_cache"
OUT_DIR = REPO / "data" / "processed" / "distillation_cache"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--triplets", required=True)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--limit", type=int, default=None, help="Cap rows for smoke testing")
    p.add_argument(
        "--teacher",
        default="hf-hub:Marqo/marqo-fashionSigLIP",
        help="open_clip identifier for the teacher",
    )
    args = p.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "teacher_embeddings.pt"
    meta_path = OUT_DIR / "meta.json"

    if out_path.exists():
        log.info("cache already exists at %s — delete to recompute", out_path)
        return

    import open_clip
    from PIL import Image

    log.info("loading teacher: %s", args.teacher)
    model, _, preprocess = open_clip.create_model_and_transforms(
        args.teacher, cache_dir=str(HF_CACHE),
    )
    tokenizer = open_clip.get_tokenizer(args.teacher)
    model.eval().to(DEVICE)
    for p_ in model.parameters():
        p_.requires_grad = False

    # Load triplets
    rows: list[dict] = []
    with open(args.triplets) as f:
        for line in f:
            rows.append(json.loads(line))
    if args.limit:
        rows = rows[: args.limit]
    log.info("loaded %d rows from %s", len(rows), args.triplets)

    # Collate batches
    queries: list[str] = []
    image_paths: list[str] = []
    img_feats: list[torch.Tensor] = []
    txt_feats: list[torch.Tensor] = []

    t0 = time.time()
    n_done = 0
    n_failed = 0
    with torch.no_grad():
        batch_imgs: list[torch.Tensor] = []
        batch_qs: list[str] = []
        batch_paths: list[str] = []

        def flush():
            nonlocal n_done
            if not batch_imgs:
                return
            imgs = torch.stack(batch_imgs).to(DEVICE)
            tokens = tokenizer(batch_qs).to(DEVICE)
            ifeat = model.encode_image(imgs)
            tfeat = model.encode_text(tokens)
            img_feats.append(ifeat.float().cpu())
            txt_feats.append(tfeat.float().cpu())
            queries.extend(batch_qs)
            image_paths.extend(batch_paths)
            n_done += len(batch_imgs)
            batch_imgs.clear()
            batch_qs.clear()
            batch_paths.clear()

        for i, r in enumerate(rows):
            try:
                img = Image.open(r["image_path"]).convert("RGB")
                batch_imgs.append(preprocess(img))
                batch_qs.append(r["query"])
                batch_paths.append(r["image_path"])
            except Exception as e:
                n_failed += 1
                continue
            if len(batch_imgs) >= args.batch_size:
                flush()
                if n_done % 500 == 0 or n_done == args.batch_size:
                    rate = n_done / max(time.time() - t0, 1e-6)
                    log.info("  encoded %d/%d (%.1f items/s, failed=%d)", n_done, len(rows), rate, n_failed)
        flush()

    img_feats_t = torch.cat(img_feats, dim=0)
    txt_feats_t = torch.cat(txt_feats, dim=0)
    log.info(
        "done. image=%s text=%s in %.1fs (failed=%d)",
        tuple(img_feats_t.shape), tuple(txt_feats_t.shape), time.time() - t0, n_failed,
    )

    # Save in fp16 to halve disk
    payload = {
        "image": img_feats_t.half(),
        "text": txt_feats_t.half(),
        "queries": queries,
        "image_paths": image_paths,
        "teacher": args.teacher,
        "embed_dim": img_feats_t.shape[-1],
    }
    tmp = out_path.with_suffix(".pt.tmp")
    torch.save(payload, tmp)
    tmp.replace(out_path)
    meta_path.write_text(json.dumps({
        "n": len(queries),
        "teacher": args.teacher,
        "embed_dim": int(img_feats_t.shape[-1]),
        "n_failed": n_failed,
        "wall_time_sec": time.time() - t0,
    }, indent=2))
    log.info("cached -> %s (%.1f MB)", out_path, out_path.stat().st_size / 1e6)


if __name__ == "__main__":
    main()
