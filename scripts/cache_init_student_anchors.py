"""
Cache init-student (open_clip ViT-B-16-SigLIP2-384/webli) embeddings for the
anchor loss in Path 2 distillation training.

Why this exists:
  Holding a frozen-init copy of a 375M-param model on MPS (alongside the
  trainable student + the K=15 image batches + probe corpus tensors) blows
  through Mac unified memory and stalls the process. We don't need it: the
  anchor loss only ever needs init embeddings for (a) every unique training
  query's text and (b) every unique training image (positives + hardnegs).
  That's at most 421 + 5000 = 5421 embeddings — encode once, save to disk.

Output: data/processed/path2/init_anchor_cache.pt
  {
    "img": [N_img, 768] L2-normed fp32  CPU tensor
    "txt": [N_txt, 768] L2-normed fp32  CPU tensor
    "image_paths": list[str]            # row order matches img tensor
    "queries":     list[str]            # row order matches txt tensor
    "model": "ViT-B-16-SigLIP2-384",
    "pretrained": "webli",
  }

Usage:
  .venv/bin/python scripts/cache_init_student_anchors.py
"""

from __future__ import annotations

import argparse
import gc
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
log = logging.getLogger("cache-init-anchor")

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
HF_CACHE = REPO / "data" / "hf_cache"


def collect_unique(hardnegs_path: Path) -> tuple[list[str], list[str]]:
    queries: list[str] = []
    seen_q = set()
    images: list[str] = []
    seen_i = set()
    with open(hardnegs_path) as f:
        for line in f:
            r = json.loads(line)
            if r["query"] not in seen_q:
                seen_q.add(r["query"])
                queries.append(r["query"])
            for p in r.get("positives", []):
                if p["image_path"] not in seen_i:
                    seen_i.add(p["image_path"])
                    images.append(p["image_path"])
            for n in r.get("hard_negatives", []):
                if n["image_path"] not in seen_i:
                    seen_i.add(n["image_path"])
                    images.append(n["image_path"])
    log.info("unique queries=%d, unique images=%d", len(queries), len(images))
    return queries, images


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hardnegs", default=str(REPO / "data/processed/path2/hardnegs.jsonl"))
    p.add_argument("--out", default=str(REPO / "data/processed/path2/init_anchor_cache.pt"))
    p.add_argument("--model", default="ViT-B-16-SigLIP2-384")
    p.add_argument("--pretrained", default="webli")
    p.add_argument("--batch-size", type=int, default=32)
    args = p.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists():
        log.info("%s already exists, skipping", out_path)
        return

    queries, images = collect_unique(Path(args.hardnegs))

    import open_clip
    from PIL import Image

    log.info("loading %s pretrained=%s", args.model, args.pretrained)
    model, _, preprocess = open_clip.create_model_and_transforms(
        args.model, pretrained=args.pretrained, cache_dir=str(HF_CACHE),
    )
    tokenizer = open_clip.get_tokenizer(args.model)
    model.eval().to(DEVICE)
    for p_ in model.parameters():
        p_.requires_grad = False

    log.info("encoding %d text queries ...", len(queries))
    t0 = time.time()
    txt_chunks: list[torch.Tensor] = []
    with torch.no_grad():
        for i in range(0, len(queries), args.batch_size):
            tokens = tokenizer(queries[i:i + args.batch_size]).to(DEVICE)
            f_ = model.encode_text(tokens)
            f_ = f_ / f_.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            txt_chunks.append(f_.float().cpu())
    txt_t = torch.cat(txt_chunks, dim=0)
    log.info("  text done: %s in %.1fs", tuple(txt_t.shape), time.time() - t0)

    log.info("encoding %d images ...", len(images))
    t0 = time.time()
    img_chunks: list[torch.Tensor] = []
    n_done = 0
    n_failed = 0
    with torch.no_grad():
        for i in range(0, len(images), args.batch_size):
            batch_paths = images[i:i + args.batch_size]
            batch_imgs = []
            ok_idx = []
            for j, p in enumerate(batch_paths):
                try:
                    img = Image.open(p).convert("RGB")
                    batch_imgs.append(preprocess(img))
                    ok_idx.append(j)
                except Exception:
                    n_failed += 1
            if not batch_imgs:
                continue
            tens = torch.stack(batch_imgs).to(DEVICE)
            f_ = model.encode_image(tens)
            f_ = f_ / f_.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            img_chunks.append(f_.float().cpu())
            n_done += len(batch_imgs)
            if n_done % 500 < args.batch_size:
                log.info("  [img] %d/%d  (%.1f items/s)",
                         n_done, len(images), n_done / max(time.time() - t0, 1e-6))
    img_t = torch.cat(img_chunks, dim=0)
    log.info("  image done: %s in %.1fs (failed=%d)", tuple(img_t.shape), time.time() - t0, n_failed)

    if n_failed:
        log.error("MUST SUCCEED on all images for index alignment; got %d failures", n_failed)
        raise RuntimeError("image encoding had failures; aborting cache write")

    payload = {
        "img": img_t,
        "txt": txt_t,
        "image_paths": images,
        "queries": queries,
        "model": args.model,
        "pretrained": args.pretrained,
        "is_l2_normed": True,
    }
    tmp = out_path.with_suffix(".pt.tmp")
    torch.save(payload, tmp)
    tmp.replace(out_path)
    log.info("saved %s (%.1f MB)", out_path, out_path.stat().st_size / 1e6)

    del model
    gc.collect()
    if DEVICE == "mps":
        try:
            torch.mps.empty_cache()
        except Exception:
            pass


if __name__ == "__main__":
    main()
