"""
Cache *fusion teacher* embeddings (FashionSigLIP + Google SigLIP-2 B/16/384)
on the Marqo-GS-10M wfash subset, for the Path 1 distillation training loop.

Why two separate caches (one per teacher) and not a precomputed score matrix:
  - The training-time score matrix is per-minibatch ([B, B]), not [N, N]. We
    compute it on the fly from cached embeddings.
  - Two encoders, two embedding spaces (FSL is fashion-tuned SigLIP, SL2 is
    SigLIP-2 webli). They live in different geometries, so we keep them apart
    and combine only at score-mean time (matching benchmark/fuse_and_eval.py).

Output layout (under data/processed/distillation_cache_fusion/):
  fashion_siglip_embeddings.pt  -> {"image": [N, 768] fp32, "text": [N, 768] fp32, ...}
  siglip2_b16_384_embeddings.pt -> same shape
  meta.json                      -> sample counts, model ids, wall times

Cache size:  5000 items * 768d * 4B * 2 (image+text) * 2 (teachers) = ~62 MB total.
Recompute time on MPS: ~5-10 min per teacher.

Usage:
  .venv/bin/python scripts/cache_teacher_fusion_scores.py \
      --triplets data/processed/marqo_gs_wfash_subset/triplets.jsonl

Reuses the same triplet input as scripts/cache_teacher_fashionsiglip.py so the
indexing matches across both caches (same row order = same N).
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
log = logging.getLogger("cache-fusion-teacher")

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
HF_CACHE = REPO / "data" / "hf_cache"
OUT_DIR = REPO / "data" / "processed" / "distillation_cache_fusion"

# (cache_filename, open_clip model_name, pretrained_tag)
TEACHERS = [
    ("fashion_siglip_embeddings.pt",  "hf-hub:Marqo/marqo-fashionSigLIP", None),
    ("siglip2_b16_384_embeddings.pt", "ViT-B-16-SigLIP2-384",             "webli"),
]


def encode_one_teacher(
    cache_filename: str,
    model_name: str,
    pretrained: str | None,
    rows: list[dict],
    batch_size: int,
    out_dir: Path,
) -> dict:
    """Encode all (query, image) pairs through one teacher and save to disk."""
    import open_clip
    from PIL import Image

    out_path = out_dir / cache_filename
    if out_path.exists():
        log.info("[%s] cache already exists at %s — skipping", model_name, out_path)
        # Load to compute meta
        d = torch.load(out_path, map_location="cpu", weights_only=False)
        return {
            "n": len(d["queries"]),
            "model": model_name,
            "embed_dim": int(d["image"].shape[-1]),
            "wall_time_sec": 0.0,
            "skipped": True,
        }

    log.info("[%s] loading teacher (pretrained=%s)", model_name, pretrained)
    if pretrained is None:
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name, cache_dir=str(HF_CACHE),
        )
    else:
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, cache_dir=str(HF_CACHE),
        )
    tokenizer = open_clip.get_tokenizer(model_name)
    model.eval().to(DEVICE)
    for p_ in model.parameters():
        p_.requires_grad = False

    queries: list[str] = []
    image_paths: list[str] = []
    img_feats: list[torch.Tensor] = []
    txt_feats: list[torch.Tensor] = []

    t0 = time.time()
    n_done = 0
    n_failed = 0

    batch_imgs: list[torch.Tensor] = []
    batch_qs: list[str] = []
    batch_paths: list[str] = []

    def flush():
        nonlocal n_done
        if not batch_imgs:
            return
        imgs = torch.stack(batch_imgs).to(DEVICE)
        tokens = tokenizer(batch_qs).to(DEVICE)
        with torch.no_grad():
            ifeat = model.encode_image(imgs)
            tfeat = model.encode_text(tokens)
        # Normalize once at cache time so the trainer can just dot-product.
        ifeat = ifeat / ifeat.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        tfeat = tfeat / tfeat.norm(dim=-1, keepdim=True).clamp_min(1e-12)
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
        except Exception:
            n_failed += 1
            continue
        if len(batch_imgs) >= batch_size:
            flush()
            if n_done % 500 == 0 or n_done == batch_size:
                rate = n_done / max(time.time() - t0, 1e-6)
                log.info("  [%s] encoded %d/%d (%.1f items/s, failed=%d)",
                         model_name, n_done, len(rows), rate, n_failed)
    flush()

    img_t = torch.cat(img_feats, dim=0)
    txt_t = torch.cat(txt_feats, dim=0)
    log.info("[%s] done. image=%s text=%s in %.1fs (failed=%d)",
             model_name, tuple(img_t.shape), tuple(txt_t.shape), time.time() - t0, n_failed)

    payload = {
        "image": img_t,                    # already L2-normed, fp32
        "text": txt_t,                     # already L2-normed, fp32
        "queries": queries,
        "image_paths": image_paths,
        "teacher": model_name,
        "pretrained": pretrained,
        "embed_dim": int(img_t.shape[-1]),
        "is_l2_normed": True,
    }
    tmp = out_path.with_suffix(".pt.tmp")
    torch.save(payload, tmp)
    tmp.replace(out_path)
    log.info("[%s] cached -> %s (%.1f MB)", model_name, out_path, out_path.stat().st_size / 1e6)

    # Free GPU memory before the next teacher
    del model
    if DEVICE == "mps":
        torch.mps.empty_cache()

    return {
        "n": len(queries),
        "model": model_name,
        "pretrained": pretrained,
        "embed_dim": int(img_t.shape[-1]),
        "wall_time_sec": time.time() - t0,
        "n_failed": n_failed,
        "skipped": False,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--triplets",
        default=str(REPO / "data/processed/marqo_gs_wfash_subset/triplets.jsonl"),
    )
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--limit", type=int, default=None,
                   help="Cap rows for smoke testing")
    p.add_argument("--out-dir", default=str(OUT_DIR),
                   help="Where to write the cache (default: distillation_cache_fusion)")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    with open(args.triplets) as f:
        for line in f:
            rows.append(json.loads(line))
    if args.limit:
        rows = rows[: args.limit]
    log.info("loaded %d rows from %s", len(rows), args.triplets)

    metas = []
    for cache_filename, model_name, pretrained in TEACHERS:
        m = encode_one_teacher(
            cache_filename, model_name, pretrained, rows, args.batch_size, out_dir,
        )
        metas.append(m)

    meta_path = out_dir / "meta.json"
    meta_path.write_text(json.dumps({
        "triplets_source": str(args.triplets),
        "n_input_rows": len(rows),
        "teachers": metas,
        "device": DEVICE,
    }, indent=2))
    log.info("wrote %s", meta_path)
    log.info("DONE — %d teachers cached under %s", len(metas), out_dir)


if __name__ == "__main__":
    main()
