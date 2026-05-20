"""
Phase A.3 — Cache SigLIP-2 student image embeddings on MPS.

Forwards every image in pairs_50k.jsonl through ViT-B-16-SigLIP2/webli image
tower ONCE. Saves the (pre-projection) embeddings as fp16 indexed by
the order in pairs_50k.jsonl.

Why pre-projection: the trainable image projection head is part of the student
model and runs live during training. We cache only the frozen tower output
(768-d) so the projection still fires on the live (trainable) weights.

Usage:
    python scripts/v5/phase_a_cache_student_image_emb.py
    python scripts/v5/phase_a_cache_student_image_emb.py --batch_size 32
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PAIRS = REPO_ROOT / "data" / "processed" / "v5_multifield" / "pairs_50k.jsonl"
DEFAULT_IMG_DIR = REPO_ROOT / "data" / "processed" / "v4_pattern_targeted" / "images"
DEFAULT_OUT = REPO_ROOT / "data" / "processed" / "v5_multifield" / "student_image_emb.pt"
DEFAULT_INDEX_OUT = REPO_ROOT / "data" / "processed" / "v5_multifield" / "student_image_index.json"


def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_pairs(path: Path) -> list[dict]:
    pairs = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                pairs.append(json.loads(line))
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=Path, default=DEFAULT_PAIRS)
    ap.add_argument("--img_dir", type=Path, default=DEFAULT_IMG_DIR)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--index_out", type=Path, default=DEFAULT_INDEX_OUT)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    device = args.device or pick_device()
    print(f"Device: {device}")

    print("Loading model: ViT-B-16-SigLIP2 / webli ...")
    import open_clip
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-16-SigLIP2-384", pretrained="webli"
    )
    model = model.to(device).eval()

    print(f"Loading pairs from {args.pairs}")
    pairs = load_pairs(args.pairs)
    print(f"  {len(pairs):,} pairs")

    # Validate that images exist (and skip ones that don't)
    valid_pairs = []
    n_missing = 0
    for p in pairs:
        img_path = args.img_dir / p["image_file"]
        if img_path.exists():
            valid_pairs.append(p)
        else:
            n_missing += 1
    if n_missing:
        print(f"  WARNING: {n_missing} images missing, will skip")

    n_total = len(valid_pairs)
    if n_total == 0:
        sys.exit("ERROR: no valid images")

    # Determine output dim from a single-image probe
    with torch.no_grad():
        probe_img = Image.open(args.img_dir / valid_pairs[0]["image_file"]).convert("RGB")
        probe_tensor = preprocess(probe_img).unsqueeze(0).to(device)
        # encode_image returns the projected image features (768-d)
        # We want the projected output because that's what the loss compares against text.
        # The "trainable image projection" mentioned in the plan refers to the small
        # head on top — for a frozen backbone we cache the final image features here.
        probe_emb = model.encode_image(probe_tensor)
        emb_dim = probe_emb.shape[-1]
    print(f"Embedding dim: {emb_dim}")

    print(f"Allocating output tensor: ({n_total}, {emb_dim}) fp16 ≈ "
          f"{n_total * emb_dim * 2 / 1024 / 1024:.1f} MB")
    out_tensor = torch.zeros((n_total, emb_dim), dtype=torch.float16)

    # Write the index so we know which pair_id maps to which row
    index = {p["pair_id"]: i for i, p in enumerate(valid_pairs)}

    t_start = time.time()
    n_done = 0
    with torch.inference_mode():
        for batch_start in tqdm(range(0, n_total, args.batch_size), desc="encoding"):
            batch = valid_pairs[batch_start : batch_start + args.batch_size]
            tensors = []
            for p in batch:
                try:
                    img = Image.open(args.img_dir / p["image_file"]).convert("RGB")
                    tensors.append(preprocess(img))
                except Exception as e:
                    print(f"  failed to load {p['image_file']}: {e}", file=sys.stderr)
                    tensors.append(torch.zeros(3, 224, 224))
            stack = torch.stack(tensors).to(device)
            emb = model.encode_image(stack)
            emb = F.normalize(emb, dim=-1)
            out_tensor[batch_start : batch_start + len(batch)] = emb.detach().cpu().to(torch.float16)
            n_done += len(batch)

    elapsed = time.time() - t_start
    rate = n_done / elapsed if elapsed > 0 else 0
    print(f"\nEncoded {n_done:,} images in {elapsed:.1f}s ({rate:.1f}/sec)")

    print(f"Saving cache to {args.out} ...")
    torch.save(out_tensor, args.out)
    print(f"Saving index to {args.index_out} ...")
    args.index_out.write_text(json.dumps(index))
    print(f"Done. Cache shape={tuple(out_tensor.shape)}, dtype={out_tensor.dtype}")


if __name__ == "__main__":
    main()
