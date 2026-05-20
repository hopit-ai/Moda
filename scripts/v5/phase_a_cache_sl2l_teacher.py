"""
v4-alt — Cache SL2-L/16/384 (large) teacher embeddings.

Creates a stronger fusion teacher: (FSL + SL2-L)/2 instead of (FSL + SL2-B)/2.
SL2-L is ~400M params vs SL2-B's 203M, generally stronger zero-shot.

Outputs:
  - teacher_sl2l_img_emb.pt   (50K × D fp16) — replaces "SL2-B image=student cache" trick
  - teacher_sl2l_text_emb.pt  dict{query: (D,) fp16}

Usage:
    python scripts/v5/phase_a_cache_sl2l_teacher.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data" / "processed" / "v5_multifield"
DEFAULT_PAIRS = DATA / "pairs_50k.jsonl"
DEFAULT_IMG_DIR = REPO / "data" / "processed" / "v4_pattern_targeted" / "images"


def pick_device():
    if torch.backends.mps.is_available(): return "mps"
    if torch.cuda.is_available(): return "cuda"
    return "cpu"


def main():
    device = pick_device()
    print(f"Device: {device}")

    img_out = DATA / "teacher_sl2l_img_emb.pt"
    text_out = DATA / "teacher_sl2l_text_emb.pt"

    pairs_path = DEFAULT_PAIRS
    pairs = []
    queries = set()
    with pairs_path.open() as f:
        for line in f:
            line = line.strip()
            if not line: continue
            r = json.loads(line)
            pairs.append(r)
            if r.get("query"):
                queries.add(r["query"])
    queries = sorted(queries)
    print(f"{len(pairs):,} pairs, {len(queries):,} unique queries")

    valid = [p for p in pairs if (DEFAULT_IMG_DIR / p["image_file"]).exists()]
    print(f"{len(valid):,} pairs with images")

    import open_clip
    print("\nLoading ViT-L-16-SigLIP2-384/webli ...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-L-16-SigLIP2-384", pretrained="webli"
    )
    tok = open_clip.get_tokenizer("ViT-L-16-SigLIP2-384")
    model = model.to(device).eval()

    # ---- Image cache ----
    if img_out.exists():
        print(f"Skipping image cache (already exists at {img_out})")
    else:
        print("Encoding images ...")
        n = len(valid)
        out = None
        t0 = time.time()
        with torch.inference_mode():
            for j in tqdm(range(0, n, 32), desc="images"):
                batch = valid[j:j+32]
                tensors = []
                for p in batch:
                    try:
                        img = Image.open(DEFAULT_IMG_DIR / p["image_file"]).convert("RGB")
                        tensors.append(preprocess(img))
                    except Exception as e:
                        print(f"  failed {p['image_file']}: {e}", file=sys.stderr)
                        tensors.append(torch.zeros(3, 384, 384))
                stack = torch.stack(tensors).to(device)
                emb = F.normalize(model.encode_image(stack), dim=-1)
                if out is None:
                    out = torch.zeros((n, emb.shape[-1]), dtype=torch.float16)
                out[j:j+len(batch)] = emb.detach().cpu().to(torch.float16)
        print(f"Image cache: {tuple(out.shape)} in {time.time()-t0:.0f}s")
        torch.save(out, img_out)

    # ---- Text cache ----
    if text_out.exists():
        print(f"Skipping text cache (already exists at {text_out})")
    else:
        print("\nEncoding text ...")
        out = {}
        t0 = time.time()
        with torch.inference_mode():
            for j in tqdm(range(0, len(queries), 128), desc="text"):
                batch = queries[j:j+128]
                tokens = tok(batch).to(device)
                emb = F.normalize(model.encode_text(tokens), dim=-1).detach().cpu().to(torch.float16)
                for i, q in enumerate(batch):
                    out[q] = emb[i]
        print(f"Text cache: {len(out):,} queries in {time.time()-t0:.0f}s")
        torch.save(out, text_out)

    print("\nDone. SL2-L teacher caches ready for v4-alt.")


if __name__ == "__main__":
    main()
