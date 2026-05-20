"""
Targeted re-cache: SL2 text teacher only.

Used after switching the SL2 backbone (e.g., 224 → 384). Re-encodes the unique
queries from pairs_50k.jsonl with the new SL2 model. FSL caches are unchanged.

Usage:
    python scripts/v5/phase_a_cache_sl2_text_only.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data" / "processed" / "v5_multifield"


def pick_device():
    if torch.backends.mps.is_available(): return "mps"
    if torch.cuda.is_available(): return "cuda"
    return "cpu"


def main():
    device = pick_device()
    print(f"Device: {device}")

    pairs_path = DATA / "pairs_50k.jsonl"
    out_path = DATA / "teacher_sl2_text_emb.pt"

    # Collect unique queries
    queries = set()
    with pairs_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("query"):
                queries.add(r["query"])
    queries = sorted(queries)
    print(f"{len(queries):,} unique queries")

    import open_clip
    print("Loading ViT-B-16-SigLIP2-384/webli ...")
    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-16-SigLIP2-384", pretrained="webli"
    )
    tok = open_clip.get_tokenizer("ViT-B-16-SigLIP2-384")
    model = model.to(device).eval()

    out = {}
    t0 = time.time()
    with torch.inference_mode():
        for j in tqdm(range(0, len(queries), 128), desc="encoding"):
            batch = queries[j : j + 128]
            tokens = tok(batch).to(device)
            emb = model.encode_text(tokens)
            emb = F.normalize(emb, dim=-1).detach().cpu().to(torch.float16)
            for i, q in enumerate(batch):
                out[q] = emb[i]
    print(f"Encoded {len(out):,} queries in {time.time() - t0:.1f}s")

    torch.save(out, out_path)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
