"""
Phase A.5 — Cache fusion teacher embeddings (FSL + SL2 text).

Outputs:
  - teacher_fsl_img_emb.pt        (50K, 768) fp16 — FSL image tower
  - teacher_fsl_text_emb.pt       dict{query: (768,) fp16} — FSL text tower
  - teacher_sl2_text_emb.pt       dict{query: (768,) fp16} — SL2 text tower
  - teacher_fsl_img_index.json    {pair_id: row_idx}

Note: the SL2 image teacher embedding equals the student image cache (both are
SL2-zeroshot post-projection L2-normalized image embeddings). We reuse
student_image_emb.pt for SL2 image teacher purposes during training — no
separate SL2 image cache is built.
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
OUT_DIR = REPO_ROOT / "data" / "processed" / "v5_multifield"


def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_pairs(path: Path) -> list[dict]:
    out = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def encode_images(model, preprocess, pairs, img_dir, device, batch_size=32):
    """Forward all images in `pairs` through the image tower, return (N, D) fp16."""
    n = len(pairs)
    out = None
    with torch.inference_mode():
        for batch_start in tqdm(range(0, n, batch_size), desc="image encoding"):
            batch = pairs[batch_start : batch_start + batch_size]
            tensors = []
            for p in batch:
                try:
                    img = Image.open(img_dir / p["image_file"]).convert("RGB")
                    tensors.append(preprocess(img))
                except Exception as e:
                    print(f"  failed {p['image_file']}: {e}", file=sys.stderr)
                    tensors.append(torch.zeros(3, 224, 224))
            stack = torch.stack(tensors).to(device)
            emb = model.encode_image(stack)
            emb = F.normalize(emb, dim=-1)
            if out is None:
                out = torch.zeros((n, emb.shape[-1]), dtype=torch.float16)
            out[batch_start : batch_start + len(batch)] = emb.detach().cpu().to(torch.float16)
    return out


def encode_texts(model, tokenizer, queries, device, batch_size=128):
    """Forward unique queries through text tower, return dict{query: (D,) fp16}."""
    queries = list(queries)
    out = {}
    with torch.inference_mode():
        for batch_start in tqdm(range(0, len(queries), batch_size), desc="text encoding"):
            batch = queries[batch_start : batch_start + batch_size]
            tokens = tokenizer(batch).to(device)
            emb = model.encode_text(tokens)
            emb = F.normalize(emb, dim=-1)
            emb_cpu = emb.detach().cpu().to(torch.float16)
            for i, q in enumerate(batch):
                out[q] = emb_cpu[i]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=Path, default=DEFAULT_PAIRS)
    ap.add_argument("--img_dir", type=Path, default=DEFAULT_IMG_DIR)
    ap.add_argument("--out_dir", type=Path, default=OUT_DIR)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = args.device or pick_device()
    print(f"Device: {device}")

    pairs = load_pairs(args.pairs)
    # Filter to pairs whose image file actually exists
    valid = [p for p in pairs if (args.img_dir / p["image_file"]).exists()]
    print(f"{len(valid):,}/{len(pairs):,} pairs with images present")
    queries = sorted({p["query"] for p in valid})
    print(f"{len(queries):,} unique queries")

    import open_clip

    # ─────────────────────────────────────────────────────────────────────
    # FSL teacher: Marqo/marqo-fashionSigLIP (ViT-B-16-SigLIP @ 224)
    # ─────────────────────────────────────────────────────────────────────
    print("\n=== FSL teacher (Marqo/marqo-fashionSigLIP) ===")
    fsl_model, _, fsl_pre = open_clip.create_model_and_transforms(
        "hf-hub:Marqo/marqo-fashionSigLIP"
    )
    fsl_tok = open_clip.get_tokenizer("hf-hub:Marqo/marqo-fashionSigLIP")
    fsl_model = fsl_model.to(device).eval()

    t0 = time.time()
    fsl_img = encode_images(fsl_model, fsl_pre, valid, args.img_dir, device, args.batch_size)
    print(f"FSL image cache: {tuple(fsl_img.shape)}, {time.time() - t0:.1f}s")
    torch.save(fsl_img, args.out_dir / "teacher_fsl_img_emb.pt")

    t0 = time.time()
    fsl_text = encode_texts(fsl_model, fsl_tok, queries, device)
    print(f"FSL text cache: {len(fsl_text):,} queries, {time.time() - t0:.1f}s")
    torch.save(fsl_text, args.out_dir / "teacher_fsl_text_emb.pt")

    # Drop FSL from memory before loading SL2
    del fsl_model
    if device == "mps":
        torch.mps.empty_cache()

    # ─────────────────────────────────────────────────────────────────────
    # SL2 teacher: text only — image side is reused from student cache
    # ─────────────────────────────────────────────────────────────────────
    print("\n=== SL2 teacher (ViT-B-16-SigLIP2/webli) — text only ===")
    sl2_model, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-16-SigLIP2-384", pretrained="webli"
    )
    sl2_tok = open_clip.get_tokenizer("ViT-B-16-SigLIP2-384")
    sl2_model = sl2_model.to(device).eval()

    t0 = time.time()
    sl2_text = encode_texts(sl2_model, sl2_tok, queries, device)
    print(f"SL2 text cache: {len(sl2_text):,} queries, {time.time() - t0:.1f}s")
    torch.save(sl2_text, args.out_dir / "teacher_sl2_text_emb.pt")

    # Index file mapping pair_id → row in the FSL image cache
    index = {p["pair_id"]: i for i, p in enumerate(valid)}
    (args.out_dir / "teacher_fsl_img_index.json").write_text(json.dumps(index))

    print("\nAll teacher caches saved to", args.out_dir)
    print("  teacher_fsl_img_emb.pt — (N, 768) fp16, indexed by teacher_fsl_img_index.json")
    print("  teacher_fsl_text_emb.pt — dict{query: (768,) fp16}")
    print("  teacher_sl2_text_emb.pt — dict{query: (768,) fp16}")
    print("  (SL2 image teacher reuses student_image_emb.pt — same output)")


if __name__ == "__main__":
    main()
