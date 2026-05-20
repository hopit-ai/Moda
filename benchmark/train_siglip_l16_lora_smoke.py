"""
MODA Phase 4 smoke run — LoRA fine-tune of SigLIP L/16-384 on a tiny mix of
Marqo-curated DeepFashion data (deepfashion-multimodal + deepfashion-inshop
HF snapshots already present under ``data/raw/``).

Purpose (per PLAN_SCREEN_DISTILL_QUANTIZE.md Appendix B / Appendix C):
  - Validate the LoRA + dataloader + loss wiring end-to-end on free hardware
  - NOT a real GCL run. Loss here is plain InfoNCE on (image, caption) pairs.
  - Output is an open_clip-compatible state_dict, so the existing screener
    (benchmark/eval_marqo_subsample.py) can evaluate it identically to the
    other MoDA SigLIP checkpoints.

Defaults are intentionally tiny (~5k pairs, 500 steps, batch 32) so the run
finishes in ~1-2h on MPS. Bump --num-train-samples / --max-steps / --batch-size
once the smoke passes.

Usage (smoke):
  .venv/bin/python benchmark/train_siglip_l16_lora_smoke.py

Usage (override):
  .venv/bin/python benchmark/train_siglip_l16_lora_smoke.py \
      --num-train-samples 20000 --max-steps 1500 --batch-size 64

After training, evaluate with:
  .venv/bin/python benchmark/eval_marqo_subsample.py \
      --models siglip-l16-lora-smoke --datasets fashion200k \
      --corpus-size 10000
  (You'll need to add the run to MODEL_CONFIGS pointing at the saved
  state_dict — see the printed instructions at end of training.)
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import sys
import time
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset as HFDataset
from PIL import Image
from torch.utils.data import DataLoader, Dataset, IterableDataset

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("siglip-l16-lora-smoke")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DFMM_ARROW = (
    _REPO_ROOT
    / "data/raw/deepfashion_multimodal/Marqo___deepfashion-multimodal/default/0.0.0"
    / "21601c432a281c8a7deaa2afdda952b4b9ef3fd7/deepfashion-multimodal-data.arrow"
)
DFIN_ARROW = (
    _REPO_ROOT
    / "data/raw/deepfashion_inshop/Marqo___deepfashion-inshop/default/0.0.0"
    / "886fbe26ceac0b2bc8195d606e3fde641e7157f8/deepfashion-inshop-data.arrow"
)

OUTPUT_DIR = _REPO_ROOT / "models" / "moda-siglip-l16-lora-smoke"
HF_CACHE = _REPO_ROOT / "data" / "hf_cache"

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class PairDataset(Dataset):
    """In-memory list of (PIL image, caption) tuples."""

    def __init__(self, samples: list[tuple[Image.Image, str]], preprocess):
        self.samples = samples
        self.preprocess = preprocess

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img, txt = self.samples[idx]
        return self.preprocess(img.convert("RGB")), txt


def build_pair_samples(
    n_total: int,
    seed: int,
    min_chars: int = 10,
    max_chars: int = 300,
) -> list[tuple[Image.Image, str]]:
    """Sample (image, caption) pairs from the two Marqo DF snapshots, half each."""
    rng = random.Random(seed)

    def load(path: Path) -> HFDataset:
        log.info("loading %s", path.parent.name)
        return HFDataset.from_file(str(path))

    dfmm = load(DFMM_ARROW)
    dfin = load(DFIN_ARROW)

    n_each = n_total // 2
    out: list[tuple[Image.Image, str]] = []

    for ds, n_pick in [(dfmm, n_each), (dfin, n_total - n_each)]:
        idxs = list(range(len(ds)))
        rng.shuffle(idxs)
        picked = 0
        for i in idxs:
            if picked >= n_pick:
                break
            row = ds[i]
            txt = row.get("text") or ""
            txt = txt.strip()
            if not (min_chars <= len(txt) <= max_chars):
                continue
            img = row["image"]
            if img is None:
                continue
            out.append((img, txt))
            picked += 1
        log.info("  picked %d pairs", picked)

    rng.shuffle(out)
    log.info("total pairs: %d", len(out))
    return out


# ---------------------------------------------------------------------------
# Model + LoRA
# ---------------------------------------------------------------------------


def build_model_and_lora(device: str, lora_r: int, lora_alpha: int, lora_dropout: float):
    import open_clip
    from peft import LoraConfig, get_peft_model

    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-L-16-SigLIP-384",
        pretrained="webli",
        cache_dir=str(HF_CACHE),
    )
    tokenizer = open_clip.get_tokenizer("ViT-L-16-SigLIP-384")

    # Freeze everything; LoRA will inject trainable adapters
    for p in model.parameters():
        p.requires_grad = False

    # Suffix-only target list — peft matches the trailing module name.
    # vision blocks: qkv, proj, fc1, fc2
    # text blocks:   q, kv, out_proj, c_fc, c_proj
    target_modules = ["qkv", "proj", "fc1", "fc2", "q", "kv", "out_proj", "c_fc", "c_proj"]

    lora_cfg = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        target_modules=target_modules,
    )
    model = get_peft_model(model, lora_cfg)

    # Gradient checkpointing on vision trunk (the memory hog at 384x384) to fit
    # on MPS unified memory at usable batch sizes. Safe with LoRA.
    try:
        model.base_model.model.visual.trunk.set_grad_checkpointing(True)
        log.info("enabled gradient checkpointing on visual.trunk")
    except Exception as e:  # pragma: no cover - best effort
        log.warning("could not enable grad checkpointing: %s", e)

    # Sanity: count trainable params
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    log.info(
        "LoRA wired. trainable=%.2fM / total=%.2fM (%.2f%%)",
        trainable / 1e6,
        total / 1e6,
        100.0 * trainable / total,
    )

    model.to(device)
    return model, preprocess, tokenizer


# ---------------------------------------------------------------------------
# Training step (InfoNCE, both directions)
# ---------------------------------------------------------------------------


def info_nce_loss(image_feat: torch.Tensor, text_feat: torch.Tensor, temperature: float) -> torch.Tensor:
    image_feat = F.normalize(image_feat, dim=-1)
    text_feat = F.normalize(text_feat, dim=-1)
    logits = image_feat @ text_feat.t() / temperature
    targets = torch.arange(image_feat.size(0), device=image_feat.device)
    return 0.5 * (F.cross_entropy(logits, targets) + F.cross_entropy(logits.t(), targets))


def collate(batch, tokenizer, context_length: int = 64):
    images = torch.stack([b[0] for b in batch])
    texts = [b[1] for b in batch]
    tokens = tokenizer(texts, context_length=context_length)
    return images, tokens


def train(args):
    device = args.device
    log.info("device=%s", device)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- model
    model, preprocess, tokenizer = build_model_and_lora(
        device, args.lora_r, args.lora_alpha, args.lora_dropout
    )

    # ---- data
    samples = build_pair_samples(args.num_train_samples, args.seed)
    dataset = PairDataset(samples, preprocess)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=lambda b: collate(b, tokenizer),
        drop_last=True,
        pin_memory=False,
    )

    # ---- optim
    trainable = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.max_steps)

    # ---- loop
    model.train()
    step = 0
    t0 = time.time()
    losses: list[float] = []
    log.info("starting smoke training: max_steps=%d, bs=%d", args.max_steps, args.batch_size)

    while step < args.max_steps:
        for images, tokens in loader:
            if step >= args.max_steps:
                break
            images = images.to(device, non_blocking=True)
            tokens = tokens.to(device, non_blocking=True)

            image_feat = model.get_image_features(images) if hasattr(model, "get_image_features") else model.encode_image(images)
            text_feat = model.get_text_features(tokens) if hasattr(model, "get_text_features") else model.encode_text(tokens)

            loss = info_nce_loss(image_feat, text_feat, args.temperature)

            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optim.step()
            sched.step()

            losses.append(loss.item())
            step += 1
            if step % args.log_every == 0 or step == 1:
                elapsed = time.time() - t0
                window = losses[-args.log_every :] if len(losses) >= args.log_every else losses
                log.info(
                    "step %4d/%d  loss=%.4f (avg%d=%.4f)  lr=%.2e  elapsed=%.1fs  step/s=%.2f",
                    step,
                    args.max_steps,
                    losses[-1],
                    len(window),
                    sum(window) / len(window),
                    sched.get_last_lr()[0],
                    elapsed,
                    step / max(elapsed, 1e-6),
                )

    log.info("training done in %.1fs", time.time() - t0)

    # ---- merge LoRA weights into base, save as open_clip state_dict
    log.info("merging LoRA into base model and saving state_dict ...")
    merged = model.merge_and_unload()  # returns the underlying open_clip model
    save_path = OUTPUT_DIR / "best" / "model_state_dict.pt"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(merged.state_dict(), save_path)
    log.info("saved merged state_dict -> %s", save_path)

    # save run metadata
    meta = {
        "backbone": "ViT-L-16-SigLIP-384 / webli",
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        "max_steps": args.max_steps,
        "num_train_samples": args.num_train_samples,
        "temperature": args.temperature,
        "seed": args.seed,
        "device": device,
        "loss": "InfoNCE (in-batch, bidirectional)",
        "data": "Marqo-curated deepfashion-multimodal + deepfashion-inshop (50/50 sample)",
        "final_loss": losses[-1] if losses else None,
        "wall_time_sec": time.time() - t0,
    }
    (OUTPUT_DIR / "run_meta.json").write_text(json.dumps(meta, indent=2))
    log.info("wrote %s", OUTPUT_DIR / "run_meta.json")

    print()
    print("=" * 78)
    print("Smoke FT done. To evaluate on the fashion200k 10K screener:")
    print()
    print("  1. Add this entry to MODEL_CONFIGS in benchmark/eval_marqo_7dataset.py:")
    print()
    print('    "siglip-l16-lora-smoke": {')
    print('        "model_name": "ViT-L-16-SigLIP-384",')
    print(f'        "pretrained": "{save_path}",')
    print('        "run_name": "MoDA-SigLIP-L16-LoRA-Smoke",')
    print('        "label": "MoDA SigLIP L/16-384 LoRA smoke (DF mix, InfoNCE)",')
    print("    },")
    print()
    print("  2. Run the screener:")
    print()
    print("    .venv/bin/python benchmark/eval_marqo_subsample.py \\")
    print("        --models google-siglip-l16-384 siglip-l16-lora-smoke \\")
    print("        --datasets fashion200k --corpus-size 10000")
    print()
    print("  Compare MAP@10 of the FT'd model vs the zero-shot L/16-384 baseline.")
    print("=" * 78)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--num-train-samples", type=int, default=5000)
    p.add_argument("--batch-size", type=int, default=12)
    p.add_argument("--max-steps", type=int, default=500)
    p.add_argument("--lr", type=float, default=1e-4)  # adapter LR; base is frozen
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--temperature", type=float, default=0.07)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="0 by default — Python 3.14 spawn can't pickle the lambda collate_fn",
    )
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--device",
        default=("mps" if torch.backends.mps.is_available() else "cpu"),
        choices=["cpu", "mps", "cuda"],
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train(args)
