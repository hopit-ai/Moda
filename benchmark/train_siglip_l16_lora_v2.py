"""
MODA Phase 4 v2 — LoRA fine-tune of SigLIP L/16-384 on Marqo-GS triplets,
with adapter-only checkpointing every N steps and clean resume support.

Why this exists:
  v1 (train_siglip_l16_lora_smoke.py) used DeepFashion prose captions and an
  aggressive LR (1e-4); the resulting fine-tune REGRESSED fashion200k MAP@10
  by 83%. Diagnosis: data shape mismatch (DF prose vs fashion200k catalog
  text) + LR too hot. v2 fixes both:
    - Data: Marqo-GS wfash triplets (Google-Shopping queries → product image)
            built by scripts/build_marqo_gs_smoke_subset.py
    - LR: 1e-5 default
    - Checkpointing: adapter-only (~50 MB) every N steps, with optimizer +
      scheduler + RNG state so a crash resumes cleanly
    - Loss: plain InfoNCE for the smoke; --use-weights enables a poor-man's
      GCL (per-pair weight from the dataset's ranking score)

Usage (smoke):
  .venv/bin/python benchmark/train_siglip_l16_lora_v2.py \
      --triplets data/processed/marqo_gs_wfash_subset/triplets.jsonl \
      --max-steps 200 --batch-size 12 --lr 1e-5 \
      --output-dir models/moda-siglip-l16-lora-v2-smoke

Resume after a crash (auto-picks latest checkpoint):
  .venv/bin/python benchmark/train_siglip_l16_lora_v2.py \
      --triplets ... --output-dir ... --resume auto

The final model_state_dict.pt (open_clip-format, LoRA merged into base) is
written to <output-dir>/best/model_state_dict.pt — directly loadable by
benchmark/eval_marqo_subsample.py.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("train-l16-v2")

HF_CACHE = _REPO_ROOT / "data" / "hf_cache"


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class TripletJsonlDataset(Dataset):
    """Reads {query, image_path, weight} records and serves (image_tensor, query, weight)."""

    def __init__(self, triplets_path: Path, preprocess):
        self.records: list[dict] = []
        with triplets_path.open() as f:
            for line in f:
                r = json.loads(line)
                if Path(r["image_path"]).exists():
                    self.records.append(r)
        self.preprocess = preprocess
        log.info("loaded %d triplets from %s (with valid image paths)", len(self.records), triplets_path)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        r = self.records[idx]
        try:
            img = Image.open(r["image_path"]).convert("RGB")
        except Exception:
            # Return a black image as a fallback to keep the batch shape; the
            # corresponding loss term will be near-uninformative but won't crash.
            img = Image.new("RGB", (384, 384), (0, 0, 0))
        return self.preprocess(img), r["query"], float(r.get("weight", 1.0))


def collate(batch, tokenizer, context_length: int = 64):
    images = torch.stack([b[0] for b in batch])
    texts = [b[1] for b in batch]
    weights = torch.tensor([b[2] for b in batch], dtype=torch.float32)
    tokens = tokenizer(texts, context_length=context_length)
    return images, tokens, weights


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

    for p in model.parameters():
        p.requires_grad = False

    target_modules = ["qkv", "proj", "fc1", "fc2", "q", "kv", "out_proj", "c_fc", "c_proj"]
    lora_cfg = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        target_modules=target_modules,
    )
    model = get_peft_model(model, lora_cfg)

    try:
        model.base_model.model.visual.trunk.set_grad_checkpointing(True)
        log.info("enabled gradient checkpointing on visual.trunk")
    except Exception as e:
        log.warning("could not enable grad checkpointing: %s", e)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    log.info(
        "LoRA wired. trainable=%.2fM / total=%.2fM (%.2f%%)",
        trainable / 1e6, total / 1e6, 100.0 * trainable / total,
    )
    model.to(device)
    return model, preprocess, tokenizer


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------


def info_nce_loss(image_feat, text_feat, temperature, weights: Optional[torch.Tensor] = None):
    """In-batch InfoNCE. If weights are given, scale per-row losses by weight."""
    image_feat = F.normalize(image_feat, dim=-1)
    text_feat = F.normalize(text_feat, dim=-1)
    logits = image_feat @ text_feat.t() / temperature
    targets = torch.arange(image_feat.size(0), device=image_feat.device)
    l_i2t = F.cross_entropy(logits, targets, reduction="none")
    l_t2i = F.cross_entropy(logits.t(), targets, reduction="none")
    per_row = 0.5 * (l_i2t + l_t2i)
    if weights is None:
        return per_row.mean()
    w = weights.to(per_row.device)
    return (per_row * w).sum() / (w.sum() + 1e-8)


# ---------------------------------------------------------------------------
# Checkpoint I/O
# ---------------------------------------------------------------------------


def atomic_save(obj, path: Path):
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def save_checkpoint(model, optimizer, scheduler, step: int, args, output_dir: Path):
    """Save adapter weights + optimizer + scheduler + RNG state. Adapter-only is ~50 MB."""
    ckpt_dir = output_dir / f"step_{step:06d}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    # PEFT adapter (small)
    model.save_pretrained(str(ckpt_dir / "adapter"))
    state = {
        "step": step,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "rng_python": random.getstate(),
        "rng_numpy": np.random.get_state(),
        "rng_torch": torch.get_rng_state(),
        "args": vars(args),
    }
    atomic_save(state, ckpt_dir / "trainer_state.pt")
    log.info("saved checkpoint -> %s", ckpt_dir)
    # symlink "latest" for fast lookup
    latest = output_dir / "latest"
    if latest.is_symlink() or latest.exists():
        try:
            latest.unlink()
        except IsADirectoryError:
            pass
    try:
        latest.symlink_to(ckpt_dir.name)
    except Exception:
        pass


def find_latest_checkpoint(output_dir: Path) -> Optional[Path]:
    if not output_dir.exists():
        return None
    candidates = sorted(output_dir.glob("step_*"))
    return candidates[-1] if candidates else None


def load_checkpoint(model, optimizer, scheduler, ckpt_dir: Path):
    from peft import PeftModel
    log.info("loading checkpoint from %s", ckpt_dir)
    base_model = model.get_base_model() if hasattr(model, "get_base_model") else model.base_model.model
    # Reload adapter on top of base
    new_model = PeftModel.from_pretrained(base_model, str(ckpt_dir / "adapter"), is_trainable=True)
    state = torch.load(ckpt_dir / "trainer_state.pt", map_location="cpu", weights_only=False)
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    random.setstate(state["rng_python"])
    np.random.set_state(state["rng_numpy"])
    torch.set_rng_state(state["rng_torch"])
    log.info("resumed at step %d", state["step"])
    return new_model, state["step"]


# ---------------------------------------------------------------------------
# Train loop
# ---------------------------------------------------------------------------


def train(args):
    device = args.device
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log.info("device=%s output=%s", device, output_dir)

    model, preprocess, tokenizer = build_model_and_lora(
        device, args.lora_r, args.lora_alpha, args.lora_dropout
    )

    triplets_path = Path(args.triplets)
    dataset = TripletJsonlDataset(triplets_path, preprocess)
    if len(dataset) == 0:
        log.error("no triplets loaded — check %s", triplets_path)
        sys.exit(2)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,  # py3.14 spawn can't pickle lambda collate
        collate_fn=lambda b: collate(b, tokenizer),
        drop_last=True,
    )

    trainable = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.max_steps)

    # Resume?
    start_step = 0
    if args.resume:
        if args.resume == "auto":
            ckpt = find_latest_checkpoint(output_dir)
        else:
            ckpt = Path(args.resume)
        if ckpt and ckpt.exists():
            model, start_step = load_checkpoint(model, optim, sched, ckpt)
            model.to(device)
            trainable = [p for p in model.parameters() if p.requires_grad]
        else:
            log.warning("--resume=%s requested but no checkpoint found, starting fresh", args.resume)

    model.train()
    step = start_step
    t0 = time.time()
    losses: list[float] = []
    log.info("training: %d -> %d steps, bs=%d, lr=%.1e, weights=%s",
             step, args.max_steps, args.batch_size, args.lr, args.use_weights)

    while step < args.max_steps:
        for images, tokens, weights in loader:
            if step >= args.max_steps:
                break
            images = images.to(device, non_blocking=True)
            tokens = tokens.to(device, non_blocking=True)

            image_feat = model.get_image_features(images) if hasattr(model, "get_image_features") else model.encode_image(images)
            text_feat = model.get_text_features(tokens) if hasattr(model, "get_text_features") else model.encode_text(tokens)

            loss = info_nce_loss(
                image_feat, text_feat, args.temperature,
                weights=weights if args.use_weights else None,
            )

            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optim.step()
            sched.step()

            losses.append(loss.item())
            step += 1
            if step % args.log_every == 0 or step == start_step + 1:
                elapsed = time.time() - t0
                window = losses[-args.log_every :] if len(losses) >= args.log_every else losses
                log.info(
                    "step %4d/%d  loss=%.4f (avg%d=%.4f)  lr=%.2e  elapsed=%.1fs  step/s=%.2f",
                    step, args.max_steps, losses[-1], len(window),
                    sum(window) / len(window), sched.get_last_lr()[0],
                    elapsed, (step - start_step) / max(elapsed, 1e-6),
                )
            if args.save_every and step % args.save_every == 0:
                save_checkpoint(model, optim, sched, step, args, output_dir)

    log.info("training done in %.1fs (final step %d)", time.time() - t0, step)

    # Final save: the open_clip-format merged state_dict for the eval harness.
    log.info("merging LoRA into base and saving open_clip-format state_dict ...")
    merged = model.merge_and_unload()
    final_path = output_dir / "best" / "model_state_dict.pt"
    final_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_save(merged.state_dict(), final_path)
    log.info("saved merged state_dict -> %s", final_path)

    meta = {
        "backbone": "ViT-L-16-SigLIP-384 / webli",
        "triplets": str(triplets_path),
        "n_triplets": len(dataset),
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        "max_steps": args.max_steps,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "temperature": args.temperature,
        "use_weights": args.use_weights,
        "loss": "weighted-InfoNCE" if args.use_weights else "InfoNCE (binary)",
        "final_loss": losses[-1] if losses else None,
        "wall_time_sec": time.time() - t0,
    }
    (output_dir / "run_meta.json").write_text(json.dumps(meta, indent=2))
    log.info("wrote %s", output_dir / "run_meta.json")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--triplets", required=True, help="Path to JSONL with {query, image_path, weight}")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--batch-size", type=int, default=12)
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--temperature", type=float, default=0.07)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--save-every", type=int, default=50, help="Save adapter checkpoint every N steps (0 = off)")
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--use-weights", action="store_true", help="Enable poor-man's GCL: weight per-pair loss by triplet 'weight' column")
    p.add_argument("--resume", default=None, help="Path to step_NNNNNN/ dir, or 'auto' to pick latest in output_dir")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--device",
        default=("mps" if torch.backends.mps.is_available() else "cpu"),
        choices=["cpu", "mps", "cuda"],
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if not args.resume:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
    train(args)
