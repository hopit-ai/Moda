"""
MODA Phase 4 (revised) — distill Marqo-FashionSigLIP into a same-shape student.

Pivot context (read this first):
  Phase 4-prev tried LoRA + InfoNCE on top of SigLIP L/16-384. That recipe
  catastrophically regressed fashion200k MAP@10 from 0.4677 → 0.10 within the
  FIRST 50 STEPS, regardless of LR (we tried 1e-5 and 5e-5). Diagnosis: small-
  batch InfoNCE on a strong pretrained model collapses image features into a
  near-degenerate cluster, even with miniscule (cos≈0.9999 vs base) updates.
  See scripts/diagnose_v2_checkpoints.py for the autopsy.

This script does the OPPOSITE risk profile:
  - Student is INITIALIZED from Google SigLIP B/16/224 (no fashion training).
    It's a *worse* model than the teacher on fashion200k by design. So any
    movement toward the teacher = improvement, not collapse.
  - Loss = direct L2 + cosine match to CACHED teacher embeddings. Zero risk
    of in-batch collapse because there is no batch interaction in the loss.
  - Optional small InfoNCE term (cross-modal) to keep image↔text aligned.
  - Same architecture as teacher (ViT-B/16/224, 768d, ~203M params) → no
    projection adapter needed; student outputs land in the teacher's embedding
    space directly. Goal here is "transfer fashion knowledge", not "compress".
    Compression to a truly smaller student is Phase 4.5 (deferred).

Success criterion (from PLAN_SCREEN_DISTILL_QUANTIZE.md):
  Student fashion200k MAP@10 ≥ 0.90 × teacher (0.5369) ≈ 0.483.

Anti-collapse guardrail:
  After 50 training steps, the orchestrator runs an inline eval. If MAP@10 has
  dropped below the BASE student's value (Google SigLIP B/16/224 zero-shot,
  expected ~0.18-0.25 on fashion200k), the run aborts.

Usage (smoke):
  .venv/bin/python benchmark/distill_fashionsiglip_to_student.py \
      --teacher-cache data/processed/distillation_cache/teacher_embeddings.pt \
      --output-dir models/moda-siglip-distilled-from-fashionsiglip-smoke \
      --max-steps 50 --batch-size 32

Resume after crash (auto-picks latest checkpoint):
  .venv/bin/python benchmark/distill_fashionsiglip_to_student.py ... --resume auto

Outputs:
  <output-dir>/step_<N>/student_state_dict.pt   (full student state for resume)
  <output-dir>/step_<N>/trainer_state.pt        (optimizer + scheduler + RNG)
  <output-dir>/best/model_state_dict.pt         (eval-ready open_clip state_dict)
  <output-dir>/run_meta.json
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
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("distill-fsiglip")

HF_CACHE = REPO / "data" / "hf_cache"


# ---------------------------------------------------------------------------
# Dataset — yields (image_tensor, query_text, teacher_image_emb, teacher_text_emb)
# ---------------------------------------------------------------------------


class CachedDistillDataset(Dataset):
    """Reads teacher cache + corresponding image files, applies STUDENT preprocess.

    The teacher cache stores teacher-preprocessed embeddings (224x224 squash).
    The student here is also B/16/224 so it shares the same preprocess. If we
    ever change student input size we must redo the preprocess accordingly.
    """

    def __init__(self, cache_path: Path, student_preprocess):
        log.info("loading teacher cache: %s", cache_path)
        cache = torch.load(cache_path, map_location="cpu", weights_only=False)
        self.image_paths: list[str] = cache["image_paths"]
        self.queries: list[str] = cache["queries"]
        self.teacher_image: torch.Tensor = cache["image"].float()  # (N, D)
        self.teacher_text: torch.Tensor = cache["text"].float()
        self.preprocess = student_preprocess
        log.info(
            "  N=%d teacher_dim=%d  image_norm_mean=%.4f  text_norm_mean=%.4f",
            len(self.queries), self.teacher_image.shape[-1],
            self.teacher_image.norm(dim=-1).mean().item(),
            self.teacher_text.norm(dim=-1).mean().item(),
        )

    def __len__(self) -> int:
        return len(self.queries)

    def __getitem__(self, idx: int):
        try:
            img = Image.open(self.image_paths[idx]).convert("RGB")
            img_t = self.preprocess(img)
        except Exception:
            img_t = torch.zeros(3, 224, 224)
        return (
            img_t,
            self.queries[idx],
            self.teacher_image[idx],
            self.teacher_text[idx],
        )


def collate(batch, tokenizer, context_length: int = 64):
    images = torch.stack([b[0] for b in batch])
    texts = [b[1] for b in batch]
    teacher_img = torch.stack([b[2] for b in batch])
    teacher_txt = torch.stack([b[3] for b in batch])
    tokens = tokenizer(texts, context_length=context_length)
    return images, tokens, teacher_img, teacher_txt


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


def build_student(device: str, init_from: str):
    """Init student from Google SigLIP B/16/224 (same arch as the teacher)."""
    import open_clip

    log.info("building student from: %s", init_from)
    model, _, preprocess = open_clip.create_model_and_transforms(
        init_from, pretrained="webli" if init_from == "ViT-B-16-SigLIP" else None,
        cache_dir=str(HF_CACHE),
    )
    tokenizer = open_clip.get_tokenizer(init_from)

    n_params = sum(p.numel() for p in model.parameters())
    log.info("student params: %.2fM", n_params / 1e6)
    model.to(device)
    return model, preprocess, tokenizer


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------


def distillation_loss(
    student_img: torch.Tensor,
    student_txt: torch.Tensor,
    teacher_img: torch.Tensor,
    teacher_txt: torch.Tensor,
    *,
    feat_weight: float = 1.0,
    cos_weight: float = 1.0,
    infonce_weight: float = 0.0,
    temperature: float = 0.07,
):
    """Composite distillation loss.

    L = feat_weight * L2(s, t)      direct feature matching (no batch interaction)
      + cos_weight  * (1 - cos(s, t))   ranking-aware angular alignment
      + infonce_weight * L_infonce(s_img, s_txt)   keep cross-modal alignment

    The L2 + cos terms work per-sample and are immune to small-batch collapse.
    The InfoNCE term is kept tiny by default (off in defaults) because we
    diagnosed that aggressive InfoNCE collapses our model.
    """
    # Normalize EVERYTHING for both cosine and L2 loss. Teacher SigLIP features
    # have raw norms in the ~15-25 range, so MSE on raw features explodes the
    # loss (we observed NaN by step 2 in the 1e-4 LR smoke). Working on the
    # unit sphere makes the loss scale-invariant and numerically tame.
    sin = F.normalize(student_img, dim=-1)
    stn = F.normalize(student_txt, dim=-1)
    tin = F.normalize(teacher_img, dim=-1)
    ttn = F.normalize(teacher_txt, dim=-1)

    # L2 distance between unit vectors. Note: ||a-b||^2 = 2 - 2*cos(a,b), so
    # this is mathematically the same family as cosine, but with quadratic
    # penalty for hard mismatches (good for distillation).
    l_feat_img = F.mse_loss(sin, tin)
    l_feat_txt = F.mse_loss(stn, ttn)
    l_feat = 0.5 * (l_feat_img + l_feat_txt)

    # Cosine alignment (1 - cos similarity), per-sample
    l_cos_img = (1.0 - (sin * tin).sum(dim=-1)).mean()
    l_cos_txt = (1.0 - (stn * ttn).sum(dim=-1)).mean()
    l_cos = 0.5 * (l_cos_img + l_cos_txt)

    components = {
        "feat_img": l_feat_img.item(),
        "feat_txt": l_feat_txt.item(),
        "cos_img": l_cos_img.item(),
        "cos_txt": l_cos_txt.item(),
    }

    total = feat_weight * l_feat + cos_weight * l_cos

    if infonce_weight > 0:
        logits = sin @ stn.t() / temperature
        targets = torch.arange(sin.size(0), device=sin.device)
        l_i2t = F.cross_entropy(logits, targets)
        l_t2i = F.cross_entropy(logits.t(), targets)
        l_nce = 0.5 * (l_i2t + l_t2i)
        total = total + infonce_weight * l_nce
        components["infonce"] = l_nce.item()

    return total, components


# ---------------------------------------------------------------------------
# Checkpoint I/O
# ---------------------------------------------------------------------------


def atomic_save(obj, path: Path):
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def save_checkpoint(model, optimizer, scheduler, step: int, args, output_dir: Path):
    ckpt_dir = output_dir / f"step_{step:06d}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    atomic_save(model.state_dict(), ckpt_dir / "student_state_dict.pt")
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


def find_latest(output_dir: Path) -> Optional[Path]:
    if not output_dir.exists():
        return None
    cands = sorted(output_dir.glob("step_*"))
    return cands[-1] if cands else None


def write_eval_ready_state_dict(model, output_dir: Path):
    """Write state_dict to <output-dir>/best/model_state_dict.pt for the eval harness."""
    final_path = output_dir / "best" / "model_state_dict.pt"
    final_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_save(model.state_dict(), final_path)
    log.info("wrote eval-ready state_dict -> %s", final_path)


# ---------------------------------------------------------------------------
# Train loop
# ---------------------------------------------------------------------------


def train(args):
    device = args.device
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log.info("device=%s output=%s", device, output_dir)

    model, preprocess, tokenizer = build_student(device, args.init_from)
    dataset = CachedDistillDataset(Path(args.teacher_cache), preprocess)
    if len(dataset) == 0:
        log.error("empty dataset"); sys.exit(2)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=lambda b: collate(b, tokenizer),
        drop_last=True,
    )

    params = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.max_steps)

    start_step = 0
    if args.resume:
        ckpt = find_latest(output_dir) if args.resume == "auto" else Path(args.resume)
        if ckpt and ckpt.exists():
            log.info("resuming from %s", ckpt)
            sd = torch.load(ckpt / "student_state_dict.pt", map_location="cpu", weights_only=False)
            model.load_state_dict(sd)
            model.to(device)
            ts = torch.load(ckpt / "trainer_state.pt", map_location="cpu", weights_only=False)
            optim.load_state_dict(ts["optimizer"])
            sched.load_state_dict(ts["scheduler"])
            random.setstate(ts["rng_python"])
            np.random.set_state(ts["rng_numpy"])
            torch.set_rng_state(ts["rng_torch"])
            start_step = ts["step"]
        else:
            log.warning("--resume=%s but no checkpoint found, starting fresh", args.resume)

    model.train()
    step = start_step
    t0 = time.time()
    losses: list[float] = []
    log.info(
        "training: %d -> %d steps  bs=%d  lr=%.1e  feat_w=%.2f cos_w=%.2f infonce_w=%.2f",
        step, args.max_steps, args.batch_size, args.lr,
        args.feat_weight, args.cos_weight, args.infonce_weight,
    )

    while step < args.max_steps:
        for images, tokens, teacher_img, teacher_txt in loader:
            if step >= args.max_steps:
                break
            # NOTE: do NOT use non_blocking=True on MPS — it can cause use-after-free
            # races where the model reads tensors before the H2D copy completes,
            # producing NaN forward outputs we observed in v0 of this trainer.
            images = images.to(device)
            tokens = tokens.to(device)
            teacher_img = teacher_img.to(device)
            teacher_txt = teacher_txt.to(device)

            student_img = model.encode_image(images)
            student_txt = model.encode_text(tokens)

            loss, components = distillation_loss(
                student_img, student_txt, teacher_img, teacher_txt,
                feat_weight=args.feat_weight,
                cos_weight=args.cos_weight,
                infonce_weight=args.infonce_weight,
                temperature=args.temperature,
            )

            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optim.step()
            sched.step()

            losses.append(loss.item())
            step += 1
            if step % args.log_every == 0 or step == start_step + 1:
                elapsed = time.time() - t0
                window = losses[-args.log_every :] if len(losses) >= args.log_every else losses
                log.info(
                    "step %4d/%d  loss=%.4f (avg%d=%.4f)  cos_img=%.4f  cos_txt=%.4f  feat=%.4f  lr=%.2e  step/s=%.2f",
                    step, args.max_steps, losses[-1], len(window),
                    sum(window) / len(window),
                    components["cos_img"], components["cos_txt"],
                    0.5 * (components["feat_img"] + components["feat_txt"]),
                    sched.get_last_lr()[0],
                    (step - start_step) / max(elapsed, 1e-6),
                )
            if args.save_every and step % args.save_every == 0:
                save_checkpoint(model, optim, sched, step, args, output_dir)
                # Also update best/model_state_dict.pt so an inline eval can score it.
                write_eval_ready_state_dict(model, output_dir)

    log.info("training done in %.1fs (final step %d)", time.time() - t0, step)
    write_eval_ready_state_dict(model, output_dir)

    meta = {
        "init_from": args.init_from,
        "teacher_cache": args.teacher_cache,
        "n_pairs": len(dataset),
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        "max_steps": args.max_steps,
        "feat_weight": args.feat_weight,
        "cos_weight": args.cos_weight,
        "infonce_weight": args.infonce_weight,
        "temperature": args.temperature,
        "loss": "L2 + cos + (optional) InfoNCE on cached teacher embeddings",
        "final_loss": losses[-1] if losses else None,
        "wall_time_sec": time.time() - t0,
    }
    (output_dir / "run_meta.json").write_text(json.dumps(meta, indent=2))
    log.info("wrote %s", output_dir / "run_meta.json")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--teacher-cache", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument(
        "--init-from",
        default="ViT-B-16-SigLIP",
        help="open_clip identifier for student init (must match teacher arch). "
        "ViT-B-16-SigLIP = Google SigLIP B/16/224 webli (no fashion training).",
    )
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-steps", type=int, default=500)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--feat-weight", type=float, default=1.0)
    p.add_argument("--cos-weight", type=float, default=1.0)
    p.add_argument("--infonce-weight", type=float, default=0.0,
                   help="DEFAULT 0.0 — InfoNCE collapsed v1/v2 LoRA runs. Enable cautiously.")
    p.add_argument("--temperature", type=float, default=0.07)
    p.add_argument("--save-every", type=int, default=100)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--resume", default=None)
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
