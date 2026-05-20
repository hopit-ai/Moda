"""
Phase C — smoke training loop for v5.

Goals (PLAN_V5 §4):
  - Verify the full forward + loss + backward + optimizer step works on MPS
  - Verify training loss decreases over N steps
  - Verify cache reads + multi-field encoding + KL teacher all interoperate
  - Produce a checkpoint usable for downstream eval

This script is intentionally minimal — no eval probe yet. It writes
training-loss curves to logs/v5_phase_c_smoke.jsonl and a final checkpoint.

Defaults run on the 50K dataset with v4 regex labels if pairs_labeled.jsonl
isn't ready yet, so we can validate the training loop end-to-end before
the full LLM extraction completes.

Usage:
    python scripts/v5/phase_c_smoke.py --steps 200 --batch_K 8 --batch_N 16
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from v5_dataset import V5Dataset
from v5_loss import (
    anchor_text_loss, fusion_kl_loss, get_loss_coefficients, grouped_gcl_loss,
)
from v5_model import build_student, count_trainable, trainable_parameter_groups

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data" / "processed" / "v5_multifield"
LOGS = REPO / "logs"
CKPT_DIR = REPO / "checkpoints" / "v5"


def pick_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=Path, default=None,
                    help="Defaults to pairs_50k_labeled.jsonl if present, else pairs_50k.jsonl")
    ap.add_argument("--steps", type=int, default=200, help="Total training steps")
    ap.add_argument("--batch_K", type=int, default=8, help="Queries per batch")
    ap.add_argument("--batch_N", type=int, default=16, help="Products per query per batch")
    ap.add_argument("--anchor_size", type=int, default=256,
                    help="Number of anchor queries per drift-loss step")
    ap.add_argument("--checkpoint_every", type=int, default=200)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    LOGS.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOGS / "v5_phase_c_smoke.jsonl"
    ckpt_path = CKPT_DIR / "smoke_final.pt"

    # ─── data path resolution ────────────────────────────────────────────
    labeled = DATA / "pairs_50k_labeled.jsonl"
    unlabeled = DATA / "pairs_50k.jsonl"
    if args.pairs is not None:
        pairs_path = args.pairs
    elif labeled.exists():
        # Use labeled file ONLY if it has substantial coverage
        n_labeled = sum(1 for _ in labeled.open())
        if n_labeled >= 5000:
            pairs_path = labeled
            print(f"Using labeled pairs ({n_labeled:,} records)")
        else:
            pairs_path = unlabeled
            print(f"Labeled pairs only at {n_labeled:,} records; falling back to "
                  f"unlabeled (pairs_50k.jsonl) with v4 regex categories")
    else:
        pairs_path = unlabeled
        print("Using unlabeled pairs (pairs_50k.jsonl) with v4 regex categories")

    device = args.device or pick_device()
    print(f"Device: {device}")

    # ─── caches ──────────────────────────────────────────────────────────
    print("Loading caches into RAM ...")
    student_img_cache = torch.load(DATA / "student_image_emb.pt", map_location="cpu").to(device)
    fsl_img_cache = torch.load(DATA / "teacher_fsl_img_emb.pt", map_location="cpu").to(device)
    fsl_text_cache = torch.load(DATA / "teacher_fsl_text_emb.pt", map_location="cpu",
                                 weights_only=False)
    sl2_text_cache = torch.load(DATA / "teacher_sl2_text_emb.pt", map_location="cpu",
                                 weights_only=False)
    print(f"  student_img_cache: {tuple(student_img_cache.shape)} {student_img_cache.dtype}")
    print(f"  fsl_img_cache:     {tuple(fsl_img_cache.shape)} {fsl_img_cache.dtype}")
    print(f"  fsl_text_cache:    {len(fsl_text_cache):,} queries")
    print(f"  sl2_text_cache:    {len(sl2_text_cache):,} queries")

    # ─── model ───────────────────────────────────────────────────────────
    print("Building student ...")
    model, tokenizer = build_student(device=device)
    counts = count_trainable(model)
    print(f"  trainable: {counts['total']/1e6:.1f}M params")

    # ─── dataset ─────────────────────────────────────────────────────────
    image_index_path = DATA / "student_image_index.json"
    ds = V5Dataset(pairs_path, image_index_path,
                   K=args.batch_K, N=args.batch_N, min_products_per_query=2)
    print(f"  dataset: {ds.stats()}")

    # Cache an anchor query set: snapshot frozen SL2 text embeddings for ~256 queries
    anchor_queries = list(sl2_text_cache.keys())[: args.anchor_size]
    anchor_init = torch.stack([sl2_text_cache[q] for q in anchor_queries]).to(device).float()
    anchor_init = F.normalize(anchor_init, dim=-1)
    print(f"  anchor set: {len(anchor_queries)} queries")

    # ─── optimizer ───────────────────────────────────────────────────────
    optim = torch.optim.AdamW(trainable_parameter_groups(model))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optim, lr_lambda=lambda step: max(0.1, 1.0 - step / max(1, args.steps))
    )

    # ─── training loop ───────────────────────────────────────────────────
    log_f = log_path.open("w")
    model.train()
    step = 0
    t_start = time.time()
    epoch = 0
    pbar = tqdm(total=args.steps, desc="smoke")
    while step < args.steps:
        epoch += 1
        for b in ds.iter_batches(shuffle=True):
            if step >= args.steps:
                break

            # ---- text encoding (the only model forward each step) ----
            tok_q = tokenizer(b.query).to(device)
            tok_t = tokenizer(b.title).to(device)
            tok_c = tokenizer(b.category_l2).to(device)
            text_q = model.encode_text(tok_q)
            text_t = model.encode_text(tok_t)
            text_c = model.encode_text(tok_c)
            text_multi = F.normalize(0.6 * text_q + 0.3 * text_t + 0.1 * text_c, dim=-1)

            # ---- cache lookups (no model forward) ----
            img_emb = F.normalize(student_img_cache[b.image_idx].float(), dim=-1)

            # Plain cosine for the KL term (both sides comparable);
            # scaled+biased form for the SigLIP-style GCL term.
            cos_scores = text_multi @ img_emb.T
            scale = model.logit_scale.exp()
            bias = model.logit_bias if hasattr(model, "logit_bias") else 0.0
            scores = cos_scores * scale + bias

            # ---- losses ----
            l_gcl = grouped_gcl_loss(
                scores, b.query_idx.to(device), b.score_linear.to(device), b.K
            )

            # Anchor: re-encode anchor queries, compare to frozen init
            tok_a = tokenizer(anchor_queries).to(device)
            anchor_now = F.normalize(model.encode_text(tok_a), dim=-1)
            l_anchor = anchor_text_loss(anchor_now, anchor_init)

            # Fusion KL: build teacher score matrix from cached embeddings
            fsl_img = F.normalize(fsl_img_cache[b.image_idx].float(), dim=-1)
            sl2_img = img_emb  # already normalized
            # Teacher text: stack cached fsl/sl2 text per query in the batch
            try:
                fsl_t = torch.stack([fsl_text_cache[q] for q in b.query]).to(device).float()
                sl2_t = torch.stack([sl2_text_cache[q] for q in b.query]).to(device).float()
                fsl_t = F.normalize(fsl_t, dim=-1)
                sl2_t = F.normalize(sl2_t, dim=-1)
                # Pass PLAIN cosine for student so it's dimensionally comparable
                # to the teacher cosine score matrix.
                l_kl = fusion_kl_loss(cos_scores, fsl_t, fsl_img, sl2_t, sl2_img)
            except KeyError as e:
                # Some query missing from teacher cache (would only happen if dataset
                # somehow drifted from cache build); skip KL term for this step
                l_kl = torch.tensor(0.0, device=device)

            lam_anchor, lam_kl = get_loss_coefficients(step)
            loss = l_gcl + lam_anchor * l_anchor + lam_kl * l_kl

            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], max_norm=1.0
            )
            optim.step()
            scheduler.step()

            entry = {
                "step": step,
                "epoch": epoch,
                "loss": loss.item(),
                "loss_gcl": l_gcl.item(),
                "loss_anchor": l_anchor.item(),
                "loss_kl": float(l_kl.item()) if torch.is_tensor(l_kl) else l_kl,
                "lam_anchor": lam_anchor,
                "lam_kl": lam_kl,
                "lr": scheduler.get_last_lr()[0],
                "elapsed_s": time.time() - t_start,
            }
            log_f.write(json.dumps(entry) + "\n")
            log_f.flush()
            pbar.update(1)
            pbar.set_postfix(loss=f"{loss.item():.3f}",
                             gcl=f"{l_gcl.item():.3f}",
                             anc=f"{l_anchor.item():.6f}",
                             kl=f"{l_kl.detach().item():.3f}" if torch.is_tensor(l_kl) else f"{l_kl:.3f}")
            step += 1

            if args.checkpoint_every > 0 and step % args.checkpoint_every == 0:
                torch.save(
                    {"step": step,
                     "model_trainable": {n: p.detach().cpu()
                                         for n, p in model.named_parameters()
                                         if p.requires_grad}},
                    ckpt_path,
                )

    pbar.close()
    log_f.close()

    elapsed = time.time() - t_start
    print(f"\nSmoke done in {elapsed:.1f}s ({step} steps, {step/elapsed:.2f} step/s)")
    print(f"Log:        {log_path}")
    print(f"Checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()
