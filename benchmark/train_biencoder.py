"""
MODA Phase 3C — Bi-Encoder Fine-Tuning with Retriever-Mined Hard Negatives

Fine-tunes FashionCLIP (Marqo/marqo-fashionCLIP) text encoder using InfoNCE
contrastive loss. Hard negatives are mined from FashionCLIP's own top-K
retrieval results, then scored by GPT-4o-mini — so the model learns exactly
where it currently fails.

Training data structure (from generate_biencoder_labels.py):
  For each query:
    - Positives: products scored 2-3 by LLM (good/exact match)
    - Hard negatives: products scored 0 by LLM but ranked highly by retriever

Loss: InfoNCE with in-batch negatives + one mined hard negative per query.
This is the same contrastive objective used in CLIP pre-training, applied
specifically to our fashion domain.

Output:
  models/moda-fashionclip-finetuned/    fine-tuned model checkpoint

Usage:
  python benchmark/train_biencoder.py                       # full run
  python benchmark/train_biencoder.py --epochs 1 --quick    # quick test
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

PROCESSED_DIR = _REPO_ROOT / "data" / "processed"
MODEL_DIR = _REPO_ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

LABELS_PATH = PROCESSED_DIR / "biencoder_retriever_labels.jsonl"
SPLIT_PATH = PROCESSED_DIR / "query_splits.json"

OUTPUT_DIR = MODEL_DIR / "moda-fashionclip-finetuned"
RANDOM_SEED = 42


class ContrastiveTripletDataset(Dataset):
    """Dataset that yields (query, positive, hard_negative) text triplets."""

    def __init__(self, triplets: list[tuple[str, str, str]], tokenizer):
        self.triplets = triplets
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        query, pos, neg = self.triplets[idx]
        q_tok = self.tokenizer([query])[0]
        p_tok = self.tokenizer([pos])[0]
        n_tok = self.tokenizer([neg])[0]
        return q_tok, p_tok, n_tok


def load_triplets(
    max_pairs: int | None = None,
    seed: int = RANDOM_SEED,
) -> tuple[list[tuple[str, str, str]], list[tuple[str, str, str]]]:
    """Build contrastive triplets from LLM-labeled retriever results.

    Groups labels by query, selects positives (score >= 2) and
    hard negatives (score == 0), then forms all valid triplets.
    Uses query_splits.json to separate train/val (no test leakage).
    """
    rng = random.Random(seed)

    splits = json.loads(SPLIT_PATH.read_text())
    train_qids = set(splits["train"])
    val_qids = set(splits["val"])

    labels_by_query: dict[str, list[dict]] = defaultdict(list)
    with open(LABELS_PATH) as f:
        for line in f:
            if line.strip():
                obj = json.loads(line)
                labels_by_query[obj["query_id"]].append(obj)

    log.info("Loaded labels for %d unique queries", len(labels_by_query))

    def extract_triplets(qids: set[str]) -> list[tuple[str, str, str]]:
        triplets = []
        for qid, items in labels_by_query.items():
            if qid not in qids:
                continue
            query_text = items[0]["query_text"]
            positives = [it["product_text"] for it in items if it["score"] >= 2]
            hard_negs = [it["product_text"] for it in items if it["score"] == 0]
            if not positives or not hard_negs:
                continue
            for pos in positives:
                neg = rng.choice(hard_negs)
                triplets.append((query_text, pos, neg))
        return triplets

    train_triplets = extract_triplets(train_qids)
    val_triplets = extract_triplets(val_qids)

    rng.shuffle(train_triplets)
    rng.shuffle(val_triplets)

    if max_pairs and len(train_triplets) > max_pairs:
        train_triplets = train_triplets[:max_pairs]

    if not val_triplets and train_triplets:
        n_val = max(200, int(len(train_triplets) * 0.10))
        val_triplets = train_triplets[:n_val]
        train_triplets = train_triplets[n_val:]
        log.info("No val-split labels; held out %d from train for validation", n_val)

    log.info("Train triplets: %d  Val triplets: %d", len(train_triplets), len(val_triplets))
    return train_triplets, val_triplets


def infonce_loss_with_hard_neg(
    q_emb: torch.Tensor,
    pos_emb: torch.Tensor,
    neg_emb: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """InfoNCE loss with in-batch negatives + one hard negative per query.

    For each query i:
      numerator = exp(sim(q_i, pos_i) / tau)
      denominator = numerator
                  + sum_j!=i exp(sim(q_i, pos_j) / tau)    [in-batch negs]
                  + exp(sim(q_i, hard_neg_i) / tau)         [mined hard neg]
    """
    batch_size = q_emb.shape[0]

    q_emb = F.normalize(q_emb, dim=-1)
    pos_emb = F.normalize(pos_emb, dim=-1)
    neg_emb = F.normalize(neg_emb, dim=-1)

    pos_sim = (q_emb * pos_emb).sum(dim=-1, keepdim=True) / temperature
    inbatch_sim = (q_emb @ pos_emb.T) / temperature
    hard_neg_sim = (q_emb * neg_emb).sum(dim=-1, keepdim=True) / temperature

    all_logits = torch.cat([inbatch_sim, hard_neg_sim], dim=-1)
    labels = torch.arange(batch_size, device=q_emb.device)

    return F.cross_entropy(all_logits, labels)


@torch.no_grad()
def evaluate_retrieval(model, tokenizer, val_triplets, device, batch_size=128):
    """Compute retrieval accuracy on val triplets: what fraction of times
    does the model rank the positive above the hard negative?"""
    model.eval()
    correct = 0
    total = 0
    use_amp = device == "mps"

    for start in range(0, len(val_triplets), batch_size):
        batch = val_triplets[start:start + batch_size]
        queries = [t[0] for t in batch]
        positives = [t[1] for t in batch]
        negatives = [t[2] for t in batch]

        q_tok = tokenizer(queries).to(device, non_blocking=True)
        p_tok = tokenizer(positives).to(device, non_blocking=True)
        n_tok = tokenizer(negatives).to(device, non_blocking=True)

        if use_amp:
            with torch.autocast(device_type="mps", dtype=torch.float16):
                q_emb = F.normalize(model.encode_text(q_tok), dim=-1)
                p_emb = F.normalize(model.encode_text(p_tok), dim=-1)
                n_emb = F.normalize(model.encode_text(n_tok), dim=-1)
        else:
            q_emb = F.normalize(model.encode_text(q_tok), dim=-1)
            p_emb = F.normalize(model.encode_text(p_tok), dim=-1)
            n_emb = F.normalize(model.encode_text(n_tok), dim=-1)

        pos_sim = (q_emb * p_emb).sum(dim=-1)
        neg_sim = (q_emb * n_emb).sum(dim=-1)

        correct += (pos_sim > neg_sim).sum().item()
        total += len(batch)

    accuracy = correct / total if total > 0 else 0.0
    return accuracy


def train(args):
    log.info("=" * 60)
    log.info("MODA Phase 3C — Bi-Encoder Fine-Tuning")
    log.info("Model: FashionCLIP (Marqo/marqo-fashionCLIP)")
    log.info("Loss: InfoNCE + hard negatives from retriever mining")
    log.info("=" * 60)

    import open_clip

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    use_amp = device == "mps"
    log.info("Device: %s | AMP (float16): %s", device, use_amp)

    log.info("Loading FashionCLIP...")
    model, _, _ = open_clip.create_model_and_transforms("hf-hub:Marqo/marqo-fashionCLIP")
    tokenizer = open_clip.get_tokenizer("hf-hub:Marqo/marqo-fashionCLIP")
    model = model.to(device)

    train_triplets, val_triplets = load_triplets(max_pairs=args.max_pairs)

    if not train_triplets:
        log.error("No training triplets! Check labels and splits.")
        return

    log.info("Sample triplet:")
    q, p, n = train_triplets[0]
    log.info("  Q: %s", q[:80])
    log.info("  P: %s", p[:80])
    log.info("  N: %s", n[:80])

    base_acc = evaluate_retrieval(model, tokenizer, val_triplets[:2000], device)
    log.info("Baseline val retrieval accuracy: %.3f", base_acc)

    model.train()
    for param in model.visual.parameters():
        param.requires_grad = False
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    log.info("Trainable parameters: %d / %d (%.1f%% — text encoder only)",
             trainable, total_params, 100 * trainable / total_params)

    micro_batch = args.batch_size // args.grad_accum
    dataset = ContrastiveTripletDataset(train_triplets, tokenizer)
    dataloader = DataLoader(
        dataset,
        batch_size=micro_batch,
        shuffle=True,
        num_workers=0,
        drop_last=True,
        pin_memory=False,
    )

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=0.01,
    )

    effective_steps = len(dataloader) // args.grad_accum
    total_steps = effective_steps * args.epochs
    warmup_steps = max(50, int(total_steps * 0.05))

    def lr_schedule(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)

    eval_steps = max(100, effective_steps // 3)
    log.info("Micro-batch: %d | Grad accum: %d | Effective batch: %d",
             micro_batch, args.grad_accum, args.batch_size)
    log.info("Steps/epoch: %d effective | Warmup: %d | Eval every: %d",
             effective_steps, warmup_steps, eval_steps)
    log.info("Starting training: %d epochs, lr=%s, temp=%.3f",
             args.epochs, args.lr, args.temperature)
    if device == "mps":
        mem_mb = torch.mps.current_allocated_memory() / 1e6
        log.info("MPS memory before training: %.0f MB", mem_mb)

    best_acc = base_acc
    global_step = 0
    t0 = time.time()

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        micro_steps = 0

        optimizer.zero_grad(set_to_none=True)

        for batch_idx, (q_tok, p_tok, n_tok) in enumerate(dataloader):
            q_tok = q_tok.to(device, non_blocking=True)
            p_tok = p_tok.to(device, non_blocking=True)
            n_tok = n_tok.to(device, non_blocking=True)

            if use_amp:
                with torch.autocast(device_type="mps", dtype=torch.float16):
                    q_emb = model.encode_text(q_tok)
                    p_emb = model.encode_text(p_tok)
                    n_emb = model.encode_text(n_tok)
                    loss = infonce_loss_with_hard_neg(
                        q_emb, p_emb, n_emb, args.temperature,
                    ) / args.grad_accum
            else:
                q_emb = model.encode_text(q_tok)
                p_emb = model.encode_text(p_tok)
                n_emb = model.encode_text(n_tok)
                loss = infonce_loss_with_hard_neg(
                    q_emb, p_emb, n_emb, args.temperature,
                ) / args.grad_accum

            loss.backward()
            epoch_loss += loss.item() * args.grad_accum
            micro_steps += 1

            if micro_steps % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if global_step % 50 == 0:
                    avg_loss = epoch_loss / micro_steps
                    lr = scheduler.get_last_lr()[0]
                    elapsed = time.time() - t0
                    step_ms = elapsed / global_step * 1000
                    log.info(
                        "  [epoch %d, step %d/%d] loss=%.4f  lr=%.2e  "
                        "%.0fms/step  (%.1f min elapsed)",
                        epoch + 1, global_step % effective_steps or effective_steps,
                        effective_steps, avg_loss, lr, step_ms, elapsed / 60,
                    )

                if global_step % eval_steps == 0:
                    acc = evaluate_retrieval(
                        model, tokenizer, val_triplets[:2000], device,
                    )
                    log.info("  → Val retrieval accuracy: %.3f (best: %.3f)",
                             acc, best_acc)
                    if acc > best_acc:
                        best_acc = acc
                        save_model(model, tokenizer, OUTPUT_DIR / "best")
                        log.info("  → New best! Saved checkpoint.")
                    model.train()

        avg_loss = epoch_loss / micro_steps if micro_steps > 0 else 0
        log.info("Epoch %d/%d complete — avg loss: %.4f", epoch + 1, args.epochs, avg_loss)

        acc = evaluate_retrieval(model, tokenizer, val_triplets[:2000], device)
        log.info("End-of-epoch val accuracy: %.3f (best: %.3f)", acc, best_acc)
        if acc > best_acc:
            best_acc = acc
            save_model(model, tokenizer, OUTPUT_DIR / "best")
            log.info("New best! Saved checkpoint.")

        if device == "mps":
            torch.mps.empty_cache()

    elapsed = time.time() - t0
    log.info("Training finished in %.1f min", elapsed / 60)
    log.info("Baseline accuracy: %.3f → Best accuracy: %.3f (+%.1f%%)",
             base_acc, best_acc, 100 * (best_acc - base_acc))

    save_model(model, tokenizer, OUTPUT_DIR)
    log.info("Final model saved → %s", OUTPUT_DIR)


def save_model(model, tokenizer, path: Path):
    """Save model weights and tokenizer config for later loading."""
    path.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path / "model_state_dict.pt")
    meta = {
        "base_model": "Marqo/marqo-fashionCLIP",
        "architecture": "hf-hub:Marqo/marqo-fashionCLIP",
        "fine_tuned": True,
        "phase": "3C",
    }
    with open(path / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    log.info("Model saved to %s", path)


def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune FashionCLIP bi-encoder")
    p.add_argument("--max_pairs", type=int, default=None)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=64,
                   help="Effective batch size (split across grad_accum steps)")
    p.add_argument("--grad_accum", type=int, default=4,
                   help="Gradient accumulation steps (default: 4, micro_batch = batch_size/grad_accum)")
    p.add_argument("--lr", type=float, default=1e-6,
                   help="Learning rate (default: 1e-6, conservative for CLIP)")
    p.add_argument("--temperature", type=float, default=0.07)
    p.add_argument("--quick", action="store_true",
                   help="Quick test: 5K pairs, 1 epoch")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.quick:
        args.max_pairs = 5_000
        args.epochs = 1
        log.info("QUICK MODE: 5K triplets, 1 epoch")
    train(args)
