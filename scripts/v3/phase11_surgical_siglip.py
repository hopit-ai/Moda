"""Phase 11 — Surgical text_projection training with SigLIP contrastive loss.

Trains ONLY the text_projection head (590K params) + logit_scale/logit_bias
using direct sigmoid contrastive loss on (title, image) pairs.

Strategy: Pre-compute all frozen encoder outputs ONCE, then train the projection
layer on cached embeddings. This makes each training step ~100x faster since
we skip the expensive encoder forward passes during training.

Usage:
  python3 -u scripts/v3/phase11_surgical_siglip.py
  python3 -u scripts/v3/phase11_surgical_siglip.py --epochs 20 --batch-size 256 --lr 1e-4
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, TensorDataset
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("phase11-surgical")

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data" / "processed" / "v3_phase10_500k"
CHECKPOINT_DIR = REPO_ROOT / "checkpoints" / "v3_phase11"


class ProjectionHead(nn.Module):
    """Standalone text projection head for fast training on cached embeddings."""

    def __init__(self, dim=768):
        super().__init__()
        self.proj = nn.Linear(dim, dim, bias=True)
        self.logit_scale = nn.Parameter(torch.tensor(4.6052))
        self.logit_bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, text_features):
        projected = self.proj(text_features)
        return F.normalize(projected, dim=-1)


def siglip_loss(text_embs, image_embs, logit_scale, logit_bias):
    """SigLIP sigmoid contrastive loss."""
    logits = text_embs @ image_embs.T * logit_scale.exp() + logit_bias
    B = logits.shape[0]
    labels = 2 * torch.eye(B, device=logits.device) - 1
    loss = -F.logsigmoid(labels * logits).mean()
    return loss


def precompute_embeddings(model, tokenizer, preprocess, device, pairs_jsonl, images_dir, max_pairs=None):
    """Pre-compute all text (pre-projection) and image embeddings."""
    log.info("Pre-computing embeddings (one-time cost)...")

    pairs = []
    with open(pairs_jsonl) as f:
        for line in f:
            item = json.loads(line)
            title = item.get("title", "").strip()
            img_path = item.get("image_path", "")
            if title and img_path:
                pairs.append((title, img_path, item.get("l1_category", "unknown")))

    if max_pairs and len(pairs) > max_pairs:
        # Stratified sampling by l1_category
        from collections import defaultdict
        by_cat = defaultdict(list)
        for p in pairs:
            by_cat[p[2]].append(p)
        
        total = len(pairs)
        sampled = []
        for cat, items in by_cat.items():
            proportion = len(items) / total
            n_take = max(1, int(max_pairs * proportion))
            random.shuffle(items)
            sampled.extend(items[:n_take])
        
        random.shuffle(sampled)
        pairs = sampled[:max_pairs]
        log.info("Stratified sample: %d pairs from %d categories", len(pairs), len(by_cat))
    
    pairs = [(p[0], p[1]) for p in pairs]  # drop category column

    log.info("Processing %d pairs...", len(pairs))

    model.eval()
    all_text_pre_proj = []
    all_image_embs = []

    batch_size = 128
    t0 = time.time()

    for i in tqdm(range(0, len(pairs), batch_size), desc="Encoding"):
        batch_pairs = pairs[i:i + batch_size]
        titles = [p[0] for p in batch_pairs]
        img_paths = [p[1] for p in batch_pairs]

        # Encode images
        imgs = []
        for img_path in img_paths:
            full_path = images_dir / Path(img_path).name
            if not full_path.exists():
                full_path = images_dir.parent / img_path
            try:
                img = preprocess(Image.open(full_path).convert("RGB"))
            except Exception:
                img = preprocess(Image.new("RGB", (224, 224)))
            imgs.append(img)

        img_batch = torch.stack(imgs).to(device)
        tokens = tokenizer(titles).to(device)

        with torch.no_grad():
            # Image embeddings (fully frozen, final output)
            image_embs = model.encode_image(img_batch)
            image_embs = F.normalize(image_embs, dim=-1)

            # Text: get pre-projection features
            # We need features BEFORE text_projection is applied
            text = model.text
            x = text.token_embedding(tokens)
            x = x + text.positional_embedding
            x = x.permute(1, 0, 2)  # NLD -> LND
            x = text.transformer(x)
            x = x.permute(1, 0, 2)  # LND -> NLD
            x = text.ln_final(x)
            # Pool: take features from the EOS token position
            # For SigLIP, pooling is at the argmax of token ids
            pooled = x[torch.arange(x.shape[0]), tokens.argmax(dim=-1)]

        all_text_pre_proj.append(pooled.cpu())
        all_image_embs.append(image_embs.cpu())

        if (i // batch_size) % 200 == 0 and i > 0:
            elapsed = time.time() - t0
            done = i + batch_size
            eta = elapsed / done * (len(pairs) - done)
            log.info("  %d/%d done (%.0fs elapsed, ETA %.0fs)", done, len(pairs), elapsed, eta)

    all_text_pre_proj = torch.cat(all_text_pre_proj, dim=0)
    all_image_embs = torch.cat(all_image_embs, dim=0)

    elapsed = time.time() - t0
    log.info("Embedding pre-computation done: %d pairs in %.0fs (%.0f pairs/s)",
             len(pairs), elapsed, len(pairs) / elapsed)

    return all_text_pre_proj, all_image_embs


def quick_eval_with_proj(proj_head, model, tokenizer, preprocess, device, corpus_size=3000):
    """Quick T2I eval on fashion200k using the projection head."""
    from datasets import load_dataset

    ds = load_dataset("Marqo/fashion200k", split="data")
    indices = list(range(len(ds)))
    np.random.seed(42)
    np.random.shuffle(indices)
    indices = indices[:corpus_size]
    ds = ds.select(indices)

    text_col = "caption" if "caption" in ds.column_names else "title"
    texts = [str(ds[i][text_col] or "") for i in range(len(ds))]
    valid_mask = [len(t.strip()) > 3 for t in texts]
    query_indices = [i for i, v in enumerate(valid_mask) if v][:min(2000, corpus_size)]

    model.eval()
    proj_head.eval()

    with torch.no_grad():
        # Encode images
        img_embs = []
        for i in range(0, len(ds), 64):
            batch_imgs = []
            for j in range(i, min(i + 64, len(ds))):
                try:
                    img_t = preprocess(ds[j]["image"].convert("RGB"))
                except Exception:
                    img_t = preprocess(Image.new("RGB", (224, 224)))
                batch_imgs.append(img_t)
            img_batch = torch.stack(batch_imgs).to(device)
            emb = model.encode_image(img_batch)
            emb = F.normalize(emb, dim=-1)
            img_embs.append(emb.cpu())
        img_embs = torch.cat(img_embs, dim=0)

        # Encode text with our custom projection
        query_texts = [texts[i] for i in query_indices]
        txt_embs = []
        for i in range(0, len(query_texts), 64):
            batch_texts = query_texts[i:i + 64]
            tokens = tokenizer(batch_texts).to(device)

            text_module = model.text
            x = text_module.token_embedding(tokens)
            x = x + text_module.positional_embedding
            x = x.permute(1, 0, 2)
            x = text_module.transformer(x)
            x = x.permute(1, 0, 2)
            x = text_module.ln_final(x)
            pooled = x[torch.arange(x.shape[0]), tokens.argmax(dim=-1)]

            projected = proj_head(pooled.to(device))
            txt_embs.append(projected.cpu())
        txt_embs = torch.cat(txt_embs, dim=0)

    sims = txt_embs @ img_embs.T
    recall_1 = 0
    recall_10 = 0
    mrr = 0.0
    for qi, corpus_idx in enumerate(query_indices):
        ranked = sims[qi].argsort(descending=True)
        rank = (ranked == corpus_idx).nonzero(as_tuple=True)[0].item()
        if rank < 1:
            recall_1 += 1
        if rank < 10:
            recall_10 += 1
        mrr += 1.0 / (rank + 1)

    n = len(query_indices)
    r1 = recall_1 / n
    r10 = recall_10 / n
    avg_recall = (r1 + r10) / 2
    return {"recall_1": r1, "recall_10": r10, "mrr": mrr / n, "avg_recall": avg_recall, "n": n}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--warmup-pct", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument("--eval-every", type=int, default=5, help="Eval every N epochs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-name", type=str, default="phase11_surgical")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    log.info("Device: %s", device)

    # Load FSL model
    import open_clip
    log.info("Loading Marqo-FashionSigLIP...")
    model, _, preprocess = open_clip.create_model_and_transforms("hf-hub:Marqo/marqo-fashionSigLIP")
    tokenizer = open_clip.get_tokenizer("hf-hub:Marqo/marqo-fashionSigLIP")
    model = model.to(device)
    model.eval()

    # Pre-compute all embeddings (one-time cost)
    pairs_path = DATA_DIR / "pairs.jsonl"
    images_dir = DATA_DIR / "images"

    cache_path = CHECKPOINT_DIR / "embedding_cache.pt"
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_path.exists():
        log.info("Loading cached embeddings from %s", cache_path)
        cache = torch.load(cache_path, map_location="cpu", weights_only=True)
        text_pre_proj = cache["text_pre_proj"]
        image_embs = cache["image_embs"]
    else:
        text_pre_proj, image_embs = precompute_embeddings(
            model, tokenizer, preprocess, device, pairs_path, images_dir, max_pairs=args.max_pairs
        )
        torch.save({"text_pre_proj": text_pre_proj, "image_embs": image_embs}, cache_path)
        log.info("Saved embedding cache: %s (%.1f MB)", cache_path,
                 cache_path.stat().st_size / 1e6)

    N = text_pre_proj.shape[0]
    log.info("Cached embeddings: %d pairs, text=%s, image=%s", N, text_pre_proj.shape, image_embs.shape)

    # Initialize projection head from FSL's existing text_projection
    proj_head = ProjectionHead(dim=768).to(device)
    with torch.no_grad():
        proj_head.proj.weight.copy_(model.text.text_projection.weight)
        proj_head.proj.bias.copy_(model.text.text_projection.bias)
        proj_head.logit_scale.copy_(model.logit_scale)
        proj_head.logit_bias.copy_(model.logit_bias)

    total_trainable = sum(p.numel() for p in proj_head.parameters())
    log.info("Trainable parameters: %d (%.2fK)", total_trainable, total_trainable / 1000)

    # Create fast tensor dataloader
    dataset = TensorDataset(text_pre_proj, image_embs)
    steps_per_epoch = N // args.batch_size
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(total_steps * args.warmup_pct)

    log.info("Steps/epoch: %d, Total steps: %d, Warmup: %d", steps_per_epoch, total_steps, warmup_steps)

    # Optimizer
    optimizer = torch.optim.AdamW(
        proj_head.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.98),
        eps=1e-6,
    )

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return args.min_lr / args.lr + (1 - args.min_lr / args.lr) * 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Save directory
    save_dir = CHECKPOINT_DIR / args.run_name
    save_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "min_lr": args.min_lr,
        "warmup_pct": args.warmup_pct,
        "weight_decay": args.weight_decay,
        "trainable_params": total_trainable,
        "trainable_layers": ["text_projection.weight", "text_projection.bias", "logit_scale", "logit_bias"],
        "loss": "siglip_sigmoid_contrastive",
        "dataset_size": N,
        "steps_per_epoch": steps_per_epoch,
        "total_steps": total_steps,
        "strategy": "pre-cached embeddings, train projection only",
    }
    with open(save_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Training loop
    log.info("=" * 60)
    log.info("Phase 11 — Surgical text_projection training (CACHED)")
    log.info("Loss: SigLIP sigmoid contrastive")
    log.info("Trainable: text_projection + logit_scale + logit_bias (%dK params)", total_trainable // 1000)
    log.info("Epochs: %d, Batch: %d, LR: %.1e", args.epochs, args.batch_size, args.lr)
    log.info("Strategy: Pre-cached embeddings → pure tensor training")
    log.info("=" * 60)

    global_step = 0
    best_avg_recall = 0.0
    eval_history = []

    for epoch in range(1, args.epochs + 1):
        proj_head.train()
        epoch_loss = 0.0
        epoch_steps = 0
        t0 = time.time()

        # Shuffle indices each epoch
        perm = torch.randperm(N)
        
        for i in range(0, N - args.batch_size + 1, args.batch_size):
            idx = perm[i:i + args.batch_size]
            text_batch = text_pre_proj[idx].to(device)
            img_batch = image_embs[idx].to(device)

            # Forward through projection head only
            text_projected = proj_head(text_batch)

            # SigLIP loss
            loss = siglip_loss(text_projected, img_batch, proj_head.logit_scale, proj_head.logit_bias)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(proj_head.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            epoch_steps += 1
            global_step += 1

            if global_step % 500 == 0:
                lr_now = scheduler.get_last_lr()[0]
                log.info(
                    "[%d/%d] step=%d loss=%.4f lr=%.2e scale=%.3f bias=%.3f",
                    epoch, args.epochs, global_step, loss.item(), lr_now,
                    proj_head.logit_scale.item(), proj_head.logit_bias.item(),
                )

        elapsed = time.time() - t0
        avg_loss = epoch_loss / max(epoch_steps, 1)
        log.info(
            "Epoch %d done: avg_loss=%.4f time=%.1fs (%.0f steps/s)",
            epoch, avg_loss, elapsed, epoch_steps / elapsed,
        )

        # Eval every N epochs
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            log.info("Evaluating on fashion200k (3K corpus)...")

            # Copy trained projection back to model for eval
            with torch.no_grad():
                model.text.text_projection.weight.copy_(proj_head.proj.weight)
                model.text.text_projection.bias.copy_(proj_head.proj.bias)
                model.logit_scale.copy_(proj_head.logit_scale)
                model.logit_bias.copy_(proj_head.logit_bias)

            metrics = quick_eval_with_proj(proj_head, model, tokenizer, preprocess, device, corpus_size=3000)
            log.info(
                "  R@1=%.4f R@10=%.4f MRR=%.4f AvgRecall=%.4f",
                metrics["recall_1"], metrics["recall_10"], metrics["mrr"], metrics["avg_recall"],
            )
            eval_history.append({"epoch": epoch, "loss": avg_loss, **metrics})

            # Save checkpoint
            ckpt = {
                "epoch": epoch,
                "global_step": global_step,
                "proj_state_dict": proj_head.state_dict(),
                "avg_loss": avg_loss,
                "metrics": metrics,
            }

            if metrics["avg_recall"] > best_avg_recall:
                best_avg_recall = metrics["avg_recall"]
                torch.save(ckpt, save_dir / "best.pt")
                log.info("  ** New best! AvgRecall=%.4f **", best_avg_recall)

            torch.save(ckpt, save_dir / f"epoch_{epoch:02d}.pt")

    # Save eval history
    with open(save_dir / "eval_history.json", "w") as f:
        json.dump(eval_history, f, indent=2)

    log.info("=" * 60)
    log.info("Training complete!")
    log.info("Best AvgRecall: %.4f", best_avg_recall)
    log.info("Checkpoints: %s", save_dir)
    log.info("=" * 60)


if __name__ == "__main__":
    main()
