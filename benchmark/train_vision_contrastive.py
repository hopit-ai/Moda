"""
MODA Phase 5-LB — Vision Encoder Contrastive Fine-Tuning for LookBench

Fine-tunes FashionSigLIP's VISION encoder only with image-to-image contrastive
learning. Goal: beat the zero-shot FashionSigLIP baseline on LookBench
(image-to-image retrieval).

Training strategy:
  - Positive pairs: H&M product images sharing the same product_type + colour
  - Hard negatives: images from different but visually similar categories
  - InfoNCE loss with in-batch negatives + explicit hard negatives
  - Alignment regularisation: KL(pretrained_sim || finetuned_sim) to prevent
    catastrophic forgetting of general visual features
  - Only vision encoder parameters are updated (text encoder frozen)

Output:
  models/moda-siglip-vision-finetuned/best/   — best checkpoint

Usage:
  python benchmark/train_vision_contrastive.py
  python benchmark/train_vision_contrastive.py --epochs 5 --batch_size 24 --lr 1e-5
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

HNM_DIR = _REPO_ROOT / "data" / "raw" / "hnm"
IMAGE_DIR = _REPO_ROOT / "data" / "raw" / "hnm_images"
OUTPUT_DIR = _REPO_ROOT / "models" / "moda-siglip-vision-finetuned"
RANDOM_SEED = 42

# Groups that are visually similar (for hard negative mining)
CONFUSABLE_GROUPS = {
    "Garment Upper body": ["Garment Full body"],
    "Garment Full body": ["Garment Upper body", "Garment Lower body"],
    "Garment Lower body": ["Garment Full body"],
}


def get_image_path(article_id: str) -> Path | None:
    aid = article_id.zfill(10)
    prefix = aid[:3]
    path = IMAGE_DIR / prefix / f"{aid}.jpg"
    return path if path.exists() else None


class VisionContrastiveDataset(Dataset):
    """Yields (anchor_img, positive_img, hard_negative_img) triplets."""

    def __init__(self, triplets: list[dict], preprocess):
        self.triplets = triplets
        self.preprocess = preprocess

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        t = self.triplets[idx]
        anchor = self._load(t["anchor_path"])
        positive = self._load(t["pos_path"])
        negative = self._load(t["neg_path"])
        return anchor, positive, negative

    def _load(self, path: str) -> torch.Tensor:
        try:
            img = Image.open(path).convert("RGB")
            return self.preprocess(img)
        except Exception:
            return torch.zeros(3, 224, 224)


def build_triplets(
    max_pairs: int = 50_000,
    val_frac: float = 0.1,
    seed: int = RANDOM_SEED,
) -> tuple[list[dict], list[dict]]:
    """Build image triplets from H&M articles grouped by product_type + colour."""
    import pandas as pd

    rng = random.Random(seed)
    df = pd.read_csv(HNM_DIR / "articles.csv", dtype=str).fillna("")

    # Build groups: product_type_name + colour_group_name
    groups: dict[str, list[dict]] = defaultdict(list)
    product_group_map: dict[str, str] = {}
    for _, row in df.iterrows():
        aid = str(row["article_id"]).strip()
        img_path = get_image_path(aid)
        if not img_path:
            continue
        key = f"{row['product_type_name']}|{row['colour_group_name']}"
        groups[key].append({
            "article_id": aid,
            "img_path": str(img_path),
            "product_type": row["product_type_name"],
            "colour": row["colour_group_name"],
            "product_group": row["product_group_name"],
        })
        product_group_map[aid] = row["product_group_name"]

    # Filter to groups with at least 2 items (need at least anchor + positive)
    valid_groups = {k: v for k, v in groups.items() if len(v) >= 2}
    log.info("Valid groups (>=2 items): %d / %d", len(valid_groups), len(groups))

    # Build index for hard negative mining
    by_product_group: dict[str, list[dict]] = defaultdict(list)
    for items in valid_groups.values():
        for item in items:
            by_product_group[item["product_group"]].append(item)

    all_items = [item for items in valid_groups.values() for item in items]
    log.info("Total items with images in valid groups: %d", len(all_items))

    triplets = []
    group_keys = list(valid_groups.keys())
    rng.shuffle(group_keys)

    for key in group_keys:
        if len(triplets) >= max_pairs:
            break
        items = valid_groups[key]
        if len(items) < 2:
            continue

        rng.shuffle(items)
        anchor_pg = items[0]["product_group"]

        # Hard negative: same product_group but different type+colour
        confusable_pgs = CONFUSABLE_GROUPS.get(anchor_pg, []) + [anchor_pg]
        hard_neg_pool = [
            it for pg in confusable_pgs
            for it in by_product_group.get(pg, [])
            if f"{it['product_type']}|{it['colour']}" != key
        ]

        if not hard_neg_pool:
            hard_neg_pool = [it for it in all_items
                            if f"{it['product_type']}|{it['colour']}" != key]

        # Create multiple pairs per group
        n_pairs = min(len(items) - 1, 5)
        for i in range(n_pairs):
            if len(triplets) >= max_pairs:
                break
            anchor = items[i]
            pos = items[(i + 1) % len(items)]
            neg = rng.choice(hard_neg_pool)
            triplets.append({
                "anchor_path": anchor["img_path"],
                "pos_path": pos["img_path"],
                "neg_path": neg["img_path"],
                "group_key": key,
            })

    rng.shuffle(triplets)
    split = int(len(triplets) * (1 - val_frac))
    train = triplets[:split]
    val = triplets[split:]
    log.info("Triplets: %d train, %d val", len(train), len(val))
    return train, val


def contrastive_loss(
    anchor_emb: torch.Tensor,
    pos_emb: torch.Tensor,
    neg_emb: torch.Tensor,
    temperature: float = 0.07,
) -> tuple[torch.Tensor, dict]:
    """Image-image InfoNCE loss."""
    a = F.normalize(anchor_emb, dim=-1)
    p = F.normalize(pos_emb, dim=-1)
    n = F.normalize(neg_emb, dim=-1)

    # InfoNCE: anchor vs in-batch positives + explicit hard negative
    pos_inbatch = (a @ p.T) / temperature  # (B, B)
    hard_neg = (a * n).sum(-1, keepdim=True) / temperature  # (B, 1)
    logits = torch.cat([pos_inbatch, hard_neg], dim=-1)  # (B, B+1)
    labels = torch.arange(a.shape[0], device=a.device)
    loss_nce = F.cross_entropy(logits, labels)

    return loss_nce, {"loss_nce": loss_nce.item()}


def weight_decay_reg(model, pretrained_state: dict, device: str) -> torch.Tensor:
    """L2 penalty on vision weight drift from pretrained values."""
    total = torch.tensor(0.0, device=device)
    n = 0
    for name, param in model.named_parameters():
        if param.requires_grad and name in pretrained_state:
            ref = pretrained_state[name].to(device)
            total = total + F.mse_loss(param, ref)
            n += 1
    return total / max(n, 1)


@torch.no_grad()
def evaluate(model, preprocess, val_triplets, device, max_eval=500):
    """Triplet accuracy: is anchor closer to positive than negative?"""
    model.eval()
    correct = 0
    total = 0

    for start in range(0, min(len(val_triplets), max_eval), 32):
        batch = val_triplets[start:start + 32]
        anchors, positives, negatives = [], [], []
        for t in batch:
            try:
                a = preprocess(Image.open(t["anchor_path"]).convert("RGB"))
                p = preprocess(Image.open(t["pos_path"]).convert("RGB"))
                n = preprocess(Image.open(t["neg_path"]).convert("RGB"))
                anchors.append(a)
                positives.append(p)
                negatives.append(n)
            except Exception:
                continue

        if not anchors:
            continue

        a_batch = torch.stack(anchors).to(device)
        p_batch = torch.stack(positives).to(device)
        n_batch = torch.stack(negatives).to(device)

        a_emb = F.normalize(model.encode_image(a_batch), dim=-1)
        p_emb = F.normalize(model.encode_image(p_batch), dim=-1)
        n_emb = F.normalize(model.encode_image(n_batch), dim=-1)

        pos_sim = (a_emb * p_emb).sum(-1)
        neg_sim = (a_emb * n_emb).sum(-1)
        correct += (pos_sim > neg_sim).sum().item()
        total += len(anchors)

    return correct / total if total else 0.0


def save_model(model, path: Path):
    path.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path / "model_state_dict.pt")
    meta = {
        "base_model": "Marqo/marqo-fashionSigLIP",
        "architecture": "hf-hub:Marqo/marqo-fashionSigLIP",
        "fine_tuned": True,
        "phase": "5-LB",
        "encoders_tuned": ["image"],
        "training": "image-image contrastive (InfoNCE + alignment reg)",
    }
    with open(path / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    log.info("Saved model to %s", path)


def train(args):
    import open_clip

    log.info("=" * 60)
    log.info("MODA Phase 5-LB — Vision Contrastive Fine-Tuning")
    log.info("=" * 60)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    log.info("Device: %s", device)

    # Load model to fine-tune (load this first to avoid holding two models)
    log.info("Loading FashionSigLIP (trainable)...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        "hf-hub:Marqo/marqo-fashionSigLIP")
    model = model.to(device)

    # Cache pretrained vision weights on CPU for alignment reg (avoids
    # keeping a second full model on MPS)
    log.info("Caching pretrained vision weights for alignment reg...")
    pretrained_state = {
        k: v.clone().cpu() for k, v in model.state_dict().items()
        if "visual" in k
    }
    pretrained_model = None  # no second model in GPU memory

    # Freeze text encoder — only train vision encoder
    for name, param in model.named_parameters():
        if "visual" not in name:
            param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    log.info("Trainable params: %d / %d (%.1f%%)",
             trainable, total_params, 100 * trainable / total_params)

    train_triplets, val_triplets = build_triplets(max_pairs=args.max_pairs)
    if not train_triplets:
        log.error("No triplets built — check data paths")
        return

    dataset = VisionContrastiveDataset(train_triplets, preprocess)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )

    vision_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(vision_params, lr=args.lr, weight_decay=0.01)
    total_steps = len(loader) * args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    best_acc = 0.0
    best_path = OUTPUT_DIR / "best"

    log.info("Starting training: %d epochs, %d steps/epoch, bs=%d, lr=%.1e",
             args.epochs, len(loader), args.batch_size, args.lr)

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_reg = 0.0
        n_batches = 0

        for batch_idx, (anchors, positives, negatives) in enumerate(loader):
            anchors = anchors.to(device)
            positives = positives.to(device)
            negatives = negatives.to(device)

            a_emb = model.encode_image(anchors)
            p_emb = model.encode_image(positives)
            n_emb = model.encode_image(negatives)

            loss_nce, _ = contrastive_loss(
                a_emb, p_emb, n_emb,
                temperature=args.temperature,
            )

            # Weight drift penalty every 10 steps to save compute
            loss_reg = torch.tensor(0.0, device=device)
            if args.align_weight > 0 and (batch_idx + 1) % 10 == 0:
                loss_reg = weight_decay_reg(model, pretrained_state, device)

            loss = loss_nce + args.align_weight * loss_reg

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(vision_params, 1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss_nce.item()
            epoch_reg += loss_reg.item()
            n_batches += 1

            if (batch_idx + 1) % 25 == 0:
                log.info(
                    "  [%d/%d] step %d/%d  nce=%.4f  reg=%.6f  lr=%.2e",
                    epoch + 1, args.epochs, batch_idx + 1, len(loader),
                    epoch_loss / n_batches, epoch_reg / n_batches,
                    scheduler.get_last_lr()[0],
                )
                sys.stdout.flush()

        val_acc = evaluate(model, preprocess, val_triplets, device)
        log.info(
            "Epoch %d/%d  avg_nce=%.4f  avg_reg=%.6f  val_acc=%.3f",
            epoch + 1, args.epochs,
            epoch_loss / n_batches, epoch_reg / n_batches, val_acc,
        )
        sys.stdout.flush()

        if val_acc > best_acc:
            best_acc = val_acc
            save_model(model, best_path)
            log.info("  *** New best val_acc=%.3f ***", val_acc)

    log.info("Training complete. Best val_acc=%.3f", best_acc)
    return best_path


def main():
    parser = argparse.ArgumentParser(
        description="Vision encoder contrastive fine-tuning for LookBench")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--align_weight", type=float, default=0.5)
    parser.add_argument("--max_pairs", type=int, default=50_000)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
