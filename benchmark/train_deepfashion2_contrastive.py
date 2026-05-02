"""
MODA — FashionSigLIP Vision-Encoder Fine-Tuning on DeepFashion2

Fine-tunes FashionSigLIP's vision encoder using cross-domain contrastive
learning on DeepFashion2 matching pairs (shop ↔ consumer images of the same
clothing item). This diversity is key: shop images are studio shots while
consumer images are in-the-wild, teaching the model to match across domains
— exactly the skill LookBench tests.

Training strategy:
  - Positive pair: shop image + consumer image of the same style
  - In-batch negatives: other items in the batch (InfoNCE)
  - Hard negatives: same category, different style
  - Alignment regularisation: L2 penalty on weight drift from pretrained
  - Only vision encoder is trained (text encoder frozen)
  - Crops to bounding box when available for cleaner signal

Usage:
  python benchmark/train_deepfashion2_contrastive.py
  python benchmark/train_deepfashion2_contrastive.py --epochs 3 --batch_size 32 --lr 2e-6
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
import zipfile
from collections import defaultdict
from io import BytesIO
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

DF2_DIR = _REPO_ROOT / "data" / "raw" / "deepfashion2"
PAIRS_FILE = DF2_DIR / "matching_pairs.json"
TRAIN_ZIP = DF2_DIR / "train.zip"
OUTPUT_DIR = _REPO_ROOT / "models" / "moda-siglip-deepfashion2"
RANDOM_SEED = 42


class DeepFashion2Dataset(Dataset):
    """Loads (anchor, positive, hard_negative) triplets from DeepFashion2.

    anchor = shop image, positive = consumer image of same style,
    hard_negative = consumer or shop image of different style but same category.
    """

    def __init__(self, triplets: list[dict], zip_path: Path, preprocess,
                 use_bbox: bool = True):
        self.triplets = triplets
        self.zip_path = zip_path
        self.preprocess = preprocess
        self.use_bbox = use_bbox
        self._zf = None

    def _get_zip(self):
        if self._zf is None:
            self._zf = zipfile.ZipFile(self.zip_path, "r")
        return self._zf

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        t = self.triplets[idx]
        anchor = self._load_image(t["anchor_name"], t.get("anchor_bbox"))
        positive = self._load_image(t["pos_name"], t.get("pos_bbox"))
        negative = self._load_image(t["neg_name"], t.get("neg_bbox"))
        return anchor, positive, negative

    def _load_image(self, name: str, bbox: list | None) -> torch.Tensor:
        try:
            zf = self._get_zip()
            zip_key = f"train/image/{name}"
            data = zf.read(zip_key)
            img = Image.open(BytesIO(data)).convert("RGB")
            if self.use_bbox and bbox:
                x1, y1, x2, y2 = bbox
                w, h = img.size
                x1 = max(0, min(x1, w))
                y1 = max(0, min(y1, h))
                x2 = max(x1 + 1, min(x2, w))
                y2 = max(y1 + 1, min(y2, h))
                img = img.crop((x1, y1, x2, y2))
            return self.preprocess(img)
        except Exception as e:
            return torch.zeros(3, 224, 224)


def build_triplets(
    max_pairs: int = 100_000,
    val_frac: float = 0.05,
    seed: int = RANDOM_SEED,
) -> tuple[list[dict], list[dict]]:
    """Build triplets from DeepFashion2 matching pairs."""
    log.info("Loading matching pairs from %s ...", PAIRS_FILE)
    with open(PAIRS_FILE) as f:
        pairs = json.load(f)
    log.info("Loaded %d raw pairs", len(pairs))

    rng = random.Random(seed)

    # Group pairs by pair_id (the actual product identifier) and by category
    by_pid: dict[int, list[dict]] = defaultdict(list)
    by_category: dict[int, list[dict]] = defaultdict(list)

    for p in pairs:
        pid = p["pair_id"]
        cat_id = p["category_id"]
        by_pid[pid].append(p)
        by_category[cat_id].append(p)

    log.info("Unique products (pair_id): %d, categories: %d",
             len(by_pid), len(by_category))

    # Build triplets: anchor=shop, positive=user (different pic, same product),
    # negative=different pair_id, same category (visually similar but different item)
    triplets = []
    pids = list(by_pid.keys())
    rng.shuffle(pids)

    for pid in pids:
        if len(triplets) >= max_pairs:
            break

        pid_pairs = by_pid[pid]
        cat_id = pid_pairs[0]["category_id"]

        # Hard negative pool: same category, different pair_id
        cat_pairs = by_category[cat_id]
        neg_pool = [p for p in cat_pairs if p["pair_id"] != pid]

        if not neg_pool:
            continue

        # One triplet per product to avoid in-batch positive collisions
        # (same pair_id appearing twice in a batch would make InfoNCE labels wrong)
        pair = rng.choice(pid_pairs)
        neg_pair = rng.choice(neg_pool)
        use_neg_user = rng.random() > 0.5

        triplets.append({
            "anchor_name": pair["shop_image"]["name"],
            "anchor_bbox": pair["shop_image"].get("bbox"),
            "pos_name": pair["user_image"]["name"],
            "pos_bbox": pair["user_image"].get("bbox"),
            "neg_name": (neg_pair["user_image"]["name"] if use_neg_user
                        else neg_pair["shop_image"]["name"]),
            "neg_bbox": (neg_pair["user_image"].get("bbox") if use_neg_user
                        else neg_pair["shop_image"].get("bbox")),
            "pair_id": pid,
            "category_id": cat_id,
        })

    rng.shuffle(triplets)
    split = int(len(triplets) * (1 - val_frac))
    train = triplets[:split]
    val = triplets[split:]
    log.info("Built %d train, %d val triplets", len(train), len(val))
    return train, val


def contrastive_loss(
    anchor_emb: torch.Tensor,
    pos_emb: torch.Tensor,
    neg_emb: torch.Tensor,
    temperature: float = 0.07,
) -> tuple[torch.Tensor, dict]:
    """InfoNCE: anchor vs in-batch positives + explicit hard negative."""
    a = F.normalize(anchor_emb, dim=-1)
    p = F.normalize(pos_emb, dim=-1)
    n = F.normalize(neg_emb, dim=-1)

    pos_inbatch = (a @ p.T) / temperature
    hard_neg = (a * n).sum(-1, keepdim=True) / temperature
    logits = torch.cat([pos_inbatch, hard_neg], dim=-1)
    labels = torch.arange(a.shape[0], device=a.device)
    loss = F.cross_entropy(logits, labels)
    return loss, {"nce": loss.item()}


def weight_drift_reg(model, pretrained_state: dict, device: str) -> torch.Tensor:
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
def evaluate(model, val_ds, device, max_eval=500):
    """Triplet accuracy: anchor closer to positive than negative?"""
    model.eval()
    correct = total = 0

    for idx in range(min(len(val_ds), max_eval)):
        try:
            a, p, n = val_ds[idx]
        except Exception:
            continue

        a_emb = F.normalize(model.encode_image(a.unsqueeze(0).to(device)), dim=-1)
        p_emb = F.normalize(model.encode_image(p.unsqueeze(0).to(device)), dim=-1)
        n_emb = F.normalize(model.encode_image(n.unsqueeze(0).to(device)), dim=-1)

        pos_sim = (a_emb * p_emb).sum()
        neg_sim = (a_emb * n_emb).sum()
        correct += int(pos_sim > neg_sim)
        total += 1

    return correct / total if total else 0.0


def save_model(model, path: Path, meta_extra: dict | None = None):
    path.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path / "model_state_dict.pt")
    meta = {
        "base_model": "Marqo/marqo-fashionSigLIP",
        "architecture": "hf-hub:Marqo/marqo-fashionSigLIP",
        "fine_tuned": True,
        "dataset": "DeepFashion2",
        "encoders_tuned": ["image"],
        "training": "cross-domain contrastive (shop ↔ consumer, InfoNCE + drift reg)",
    }
    if meta_extra:
        meta.update(meta_extra)
    with open(path / "meta.json", "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    log.info("Saved model to %s", path)


def train(args):
    import open_clip

    log.info("=" * 60)
    log.info("MODA — DeepFashion2 Cross-Domain Contrastive Fine-Tuning")
    log.info("=" * 60)

    if not TRAIN_ZIP.exists():
        log.error("train.zip not found at %s — download it first", TRAIN_ZIP)
        return

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    log.info("Device: %s", device)

    log.info("Loading FashionSigLIP ...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        "hf-hub:Marqo/marqo-fashionSigLIP")
    model = model.to(device)

    # Cache pretrained vision weights for alignment reg
    log.info("Caching pretrained vision weights ...")
    pretrained_state = {
        k: v.clone().cpu() for k, v in model.state_dict().items()
        if "visual" in k
    }

    # Freeze text encoder
    for name, param in model.named_parameters():
        if "visual" not in name:
            param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    log.info("Trainable: %d / %d (%.1f%%)", trainable, total, 100 * trainable / total)

    train_triplets, val_triplets = build_triplets(max_pairs=args.max_pairs)

    train_ds = DeepFashion2Dataset(train_triplets, TRAIN_ZIP, preprocess,
                                   use_bbox=args.use_bbox)
    val_ds = DeepFashion2Dataset(val_triplets, TRAIN_ZIP, preprocess,
                                  use_bbox=args.use_bbox)

    loader = DataLoader(
        train_ds,
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

    log.info("Training: %d epochs, %d steps/epoch, bs=%d, lr=%.1e, bbox=%s",
             args.epochs, len(loader), args.batch_size, args.lr, args.use_bbox)
    log.info("Alignment weight: %.3f, Temperature: %.3f",
             args.align_weight, args.temperature)

    for epoch in range(args.epochs):
        model.train()
        epoch_nce = epoch_reg = 0.0
        n_batches = 0
        t0 = time.time()

        for batch_idx, (anchors, positives, negatives) in enumerate(loader):
            anchors = anchors.to(device)
            positives = positives.to(device)
            negatives = negatives.to(device)

            a_emb = model.encode_image(anchors)
            p_emb = model.encode_image(positives)
            n_emb = model.encode_image(negatives)

            loss_nce, _ = contrastive_loss(a_emb, p_emb, n_emb, args.temperature)

            loss_reg = torch.tensor(0.0, device=device)
            if args.align_weight > 0 and (batch_idx + 1) % 10 == 0:
                loss_reg = weight_drift_reg(model, pretrained_state, device)

            loss = loss_nce + args.align_weight * loss_reg

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(vision_params, 1.0)
            optimizer.step()
            scheduler.step()

            epoch_nce += loss_nce.item()
            epoch_reg += loss_reg.item()
            n_batches += 1

            if (batch_idx + 1) % 25 == 0:
                elapsed = time.time() - t0
                rate = (batch_idx + 1) / elapsed
                eta = (len(loader) - batch_idx - 1) / rate
                log.info(
                    "  [E%d] step %d/%d  nce=%.4f  reg=%.6f  lr=%.2e  "
                    "%.1f step/s  ETA %.0fs",
                    epoch + 1, batch_idx + 1, len(loader),
                    epoch_nce / n_batches, epoch_reg / n_batches,
                    scheduler.get_last_lr()[0], rate, eta,
                )
                sys.stdout.flush()

        # Validate
        val_acc = evaluate(model, val_ds, device, max_eval=min(500, len(val_ds)))
        epoch_elapsed = time.time() - t0
        log.info(
            "Epoch %d/%d  nce=%.4f  reg=%.6f  val_acc=%.3f  (%.0fs)",
            epoch + 1, args.epochs,
            epoch_nce / n_batches, epoch_reg / n_batches,
            val_acc, epoch_elapsed,
        )
        sys.stdout.flush()

        if val_acc > best_acc:
            best_acc = val_acc
            save_model(model, best_path, {
                "epoch": epoch + 1,
                "val_acc": val_acc,
                "n_train_triplets": len(train_triplets),
            })
            log.info("  *** New best val_acc=%.3f ***", val_acc)
        elif epoch > 0:
            save_model(model, OUTPUT_DIR / f"epoch_{epoch + 1}", {
                "epoch": epoch + 1,
                "val_acc": val_acc,
            })

    log.info("Training complete. Best val_acc=%.3f", best_acc)
    log.info("Best model at: %s", best_path)


def main():
    parser = argparse.ArgumentParser(
        description="DeepFashion2 cross-domain contrastive fine-tuning")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--lr", type=float, default=2e-6)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--align_weight", type=float, default=0.3)
    parser.add_argument("--max_pairs", type=int, default=100_000)
    parser.add_argument("--use_bbox", action="store_true", default=True)
    parser.add_argument("--no_bbox", dest="use_bbox", action="store_false")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
