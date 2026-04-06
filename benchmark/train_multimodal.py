"""
MODA Phase 4F — Joint Text + Image Encoder Fine-Tuning

Fine-tunes BOTH FashionCLIP encoders (text + vision) with contrastive
loss, using LLM-judged labels from both text retrieval (Phase 3C) and
image retrieval (Phase 4E).

Key difference from Phase 3C (text-only fine-tuning):
  - Both text and image encoders are updated
  - Alignment regularisation prevents text-image embedding drift
  - Training data includes hard negatives from BOTH retrieval channels

Loss = InfoNCE(text) + InfoNCE(image) + λ·AlignmentReg

AlignmentReg = KL(pretrained text-image sim ∥ finetuned text-image sim)
This keeps the cross-modal alignment intact while improving both encoders.

Output:
  models/moda-fashionclip-multimodal/best/   — best checkpoint

Usage:
  python benchmark/train_multimodal.py
  python benchmark/train_multimodal.py --epochs 3 --batch_size 32
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

PROCESSED_DIR = _REPO_ROOT / "data" / "processed"
MODEL_DIR = _REPO_ROOT / "models"
IMAGE_DIR = _REPO_ROOT / "data" / "raw" / "hnm_images"

TEXT_LABELS_PATH = PROCESSED_DIR / "biencoder_retriever_labels.jsonl"
IMAGE_LABELS_PATH = PROCESSED_DIR / "image_retriever_labels.jsonl"
SPLIT_PATH = PROCESSED_DIR / "query_splits.json"

OUTPUT_DIR = MODEL_DIR / "moda-fashionclip-multimodal"
RANDOM_SEED = 42


def get_image_path(article_id: str) -> Path | None:
    aid = article_id.zfill(10)
    prefix = aid[:3]
    path = IMAGE_DIR / prefix / f"{aid}.jpg"
    if path.exists():
        return path
    return None


class MultimodalTripletDataset(Dataset):
    """Yields (query_text, positive_text, negative_text, positive_img_path, negative_img_path)."""

    def __init__(
        self,
        triplets: list[dict],
        tokenizer,
        preprocess,
    ):
        self.triplets = triplets
        self.tokenizer = tokenizer
        self.preprocess = preprocess

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        t = self.triplets[idx]
        q_tok = self.tokenizer([t["query_text"]])[0]
        p_tok = self.tokenizer([t["pos_text"]])[0]
        n_tok = self.tokenizer([t["neg_text"]])[0]

        p_img = self._load_image(t.get("pos_img_path"))
        n_img = self._load_image(t.get("neg_img_path"))

        return q_tok, p_tok, n_tok, p_img, n_img

    def _load_image(self, path: str | None) -> torch.Tensor:
        if path and Path(path).exists():
            try:
                img = Image.open(path).convert("RGB")
                return self.preprocess(img)
            except Exception:
                pass
        return torch.zeros(3, 224, 224)


def load_triplets(
    max_pairs: int | None = None,
    seed: int = RANDOM_SEED,
) -> tuple[list[dict], list[dict]]:
    """Build multimodal triplets from both text and image retriever labels.

    Each triplet has: query_text, pos_text, neg_text, pos_img_path, neg_img_path,
    pos_article_id, neg_article_id, source.
    """
    rng = random.Random(seed)
    splits = json.loads(SPLIT_PATH.read_text())
    train_qids = set(splits["train"])

    def load_labels(path: Path) -> dict[str, list[dict]]:
        by_query: dict[str, list[dict]] = defaultdict(list)
        if not path.exists():
            log.warning("Labels file not found: %s", path)
            return by_query
        with open(path) as f:
            for line in f:
                if line.strip():
                    obj = json.loads(line)
                    if obj["query_id"] in train_qids:
                        by_query[obj["query_id"]].append(obj)
        return by_query

    text_labels = load_labels(TEXT_LABELS_PATH)
    image_labels = load_labels(IMAGE_LABELS_PATH)

    log.info("Text retriever labels: %d queries", len(text_labels))
    log.info("Image retriever labels: %d queries", len(image_labels))

    all_queries = set(text_labels.keys()) | set(image_labels.keys())
    log.info("Union of queries: %d", len(all_queries))

    triplets = []
    for qid in all_queries:
        items = text_labels.get(qid, []) + image_labels.get(qid, [])
        positives = [it for it in items if it["score"] >= 2]
        hard_negs = [it for it in items if it["score"] == 0]

        if not positives or not hard_negs:
            continue

        for pos in positives:
            neg = rng.choice(hard_negs)
            pos_img = get_image_path(pos["article_id"])
            neg_img = get_image_path(neg["article_id"])
            triplets.append({
                "query_text": pos["query_text"],
                "pos_text": pos["product_text"],
                "neg_text": neg["product_text"],
                "pos_article_id": pos["article_id"],
                "neg_article_id": neg["article_id"],
                "pos_img_path": str(pos_img) if pos_img else None,
                "neg_img_path": str(neg_img) if neg_img else None,
                "source": pos.get("source", "unknown"),
            })

    rng.shuffle(triplets)

    if max_pairs and len(triplets) > max_pairs:
        triplets = triplets[:max_pairs]

    n_val = max(200, int(len(triplets) * 0.10))
    val_triplets = triplets[:n_val]
    train_triplets = triplets[n_val:]

    has_img = sum(1 for t in triplets if t["pos_img_path"] and t["neg_img_path"])
    log.info("Total triplets: %d (with images: %d, %.1f%%)",
             len(triplets), has_img, 100 * has_img / len(triplets) if triplets else 0)
    log.info("Train: %d | Val: %d", len(train_triplets), len(val_triplets))

    return train_triplets, val_triplets


def multimodal_loss(
    q_text_emb: torch.Tensor,
    pos_text_emb: torch.Tensor,
    neg_text_emb: torch.Tensor,
    pos_img_emb: torch.Tensor,
    neg_img_emb: torch.Tensor,
    temperature: float = 0.07,
    align_weight: float = 0.1,
    pretrained_model=None,
    pretrained_pos_text_emb: torch.Tensor | None = None,
    pretrained_pos_img_emb: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict]:
    """Combined loss: text InfoNCE + image InfoNCE + alignment regularization."""
    bs = q_text_emb.shape[0]

    q = F.normalize(q_text_emb, dim=-1)
    pt = F.normalize(pos_text_emb, dim=-1)
    nt = F.normalize(neg_text_emb, dim=-1)
    pi = F.normalize(pos_img_emb, dim=-1)
    ni = F.normalize(neg_img_emb, dim=-1)

    # Text InfoNCE: query vs positive/negative text
    text_pos = (q * pt).sum(-1, keepdim=True) / temperature
    text_inbatch = (q @ pt.T) / temperature
    text_hard = (q * nt).sum(-1, keepdim=True) / temperature
    text_logits = torch.cat([text_inbatch, text_hard], dim=-1)
    text_labels = torch.arange(bs, device=q.device)
    loss_text = F.cross_entropy(text_logits, text_labels)

    # Image InfoNCE: query text vs positive/negative image
    img_inbatch = (q @ pi.T) / temperature
    img_hard = (q * ni).sum(-1, keepdim=True) / temperature
    img_logits = torch.cat([img_inbatch, img_hard], dim=-1)
    loss_image = F.cross_entropy(img_logits, text_labels)

    # Alignment regularization: keep text-image similarity distribution
    # close to the pretrained model's distribution
    loss_align = torch.tensor(0.0, device=q.device)
    if pretrained_pos_text_emb is not None and pretrained_pos_img_emb is not None:
        ft_sim = (pt * pi).sum(-1) / temperature
        pre_pt = F.normalize(pretrained_pos_text_emb, dim=-1)
        pre_pi = F.normalize(pretrained_pos_img_emb, dim=-1)
        pre_sim = (pre_pt * pre_pi).sum(-1) / temperature

        loss_align = F.mse_loss(ft_sim, pre_sim.detach())

    total = loss_text + loss_image + align_weight * loss_align
    metrics = {
        "loss_text": loss_text.item(),
        "loss_image": loss_image.item(),
        "loss_align": loss_align.item(),
        "loss_total": total.item(),
    }
    return total, metrics


@torch.no_grad()
def evaluate_multimodal(model, tokenizer, preprocess, val_triplets, device):
    model.eval()
    correct_text = 0
    correct_image = 0
    total = 0

    for start in range(0, len(val_triplets), 64):
        batch = val_triplets[start:start + 64]
        queries = [t["query_text"] for t in batch]
        pos_texts = [t["pos_text"] for t in batch]
        neg_texts = [t["neg_text"] for t in batch]

        q_tok = tokenizer(queries).to(device)
        p_tok = tokenizer(pos_texts).to(device)
        n_tok = tokenizer(neg_texts).to(device)

        with torch.autocast(device_type="mps", dtype=torch.float16) if device == "mps" else torch.no_grad():
            q_emb = F.normalize(model.encode_text(q_tok), dim=-1)
            p_emb = F.normalize(model.encode_text(p_tok), dim=-1)
            n_emb = F.normalize(model.encode_text(n_tok), dim=-1)

        pos_sim = (q_emb * p_emb).sum(-1)
        neg_sim = (q_emb * n_emb).sum(-1)
        correct_text += (pos_sim > neg_sim).sum().item()

        # Image accuracy (for triplets with images)
        for i, t in enumerate(batch):
            if t.get("pos_img_path") and t.get("neg_img_path"):
                try:
                    p_img = preprocess(Image.open(t["pos_img_path"]).convert("RGB")).unsqueeze(0).to(device)
                    n_img = preprocess(Image.open(t["neg_img_path"]).convert("RGB")).unsqueeze(0).to(device)
                    with torch.autocast(device_type="mps", dtype=torch.float16) if device == "mps" else torch.no_grad():
                        p_img_emb = F.normalize(model.encode_image(p_img), dim=-1)
                        n_img_emb = F.normalize(model.encode_image(n_img), dim=-1)
                    q_single = q_emb[i:i+1]
                    if (q_single * p_img_emb).sum() > (q_single * n_img_emb).sum():
                        correct_image += 1
                except Exception:
                    pass

        total += len(batch)

    text_acc = correct_text / total if total else 0
    img_acc = correct_image / total if total else 0
    return text_acc, img_acc


def train(args):
    import open_clip

    log.info("=" * 60)
    log.info("MODA Phase 4F — Joint Text+Image Fine-Tuning")
    log.info("=" * 60)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    log.info("Device: %s", device)

    # Load pretrained model (for alignment regularization reference)
    log.info("Loading pretrained FashionCLIP (frozen reference)...")
    pretrained_model, pretrained_preprocess, _ = open_clip.create_model_and_transforms(
        "hf-hub:Marqo/marqo-fashionCLIP")
    pretrained_model = pretrained_model.to(device).eval()
    for p in pretrained_model.parameters():
        p.requires_grad = False

    # Load model to fine-tune
    log.info("Loading FashionCLIP (trainable)...")
    model, preprocess, _ = open_clip.create_model_and_transforms(
        "hf-hub:Marqo/marqo-fashionCLIP")
    tokenizer = open_clip.get_tokenizer("hf-hub:Marqo/marqo-fashionCLIP")
    model = model.to(device)

    train_triplets, val_triplets = load_triplets(max_pairs=args.max_pairs)
    if not train_triplets:
        log.error("No training triplets!")
        return

    dataset = MultimodalTripletDataset(train_triplets, tokenizer, preprocess)
    micro_batch = args.batch_size // args.grad_accum
    dataloader = DataLoader(
        dataset, batch_size=micro_batch, shuffle=True,
        num_workers=0, drop_last=True, pin_memory=False,
    )

    # Trainable: both text and image encoders
    model.train()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_p = sum(p.numel() for p in model.parameters())
    log.info("Trainable: %d / %d (%.1f%% — text + image encoders)",
             trainable, total_p, 100 * trainable / total_p)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=0.01,
    )

    eff_steps = len(dataloader) // args.grad_accum
    total_steps = eff_steps * args.epochs
    warmup_steps = max(50, int(total_steps * 0.05))

    def lr_sched(step):
        if step < warmup_steps:
            return step / warmup_steps
        prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * prog))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_sched)
    eval_steps = max(100, eff_steps // 3)

    log.info("Micro-batch: %d | Grad accum: %d | Effective: %d",
             micro_batch, args.grad_accum, args.batch_size)
    log.info("Steps/epoch: %d | Warmup: %d | Eval every: %d",
             eff_steps, warmup_steps, eval_steps)

    # Baseline eval
    base_text_acc, base_img_acc = evaluate_multimodal(
        model, tokenizer, preprocess, val_triplets[:1000], device)
    log.info("Baseline — text_acc: %.3f, img_acc: %.3f", base_text_acc, base_img_acc)

    best_text_acc = base_text_acc
    global_step = 0
    t0 = time.time()

    for epoch in range(args.epochs):
        model.train()
        epoch_metrics = defaultdict(float)
        micro_steps = 0
        optimizer.zero_grad(set_to_none=True)

        for batch_idx, (q_tok, p_tok, n_tok, p_img, n_img) in enumerate(dataloader):
            q_tok = q_tok.to(device, non_blocking=True)
            p_tok = p_tok.to(device, non_blocking=True)
            n_tok = n_tok.to(device, non_blocking=True)
            p_img = p_img.to(device, non_blocking=True)
            n_img = n_img.to(device, non_blocking=True)

            use_amp = device == "mps"
            if use_amp:
                with torch.autocast(device_type="mps", dtype=torch.float16):
                    q_emb = model.encode_text(q_tok)
                    pt_emb = model.encode_text(p_tok)
                    nt_emb = model.encode_text(n_tok)
                    pi_emb = model.encode_image(p_img)
                    ni_emb = model.encode_image(n_img)

                    with torch.no_grad():
                        pre_pt_emb = pretrained_model.encode_text(p_tok)
                        pre_pi_emb = pretrained_model.encode_image(p_img)

                    loss, metrics = multimodal_loss(
                        q_emb, pt_emb, nt_emb, pi_emb, ni_emb,
                        temperature=args.temperature,
                        align_weight=args.align_weight,
                        pretrained_pos_text_emb=pre_pt_emb,
                        pretrained_pos_img_emb=pre_pi_emb,
                    )
                    loss = loss / args.grad_accum
            else:
                q_emb = model.encode_text(q_tok)
                pt_emb = model.encode_text(p_tok)
                nt_emb = model.encode_text(n_tok)
                pi_emb = model.encode_image(p_img)
                ni_emb = model.encode_image(n_img)

                with torch.no_grad():
                    pre_pt_emb = pretrained_model.encode_text(p_tok)
                    pre_pi_emb = pretrained_model.encode_image(p_img)

                loss, metrics = multimodal_loss(
                    q_emb, pt_emb, nt_emb, pi_emb, ni_emb,
                    temperature=args.temperature,
                    align_weight=args.align_weight,
                    pretrained_pos_text_emb=pre_pt_emb,
                    pretrained_pos_img_emb=pre_pi_emb,
                )
                loss = loss / args.grad_accum

            loss.backward()
            for k, v in metrics.items():
                epoch_metrics[k] += v
            micro_steps += 1

            if micro_steps % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if global_step % 50 == 0:
                    avg = {k: v / micro_steps for k, v in epoch_metrics.items()}
                    elapsed = time.time() - t0
                    log.info(
                        "  [ep%d step%d/%d] total=%.4f text=%.4f img=%.4f align=%.4f "
                        "lr=%.2e %.0fms/step",
                        epoch + 1, global_step % eff_steps or eff_steps, eff_steps,
                        avg["loss_total"], avg["loss_text"], avg["loss_image"],
                        avg["loss_align"], scheduler.get_last_lr()[0],
                        elapsed / global_step * 1000,
                    )

                if global_step % eval_steps == 0:
                    t_acc, i_acc = evaluate_multimodal(
                        model, tokenizer, preprocess, val_triplets[:1000], device)
                    log.info("  → Val text_acc=%.3f img_acc=%.3f (best_text=%.3f)",
                             t_acc, i_acc, best_text_acc)
                    if t_acc > best_text_acc:
                        best_text_acc = t_acc
                        save_model(model, OUTPUT_DIR / "best")
                        log.info("  → New best! Saved.")
                    model.train()

        # End of epoch
        t_acc, i_acc = evaluate_multimodal(
            model, tokenizer, preprocess, val_triplets[:1000], device)
        log.info("Epoch %d — text_acc=%.3f img_acc=%.3f (best=%.3f)",
                 epoch + 1, t_acc, i_acc, best_text_acc)
        if t_acc > best_text_acc:
            best_text_acc = t_acc
            save_model(model, OUTPUT_DIR / "best")
            log.info("New best! Saved.")

        if device == "mps":
            torch.mps.empty_cache()

    elapsed = time.time() - t0
    log.info("Training complete in %.1f min", elapsed / 60)
    log.info("Baseline text_acc=%.3f → Best=%.3f (+%.1f%%)",
             base_text_acc, best_text_acc,
             100 * (best_text_acc - base_text_acc))
    save_model(model, OUTPUT_DIR)


def save_model(model, path: Path):
    path.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path / "model_state_dict.pt")
    meta = {
        "base_model": "Marqo/marqo-fashionCLIP",
        "architecture": "hf-hub:Marqo/marqo-fashionCLIP",
        "fine_tuned": True,
        "phase": "4F",
        "encoders_tuned": ["text", "image"],
    }
    with open(path / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--max_pairs", type=int, default=None)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=5e-7,
                   help="Lower LR since we're tuning both encoders")
    p.add_argument("--temperature", type=float, default=0.07)
    p.add_argument("--align_weight", type=float, default=0.1,
                   help="Weight for alignment regularization")
    p.add_argument("--quick", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.quick:
        args.max_pairs = 2000
        args.epochs = 1
    train(args)
