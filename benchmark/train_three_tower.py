"""
MODA Phase 4G — Three-Tower Fashion Retriever

Novel architecture with dedicated query/text/image towers trained
contrastively on LLM-judged relevance labels.

Key insight: search queries ("red dress") are fundamentally different from
product descriptions ("Ladies' Red V-Neck Summer Dress in Viscose Blend").
A shared encoder is suboptimal. The 3-tower model gives the query its own
encoder with a learned projection head, while keeping separate text and
image towers for products — all projecting into a single shared space.

Architecture:
  Query Tower  = CLIP text encoder + 2-layer projection MLP  (trained)
  Text Tower   = CLIP text encoder (separate copy)           (frozen)
  Image Tower  = CLIP vision encoder                         (frozen)

Training strategy (2 phases):
  Phase A: Precompute product embeddings (text + image towers frozen)
  Phase B: Train query tower + projection against cached product embeddings
           L = L_qt + L_qi (InfoNCE losses against frozen targets)

This is ~10x faster than training all 3 towers jointly because:
  - No image I/O during training
  - No forward pass through product encoders
  - Gradients only through the query encoder + projection MLP

Output:
  models/moda-3tower/best/

Usage:
  python benchmark/train_three_tower.py
  python benchmark/train_three_tower.py --epochs 5 --batch_size 128
  python benchmark/train_three_tower.py --quick  # fast sanity check
"""

from __future__ import annotations

import argparse
import copy
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
DISTILL_PATH = PROCESSED_DIR / "ce_distillation_pairs.jsonl"
SPLIT_PATH = PROCESSED_DIR / "query_splits.json"

OUTPUT_DIR = MODEL_DIR / "moda-3tower"
RANDOM_SEED = 42


# ─── Three-Tower Model ───────────────────────────────────────────────────────

class ThreeTowerModel(nn.Module):
    """Three-tower contrastive retrieval model.

    Tower 1 (Query):  CLIP text encoder + learned projection MLP.
    Tower 2 (Text):   Separate CLIP text encoder copy (frozen for training).
    Tower 3 (Image):  CLIP vision encoder (frozen for training).

    All three produce 512-dim embeddings in a shared space.
    Product towers are frozen during training — their embeddings are precomputed.
    """

    def __init__(self, clip_model, embed_dim: int = 512):
        super().__init__()
        self.query_encoder = copy.deepcopy(clip_model)
        self.text_encoder = copy.deepcopy(clip_model)
        self.image_encoder = clip_model

        self.query_projection = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        nn.init.zeros_(self.query_projection[-1].bias)
        nn.init.eye_(self.query_projection[-1].weight)

    def freeze_product_towers(self):
        for p in self.text_encoder.parameters():
            p.requires_grad = False
        for p in self.image_encoder.parameters():
            p.requires_grad = False

    def encode_query(self, tokens: torch.Tensor) -> torch.Tensor:
        base_emb = self.query_encoder.encode_text(tokens)
        return self.query_projection(base_emb)

    def encode_product_text(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.text_encoder.encode_text(tokens)

    def encode_product_image(self, images: torch.Tensor) -> torch.Tensor:
        return self.image_encoder.encode_image(images)


# ─── Loss ─────────────────────────────────────────────────────────────────────

def three_tower_loss(
    q_emb: torch.Tensor,
    pos_text_emb: torch.Tensor,
    neg_text_emb: torch.Tensor,
    pos_img_emb: torch.Tensor,
    neg_img_emb: torch.Tensor,
    temperature: float = 0.07,
    neg_types: list[str] | None = None,
    soft_margin: float = 0.05,
) -> tuple[torch.Tensor, dict]:
    """Two-term loss for the 3-tower model (product towers frozen).

    L_qt:  query ↔ product-text   (main retrieval objective)
    L_qi:  query ↔ product-image  (cross-modal retrieval)

    For soft negatives (score=1), uses a reduced temperature to allow
    smaller margin, preventing the model from pushing partial matches
    too far from the query while still learning the boundary.
    """
    bs = q_emb.shape[0]
    q_emb = q_emb.float()
    pos_text_emb = pos_text_emb.float()
    neg_text_emb = neg_text_emb.float()
    pos_img_emb = pos_img_emb.float()
    neg_img_emb = neg_img_emb.float()
    q = F.normalize(q_emb, dim=-1)
    pt = F.normalize(pos_text_emb.detach(), dim=-1)
    nt = F.normalize(neg_text_emb.detach(), dim=-1)
    pi = F.normalize(pos_img_emb.detach(), dim=-1)
    ni = F.normalize(neg_img_emb.detach(), dim=-1)

    labels = torch.arange(bs, device=q.device)

    # L_qt: query → product text (in-batch + hard/soft negative)
    qt_inbatch = (q @ pt.T) / temperature
    qt_hard = (q * nt).sum(-1, keepdim=True) / temperature
    qt_logits = torch.cat([qt_inbatch, qt_hard], dim=-1)
    loss_qt = F.cross_entropy(qt_logits, labels)

    # L_qi: query → product image (in-batch + hard/soft negative)
    qi_inbatch = (q @ pi.T) / temperature
    qi_hard = (q * ni).sum(-1, keepdim=True) / temperature
    qi_logits = torch.cat([qi_inbatch, qi_hard], dim=-1)
    loss_qi = F.cross_entropy(qi_logits, labels)

    total = loss_qt + loss_qi

    if neg_types is not None:
        is_soft = torch.tensor(
            [1.0 if nt == "soft" else 0.0 for nt in neg_types],
            device=q.device
        )
        if is_soft.sum() > 0:
            pos_sim_t = (q * pt).sum(-1)
            neg_sim_t = (q * nt).sum(-1)
            margin_loss = F.relu(neg_sim_t - pos_sim_t + soft_margin)
            soft_loss = (margin_loss * is_soft).sum() / max(is_soft.sum(), 1.0)
            total = total + 0.5 * soft_loss

    metrics = {
        "loss_qt": loss_qt.item(),
        "loss_qi": loss_qi.item(),
        "loss_total": total.item(),
    }
    return total, metrics


# ─── Dataset with precomputed product embeddings ─────────────────────────────

def get_image_path(article_id: str) -> Path | None:
    aid = article_id.zfill(10)
    prefix = aid[:3]
    path = IMAGE_DIR / prefix / f"{aid}.jpg"
    if path.exists():
        return path
    return None


class CachedThreeTowerDataset(Dataset):
    """Training dataset with precomputed product embeddings.

    Each sample yields:
      (query_tokens, pos_text_emb, neg_text_emb, pos_img_emb, neg_img_emb, neg_type_idx)

    Product embeddings are looked up from precomputed caches by article_id.
    Only the query needs to go through the encoder during training.
    neg_type_idx: 0 = hard negative, 1 = soft negative (score=1 partial match)
    """

    NEG_TYPE_MAP = {"hard": 0, "soft": 1}

    def __init__(
        self,
        triplets: list[dict],
        tokenizer,
        text_emb_cache: dict[str, np.ndarray],
        img_emb_cache: dict[str, np.ndarray],
        embed_dim: int = 512,
    ):
        self.triplets = triplets
        self.tokenizer = tokenizer
        self.text_cache = text_emb_cache
        self.img_cache = img_emb_cache
        self.zero_emb = np.zeros(embed_dim, dtype=np.float32)

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        t = self.triplets[idx]
        q_tok = self.tokenizer([t["query_text"]])[0]

        pos_t = torch.from_numpy(self.text_cache.get(t["pos_article_id"], self.zero_emb))
        neg_t = torch.from_numpy(self.text_cache.get(t["neg_article_id"], self.zero_emb))
        pos_i = torch.from_numpy(self.img_cache.get(t["pos_article_id"], self.zero_emb))
        neg_i = torch.from_numpy(self.img_cache.get(t["neg_article_id"], self.zero_emb))
        neg_type = self.NEG_TYPE_MAP.get(t.get("neg_type", "hard"), 0)

        return q_tok, pos_t, neg_t, pos_i, neg_i, neg_type


def load_triplets(
    max_pairs: int | None = None,
    seed: int = RANDOM_SEED,
) -> tuple[list[dict], list[dict]]:
    """Build triplets from both text and image retriever labels.

    Uses query-disjoint train/val split from query_splits.json to prevent
    leakage. Score-1 (partial match) pairs are included as soft negatives
    with a 'neg_type' field to distinguish them from hard negatives.
    """
    rng = random.Random(seed)
    splits = json.loads(SPLIT_PATH.read_text())
    train_qids = set(splits["train"])
    test_qids = set(splits["test"])

    def load_labels(path: Path) -> dict[str, list[dict]]:
        by_query: dict[str, list[dict]] = defaultdict(list)
        if not path.exists():
            log.warning("Labels file not found: %s", path)
            return by_query
        with open(path) as f:
            for line in f:
                if line.strip():
                    obj = json.loads(line)
                    if obj["query_id"] not in test_qids:
                        by_query[obj["query_id"]].append(obj)
        return by_query

    text_labels = load_labels(TEXT_LABELS_PATH)
    image_labels = load_labels(IMAGE_LABELS_PATH)

    log.info("Text retriever labels: %d queries", len(text_labels))
    log.info("Image retriever labels: %d queries", len(image_labels))

    all_label_qids = sorted(set(text_labels.keys()) | set(image_labels.keys()))
    rng.shuffle(all_label_qids)
    n_val_queries = max(200, int(len(all_label_qids) * 0.10))
    val_query_set = set(all_label_qids[:n_val_queries])
    train_query_set = set(all_label_qids[n_val_queries:])
    log.info("Query-disjoint split: %d train queries, %d val queries",
             len(train_query_set), len(val_query_set))

    def build_triplets_for_qids(qid_set: set[str]) -> list[dict]:
        matching = (set(text_labels.keys()) | set(image_labels.keys())) & qid_set
        triplets = []
        for qid in matching:
            items = text_labels.get(qid, []) + image_labels.get(qid, [])
            positives = [it for it in items if it["score"] >= 2]
            hard_negs = [it for it in items if it["score"] == 0]
            soft_negs = [it for it in items if it["score"] == 1]

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
                    "neg_type": "hard",
                })

                if soft_negs:
                    sn = rng.choice(soft_negs)
                    sn_img = get_image_path(sn["article_id"])
                    triplets.append({
                        "query_text": pos["query_text"],
                        "pos_text": pos["product_text"],
                        "neg_text": sn["product_text"],
                        "pos_article_id": pos["article_id"],
                        "neg_article_id": sn["article_id"],
                        "pos_img_path": str(pos_img) if pos_img else None,
                        "neg_img_path": str(sn_img) if sn_img else None,
                        "source": pos.get("source", "unknown"),
                        "neg_type": "soft",
                    })

        return triplets

    train_triplets = build_triplets_for_qids(train_query_set)
    val_triplets = build_triplets_for_qids(val_query_set)

    rng.shuffle(train_triplets)
    rng.shuffle(val_triplets)

    if max_pairs and len(train_triplets) > max_pairs:
        train_triplets = train_triplets[:max_pairs]

    if not val_triplets:
        raise ValueError(
            "No val-split triplets found. Check label files have enough "
            "queries to create a disjoint validation set."
        )

    all_triplets = train_triplets + val_triplets
    has_img = sum(1 for t in all_triplets if t["pos_img_path"] and t["neg_img_path"])
    n_soft = sum(1 for t in train_triplets if t.get("neg_type") == "soft")
    log.info("Total triplets: %d (with images: %d, %.1f%%)",
             len(all_triplets), has_img,
             100 * has_img / len(all_triplets) if all_triplets else 0)
    log.info("Train: %d (hard: %d, soft: %d) | Val: %d (query-disjoint)",
             len(train_triplets), len(train_triplets) - n_soft, n_soft,
             len(val_triplets))

    return train_triplets, val_triplets


class DistillationDataset(Dataset):
    """Dataset for MarginMSE distillation from CE scores.

    Each sample yields (query_tokens, doc_i_text_emb, doc_j_text_emb, ce_margin).
    The retriever should learn to produce similarity margins that match the CE's.
    """

    def __init__(
        self,
        pairs: list[dict],
        tokenizer,
        text_emb_cache: dict[str, np.ndarray],
        embed_dim: int = 512,
    ):
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.text_cache = text_emb_cache
        self.zero_emb = np.zeros(embed_dim, dtype=np.float32)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        p = self.pairs[idx]
        q_tok = self.tokenizer([p["query_text"]])[0]
        emb_i = torch.from_numpy(self.text_cache.get(p["doc_i_id"], self.zero_emb))
        emb_j = torch.from_numpy(self.text_cache.get(p["doc_j_id"], self.zero_emb))
        margin = torch.tensor(p["ce_margin"], dtype=torch.float32)
        return q_tok, emb_i, emb_j, margin


def margin_mse_loss(
    q_emb: torch.Tensor,
    doc_i_emb: torch.Tensor,
    doc_j_emb: torch.Tensor,
    ce_margin: torch.Tensor,
) -> torch.Tensor:
    """MarginMSE: minimize MSE between retriever margin and CE margin."""
    q = F.normalize(q_emb.float(), dim=-1)
    di = F.normalize(doc_i_emb.float().detach(), dim=-1)
    dj = F.normalize(doc_j_emb.float().detach(), dim=-1)
    retriever_margin = (q * di).sum(-1) - (q * dj).sum(-1)
    return F.mse_loss(retriever_margin, ce_margin.to(q.device))


def load_distillation_pairs(seed: int = RANDOM_SEED) -> list[dict]:
    """Load CE distillation pairs for MarginMSE training."""
    if not DISTILL_PATH.exists():
        log.info("No distillation data at %s — skipping distillation", DISTILL_PATH)
        return []

    splits = json.loads(SPLIT_PATH.read_text())
    train_qids = set(splits["train"])

    pairs = []
    with open(DISTILL_PATH) as f:
        for line in f:
            if line.strip():
                obj = json.loads(line)
                if obj["query_id"] in train_qids:
                    pairs.append(obj)

    log.info("Loaded %d distillation pairs", len(pairs))
    return pairs


def precompute_product_embeddings(
    model: ThreeTowerModel,
    tokenizer,
    preprocess,
    triplets: list[dict],
    device: str,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Precompute text and image embeddings for all articles in the dataset."""
    article_ids = set()
    article_texts: dict[str, str] = {}
    for t in triplets:
        for key in ("pos_article_id", "neg_article_id"):
            aid = t[key]
            article_ids.add(aid)
        article_texts[t["pos_article_id"]] = t["pos_text"]
        article_texts[t["neg_article_id"]] = t["neg_text"]

    article_ids = sorted(article_ids)
    log.info("Precomputing embeddings for %d unique articles...", len(article_ids))

    model.eval()
    text_cache: dict[str, np.ndarray] = {}
    img_cache: dict[str, np.ndarray] = {}
    batch_size = 128

    # Text embeddings
    log.info("  Precomputing text tower embeddings...")
    texts = [article_texts.get(aid, "") for aid in article_ids]
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            batch_ids = article_ids[start:start + batch_size]
            tokens = tokenizer(batch).to(device)
            with torch.autocast(device_type="mps", dtype=torch.float16) if device == "mps" else torch.no_grad():
                emb = model.encode_product_text(tokens)
                emb = F.normalize(emb, dim=-1)
            emb_np = emb.cpu().float().numpy()
            for i, aid in enumerate(batch_ids):
                text_cache[aid] = emb_np[i]
    log.info("    %d text embeddings cached", len(text_cache))

    # Image embeddings
    log.info("  Precomputing image tower embeddings...")
    zero_img = torch.zeros(3, 224, 224)
    img_batch_size = 64
    with torch.no_grad():
        for start in range(0, len(article_ids), img_batch_size):
            batch_ids = article_ids[start:start + img_batch_size]
            imgs = []
            for aid in batch_ids:
                ipath = get_image_path(aid)
                if ipath:
                    try:
                        imgs.append(preprocess(Image.open(ipath).convert("RGB")))
                    except Exception:
                        imgs.append(zero_img)
                else:
                    imgs.append(zero_img)
            img_tensor = torch.stack(imgs).to(device)
            with torch.autocast(device_type="mps", dtype=torch.float16) if device == "mps" else torch.no_grad():
                emb = model.encode_product_image(img_tensor)
                emb = F.normalize(emb, dim=-1)
            emb_np = emb.cpu().float().numpy()
            for i, aid in enumerate(batch_ids):
                img_cache[aid] = emb_np[i]
            if device == "mps" and start % (img_batch_size * 20) == 0 and start > 0:
                torch.mps.empty_cache()
    log.info("    %d image embeddings cached", len(img_cache))

    if device == "mps":
        torch.mps.empty_cache()

    return text_cache, img_cache


# ─── Evaluation ───────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_three_tower(
    model: ThreeTowerModel,
    tokenizer,
    val_triplets: list[dict],
    text_cache: dict[str, np.ndarray],
    img_cache: dict[str, np.ndarray],
    device: str,
):
    """Accuracy on validation triplets using cached product embeddings."""
    model.eval()
    correct_text = 0
    correct_image = 0
    total = 0
    zero_emb = np.zeros(512, dtype=np.float32)

    for start in range(0, len(val_triplets), 128):
        batch = val_triplets[start:start + 128]
        queries = [t["query_text"] for t in batch]

        q_tok = tokenizer(queries).to(device)
        ctx = torch.autocast(device_type="mps", dtype=torch.float16) if device == "mps" else torch.no_grad()
        with ctx:
            q_emb = F.normalize(model.encode_query(q_tok), dim=-1).float()

        for i, t in enumerate(batch):
            q = q_emb[i]
            pt = torch.from_numpy(text_cache.get(t["pos_article_id"], zero_emb)).to(device)
            nt = torch.from_numpy(text_cache.get(t["neg_article_id"], zero_emb)).to(device)
            if (q @ pt) > (q @ nt):
                correct_text += 1

            pi = torch.from_numpy(img_cache.get(t["pos_article_id"], zero_emb)).to(device)
            ni = torch.from_numpy(img_cache.get(t["neg_article_id"], zero_emb)).to(device)
            if (q @ pi) > (q @ ni):
                correct_image += 1

        total += len(batch)

    text_acc = correct_text / total if total else 0
    img_acc = correct_image / total if total else 0
    return text_acc, img_acc


# ─── Save / Load ──────────────────────────────────────────────────────────────

def save_three_tower(model: ThreeTowerModel, path: Path):
    path.mkdir(parents=True, exist_ok=True)
    torch.save({
        "query_encoder": model.query_encoder.state_dict(),
        "text_encoder": model.text_encoder.state_dict(),
        "image_encoder": model.image_encoder.state_dict(),
        "query_projection": model.query_projection.state_dict(),
    }, path / "three_tower_state.pt")
    meta = {
        "base_model": "Marqo/marqo-fashionCLIP",
        "architecture": "three_tower",
        "phase": "4G",
        "towers": ["query", "text", "image"],
        "query_projection": "Linear(512,512)->GELU->Linear(512,512)",
        "embed_dim": 512,
    }
    with open(path / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)


def load_three_tower(path: Path, device: str = "cpu") -> ThreeTowerModel:
    import open_clip
    base_model, _, _ = open_clip.create_model_and_transforms(
        "hf-hub:Marqo/marqo-fashionCLIP")
    model = ThreeTowerModel(base_model, embed_dim=512)
    state = torch.load(path / "three_tower_state.pt", map_location=device)
    model.query_encoder.load_state_dict(state["query_encoder"])
    model.text_encoder.load_state_dict(state["text_encoder"])
    model.image_encoder.load_state_dict(state["image_encoder"])
    model.query_projection.load_state_dict(state["query_projection"])
    return model.to(device).eval()


# ─── Training Loop ────────────────────────────────────────────────────────────

def train(args):
    import open_clip

    log.info("=" * 60)
    log.info("MODA Phase 4G — Three-Tower Fashion Retriever")
    log.info("=" * 60)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    log.info("Device: %s", device)

    log.info("Loading FashionCLIP base model...")
    base_model, preprocess, _ = open_clip.create_model_and_transforms(
        "hf-hub:Marqo/marqo-fashionCLIP")
    tokenizer = open_clip.get_tokenizer("hf-hub:Marqo/marqo-fashionCLIP")

    model = ThreeTowerModel(base_model, embed_dim=512).to(device)

    # Freeze product towers — they serve as fixed targets
    model.freeze_product_towers()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_p = sum(p.numel() for p in model.parameters())
    log.info("Three-tower params: %d total, %d trainable (%.1f%%)",
             total_p, trainable, 100 * trainable / total_p)
    log.info("  Query tower:  encoder + projection MLP (TRAINED)")
    log.info("  Text tower:   CLIP text encoder (FROZEN)")
    log.info("  Image tower:  CLIP vision encoder (FROZEN)")

    train_triplets, val_triplets = load_triplets(max_pairs=args.max_pairs)
    if not train_triplets:
        log.error("No training triplets! Check label files.")
        return

    # Phase A: Precompute product embeddings
    all_triplets = train_triplets + val_triplets
    text_cache, img_cache = precompute_product_embeddings(
        model, tokenizer, preprocess, all_triplets, device)

    # Phase B: Train query tower against cached targets
    dataset = CachedThreeTowerDataset(
        train_triplets, tokenizer, text_cache, img_cache, embed_dim=512)

    micro_batch = args.batch_size // args.grad_accum
    dataloader = DataLoader(
        dataset, batch_size=micro_batch, shuffle=True,
        num_workers=2, drop_last=True, pin_memory=True,
    )

    distill_pairs = load_distillation_pairs()
    distill_loader = None
    if distill_pairs:
        distill_ds = DistillationDataset(distill_pairs, tokenizer, text_cache, 512)
        distill_loader = DataLoader(
            distill_ds, batch_size=micro_batch, shuffle=True,
            num_workers=0, drop_last=True, pin_memory=True,
        )
        log.info("Distillation dataloader: %d batches", len(distill_loader))

    param_groups = [
        {"params": model.query_projection.parameters(), "lr": args.lr * 10},
        {"params": model.query_encoder.parameters(), "lr": args.lr},
    ]
    optimizer = torch.optim.AdamW(param_groups, weight_decay=0.01)

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

    base_text_acc, base_img_acc = evaluate_three_tower(
        model, tokenizer, val_triplets[:2000], text_cache, img_cache, device)
    log.info("Baseline — text_acc=%.3f, img_acc=%.3f", base_text_acc, base_img_acc)

    best_text_acc = base_text_acc
    global_step = 0
    t0 = time.time()

    for epoch in range(args.epochs):
        model.query_encoder.train()
        model.query_projection.train()
        epoch_metrics = defaultdict(float)
        micro_steps = 0
        optimizer.zero_grad(set_to_none=True)
        distill_iter = iter(distill_loader) if distill_loader else None

        for batch_idx, (q_tok, pos_t_emb, neg_t_emb, pos_i_emb, neg_i_emb, neg_type_idx) in enumerate(dataloader):
            q_tok = q_tok.to(device, non_blocking=True)
            pos_t_emb = pos_t_emb.to(device, non_blocking=True)
            neg_t_emb = neg_t_emb.to(device, non_blocking=True)
            pos_i_emb = pos_i_emb.to(device, non_blocking=True)
            neg_i_emb = neg_i_emb.to(device, non_blocking=True)
            batch_neg_types = ["soft" if nt == 1 else "hard" for nt in neg_type_idx.tolist()]

            use_amp = device == "mps"
            if use_amp:
                with torch.autocast(device_type="mps", dtype=torch.float16):
                    q_emb = model.encode_query(q_tok)
                    loss, metrics = three_tower_loss(
                        q_emb, pos_t_emb, neg_t_emb, pos_i_emb, neg_i_emb,
                        temperature=args.temperature,
                        neg_types=batch_neg_types,
                    )
                    loss = loss / args.grad_accum
            else:
                q_emb = model.encode_query(q_tok)
                loss, metrics = three_tower_loss(
                    q_emb, pos_t_emb, neg_t_emb, pos_i_emb, neg_i_emb,
                    temperature=args.temperature,
                    neg_types=batch_neg_types,
                )
                loss = loss / args.grad_accum

            loss.backward()

            if distill_iter is not None:
                try:
                    dq_tok, d_emb_i, d_emb_j, d_margin = next(distill_iter)
                except StopIteration:
                    distill_iter = iter(distill_loader)
                    dq_tok, d_emb_i, d_emb_j, d_margin = next(distill_iter)
                dq_tok = dq_tok.to(device, non_blocking=True)
                d_emb_i = d_emb_i.to(device, non_blocking=True)
                d_emb_j = d_emb_j.to(device, non_blocking=True)
                d_margin = d_margin.to(device, non_blocking=True)
                if use_amp:
                    with torch.autocast(device_type="mps", dtype=torch.float16):
                        dq_emb = model.encode_query(dq_tok)
                else:
                    dq_emb = model.encode_query(dq_tok)
                d_loss = margin_mse_loss(dq_emb, d_emb_i, d_emb_j, d_margin)
                (args.distill_weight * d_loss / args.grad_accum).backward()
                metrics["loss_distill"] = d_loss.item()

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
                        "  [ep%d step%d/%d] total=%.4f qt=%.4f qi=%.4f "
                        "lr=%.2e %.0fms/step",
                        epoch + 1, global_step % eff_steps or eff_steps, eff_steps,
                        avg["loss_total"], avg["loss_qt"], avg["loss_qi"],
                        scheduler.get_last_lr()[0],
                        elapsed / global_step * 1000,
                    )

                if global_step % eval_steps == 0:
                    t_acc, i_acc = evaluate_three_tower(
                        model, tokenizer, val_triplets[:2000],
                        text_cache, img_cache, device)
                    log.info("  → Val text_acc=%.3f img_acc=%.3f (best=%.3f)",
                             t_acc, i_acc, best_text_acc)
                    if t_acc > best_text_acc:
                        best_text_acc = t_acc
                        save_three_tower(model, OUTPUT_DIR / "best")
                        log.info("  → New best! Saved.")
                    model.query_encoder.train()
                    model.query_projection.train()

        t_acc, i_acc = evaluate_three_tower(
            model, tokenizer, val_triplets[:2000],
            text_cache, img_cache, device)
        log.info("Epoch %d — text_acc=%.3f img_acc=%.3f (best=%.3f)",
                 epoch + 1, t_acc, i_acc, best_text_acc)
        if t_acc > best_text_acc:
            best_text_acc = t_acc
            save_three_tower(model, OUTPUT_DIR / "best")
            log.info("New best! Saved.")

        if device == "mps":
            torch.mps.empty_cache()

    elapsed = time.time() - t0
    log.info("=" * 60)
    log.info("Three-Tower training complete in %.1f min", elapsed / 60)
    log.info("Baseline text_acc=%.3f → Best=%.3f (+%.1f%%)",
             base_text_acc, best_text_acc,
             100 * (best_text_acc - base_text_acc))
    log.info("=" * 60)
    save_three_tower(model, OUTPUT_DIR)


def parse_args():
    p = argparse.ArgumentParser(
        description="MODA Phase 4G — Three-Tower Fashion Retriever")
    p.add_argument("--max_pairs", type=int, default=None)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-6)
    p.add_argument("--temperature", type=float, default=0.07)
    p.add_argument("--distill_weight", type=float, default=0.1,
                   help="Weight for MarginMSE distillation loss (0 = off)")
    p.add_argument("--quick", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.quick:
        args.max_pairs = 2000
        args.epochs = 1
    train(args)
