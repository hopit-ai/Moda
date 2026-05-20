"""
MODA Phase 3 — Fused Item Tower with Contrastive Training (Path A)

Two-tower architecture following the Vinted pattern:

ITEM TOWER (trained end-to-end):
  FashionCLIP_text(article) -> 512d -> Linear -> 256d  --|
  color_embed(colour_group_name) -> Embed(64) -> Linear -> 256d  --|-- Concat -> 1024d
  category_embed(product_type_name) -> Embed(64) -> Linear -> 256d --|     -> FusionMLP
  group_embed(product_group_name) -> Embed(32) -> Linear -> 256d   --|     -> 256d output

QUERY TOWER (frozen CLIP + trainable projection head):
  FashionCLIP_text(query) -> 512d -> ProjectionHead(GELU, LayerNorm) -> 256d

TRAINING:
  InfoNCE contrastive loss with in-batch negatives + 1 hard negative per query
  Learnable temperature parameter
  AdamW, lr=1e-4, cosine annealing, 10-20 epochs

Data: biencoder_retriever_labels.jsonl (100K labels from Phase 3.5)

Output:
  models/moda-fused-item-tower/        final model
  models/moda-fused-item-tower/best/   best checkpoint

Usage:
  python benchmark/train_fused_item_tower.py
  python benchmark/train_fused_item_tower.py --epochs 5 --quick
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
import pandas as pd
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
HNM_DIR = _REPO_ROOT / "data" / "raw" / "hnm_real"
MODEL_DIR = _REPO_ROOT / "models"

LABELS_PATH = PROCESSED_DIR / "biencoder_retriever_labels.jsonl"
SPLIT_PATH = PROCESSED_DIR / "query_splits.json"
OUTPUT_DIR = MODEL_DIR / "moda-fused-item-tower"

FUSED_DIM = 256
RANDOM_SEED = 42

FIELD_COLS = {
    "color": "colour_group_name",
    "category": "product_type_name",
    "group": "product_group_name",
}


# ─── Vocabulary ───────────────────────────────────────────────────────────────

def build_vocabularies(articles_df: pd.DataFrame) -> dict[str, dict[str, int]]:
    """Build field -> {value: index} maps. Index 0 is reserved for unknown."""
    vocabs = {}
    for field, col in FIELD_COLS.items():
        unique_vals = sorted(
            set(v.strip() for v in articles_df[col].unique() if v.strip())
        )
        val_to_idx = {"<UNK>": 0}
        for i, v in enumerate(unique_vals, start=1):
            val_to_idx[v] = i
        vocabs[field] = val_to_idx
        log.info("  %s vocabulary: %d values (+UNK)", field, len(unique_vals))
    return vocabs


# ─── Model Architecture ──────────────────────────────────────────────────────

class QueryTower(nn.Module):
    """Frozen FashionCLIP text encoder + trainable projection head."""

    def __init__(self, clip_dim: int = 512, out_dim: int = FUSED_DIM):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(clip_dim, out_dim),
            nn.GELU(),
            nn.LayerNorm(out_dim),
        )

    def forward(self, clip_emb: torch.Tensor) -> torch.Tensor:
        return self.proj(clip_emb)


class ItemTower(nn.Module):
    """Fuses FashionCLIP text embedding with learned field embeddings."""

    def __init__(
        self,
        clip_dim: int = 512,
        color_vocab_size: int = 51,
        category_vocab_size: int = 132,
        group_vocab_size: int = 20,
        field_embed_dim: int = 64,
        group_embed_dim: int = 32,
        proj_dim: int = 256,
        out_dim: int = FUSED_DIM,
    ):
        super().__init__()

        self.clip_proj = nn.Linear(clip_dim, proj_dim)

        self.color_embed = nn.Embedding(color_vocab_size, field_embed_dim)
        self.color_proj = nn.Linear(field_embed_dim, proj_dim)

        self.cat_embed = nn.Embedding(category_vocab_size, field_embed_dim)
        self.cat_proj = nn.Linear(field_embed_dim, proj_dim)

        self.group_embed = nn.Embedding(group_vocab_size, group_embed_dim)
        self.group_proj = nn.Linear(group_embed_dim, proj_dim)

        fusion_in = proj_dim * 4  # 4 projected streams concatenated
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, fusion_in // 2),
            nn.GELU(),
            nn.LayerNorm(fusion_in // 2),
            nn.Dropout(0.1),
            nn.Linear(fusion_in // 2, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(
        self,
        clip_emb: torch.Tensor,
        color_idx: torch.Tensor,
        cat_idx: torch.Tensor,
        group_idx: torch.Tensor,
    ) -> torch.Tensor:
        h_clip = self.clip_proj(clip_emb)
        h_color = self.color_proj(self.color_embed(color_idx))
        h_cat = self.cat_proj(self.cat_embed(cat_idx))
        h_group = self.group_proj(self.group_embed(group_idx))

        fused = torch.cat([h_clip, h_color, h_cat, h_group], dim=-1)
        return self.fusion(fused)


class FusedRetriever(nn.Module):
    """Two-tower retriever with fused item tower and query projection head."""

    def __init__(
        self,
        clip_dim: int = 512,
        color_vocab_size: int = 51,
        category_vocab_size: int = 132,
        group_vocab_size: int = 20,
        out_dim: int = FUSED_DIM,
    ):
        super().__init__()
        self.query_tower = QueryTower(clip_dim, out_dim)
        self.item_tower = ItemTower(
            clip_dim=clip_dim,
            color_vocab_size=color_vocab_size,
            category_vocab_size=category_vocab_size,
            group_vocab_size=group_vocab_size,
            out_dim=out_dim,
        )
        self.log_temperature = nn.Parameter(torch.tensor(math.log(1 / 0.07)))

    @property
    def temperature(self) -> torch.Tensor:
        return self.log_temperature.exp().clamp(min=0.01, max=0.5)

    def encode_query(self, clip_emb: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.query_tower(clip_emb), dim=-1)

    def encode_item(
        self,
        clip_emb: torch.Tensor,
        color_idx: torch.Tensor,
        cat_idx: torch.Tensor,
        group_idx: torch.Tensor,
    ) -> torch.Tensor:
        return F.normalize(
            self.item_tower(clip_emb, color_idx, cat_idx, group_idx), dim=-1
        )


# ─── Dataset ─────────────────────────────────────────────────────────────────

class FusedTripletDataset(Dataset):
    """Yields (query_clip_emb, pos_clip_emb, neg_clip_emb,
               pos_color, pos_cat, pos_group,
               neg_color, neg_cat, neg_group)."""

    def __init__(
        self,
        triplets: list[dict],
        article_clip_embs: dict[str, np.ndarray],
        article_fields: dict[str, dict[str, int]],
        query_clip_embs: dict[str, np.ndarray],
    ):
        self.triplets = triplets
        self.article_clip_embs = article_clip_embs
        self.article_fields = article_fields
        self.query_clip_embs = query_clip_embs

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        t = self.triplets[idx]
        qid = t["query_id"]
        pos_id = t["pos_id"]
        neg_id = t["neg_id"]

        q_emb = torch.from_numpy(self.query_clip_embs[qid])
        p_emb = torch.from_numpy(self.article_clip_embs[pos_id])
        n_emb = torch.from_numpy(self.article_clip_embs[neg_id])

        p_fields = self.article_fields[pos_id]
        n_fields = self.article_fields[neg_id]

        return (
            q_emb,
            p_emb, n_emb,
            p_fields["color"], p_fields["category"], p_fields["group"],
            n_fields["color"], n_fields["category"], n_fields["group"],
        )


# ─── Data Loading ────────────────────────────────────────────────────────────

def load_data(
    vocabs: dict[str, dict[str, int]],
    articles_df: pd.DataFrame,
    max_pairs: int | None = None,
    seed: int = RANDOM_SEED,
) -> tuple[list[dict], list[dict]]:
    """Load triplets from biencoder labels, split by query_splits.json."""
    rng = random.Random(seed)

    splits = json.loads(SPLIT_PATH.read_text())
    train_qids = set(splits["train"])
    val_qids = set(splits["val"])

    art_field_map = {}
    for _, row in articles_df.iterrows():
        aid = str(row["article_id"]).strip()
        art_field_map[aid] = {
            "color": vocabs["color"].get(str(row[FIELD_COLS["color"]]).strip(), 0),
            "category": vocabs["category"].get(str(row[FIELD_COLS["category"]]).strip(), 0),
            "group": vocabs["group"].get(str(row[FIELD_COLS["group"]]).strip(), 0),
        }

    labels_by_query: dict[str, list[dict]] = defaultdict(list)
    with open(LABELS_PATH) as f:
        for line in f:
            if line.strip():
                obj = json.loads(line)
                labels_by_query[obj["query_id"]].append(obj)

    log.info("Loaded labels for %d unique queries", len(labels_by_query))

    def extract_triplets(qids: set[str]) -> list[dict]:
        triplets = []
        for qid, items in labels_by_query.items():
            if qid not in qids:
                continue
            positives = [it for it in items if it["score"] >= 2]
            hard_negs = [it for it in items if it["score"] == 0]
            if not positives or not hard_negs:
                continue
            for pos in positives:
                pos_aid = pos["article_id"]
                if pos_aid not in art_field_map:
                    continue
                neg = rng.choice(hard_negs)
                neg_aid = neg["article_id"]
                if neg_aid not in art_field_map:
                    continue
                triplets.append({
                    "query_id": qid,
                    "query_text": pos["query_text"],
                    "pos_id": pos_aid,
                    "neg_id": neg_aid,
                    "pos_text": pos["product_text"],
                    "neg_text": neg["product_text"],
                })
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

    log.info("Train triplets: %d  Val triplets: %d", len(train_triplets), len(val_triplets))
    return train_triplets, val_triplets, art_field_map


def precompute_clip_embeddings(
    triplets: list[dict],
    articles_df: pd.DataFrame,
    device: str,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Pre-compute FashionCLIP embeddings for all queries and articles in the data."""
    from benchmark.models import load_clip_model, encode_texts_clip
    from benchmark.embed_hnm import build_article_text

    ft_path = _REPO_ROOT / "models" / "moda-fashionclip-finetuned" / "best"
    use_ft = (ft_path / "model_state_dict.pt").exists()

    log.info("Loading FashionCLIP%s...", " (fine-tuned)" if use_ft else "")
    model, _, tokenizer = load_clip_model("fashion-clip", device=device)
    if use_ft:
        sd = torch.load(ft_path / "model_state_dict.pt", map_location="cpu")
        model.load_state_dict(sd, strict=False)
        model = model.to(device).eval()

    unique_queries = {}
    unique_articles = {}
    for t in triplets:
        unique_queries[t["query_id"]] = t["query_text"]
        unique_articles[t["pos_id"]] = t["pos_text"]
        unique_articles[t["neg_id"]] = t["neg_text"]

    art_map = {str(r["article_id"]).strip(): r.to_dict()
               for _, r in articles_df.iterrows()}

    log.info("Encoding %d unique queries...", len(unique_queries))
    q_ids = list(unique_queries.keys())
    q_texts = [unique_queries[qid] for qid in q_ids]
    q_embs = encode_texts_clip(q_texts, model, tokenizer, device, batch_size=128)
    query_emb_map = {qid: q_embs[i] for i, qid in enumerate(q_ids)}

    log.info("Encoding %d unique articles...", len(unique_articles))
    a_ids = list(unique_articles.keys())
    a_texts = []
    for aid in a_ids:
        row = art_map.get(aid)
        if row:
            a_texts.append(build_article_text(row))
        else:
            a_texts.append(unique_articles[aid])
    a_embs = encode_texts_clip(a_texts, model, tokenizer, device, batch_size=128)
    article_emb_map = {aid: a_embs[i] for i, aid in enumerate(a_ids)}

    del model, tokenizer
    if device == "mps":
        torch.mps.empty_cache()

    return query_emb_map, article_emb_map


# ─── Training ─────────────────────────────────────────────────────────────────

def infonce_loss(
    q_emb: torch.Tensor,
    pos_emb: torch.Tensor,
    neg_emb: torch.Tensor,
    temperature: torch.Tensor,
) -> torch.Tensor:
    """InfoNCE with in-batch negatives + 1 hard negative per query."""
    batch_size = q_emb.shape[0]

    inbatch_sim = (q_emb @ pos_emb.T) / temperature
    hard_neg_sim = (q_emb * neg_emb).sum(dim=-1, keepdim=True) / temperature

    all_logits = torch.cat([inbatch_sim, hard_neg_sim], dim=-1)
    labels = torch.arange(batch_size, device=q_emb.device)
    return F.cross_entropy(all_logits, labels)


def uniformity_loss(emb: torch.Tensor, t: float = 2.0) -> torch.Tensor:
    """Penalize embeddings being too close (anti-collapse regularizer).

    Wang & Isola (2020) "Understanding Contrastive Representation Learning
    through Alignment and Uniformity on the Hypersphere".
    """
    sq_pdist = torch.cdist(emb, emb, p=2).pow(2)
    return sq_pdist.mul(-t).exp().mean().log()


@torch.no_grad()
def evaluate_retrieval(
    fused_model: FusedRetriever,
    val_triplets: list[dict],
    query_embs: dict[str, np.ndarray],
    article_embs: dict[str, np.ndarray],
    article_fields: dict[str, dict[str, int]],
    device: str,
    batch_size: int = 256,
) -> float:
    """Compute retrieval accuracy: fraction of queries where pos ranks above neg."""
    fused_model.eval()
    correct = 0
    total = 0

    for start in range(0, len(val_triplets), batch_size):
        batch = val_triplets[start:start + batch_size]

        q_emb = torch.stack([torch.from_numpy(query_embs[t["query_id"]]) for t in batch]).to(device)
        p_clip = torch.stack([torch.from_numpy(article_embs[t["pos_id"]]) for t in batch]).to(device)
        n_clip = torch.stack([torch.from_numpy(article_embs[t["neg_id"]]) for t in batch]).to(device)

        p_color = torch.tensor([article_fields[t["pos_id"]]["color"] for t in batch], device=device)
        p_cat = torch.tensor([article_fields[t["pos_id"]]["category"] for t in batch], device=device)
        p_group = torch.tensor([article_fields[t["pos_id"]]["group"] for t in batch], device=device)

        n_color = torch.tensor([article_fields[t["neg_id"]]["color"] for t in batch], device=device)
        n_cat = torch.tensor([article_fields[t["neg_id"]]["category"] for t in batch], device=device)
        n_group = torch.tensor([article_fields[t["neg_id"]]["group"] for t in batch], device=device)

        q_vec = fused_model.encode_query(q_emb)
        p_vec = fused_model.encode_item(p_clip, p_color, p_cat, p_group)
        n_vec = fused_model.encode_item(n_clip, n_color, n_cat, n_group)

        pos_sim = (q_vec * p_vec).sum(dim=-1)
        neg_sim = (q_vec * n_vec).sum(dim=-1)

        correct += (pos_sim > neg_sim).sum().item()
        total += len(batch)

    return correct / total if total > 0 else 0.0


def train(args):
    log.info("=" * 60)
    log.info("MODA — Fused Item Tower with Contrastive Training (Path A)")
    log.info("Architecture: Two-tower, Item=FusionMLP(CLIP+color+cat+group)")
    log.info("Loss: InfoNCE + in-batch + hard negatives")
    log.info("=" * 60)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    log.info("Device: %s", device)

    log.info("Loading articles...")
    articles_df = pd.read_csv(HNM_DIR / "articles.csv", dtype=str).fillna("")
    vocabs = build_vocabularies(articles_df)

    train_triplets, val_triplets, art_field_map = load_data(
        vocabs, articles_df, max_pairs=args.max_pairs,
    )

    if not train_triplets:
        log.error("No training triplets!")
        return

    all_triplets = train_triplets + val_triplets
    query_embs, article_embs = precompute_clip_embeddings(
        all_triplets, articles_df, device,
    )

    log.info("Building model...")
    model = FusedRetriever(
        clip_dim=512,
        color_vocab_size=len(vocabs["color"]),
        category_vocab_size=len(vocabs["category"]),
        group_vocab_size=len(vocabs["group"]),
        out_dim=FUSED_DIM,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info("Model parameters: %d total, %d trainable", total_params, trainable_params)

    dataset = FusedTripletDataset(
        train_triplets, article_embs, art_field_map, query_embs,
    )
    micro_batch = args.batch_size // args.grad_accum
    dataloader = DataLoader(
        dataset,
        batch_size=micro_batch,
        shuffle=True,
        num_workers=0,
        drop_last=True,
        pin_memory=False,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    effective_steps = len(dataloader) // args.grad_accum
    total_steps = effective_steps * args.epochs
    warmup_steps = max(50, int(total_steps * 0.05))

    def lr_schedule(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)
    eval_steps = max(50, effective_steps // 3)

    log.info("Micro-batch: %d | Grad accum: %d | Effective batch: %d",
             micro_batch, args.grad_accum, args.batch_size)
    log.info("Steps/epoch: %d | Warmup: %d | Eval every: %d",
             effective_steps, warmup_steps, eval_steps)

    base_acc = evaluate_retrieval(
        model, val_triplets[:2000], query_embs, article_embs,
        art_field_map, device,
    )
    log.info("Baseline val accuracy (random init): %.3f", base_acc)

    best_acc = base_acc
    global_step = 0
    t0 = time.time()

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        micro_steps = 0
        optimizer.zero_grad(set_to_none=True)

        for batch_idx, batch_data in enumerate(dataloader):
            (q_emb, p_clip, n_clip,
             p_color, p_cat, p_group,
             n_color, n_cat, n_group) = [x.to(device) for x in batch_data]

            q_vec = model.encode_query(q_emb)
            p_vec = model.encode_item(p_clip, p_color, p_cat, p_group)
            n_vec = model.encode_item(n_clip, n_color, n_cat, n_group)

            ce_loss = infonce_loss(q_vec, p_vec, n_vec, model.temperature)
            uni_loss = (uniformity_loss(p_vec) + uniformity_loss(n_vec)) * 0.5
            loss = (ce_loss + 0.1 * uni_loss) / args.grad_accum

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
                    with torch.no_grad():
                        p_sim = (p_vec @ p_vec.T).fill_diagonal_(0)
                        avg_p_sim = p_sim.sum() / (p_vec.shape[0] * (p_vec.shape[0] - 1))
                    log.info(
                        "  [epoch %d, step %d/%d] loss=%.4f  lr=%.2e  temp=%.3f  "
                        "item_sim=%.3f  %.0fms/step  (%.1f min)",
                        epoch + 1, global_step % effective_steps or effective_steps,
                        effective_steps, avg_loss, lr,
                        model.temperature.item(),
                        avg_p_sim.item(),
                        elapsed / global_step * 1000, elapsed / 60,
                    )

                if global_step % eval_steps == 0:
                    acc = evaluate_retrieval(
                        model, val_triplets[:2000], query_embs, article_embs,
                        art_field_map, device,
                    )
                    log.info("  → Val accuracy: %.3f (best: %.3f)", acc, best_acc)
                    if acc > best_acc:
                        best_acc = acc
                        save_model(model, vocabs, OUTPUT_DIR / "best")
                        log.info("  → New best! Saved.")
                    model.train()

        avg_loss = epoch_loss / micro_steps if micro_steps > 0 else 0
        log.info("Epoch %d/%d — avg loss: %.4f", epoch + 1, args.epochs, avg_loss)

        acc = evaluate_retrieval(
            model, val_triplets[:2000], query_embs, article_embs,
            art_field_map, device,
        )
        log.info("End-of-epoch val accuracy: %.3f (best: %.3f)", acc, best_acc)
        if acc > best_acc:
            best_acc = acc
            save_model(model, vocabs, OUTPUT_DIR / "best")
            log.info("New best! Saved.")

        if device == "mps":
            torch.mps.empty_cache()

    elapsed = time.time() - t0
    log.info("Training finished in %.1f min", elapsed / 60)
    log.info("Baseline accuracy: %.3f → Best accuracy: %.3f (+%.1f%%)",
             base_acc, best_acc, 100 * (best_acc - base_acc))

    save_model(model, vocabs, OUTPUT_DIR)
    log.info("Final model saved → %s", OUTPUT_DIR)


def save_model(model: FusedRetriever, vocabs: dict, path: Path):
    path.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path / "model_state_dict.pt")
    meta = {
        "architecture": "FusedRetriever",
        "fused_dim": FUSED_DIM,
        "clip_dim": 512,
        "vocabs": {k: list(v.keys()) for k, v in vocabs.items()},
        "vocab_sizes": {k: len(v) for k, v in vocabs.items()},
        "phase": "3-PathA",
    }
    with open(path / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    log.info("Model saved to %s", path)


def parse_args():
    p = argparse.ArgumentParser(description="Train Fused Item Tower (Path A)")
    p.add_argument("--max_pairs", type=int, default=None)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch_size", type=int, default=256,
                   help="Effective batch size (larger = more in-batch negatives)")
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--quick", action="store_true",
                   help="Quick test: 5K pairs, 3 epochs")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.quick:
        args.max_pairs = 5_000
        args.epochs = 3
        log.info("QUICK MODE: 5K triplets, 3 epochs")
    train(args)
