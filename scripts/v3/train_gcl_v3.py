"""Phase 2/3/4 — Grouped Listwise GCL Training for beating FashionSigLIP.

Architecture:
  - Base: ViT-B-16-SigLIP/webli (203M, same starting point as FSL)
  - Configurable training scope (text-only, heads-only, text+image-light, symmetric)

Loss:
  - Grouped Listwise GCL: K queries × N products/query, weighted sigmoid CE
  - Anchor regularization: MSE against pretrained embeddings (drift penalty)
  - Multi-field text: query + title + composite encoded separately, weighted sum

Usage:
  # Smoke test (Phase 3)
  python3 scripts/v3/train_gcl_v3.py --scope text-only --epochs 1 --max-pairs 5000

  # Full training (Phase 4)
  python3 scripts/v3/train_gcl_v3.py --scope text-only --epochs 5

  # With specific device
  python3 scripts/v3/train_gcl_v3.py --scope text-only --device mps
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import logging
import math
import os
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("train-gcl-v3")

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data" / "processed" / "v3_stratified_multifield"
CHECKPOINT_DIR = REPO_ROOT / "checkpoints" / "v3_gcl"

# ── Training scopes ──────────────────────────────────────────────────────────

SCOPES = {
    "text-only": {
        "description": "Text tower last 4 blocks + text projection",
        "train_patterns": [
            "text.transformer.resblocks.8.",
            "text.transformer.resblocks.9.",
            "text.transformer.resblocks.10.",
            "text.transformer.resblocks.11.",
            "text.text_projection",
            "text.ln_final",
            "logit_scale",
            "logit_bias",
        ],
    },
    "heads-only": {
        "description": "Image proj + text proj + logit scale/bias only",
        "train_patterns": [
            "visual.trunk.attn_pool",
            "visual.head",
            "text.text_projection",
            "text.ln_final",
            "logit_scale",
            "logit_bias",
        ],
    },
    "text-image-light": {
        "description": "Text last 4 + image last 1 block + both projections",
        "train_patterns": [
            "text.transformer.resblocks.8.",
            "text.transformer.resblocks.9.",
            "text.transformer.resblocks.10.",
            "text.transformer.resblocks.11.",
            "text.text_projection",
            "text.ln_final",
            "visual.trunk.blocks.11.",
            "visual.trunk.norm",
            "visual.trunk.attn_pool",
            "visual.head",
            "logit_scale",
            "logit_bias",
        ],
    },
    "symmetric-last2": {
        "description": "Image last 2 + text last 2 + projections",
        "train_patterns": [
            "text.transformer.resblocks.10.",
            "text.transformer.resblocks.11.",
            "text.text_projection",
            "text.ln_final",
            "visual.trunk.blocks.10.",
            "visual.trunk.blocks.11.",
            "visual.trunk.norm",
            "visual.trunk.attn_pool",
            "visual.head",
            "logit_scale",
            "logit_bias",
        ],
    },
}


# ── Dataset ──────────────────────────────────────────────────────────────────


class GroupedQueryDataset(Dataset):
    """Loads pairs.jsonl and groups by query for structured batch sampling.

    Each __getitem__ returns one query group: (query_text, title, composite,
    list_of_images, list_of_weights).
    """

    def __init__(
        self,
        data_dir: Path,
        max_pairs: int | None = None,
        products_per_query: int = 16,
        min_products: int = 4,
        seed: int = 42,
    ):
        self.data_dir = data_dir
        self.products_per_query = products_per_query
        self.rng = random.Random(seed)

        log.info("Loading pairs.jsonl...")
        groups: dict[str, list[dict]] = defaultdict(list)
        jsonl_path = data_dir / "pairs.jsonl"

        total_loaded = 0
        with open(jsonl_path) as f:
            for line in f:
                row = json.loads(line)
                groups[row["query"]].append(row)
                total_loaded += 1
                if max_pairs and total_loaded >= max_pairs:
                    break

        # Filter out queries with too few products
        self.query_groups: list[tuple[str, list[dict]]] = [
            (q, prods) for q, prods in groups.items()
            if len(prods) >= min_products
        ]
        self.rng.shuffle(self.query_groups)

        log.info(
            "Dataset: %d queries (of %d total), %d pairs loaded, %d products/query target",
            len(self.query_groups), len(groups), total_loaded, products_per_query,
        )

    def __len__(self) -> int:
        return len(self.query_groups)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        query_text, products = self.query_groups[idx]

        # Sample N products (with replacement if needed)
        if len(products) >= self.products_per_query:
            sampled = self.rng.sample(products, self.products_per_query)
        else:
            sampled = products + self.rng.choices(
                products, k=self.products_per_query - len(products)
            )

        images = []
        weights = []
        titles = []
        composites = []
        l1_categories = []

        for prod in sampled:
            img_path = self.data_dir / prod["image_path"]
            try:
                img = Image.open(img_path).convert("RGB")
            except Exception:
                img = Image.new("RGB", (224, 224), (128, 128, 128))
            images.append(img)
            weights.append(prod.get("weight", 0.5))
            titles.append(prod.get("title", ""))
            composites.append(prod.get("composite", ""))
            l1_categories.append(prod.get("l1_category", "other"))

        # Use the most common L1 category as the group's category label
        from collections import Counter
        cat_counts = Counter(l1_categories)
        group_l1 = cat_counts.most_common(1)[0][0]

        return {
            "query": query_text,
            "titles": titles,
            "composites": composites,
            "images": images,
            "weights": weights,
            "l1_category": group_l1,
        }


def collate_grouped(batch: list[dict]) -> dict[str, Any]:
    """Collate K query groups into a flat batch."""
    all_queries = []
    all_titles = []
    all_composites = []
    all_images = []
    all_weights = []
    all_l1_categories = []
    group_ids = []

    for gid, item in enumerate(batch):
        n = len(item["images"])
        all_queries.extend([item["query"]] * n)
        all_titles.extend(item["titles"])
        all_composites.extend(item["composites"])
        all_images.extend(item["images"])
        all_weights.extend(item["weights"])
        all_l1_categories.extend([item["l1_category"]] * n)
        group_ids.extend([gid] * n)

    return {
        "queries": all_queries,
        "titles": all_titles,
        "composites": all_composites,
        "images": all_images,
        "weights": torch.tensor(all_weights, dtype=torch.float32),
        "group_ids": torch.tensor(group_ids, dtype=torch.long),
        "l1_categories": all_l1_categories,
    }


# ── Loss Functions ───────────────────────────────────────────────────────────


class GroupedGCLLoss(nn.Module):
    """Grouped Generalized Contrastive Loss.

    For each query in the batch, all products under the same query are positives
    (weighted by score_linear), all products under different queries are negatives.
    Uses sigmoid cross-entropy (SigLIP-style) rather than softmax.
    """

    def forward(
        self,
        text_embs: torch.Tensor,   # (B, D) - one per product (from query text)
        image_embs: torch.Tensor,  # (B, D) - one per product
        group_ids: torch.Tensor,   # (B,) - which query group each product belongs to
        weights: torch.Tensor,     # (B,) - relevance weight per product
        logit_scale: torch.Tensor,
        logit_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Compute pairwise similarities
        text_embs = F.normalize(text_embs, dim=-1)
        image_embs = F.normalize(image_embs, dim=-1)

        # (B, B) similarity matrix
        logits = text_embs @ image_embs.T * logit_scale.exp()
        if logit_bias is not None:
            logits = logits + logit_bias

        # Labels: 1.0 if same group, 0.0 if different group
        # Weight positives by their relevance score
        labels = (group_ids.unsqueeze(0) == group_ids.unsqueeze(1)).float()

        # Weight matrix: positives weighted by product relevance, negatives = 1.0
        weight_matrix = torch.where(
            labels > 0,
            weights.unsqueeze(0).expand_as(labels),
            torch.ones_like(labels),
        )

        # Sigmoid cross-entropy (SigLIP-style)
        loss = -weight_matrix * (
            labels * F.logsigmoid(logits) +
            (1 - labels) * F.logsigmoid(-logits)
        )

        return loss.mean()


class AnchorRegLoss(nn.Module):
    """Drift penalty: MSE between current and pretrained embeddings."""

    def forward(
        self,
        current_embs: torch.Tensor,
        anchor_embs: torch.Tensor,
    ) -> torch.Tensor:
        return F.mse_loss(current_embs, anchor_embs.detach())


# ── Training Logic ───────────────────────────────────────────────────────────


def apply_training_scope(model, scope_name: str) -> int:
    """Freeze all params, then unfreeze those matching the scope patterns."""
    scope = SCOPES[scope_name]

    # Freeze everything
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze matching patterns
    trainable = 0
    for name, param in model.named_parameters():
        for pattern in scope["train_patterns"]:
            if pattern in name:
                param.requires_grad = True
                trainable += param.numel()
                break

    log.info("Scope '%s': %s", scope_name, scope["description"])
    log.info("Trainable parameters: %d (%.1fM)", trainable, trainable / 1e6)
    return trainable


def get_anchor_embeddings(
    model, tokenizer, preprocess, images: list, device: torch.device
) -> torch.Tensor:
    """Compute anchor embeddings from pretrained model (before training)."""
    model.eval()
    with torch.no_grad():
        processed = torch.stack([preprocess(img) for img in images]).to(device)
        embs = model.encode_image(processed)
        embs = F.normalize(embs, dim=-1)
    return embs


def train_one_epoch(
    model,
    tokenizer,
    preprocess,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    gcl_loss_fn: GroupedGCLLoss,
    anchor_loss_fn: AnchorRegLoss,
    anchor_model,
    device: torch.device,
    epoch: int,
    lambda_anchor: float,
    multi_field_weights: tuple[float, float, float] = (0.6, 0.3, 0.1),
) -> dict[str, float]:
    """Train one epoch with grouped GCL + anchor reg."""
    model.train()
    total_loss = 0.0
    total_gcl = 0.0
    total_anchor = 0.0
    n_batches = 0

    for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch}")):
        # Prepare images
        images_pil = batch["images"]
        images_tensor = torch.stack([preprocess(img) for img in images_pil]).to(device)

        # Encode images
        image_embs = model.encode_image(images_tensor)

        # ── Loss 1: Original search query → image matching ──
        w_query, w_title, w_composite = multi_field_weights

        queries_tok = tokenizer(batch["queries"]).to(device)
        query_embs = model.encode_text(queries_tok)

        # Title embeddings
        if w_title > 0 and any(t.strip() for t in batch["titles"]):
            titles_tok = tokenizer(batch["titles"]).to(device)
            title_embs = model.encode_text(titles_tok)
        else:
            title_embs = torch.zeros_like(query_embs)

        # Composite embeddings (color + material + garment)
        if w_composite > 0 and any(c.strip() for c in batch["composites"]):
            composites_tok = tokenizer(batch["composites"]).to(device)
            composite_embs = model.encode_text(composites_tok)
        else:
            composite_embs = torch.zeros_like(query_embs)

        # Weighted text embedding for search queries
        text_embs_query = (
            w_query * F.normalize(query_embs, dim=-1) +
            w_title * F.normalize(title_embs, dim=-1) +
            w_composite * F.normalize(composite_embs, dim=-1)
        )

        # Get logit scale and bias
        logit_scale = model.logit_scale if hasattr(model, "logit_scale") else torch.tensor(4.6, device=device)
        logit_bias = model.logit_bias if hasattr(model, "logit_bias") else None

        group_ids = batch["group_ids"].to(device)
        weights = batch["weights"].to(device)

        loss_query = gcl_loss_fn(text_embs_query, image_embs, group_ids, weights, logit_scale, logit_bias)

        # ── Loss 2: Category label → image matching (benchmark-style) ──
        # Group products by L1 category and use category as query
        l1_cats = batch["l1_categories"]
        cat_labels = list(set(l1_cats))
        # Create category group IDs (items with same L1 are positives)
        cat_to_id = {c: i for i, c in enumerate(cat_labels)}
        cat_group_ids = torch.tensor([cat_to_id[c] for c in l1_cats], dtype=torch.long, device=device)

        # Encode category labels as text
        cat_texts = [l1_cats[i] for i in range(len(l1_cats))]
        cat_tok = tokenizer(cat_texts).to(device)
        cat_embs = model.encode_text(cat_tok)
        text_embs_cat = F.normalize(cat_embs, dim=-1)

        loss_category = gcl_loss_fn(text_embs_cat, image_embs, cat_group_ids, weights, logit_scale, logit_bias)

        # ── Combined GCL loss: 50% query-based + 50% category-based ──
        loss_gcl = 0.5 * loss_query + 0.5 * loss_category

        # Anchor regularization
        if anchor_model is not None and lambda_anchor > 0:
            with torch.no_grad():
                anchor_embs = anchor_model.encode_image(images_tensor)
                anchor_embs = F.normalize(anchor_embs, dim=-1)
            loss_anchor = anchor_loss_fn(F.normalize(image_embs, dim=-1), anchor_embs)
        else:
            loss_anchor = torch.tensor(0.0, device=device)

        # Total loss
        loss = loss_gcl + lambda_anchor * loss_anchor

        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], max_norm=1.0
        )

        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        total_gcl += loss_gcl.item()
        total_anchor += loss_anchor.item()
        n_batches += 1

        if (batch_idx + 1) % 10 == 0:
            log.info(
                "  [%d/%d] loss=%.4f gcl=%.4f anchor=%.4f lr=%.2e",
                batch_idx + 1, len(dataloader),
                loss.item(), loss_gcl.item(), loss_anchor.item(),
                optimizer.param_groups[0]["lr"],
            )

        # Free memory
        del images_tensor, image_embs, query_embs, title_embs, composite_embs, text_embs_query, text_embs_cat, cat_embs
        if device.type == "mps":
            torch.mps.empty_cache()

    return {
        "loss": total_loss / max(n_batches, 1),
        "gcl_loss": total_gcl / max(n_batches, 1),
        "anchor_loss": total_anchor / max(n_batches, 1),
        "n_batches": n_batches,
    }


# ── Evaluation ───────────────────────────────────────────────────────────────


@torch.no_grad()
def quick_eval(
    model, tokenizer, preprocess, device: torch.device, n_queries: int = 100
) -> dict[str, float]:
    """Quick MAP@10 on a random subset of the training data (sanity check only)."""
    model.eval()

    jsonl_path = DATA_DIR / "pairs.jsonl"
    groups: dict[str, list[dict]] = defaultdict(list)
    with open(jsonl_path) as f:
        for i, line in enumerate(f):
            if i >= 20000:
                break
            row = json.loads(line)
            groups[row["query"]].append(row)

    # Pick queries with enough products
    eligible = [(q, ps) for q, ps in groups.items() if len(ps) >= 10]
    random.shuffle(eligible)
    eligible = eligible[:n_queries]

    if not eligible:
        return {"map10": 0.0}

    aps = []
    for query_text, products in eligible:
        # Encode query
        tok = tokenizer([query_text]).to(device)
        q_emb = model.encode_text(tok)
        q_emb = F.normalize(q_emb, dim=-1)

        # Encode all product images
        imgs = []
        relevances = []
        for prod in products[:50]:
            img_path = DATA_DIR / prod["image_path"]
            try:
                img = Image.open(img_path).convert("RGB")
                imgs.append(preprocess(img))
                relevances.append(prod.get("weight", 0.5))
            except Exception:
                continue

        if len(imgs) < 5:
            continue

        img_tensor = torch.stack(imgs).to(device)
        img_embs = model.encode_image(img_tensor)
        img_embs = F.normalize(img_embs, dim=-1)

        # Compute similarities and rank
        sims = (q_emb @ img_embs.T).squeeze(0).cpu().numpy()
        ranked_indices = np.argsort(-sims)[:10]

        # AP@10: higher weight products ranked higher = better
        sorted_relevances = [relevances[i] for i in ranked_indices]
        threshold = 0.5
        hits = 0
        precision_sum = 0.0
        for rank, rel in enumerate(sorted_relevances, 1):
            if rel >= threshold:
                hits += 1
                precision_sum += hits / rank
        ap = precision_sum / min(10, sum(1 for r in relevances if r >= threshold)) if hits > 0 else 0.0
        aps.append(ap)

        del img_tensor, img_embs
        if device.type == "mps":
            torch.mps.empty_cache()

    model.train()
    return {"map10": np.mean(aps) if aps else 0.0}


# ── Main ─────────────────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(description="Phase 2/3/4 GCL Training")
    p.add_argument("--scope", choices=list(SCOPES.keys()), default="text-only")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--max-pairs", type=int, default=None, help="Limit training pairs (for smoke test)")
    p.add_argument("--queries-per-batch", type=int, default=8, help="K queries per batch")
    p.add_argument("--products-per-query", type=int, default=16, help="N products per query")
    p.add_argument("--lr", type=float, default=5e-6, help="Peak learning rate")
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--lambda-anchor-start", type=float, default=0.5, help="Anchor reg weight at start")
    p.add_argument("--lambda-anchor-end", type=float, default=0.1, help="Anchor reg weight at end")
    p.add_argument("--warmup-pct", type=float, default=0.05, help="Warmup fraction of total steps")
    p.add_argument("--device", type=str, default=None, help="Force device (cuda/mps/cpu)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval-every", type=int, default=50, help="Eval every N batches")
    p.add_argument("--save-every-epoch", action="store_true", default=True)
    p.add_argument("--run-name", type=str, default=None, help="Custom run name for checkpoints")
    return p.parse_args()


def main():
    args = parse_args()
    t0 = time.time()

    # Seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Device
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    log.info("Device: %s", device)

    # Load model
    import open_clip
    log.info("Loading ViT-B-16-SigLIP/webli...")
    model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
        "ViT-B-16-SigLIP", pretrained="webli"
    )
    tokenizer = open_clip.get_tokenizer("ViT-B-16-SigLIP")
    model = model.to(device)

    # Create frozen anchor model (for regularization)
    log.info("Creating frozen anchor model...")
    anchor_model = copy.deepcopy(model)
    anchor_model.eval()
    for p in anchor_model.parameters():
        p.requires_grad = False

    # Apply training scope
    trainable_params = apply_training_scope(model, args.scope)

    # Dataset
    dataset = GroupedQueryDataset(
        DATA_DIR,
        max_pairs=args.max_pairs,
        products_per_query=args.products_per_query,
        min_products=4,
        seed=args.seed,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.queries_per_batch,
        shuffle=True,
        collate_fn=collate_grouped,
        num_workers=0,
        drop_last=True,
    )

    # Optimizer
    trainable_param_list = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_param_list,
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.98),
        eps=1e-6,
    )

    # Scheduler: cosine with warmup
    total_steps = len(dataloader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_pct)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Loss functions
    gcl_loss_fn = GroupedGCLLoss()
    anchor_loss_fn = AnchorRegLoss()

    # Run name
    run_name = args.run_name or f"gcl_v3_{args.scope}_{int(time.time())}"
    run_dir = CHECKPOINT_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config = vars(args)
    config["trainable_params"] = trainable_params
    config["total_queries"] = len(dataset)
    config["total_steps"] = total_steps
    config["device"] = str(device)
    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    log.info("=" * 60)
    log.info("Run: %s", run_name)
    log.info("Scope: %s (%s)", args.scope, SCOPES[args.scope]["description"])
    log.info("Epochs: %d, Steps: %d, Batch: %d queries × %d products = %d",
             args.epochs, total_steps, args.queries_per_batch,
             args.products_per_query, args.queries_per_batch * args.products_per_query)
    log.info("LR: %.2e, Anchor λ: %.2f→%.2f", args.lr, args.lambda_anchor_start, args.lambda_anchor_end)
    log.info("=" * 60)

    # Training loop
    history = []
    best_map10 = 0.0

    for epoch in range(1, args.epochs + 1):
        # Linear decay of anchor lambda
        progress = (epoch - 1) / max(args.epochs - 1, 1)
        lambda_anchor = args.lambda_anchor_start + progress * (args.lambda_anchor_end - args.lambda_anchor_start)
        log.info("Epoch %d/%d — lambda_anchor=%.3f", epoch, args.epochs, lambda_anchor)

        epoch_stats = train_one_epoch(
            model=model,
            tokenizer=tokenizer,
            preprocess=preprocess_train,
            dataloader=dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            gcl_loss_fn=gcl_loss_fn,
            anchor_loss_fn=anchor_loss_fn,
            anchor_model=anchor_model,
            device=device,
            epoch=epoch,
            lambda_anchor=lambda_anchor,
        )

        # Quick eval
        eval_stats = quick_eval(model, tokenizer, preprocess_val, device, n_queries=50)
        epoch_stats.update(eval_stats)
        epoch_stats["epoch"] = epoch
        epoch_stats["lambda_anchor"] = lambda_anchor
        history.append(epoch_stats)

        log.info(
            "Epoch %d results: loss=%.4f, gcl=%.4f, anchor=%.4f, MAP@10=%.4f",
            epoch, epoch_stats["loss"], epoch_stats["gcl_loss"],
            epoch_stats["anchor_loss"], epoch_stats["map10"],
        )

        # Save checkpoint
        if args.save_every_epoch:
            ckpt_path = run_dir / f"epoch_{epoch}.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": {
                    k: v for k, v in model.state_dict().items()
                    if any(p in k for p in SCOPES[args.scope]["train_patterns"])
                },
                "optimizer_state_dict": optimizer.state_dict(),
                "stats": epoch_stats,
            }, ckpt_path)
            log.info("Saved checkpoint: %s", ckpt_path)

        # Save best (trainable params only)
        if epoch_stats["map10"] > best_map10:
            best_map10 = epoch_stats["map10"]
            best_path = run_dir / "best.pt"
            best_state = {
                k: v for k, v in model.state_dict().items()
                if any(p in k for p in SCOPES[args.scope]["train_patterns"])
            }
            torch.save({
                "epoch": epoch,
                "model_state_dict": best_state,
                "stats": epoch_stats,
                "scope": args.scope,
            }, best_path)
            log.info("New best MAP@10=%.4f — saved to %s", best_map10, best_path)

        # Force garbage collection between epochs
        gc.collect()
        if device.type == "mps":
            torch.mps.empty_cache()

    # Save final model (trainable params only to save disk)
    final_path = run_dir / "final.pt"
    trainable_state = {
        k: v for k, v in model.state_dict().items()
        if any(p in k for p in SCOPES[args.scope]["train_patterns"])
    }
    torch.save({
        "model_state_dict": trainable_state,
        "scope": args.scope,
        "config": config,
        "history": history,
    }, final_path)
    log.info("Saved final checkpoint (%d keys): %s", len(trainable_state), final_path)

    # Save training history
    with open(run_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    elapsed = time.time() - t0
    log.info("=" * 60)
    log.info("Training complete in %.1f minutes", elapsed / 60)
    log.info("Best MAP@10: %.4f", best_map10)
    log.info("Checkpoints: %s", run_dir)
    log.info("=" * 60)

    print(f"\nDone: {run_name}")
    print(f"  Best MAP@10: {best_map10:.4f}")
    print(f"  Checkpoint dir: {run_dir}")
    print(f"  Time: {elapsed/60:.1f} min")


if __name__ == "__main__":
    main()
