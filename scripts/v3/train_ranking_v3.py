"""Phase 8 — Ranking-Aware Training for beating FashionSigLIP.

Based on Phase 7 error analysis finding that the gap is RANKING QUALITY,
not recall. FSL wins by placing gold at rank 5.5 vs base's rank 20.

Key changes from train_gcl_v3.py:
  1. Replaces GroupedGCLLoss (sigmoid CE) with SmoothAP loss
     - Directly optimizes Average Precision via differentiable sigmoid approximation
     - Teaches model to rank relevant items HIGHER, not just match/no-match
  2. Adds ListNet auxiliary loss for cross-category discrimination
  3. Keeps dual-loss (query + category label), frozen image tower, anchor reg

Usage:
  python3 scripts/v3/train_ranking_v3.py --scope text-only --epochs 2
  python3 scripts/v3/train_ranking_v3.py --scope text-only --epochs 2 --max-pairs 5000  # smoke test
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import logging
import math
import random
import time
from collections import Counter, defaultdict
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
log = logging.getLogger("train-ranking-v3")

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data" / "processed" / "v3_stratified_multifield"
CHECKPOINT_DIR = REPO_ROOT / "checkpoints" / "v3_ranking"

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


# ── Dataset (reused from train_gcl_v3) ──────────────────────────────────────


class GroupedQueryDataset(Dataset):
    def __init__(self, data_dir: Path, max_pairs: int | None = None,
                 products_per_query: int = 16, min_products: int = 4, seed: int = 42):
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

        self.query_groups = [(q, prods) for q, prods in groups.items() if len(prods) >= min_products]
        self.rng.shuffle(self.query_groups)
        log.info("Dataset: %d queries, %d pairs loaded", len(self.query_groups), total_loaded)

    def __len__(self): return len(self.query_groups)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        query_text, products = self.query_groups[idx]
        if len(products) >= self.products_per_query:
            sampled = self.rng.sample(products, self.products_per_query)
        else:
            sampled = products + self.rng.choices(products, k=self.products_per_query - len(products))

        images, weights, titles, composites, l1_categories = [], [], [], [], []
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

        cat_counts = Counter(l1_categories)
        group_l1 = cat_counts.most_common(1)[0][0]
        return {
            "query": query_text, "titles": titles, "composites": composites,
            "images": images, "weights": weights, "l1_category": group_l1,
        }


def collate_grouped(batch: list[dict]) -> dict[str, Any]:
    all_queries, all_titles, all_composites = [], [], []
    all_images, all_weights, all_l1_categories = [], [], []
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
        "queries": all_queries, "titles": all_titles, "composites": all_composites,
        "images": all_images, "weights": torch.tensor(all_weights, dtype=torch.float32),
        "group_ids": torch.tensor(group_ids, dtype=torch.long),
        "l1_categories": all_l1_categories,
    }


# ── Ranking-Aware Loss Functions ─────────────────────────────────────────────


class SmoothAPLoss(nn.Module):
    """Differentiable approximation of Average Precision (vectorized).

    Uses sigmoid approximation to make the rank indicator differentiable.
    For each query, computes AP over the gallery using soft ranks.

    Based on: "Smooth-AP: Smoothing the Path Towards Large-Scale Image Retrieval"
    (Brown et al., ECCV 2020)
    """

    def __init__(self, temperature: float = 0.01):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        text_embs: torch.Tensor,   # (B, D) query embeddings
        image_embs: torch.Tensor,  # (B, D) gallery embeddings
        group_ids: torch.Tensor,   # (B,) group assignment for each gallery item
        weights: torch.Tensor,     # (B,) relevance weight per gallery item
        logit_scale: torch.Tensor,
        logit_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B = text_embs.shape[0]
        device = text_embs.device

        sims = text_embs @ image_embs.T * logit_scale.exp()
        if logit_bias is not None:
            sims = sims + logit_bias

        labels = (group_ids.unsqueeze(0) == group_ids.unsqueeze(1)).float()
        relevance = labels * weights.unsqueeze(0)
        binary_rel = (relevance > 0.3).float()  # (B, B)

        total_ap = torch.tensor(0.0, device=device)
        n_valid = 0

        for i in range(B):
            rel_i = binary_rel[i]  # (B,)
            n_pos = rel_i.sum()
            if n_pos < 1:
                continue

            sim_i = sims[i]  # (B,)

            # Pairwise difference matrix (B, B): diff[k, j] = sim[k] - sim[j]
            diff = sim_i.unsqueeze(0) - sim_i.unsqueeze(1)  # (B, B)
            soft_ind = torch.sigmoid(diff / self.temperature)  # (B, B)

            # For each column j: rank_all(j) = 1 + sum_k(soft_ind[k,j]) - self-count
            diag = soft_ind.diag()  # (B,)
            rank_all = 1.0 + soft_ind.sum(dim=0) - diag  # (B,)

            # rank_pos(j) = 1 + sum_k(soft_ind[k,j] * rel[k]) - self
            rank_pos = 1.0 + (soft_ind * rel_i.unsqueeze(1)).sum(dim=0) - diag * rel_i  # (B,)

            # AP = sum over positives of (rank_pos / rank_all) / n_pos
            precision_at_pos = (rank_pos / rank_all.clamp(min=1e-6)) * rel_i  # (B,)
            ap_i = precision_at_pos.sum() / n_pos

            total_ap = total_ap + ap_i
            n_valid += 1

        if n_valid == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        return 1.0 - total_ap / n_valid


class ListNetLoss(nn.Module):
    """ListNet loss: KL divergence between target and predicted top-1 probability distributions.

    For each query, computes a softmax distribution over gallery items from both
    the target relevance scores and the predicted similarities, then minimizes
    the cross-entropy between them.

    This teaches the model to produce similarity scores that preserve the ranking
    order of the relevance labels.
    """

    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        text_embs: torch.Tensor,   # (B, D)
        image_embs: torch.Tensor,  # (B, D)
        group_ids: torch.Tensor,   # (B,)
        weights: torch.Tensor,     # (B,) relevance weight
        logit_scale: torch.Tensor,
        logit_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B = text_embs.shape[0]

        sims = text_embs @ image_embs.T * logit_scale.exp()
        if logit_bias is not None:
            sims = sims + logit_bias

        # For each query (row), get relevance scores for all gallery items
        labels = (group_ids.unsqueeze(0) == group_ids.unsqueeze(1)).float()
        relevance = labels * weights.unsqueeze(0)  # (B, B) weighted relevance

        # ListNet: compute cross-entropy between target distribution and predicted distribution
        # Target distribution: softmax over relevance scores (row-wise)
        # Predicted distribution: softmax over similarity scores (row-wise)
        target_dist = F.softmax(relevance / self.temperature, dim=1)
        pred_log_dist = F.log_softmax(sims / self.temperature, dim=1)

        # Cross-entropy: -sum(target * log(pred))
        loss = -(target_dist * pred_log_dist).sum(dim=1).mean()
        return loss


class AnchorRegLoss(nn.Module):
    def forward(self, current_embs: torch.Tensor, anchor_embs: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(current_embs, anchor_embs.detach())


# ── Training Logic ───────────────────────────────────────────────────────────


def apply_training_scope(model, scope_name: str) -> int:
    scope = SCOPES[scope_name]
    for param in model.parameters():
        param.requires_grad = False
    trainable = 0
    for name, param in model.named_parameters():
        for pattern in scope["train_patterns"]:
            if pattern in name:
                param.requires_grad = True
                trainable += param.numel()
                break
    log.info("Scope '%s': %s — %d trainable params (%.1fM)",
             scope_name, scope["description"], trainable, trainable / 1e6)
    return trainable


def train_one_epoch(
    model, tokenizer, preprocess, dataloader: DataLoader,
    optimizer, scheduler,
    smoothap_loss_fn: SmoothAPLoss,
    listnet_loss_fn: ListNetLoss,
    anchor_loss_fn: AnchorRegLoss,
    anchor_model,
    device: torch.device,
    epoch: int,
    lambda_anchor: float,
    loss_weights: dict[str, float],
    multi_field_weights: tuple[float, float, float] = (0.6, 0.3, 0.1),
) -> dict[str, float]:
    """Train one epoch with SmoothAP + ListNet + anchor reg."""
    model.train()
    total_loss = 0.0
    total_smoothap = 0.0
    total_listnet = 0.0
    total_anchor = 0.0
    n_batches = 0

    w_smoothap = loss_weights.get("smoothap", 0.5)
    w_listnet = loss_weights.get("listnet", 0.3)
    w_cat = loss_weights.get("category", 0.2)

    for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch}")):
        images_pil = batch["images"]
        images_tensor = torch.stack([preprocess(img) for img in images_pil]).to(device)
        image_embs = model.encode_image(images_tensor)
        image_embs_norm = F.normalize(image_embs, dim=-1)

        # Multi-field text encoding
        w_query, w_title, w_composite = multi_field_weights
        queries_tok = tokenizer(batch["queries"]).to(device)
        query_embs = model.encode_text(queries_tok)

        if w_title > 0 and any(t.strip() for t in batch["titles"]):
            titles_tok = tokenizer(batch["titles"]).to(device)
            title_embs = model.encode_text(titles_tok)
        else:
            title_embs = torch.zeros_like(query_embs)

        if w_composite > 0 and any(c.strip() for c in batch["composites"]):
            composites_tok = tokenizer(batch["composites"]).to(device)
            composite_embs = model.encode_text(composites_tok)
        else:
            composite_embs = torch.zeros_like(query_embs)

        text_embs_query = F.normalize(
            w_query * F.normalize(query_embs, dim=-1) +
            w_title * F.normalize(title_embs, dim=-1) +
            w_composite * F.normalize(composite_embs, dim=-1),
            dim=-1,
        )

        logit_scale = model.logit_scale if hasattr(model, "logit_scale") else torch.tensor(4.6, device=device)
        logit_bias = model.logit_bias if hasattr(model, "logit_bias") else None
        group_ids = batch["group_ids"].to(device)
        weights = batch["weights"].to(device)

        # ── Loss 1: SmoothAP on search queries ──
        loss_sap = smoothap_loss_fn(text_embs_query, image_embs_norm, group_ids, weights, logit_scale, logit_bias)

        # ── Loss 2: ListNet on search queries (ranking cross-entropy) ──
        loss_ln = listnet_loss_fn(text_embs_query, image_embs_norm, group_ids, weights, logit_scale, logit_bias)

        # ── Loss 3: Category label ListNet (benchmark-style ranking) ──
        l1_cats = batch["l1_categories"]
        cat_labels = list(set(l1_cats))
        cat_to_id = {c: i for i, c in enumerate(cat_labels)}
        cat_group_ids = torch.tensor([cat_to_id[c] for c in l1_cats], dtype=torch.long, device=device)
        cat_texts = l1_cats
        cat_tok = tokenizer(cat_texts).to(device)
        cat_embs = model.encode_text(cat_tok)
        text_embs_cat = F.normalize(cat_embs, dim=-1)
        loss_cat = listnet_loss_fn(text_embs_cat, image_embs_norm, cat_group_ids, weights, logit_scale, logit_bias)

        # Combined ranking loss
        loss_ranking = w_smoothap * loss_sap + w_listnet * loss_ln + w_cat * loss_cat

        # Anchor regularization
        if anchor_model is not None and lambda_anchor > 0:
            with torch.no_grad():
                anchor_embs = anchor_model.encode_image(images_tensor)
                anchor_embs = F.normalize(anchor_embs, dim=-1)
            loss_anchor = anchor_loss_fn(image_embs_norm, anchor_embs)
        else:
            loss_anchor = torch.tensor(0.0, device=device)

        loss = loss_ranking + lambda_anchor * loss_anchor

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], max_norm=1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        total_smoothap += loss_sap.item()
        total_listnet += loss_ln.item()
        total_anchor += loss_anchor.item()
        n_batches += 1

        if (batch_idx + 1) % 10 == 0:
            log.info(
                "  [%d/%d] loss=%.4f sap=%.4f ln=%.4f cat=%.4f anchor=%.4f lr=%.2e",
                batch_idx + 1, len(dataloader),
                loss.item(), loss_sap.item(), loss_ln.item(), loss_cat.item(),
                loss_anchor.item(), optimizer.param_groups[0]["lr"],
            )

        del images_tensor, image_embs, image_embs_norm, query_embs, title_embs, composite_embs
        del text_embs_query, text_embs_cat, cat_embs
        if device.type == "mps":
            torch.mps.empty_cache()

    return {
        "loss": total_loss / max(n_batches, 1),
        "smoothap_loss": total_smoothap / max(n_batches, 1),
        "listnet_loss": total_listnet / max(n_batches, 1),
        "anchor_loss": total_anchor / max(n_batches, 1),
        "n_batches": n_batches,
    }


@torch.no_grad()
def quick_eval(model, tokenizer, preprocess, device: torch.device, n_queries: int = 100) -> dict[str, float]:
    model.eval()
    jsonl_path = DATA_DIR / "pairs.jsonl"
    groups: dict[str, list[dict]] = defaultdict(list)
    with open(jsonl_path) as f:
        for i, line in enumerate(f):
            if i >= 20000:
                break
            row = json.loads(line)
            groups[row["query"]].append(row)

    eligible = [(q, ps) for q, ps in groups.items() if len(ps) >= 10]
    random.shuffle(eligible)
    eligible = eligible[:n_queries]
    if not eligible:
        return {"map10": 0.0}

    aps = []
    for query_text, products in eligible:
        tok = tokenizer([query_text]).to(device)
        q_emb = F.normalize(model.encode_text(tok), dim=-1)

        imgs, relevances = [], []
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
        img_embs = F.normalize(model.encode_image(img_tensor), dim=-1)
        sims = (q_emb @ img_embs.T).squeeze(0).cpu().numpy()
        ranked_indices = np.argsort(-sims)[:10]

        sorted_relevances = [relevances[i] for i in ranked_indices]
        hits, precision_sum = 0, 0.0
        for rank, rel in enumerate(sorted_relevances, 1):
            if rel >= 0.5:
                hits += 1
                precision_sum += hits / rank
        n_rel = sum(1 for r in relevances if r >= 0.5)
        ap = precision_sum / min(10, n_rel) if hits > 0 else 0.0
        aps.append(ap)

        del img_tensor, img_embs
        if device.type == "mps":
            torch.mps.empty_cache()

    model.train()
    return {"map10": np.mean(aps) if aps else 0.0}


def parse_args():
    p = argparse.ArgumentParser(description="Phase 8: Ranking-Aware Training")
    p.add_argument("--scope", choices=list(SCOPES.keys()), default="text-only")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--max-pairs", type=int, default=None)
    p.add_argument("--queries-per-batch", type=int, default=8)
    p.add_argument("--products-per-query", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-6)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--lambda-anchor", type=float, default=0.3)
    p.add_argument("--smoothap-temp", type=float, default=0.01)
    p.add_argument("--listnet-temp", type=float, default=1.0)
    p.add_argument("--w-smoothap", type=float, default=0.5)
    p.add_argument("--w-listnet", type=float, default=0.3)
    p.add_argument("--w-category", type=float, default=0.2)
    p.add_argument("--warmup-pct", type=float, default=0.05)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--run-name", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    t0 = time.time()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    log.info("Device: %s", device)

    import open_clip
    log.info("Loading ViT-B-16-SigLIP/webli...")
    model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
        "ViT-B-16-SigLIP", pretrained="webli"
    )
    tokenizer = open_clip.get_tokenizer("ViT-B-16-SigLIP")
    model = model.to(device)

    log.info("Creating frozen anchor model...")
    anchor_model = copy.deepcopy(model)
    anchor_model.eval()
    for p in anchor_model.parameters():
        p.requires_grad = False

    trainable_params = apply_training_scope(model, args.scope)

    dataset = GroupedQueryDataset(DATA_DIR, max_pairs=args.max_pairs,
                                  products_per_query=args.products_per_query, seed=args.seed)
    dataloader = DataLoader(dataset, batch_size=args.queries_per_batch, shuffle=True,
                            collate_fn=collate_grouped, num_workers=0, drop_last=True)

    trainable_param_list = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_param_list, lr=args.lr,
                                   weight_decay=args.weight_decay, betas=(0.9, 0.98), eps=1e-6)

    total_steps = len(dataloader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_pct)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    smoothap_loss_fn = SmoothAPLoss(temperature=args.smoothap_temp)
    listnet_loss_fn = ListNetLoss(temperature=args.listnet_temp)
    anchor_loss_fn = AnchorRegLoss()

    loss_weights = {"smoothap": args.w_smoothap, "listnet": args.w_listnet, "category": args.w_category}

    run_name = args.run_name or f"ranking_v3_{args.scope}_{int(time.time())}"
    run_dir = CHECKPOINT_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    config = vars(args)
    config["trainable_params"] = trainable_params
    config["total_queries"] = len(dataset)
    config["total_steps"] = total_steps
    config["device"] = str(device)
    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    log.info("=" * 60)
    log.info("Run: %s", run_name)
    log.info("Loss: SmoothAP(%.1f) + ListNet(%.1f) + Category(%.1f)", args.w_smoothap, args.w_listnet, args.w_category)
    log.info("Scope: %s, Epochs: %d, Steps: %d", args.scope, args.epochs, total_steps)
    log.info("Batch: %d queries x %d products = %d items", args.queries_per_batch, args.products_per_query,
             args.queries_per_batch * args.products_per_query)
    log.info("LR: %.2e, Anchor lambda: %.2f", args.lr, args.lambda_anchor)
    log.info("=" * 60)

    history = []
    best_map10 = 0.0

    for epoch in range(1, args.epochs + 1):
        log.info("Epoch %d/%d", epoch, args.epochs)

        epoch_stats = train_one_epoch(
            model=model, tokenizer=tokenizer, preprocess=preprocess_train,
            dataloader=dataloader, optimizer=optimizer, scheduler=scheduler,
            smoothap_loss_fn=smoothap_loss_fn, listnet_loss_fn=listnet_loss_fn,
            anchor_loss_fn=anchor_loss_fn, anchor_model=anchor_model,
            device=device, epoch=epoch, lambda_anchor=args.lambda_anchor,
            loss_weights=loss_weights,
        )

        eval_stats = quick_eval(model, tokenizer, preprocess_val, device, n_queries=50)
        epoch_stats.update(eval_stats)
        epoch_stats["epoch"] = epoch
        history.append(epoch_stats)

        log.info("Epoch %d: loss=%.4f, sap=%.4f, ln=%.4f, anchor=%.4f, MAP@10=%.4f",
                 epoch, epoch_stats["loss"], epoch_stats["smoothap_loss"],
                 epoch_stats["listnet_loss"], epoch_stats["anchor_loss"], epoch_stats["map10"])

        # Save epoch checkpoint (trainable params only)
        ckpt_path = run_dir / f"epoch_{epoch}.pt"
        torch.save({
            "epoch": epoch,
            "model_state_dict": {
                k: v for k, v in model.state_dict().items()
                if any(p in k for p in SCOPES[args.scope]["train_patterns"])
            },
            "optimizer_state_dict": optimizer.state_dict(),
            "stats": epoch_stats,
            "scope": args.scope,
        }, ckpt_path)
        log.info("Saved: %s", ckpt_path)

        if epoch_stats["map10"] > best_map10:
            best_map10 = epoch_stats["map10"]
            best_path = run_dir / "best.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": {
                    k: v for k, v in model.state_dict().items()
                    if any(p in k for p in SCOPES[args.scope]["train_patterns"])
                },
                "stats": epoch_stats,
                "scope": args.scope,
            }, best_path)
            log.info("New best MAP@10=%.4f", best_map10)

        gc.collect()
        if device.type == "mps":
            torch.mps.empty_cache()

    with open(run_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    elapsed = time.time() - t0
    log.info("=" * 60)
    log.info("Training complete in %.1f minutes", elapsed / 60)
    log.info("Best MAP@10: %.4f", best_map10)
    log.info("Checkpoints: %s", run_dir)
    log.info("=" * 60)
    print(f"\nDone: {run_name}\n  Best MAP@10: {best_map10:.4f}\n  Time: {elapsed/60:.1f} min")


if __name__ == "__main__":
    main()
