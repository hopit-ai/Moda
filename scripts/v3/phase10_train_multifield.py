"""Phase 10 — Multi-field GCL training.

Implements GCL Algorithm 2 (multi-field) with weighted cross-entropy loss (Eq. 4).

Supports three training configurations:
  --text-mode query-primary  : Original — query is primary LHS (0.35), title is RHS only
  --text-mode title-primary  : NEW — title moves to LHS as primary (0.45), query demoted (0.15)
  --training-scope text-only : Only last 4 text transformer blocks trainable
  --training-scope text-image-light : Text + last 2 vision blocks trainable (differential LR)

Usage:
  # Title-primary (matches atlas/KAGL eval pattern)
  python3 -u scripts/v3/phase10_train_multifield.py --model-source fsl --text-mode title-primary

  # Text-image-light (both encoders, differential LR)
  python3 -u scripts/v3/phase10_train_multifield.py --model-source fsl --training-scope text-image-light
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
import sys
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
log = logging.getLogger("phase10-train")

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data" / "processed" / "v3_phase10_500k"
CHECKPOINT_DIR = REPO_ROOT / "checkpoints" / "v3_phase10"
PHASE4B_CHECKPOINT = REPO_ROOT / "checkpoints" / "v3_gcl" / "phase4b_dual_loss" / "best.pt"

TEXT_ONLY_PATTERNS = [
    "text.transformer.resblocks.8.",
    "text.transformer.resblocks.9.",
    "text.transformer.resblocks.10.",
    "text.transformer.resblocks.11.",
    "text.text_projection",
    "text.ln_final",
    "logit_scale",
    "logit_bias",
]

VISION_LIGHT_PATTERNS = [
    "visual.trunk.blocks.10.",
    "visual.trunk.blocks.11.",
    "visual.trunk.norm.",
    "visual.trunk.attn_pool.",
]

# LHS/RHS weights per text-mode
LHS_WEIGHTS_QUERY_PRIMARY = {
    "query": 0.35, "l1_category": 0.25, "subcategory3": 0.15,
    "color_str": 0.15, "material_str": 0.10,
}
LHS_WEIGHTS_TITLE_PRIMARY = {
    "title": 0.45, "query": 0.15, "l1_category": 0.15,
    "subcategory3": 0.10, "color_str": 0.10, "material_str": 0.05,
}

RHS_WEIGHTS_QUERY_PRIMARY = {"image": 0.6, "title": 0.4}
RHS_WEIGHTS_TITLE_PRIMARY = {"image": 0.7, "query": 0.3}


# ── Dataset ──────────────────────────────────────────────────────────────────

class MultiFieldDataset(Dataset):
    """Loads pairs.jsonl grouped by query for multi-field GCL training."""

    def __init__(
        self,
        data_dir: Path,
        max_pairs: int | None = None,
        products_per_query: int = 16,
        min_products: int = 4,
        seed: int = 42,
        enriched_path: Path | None = None,
        synthetic_path: Path | None = None,
    ):
        self.data_dir = data_dir
        self.products_per_query = products_per_query
        self.rng = random.Random(seed)

        log.info("Loading pairs.jsonl from %s ...", data_dir)
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

        # Load enriched descriptions (additional query-image pairs)
        n_enriched = 0
        if enriched_path and enriched_path.exists():
            log.info("Loading enriched descriptions from %s ...", enriched_path)
            with open(enriched_path) as f:
                for line in f:
                    row = json.loads(line)
                    eq = row.get("enriched_query", "")
                    if eq and row.get("image_path"):
                        groups[eq].append({
                            "query": eq,
                            "title": row.get("original_title", ""),
                            "image_path": row["image_path"],
                            "weight": row.get("weight", 0.5),
                            "l1_category": row.get("l1_category", "other"),
                            "color_str": row.get("color_str", ""),
                            "material_str": row.get("material_str", ""),
                            "subcategory3": "",
                        })
                        n_enriched += 1
            log.info("  Added %d enriched pairs", n_enriched)

        # Load synthetic texts and inject into existing groups by category
        n_synthetic = 0
        if synthetic_path and synthetic_path.exists():
            log.info("Loading synthetic texts from %s ...", synthetic_path)
            cat_to_groups: dict[str, list[str]] = defaultdict(list)
            for q, prods in groups.items():
                for p in prods:
                    cat = p.get("l1_category", "other")
                    cat_to_groups[cat].append(q)
            
            with open(synthetic_path) as f:
                for line in f:
                    row = json.loads(line)

                    # Support two formats:
                    # Format A (old): {"text": ..., "style": ..., "category": ...}
                    # Format B (new): {"query": ..., "image_path": ..., "l1_category": ...}
                    synth_text = row.get("text") or row.get("query", "")
                    if not synth_text or len(synth_text) < 5:
                        continue

                    # Format B: has a direct image_path — create a product entry directly
                    if "image_path" in row and row["image_path"]:
                        img_path = row["image_path"]
                        l1_cat = row.get("l1_category", "other")
                        synth_prod = {
                            "query": synth_text,
                            "title": synth_text,
                            "image_path": img_path,
                            "l1_category": l1_cat,
                            "color_str": "",
                            "material_str": "",
                            "subcategory3": "",
                        }
                        groups[synth_text].append(synth_prod)
                        n_synthetic += 1
                        continue

                    # Format A: category-based injection (old behavior)
                    style = row.get("style", "")
                    cat = row.get("category", "other").lower()

                    # Map synthetic category to our l1 categories
                    target_cat = "other"
                    if style == "atlas":
                        if cat in ("ethnic wear", "sarees", "kurta", "lehenga choli", "salwar kameez"):
                            target_cat = "ethnic_wear"
                        else:
                            target_cat = "tops"
                    elif style == "kagl":
                        if cat in ("footwear",):
                            target_cat = "shoes"
                        elif cat in ("accessories",):
                            target_cat = "accessories"
                        elif cat in ("bags",):
                            target_cat = "bags"
                        elif cat in ("personal care",):
                            target_cat = "beauty"

                    # Add synthetic text as a new query pointing to random products from matching category
                    matching_queries = cat_to_groups.get(target_cat, [])
                    if not matching_queries:
                        matching_queries = cat_to_groups.get("other", [])
                    if matching_queries:
                        donor_q = self.rng.choice(matching_queries)
                        donor_prods = groups[donor_q]
                        synth_prods = []
                        for p in donor_prods[:4]:
                            sp = dict(p)
                            sp["query"] = synth_text
                            synth_prods.append(sp)
                        if synth_prods:
                            groups[synth_text].extend(synth_prods)
                            n_synthetic += 1
            log.info("  Added %d synthetic query groups", n_synthetic)

        self.query_groups: list[tuple[str, list[dict]]] = [
            (q, prods) for q, prods in groups.items()
            if len(prods) >= min_products
        ]
        self.rng.shuffle(self.query_groups)

        log.info(
            "Dataset: %d queries (of %d), %d pairs + %d enriched, %d products/query",
            len(self.query_groups), len(groups), total_loaded, n_enriched, products_per_query,
        )

    def __len__(self) -> int:
        return len(self.query_groups)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        query_text, products = self.query_groups[idx]

        if len(products) >= self.products_per_query:
            sampled = self.rng.sample(products, self.products_per_query)
        else:
            sampled = products + self.rng.choices(
                products, k=self.products_per_query - len(products)
            )

        images = []
        weights = []
        titles = []
        l1_categories = []
        color_strs = []
        material_strs = []
        subcategory3s = []

        for prod in sampled:
            img_path = self.data_dir / prod["image_path"]
            try:
                img = Image.open(img_path).convert("RGB")
            except Exception:
                img = Image.new("RGB", (224, 224), (128, 128, 128))
            images.append(img)
            weights.append(prod.get("weight", 0.5))
            titles.append(prod.get("title", ""))
            l1_categories.append(prod.get("l1_category", "other"))
            color_strs.append(prod.get("color_str", ""))
            material_strs.append(prod.get("material_str", ""))
            subcategory3s.append(prod.get("subcategory3", ""))

        cat_counts = Counter(l1_categories)
        group_l1 = cat_counts.most_common(1)[0][0]

        # Pick a representative title for this group (longest non-empty title)
        group_title = max(titles, key=len) if any(titles) else ""

        return {
            "query": query_text,
            "group_title": group_title,
            "titles": titles,
            "images": images,
            "weights": weights,
            "l1_category": group_l1,
            "l1_categories": l1_categories,
            "color_strs": color_strs,
            "material_strs": material_strs,
            "subcategory3s": subcategory3s,
        }


def collate_multifield(batch: list[dict]) -> dict[str, Any]:
    all_queries = []
    all_group_titles = []
    all_titles = []
    all_images = []
    all_weights = []
    all_l1_categories = []
    all_color_strs = []
    all_material_strs = []
    all_subcategory3s = []
    group_ids = []

    for gid, item in enumerate(batch):
        n = len(item["images"])
        all_queries.extend([item["query"]] * n)
        all_group_titles.extend([item["group_title"]] * n)
        all_titles.extend(item["titles"])
        all_images.extend(item["images"])
        all_weights.extend(item["weights"])
        all_l1_categories.extend(item["l1_categories"])
        all_color_strs.extend(item["color_strs"])
        all_material_strs.extend(item["material_strs"])
        all_subcategory3s.extend(item["subcategory3s"])
        group_ids.extend([gid] * n)

    return {
        "queries": all_queries,
        "group_titles": all_group_titles,
        "titles": all_titles,
        "images": all_images,
        "weights": torch.tensor(all_weights, dtype=torch.float32),
        "group_ids": torch.tensor(group_ids, dtype=torch.long),
        "l1_categories": all_l1_categories,
        "color_strs": all_color_strs,
        "material_strs": all_material_strs,
        "subcategory3s": all_subcategory3s,
    }


# ── Loss: Paper-faithful Weighted Cross-Entropy (Eq. 4) ─────────────────────

class WeightedCELoss(nn.Module):
    """GCL paper Eq. 4: weighted cross-entropy over similarity matrix.

    Unlike sigmoid CE (pairwise), this is a softmax-based loss that:
    - Treats the diagonal as "correct class" (query i → document i)
    - Weights the loss per-sample by relevance score w_i
    - Computes both row-wise (query→doc) and column-wise (doc→query) losses
    """

    def forward(
        self,
        Z: torch.Tensor,          # (B, B) similarity matrix
        weights: torch.Tensor,     # (B,) per-sample relevance weight
    ) -> torch.Tensor:
        B = Z.shape[0]
        targets = torch.arange(B, device=Z.device)

        # Row-wise: for each query i, classify among all documents
        log_probs_row = F.log_softmax(Z, dim=1)  # (B, B)
        loss_row = -weights * log_probs_row[torch.arange(B, device=Z.device), targets]

        # Column-wise: for each document i, classify among all queries
        log_probs_col = F.log_softmax(Z, dim=0)  # (B, B)
        loss_col = -weights * log_probs_col[targets, torch.arange(B, device=Z.device)]

        return (loss_row.mean() + loss_col.mean()) / 2.0


# ── Training Logic ───────────────────────────────────────────────────────────

def apply_training_scope(model, scope: str = "text-only") -> tuple[int, list[str], list[str]]:
    """Freeze all params, then selectively unfreeze.
    
    Returns (total_trainable, text_param_names, vision_param_names).
    """
    for param in model.parameters():
        param.requires_grad = False

    text_params, vision_params = [], []
    trainable = 0
    for name, param in model.named_parameters():
        for pattern in TEXT_ONLY_PATTERNS:
            if pattern in name:
                param.requires_grad = True
                trainable += param.numel()
                text_params.append(name)
                break

    if scope == "text-image-light":
        for name, param in model.named_parameters():
            for pattern in VISION_LIGHT_PATTERNS:
                if pattern in name:
                    param.requires_grad = True
                    trainable += param.numel()
                    vision_params.append(name)
                    break

    log.info("Scope '%s': %d trainable params (%.1fM) — %d text, %d vision",
             scope, trainable, trainable / 1e6, len(text_params), len(vision_params))
    return trainable, text_params, vision_params


def load_phase4b_checkpoint(model, checkpoint_path: Path, device: torch.device):
    """Load Phase 4b trained weights into model."""
    log.info("Loading Phase 4b checkpoint: %s", checkpoint_path)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt["model_state_dict"]

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    loaded = len(state_dict) - len(unexpected)
    log.info("Loaded %d keys from Phase 4b (%d missing, %d unexpected)",
             loaded, len(missing), len(unexpected))
    return ckpt.get("stats", {})


def encode_text_field(model, tokenizer, texts: list[str], device: torch.device) -> torch.Tensor:
    """Encode a list of text strings, returning normalized embeddings.
    Returns zero vectors for empty strings."""
    non_empty = [(i, t) for i, t in enumerate(texts) if t.strip()]

    B = len(texts)
    D = 768  # ViT-B-16-SigLIP embedding dim
    embs = torch.zeros(B, D, device=device)

    if non_empty:
        indices, valid_texts = zip(*non_empty)
        tok = tokenizer(list(valid_texts)).to(device)
        valid_embs = model.encode_text(tok)
        valid_embs = F.normalize(valid_embs, dim=-1)
        for i, idx in enumerate(indices):
            embs[idx] = valid_embs[i]

    return embs


def train_one_epoch(
    model,
    tokenizer,
    preprocess,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    wce_loss_fn: WeightedCELoss,
    anchor_model,
    device: torch.device,
    epoch: int,
    lambda_anchor: float,
    lhs_field_weights: dict[str, float],
    rhs_field_weights: dict[str, float],
    save_patterns: list[str],
    save_dir: Path | None = None,
    save_every: int = 500,
) -> dict[str, float]:
    """Train one epoch with multi-field GCL (Algorithm 2).
    
    In title-primary mode, 'title' is an LHS field and 'query' can be RHS.
    In query-primary mode, 'query' is LHS and 'title' is RHS (original behavior).
    """
    model.train()
    total_loss = 0.0
    total_avg_loss = 0.0
    total_cross_loss = 0.0
    total_anchor_loss = 0.0
    n_batches = 0

    title_is_lhs = "title" in lhs_field_weights

    for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch}")):
        images_pil = batch["images"]
        images_tensor = torch.stack([preprocess(img) for img in images_pil]).to(device)
        weights = batch["weights"].to(device)
        B = images_tensor.shape[0]

        logit_scale = model.logit_scale if hasattr(model, "logit_scale") else torch.tensor(4.6, device=device)
        logit_bias = model.logit_bias if hasattr(model, "logit_bias") else None

        image_embs = F.normalize(model.encode_image(images_tensor), dim=-1)

        # Build LHS (text) and RHS (doc/image) field embeddings
        lhs_fields = {}
        rhs_fields = {}

        # Always encode these text fields
        query_embs = encode_text_field(model, tokenizer, batch["queries"], device)
        title_embs = encode_text_field(model, tokenizer, batch["titles"], device)
        group_title_embs = encode_text_field(model, tokenizer, batch["group_titles"], device)
        l1_embs = encode_text_field(model, tokenizer, batch["l1_categories"], device)
        sub3_embs = encode_text_field(model, tokenizer, batch["subcategory3s"], device)
        color_embs = encode_text_field(model, tokenizer, batch["color_strs"], device)
        mat_embs = encode_text_field(model, tokenizer, batch["material_strs"], device)

        all_text_embs = {
            "query": query_embs, "title": group_title_embs, "l1_category": l1_embs,
            "subcategory3": sub3_embs, "color_str": color_embs, "material_str": mat_embs,
        }

        for name in lhs_field_weights:
            lhs_fields[name] = all_text_embs[name]
        
        rhs_fields["image"] = image_embs
        if "title" in rhs_field_weights:
            rhs_fields["title"] = title_embs
        if "query" in rhs_field_weights:
            rhs_fields["query"] = query_embs

        # Weighted average embeddings
        lhs_avg = sum(lhs_field_weights[k] * v for k, v in lhs_fields.items())
        lhs_avg = F.normalize(lhs_avg, dim=-1)
        rhs_avg = sum(rhs_field_weights[k] * v for k, v in rhs_fields.items())
        rhs_avg = F.normalize(rhs_avg, dim=-1)

        # Loss 1: averaged embedding matching
        Z_avg = lhs_avg @ rhs_avg.T * logit_scale.exp()
        if logit_bias is not None:
            Z_avg = Z_avg + logit_bias
        loss_avg = wce_loss_fn(Z_avg, weights)

        # Loss 2: cross-field losses
        loss_cross = torch.tensor(0.0, device=device)
        n_cross = 0
        for lhs_name, lhs_emb in lhs_fields.items():
            has_content = lhs_emb.abs().sum(dim=-1) > 1e-6
            if has_content.sum() < 2:
                continue
            for rhs_name, rhs_emb in rhs_fields.items():
                Z_jk = lhs_emb @ rhs_emb.T * logit_scale.exp()
                if logit_bias is not None:
                    Z_jk = Z_jk + logit_bias
                loss_cross = loss_cross + wce_loss_fn(Z_jk, weights)
                n_cross += 1

        if n_cross > 0:
            loss_cross = loss_cross / n_cross

        # Anchor regularization: constrain the primary text field
        if anchor_model is not None and lambda_anchor > 0:
            anchor_texts = batch["group_titles"] if title_is_lhs else batch["queries"]
            primary_embs = group_title_embs if title_is_lhs else query_embs
            with torch.no_grad():
                anchor_text_embs = F.normalize(
                    anchor_model.encode_text(tokenizer(anchor_texts).to(device)),
                    dim=-1,
                )
            loss_anchor = F.mse_loss(primary_embs, anchor_text_embs.detach())
        else:
            loss_anchor = torch.tensor(0.0, device=device)

        loss = loss_avg + loss_cross + lambda_anchor * loss_anchor

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], max_norm=1.0
        )
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        total_avg_loss += loss_avg.item()
        total_cross_loss += loss_cross.item()
        total_anchor_loss += loss_anchor.item()
        n_batches += 1

        if (batch_idx + 1) % 20 == 0:
            log.info(
                "  [%d/%d] loss=%.4f avg=%.4f cross=%.4f anchor=%.4f lr=%.2e",
                batch_idx + 1, len(dataloader),
                loss.item(), loss_avg.item(), loss_cross.item(), loss_anchor.item(),
                optimizer.param_groups[0]["lr"],
            )

        if save_dir and save_every > 0 and (batch_idx + 1) % save_every == 0:
            ckpt_path_step = save_dir / f"step_{batch_idx + 1}.pt"
            trainable_state = {
                k: v for k, v in model.state_dict().items()
                if any(p in k for p in save_patterns)
            }
            torch.save({
                "step": batch_idx + 1,
                "model_state_dict": trainable_state,
                "loss": loss.item(),
            }, ckpt_path_step)
            log.info("Checkpoint saved: %s (loss=%.4f)", ckpt_path_step, loss.item())

        del images_tensor, image_embs, lhs_avg, rhs_avg, Z_avg
        del query_embs, title_embs, group_title_embs, l1_embs, sub3_embs, color_embs, mat_embs
        for k in list(lhs_fields.keys()):
            del lhs_fields[k]
        for k in list(rhs_fields.keys()):
            del rhs_fields[k]
        if device.type == "mps":
            torch.mps.empty_cache()

    return {
        "loss": total_loss / max(n_batches, 1),
        "avg_loss": total_avg_loss / max(n_batches, 1),
        "cross_loss": total_cross_loss / max(n_batches, 1),
        "anchor_loss": total_anchor_loss / max(n_batches, 1),
        "n_batches": n_batches,
    }


# ── Quick eval (sanity check) ───────────────────────────────────────────────

@torch.no_grad()
def quick_eval(
    model, tokenizer, preprocess, device: torch.device, data_dir: Path, n_queries: int = 100
) -> dict[str, float]:
    model.eval()

    jsonl_path = data_dir / "pairs.jsonl"
    groups: dict[str, list[dict]] = defaultdict(list)
    with open(jsonl_path) as f:
        for i, line in enumerate(f):
            if i >= 30000:
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

        imgs = []
        relevances = []
        for prod in products[:50]:
            img_path = data_dir / prod["image_path"]
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
        threshold = 0.3
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
    p = argparse.ArgumentParser(description="Phase 10 Multi-field GCL Training")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--max-pairs", type=int, default=None)
    p.add_argument("--queries-per-batch", type=int, default=8)
    p.add_argument("--products-per-query", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-6)
    p.add_argument("--vision-lr", type=float, default=None,
                   help="LR for vision params (text-image-light only, default: lr/10)")
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--lambda-anchor", type=float, default=0.5)
    p.add_argument("--warmup-pct", type=float, default=0.05)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-every", type=int, default=500)
    p.add_argument("--resume-checkpoint", type=str, default=None,
                   help="Phase 4b checkpoint path (default: auto-detect)")
    p.add_argument("--model-source", type=str, default="phase4b",
                   choices=["phase4b", "fsl"],
                   help="Model initialization: 'phase4b' resumes from Phase 4b; "
                        "'fsl' fine-tunes Marqo-FashionSigLIP directly")
    p.add_argument("--text-mode", type=str, default="query-primary",
                   choices=["query-primary", "title-primary"],
                   help="Which text field is the primary LHS: query (original) or title (for atlas/KAGL)")
    p.add_argument("--training-scope", type=str, default="text-only",
                   choices=["text-only", "text-image-light"],
                   help="Which params to train: text-only or text+vision (differential LR)")
    p.add_argument("--data-dir", type=str, default=None)
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

    data_dir = Path(args.data_dir) if args.data_dir else DATA_DIR
    log.info("Data dir: %s", data_dir)

    # Select field weights based on text-mode
    if args.text_mode == "title-primary":
        lhs_field_weights = LHS_WEIGHTS_TITLE_PRIMARY
        rhs_field_weights = RHS_WEIGHTS_TITLE_PRIMARY
    else:
        lhs_field_weights = LHS_WEIGHTS_QUERY_PRIMARY
        rhs_field_weights = RHS_WEIGHTS_QUERY_PRIMARY

    # Load model
    import open_clip

    ckpt_path = None
    if args.model_source == "fsl":
        log.info("Loading Marqo-FashionSigLIP (hf-hub)...")
        model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
            "hf-hub:Marqo/marqo-fashionSigLIP"
        )
        tokenizer = open_clip.get_tokenizer("hf-hub:Marqo/marqo-fashionSigLIP")
        model = model.to(device)
        log.info("FashionSigLIP loaded — fine-tuning from FSL weights")
    else:
        log.info("Loading ViT-B-16-SigLIP/webli...")
        model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
            "ViT-B-16-SigLIP", pretrained="webli"
        )
        tokenizer = open_clip.get_tokenizer("ViT-B-16-SigLIP")
        model = model.to(device)

        ckpt_path = Path(args.resume_checkpoint) if args.resume_checkpoint else PHASE4B_CHECKPOINT
        if ckpt_path.exists():
            load_phase4b_checkpoint(model, ckpt_path, device)
        else:
            log.warning("Phase 4b checkpoint not found at %s — training from scratch!", ckpt_path)

    # Anchor model (frozen copy of initial state)
    log.info("Creating anchor model from current state...")
    anchor_model = copy.deepcopy(model)
    anchor_model.eval()
    for p in anchor_model.parameters():
        p.requires_grad = False

    # Apply training scope
    trainable_params, text_param_names, vision_param_names = apply_training_scope(
        model, args.training_scope
    )
    save_patterns = TEXT_ONLY_PATTERNS + (VISION_LIGHT_PATTERNS if args.training_scope == "text-image-light" else [])

    # Dataset
    enriched_path = data_dir / "enriched_descriptions.jsonl"
    synthetic_path = REPO_ROOT / "data" / "synthetic" / "phase10c_atlas_kagl" / "atlas_kagl_synthetic.jsonl"
    # Round 1 generic synthetic
    synth_round1 = REPO_ROOT / "data" / "synthetic" / "phase10_gap_fill" / "synthetic_all.jsonl"
    # Phase 10e: large-scale atlas/KAGL synthetic (5600 pairs, local images)
    synth_atlas_kagl_large = data_dir / "synthetic_atlas_kagl_large.jsonl"
    # Phase 10f: GS-10M filtered atlas/KAGL items (title+query pairs)
    gs10m_atlas_kagl = data_dir / "gs10m_atlas_kagl.jsonl"
    if not gs10m_atlas_kagl.exists():
        gs10m_atlas_kagl = REPO_ROOT / "data" / "processed" / "v3_phase10f_combined" / "gs10m_atlas_kagl.jsonl"

    # Merge all synthetic files into a temp combined file
    combined_synth = REPO_ROOT / "data" / "synthetic" / "combined_synthetic.jsonl"
    with open(combined_synth, "w") as out:
        for sp in [synthetic_path, synth_round1, synth_atlas_kagl_large, gs10m_atlas_kagl]:
            if sp.exists():
                with open(sp) as fin:
                    for line in fin:
                        out.write(line)
    log.info("Combined synthetic: %s", combined_synth)

    dataset = MultiFieldDataset(
        data_dir,
        max_pairs=args.max_pairs,
        products_per_query=args.products_per_query,
        min_products=4,
        seed=args.seed,
        enriched_path=enriched_path if enriched_path.exists() else None,
        synthetic_path=combined_synth if combined_synth.exists() else None,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.queries_per_batch,
        shuffle=True,
        collate_fn=collate_multifield,
        num_workers=0,
        drop_last=True,
    )

    # Optimizer — differential LR by layer depth
    if args.training_scope == "text-image-light" and vision_param_names:
        vision_lr = args.vision_lr or args.lr / 10.0
        text_params_list = [p for n, p in model.named_parameters() if n in text_param_names]
        vision_params_list = [p for n, p in model.named_parameters() if n in vision_param_names]
        param_groups = [
            {"params": text_params_list, "lr": args.lr},
            {"params": vision_params_list, "lr": vision_lr},
        ]
        optimizer = torch.optim.AdamW(
            param_groups, weight_decay=args.weight_decay, betas=(0.9, 0.98), eps=1e-6,
        )
        log.info("Differential LR: text=%.2e, vision=%.2e", args.lr, vision_lr)
    elif args.training_scope == "text-only":
        # Differential LR: top text blocks full LR, mid blocks half, early frozen
        top_params, mid_params, other_params = [], [], []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            # Text transformer blocks 8-11 → full LR
            if any(f"resblocks.{b}" in name for b in [8, 9, 10, 11]):
                top_params.append(param)
            # Text transformer blocks 4-7 → half LR
            elif any(f"resblocks.{b}" in name for b in [4, 5, 6, 7]):
                mid_params.append(param)
            else:
                other_params.append(param)

        param_groups = []
        if top_params:
            param_groups.append({"params": top_params, "lr": args.lr})
        if mid_params:
            param_groups.append({"params": mid_params, "lr": args.lr * 0.5})
        if other_params:
            param_groups.append({"params": other_params, "lr": args.lr})

        optimizer = torch.optim.AdamW(
            param_groups, weight_decay=args.weight_decay, betas=(0.9, 0.98), eps=1e-6,
        )
        log.info("Differential LR (text-only): top(8-11)=%.2e, mid(4-7)=%.2e, other=%.2e",
                 args.lr, args.lr * 0.5, args.lr)
        log.info("  top_params=%d, mid_params=%d, other_params=%d",
                 len(top_params), len(mid_params), len(other_params))
    else:
        trainable_param_list = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            trainable_param_list, lr=args.lr,
            weight_decay=args.weight_decay, betas=(0.9, 0.98), eps=1e-6,
        )

    # Scheduler
    total_steps = len(dataloader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_pct)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    wce_loss_fn = WeightedCELoss()

    # Run setup
    source_tag = "fsl" if args.model_source == "fsl" else "p4b"
    run_name = args.run_name or f"phase10_{source_tag}_{args.text_mode}_{int(time.time())}"
    run_dir = CHECKPOINT_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    config = vars(args)
    config["trainable_params"] = trainable_params
    config["total_queries"] = len(dataset)
    config["total_steps"] = total_steps
    config["device"] = str(device)
    config["data_dir"] = str(data_dir)
    config["lhs_field_weights"] = lhs_field_weights
    config["rhs_field_weights"] = rhs_field_weights
    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    log.info("=" * 60)
    log.info("Run: %s", run_name)
    log.info("Text mode: %s", args.text_mode)
    log.info("Training scope: %s", args.training_scope)
    log.info("Epochs: %d, Steps: %d, Batch: %d queries x %d products = %d",
             args.epochs, total_steps, args.queries_per_batch,
             args.products_per_query, args.queries_per_batch * args.products_per_query)
    log.info("LR: %.2e, Anchor lambda: %.2f", args.lr, args.lambda_anchor)
    log.info("LHS fields: %s", lhs_field_weights)
    log.info("RHS fields: %s", rhs_field_weights)
    log.info("=" * 60)

    history = []
    best_map10 = 0.0

    for epoch in range(1, args.epochs + 1):
        log.info("Epoch %d/%d", epoch, args.epochs)

        epoch_stats = train_one_epoch(
            model=model,
            tokenizer=tokenizer,
            preprocess=preprocess_train,
            dataloader=dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            wce_loss_fn=wce_loss_fn,
            anchor_model=anchor_model,
            device=device,
            epoch=epoch,
            lambda_anchor=args.lambda_anchor,
            lhs_field_weights=lhs_field_weights,
            rhs_field_weights=rhs_field_weights,
            save_patterns=save_patterns,
            save_dir=run_dir,
            save_every=args.save_every,
        )

        eval_stats = quick_eval(model, tokenizer, preprocess_val, device, data_dir, n_queries=80)
        epoch_stats.update(eval_stats)
        epoch_stats["epoch"] = epoch
        history.append(epoch_stats)

        log.info(
            "Epoch %d: loss=%.4f avg=%.4f cross=%.4f anchor=%.4f MAP@10=%.4f",
            epoch, epoch_stats["loss"], epoch_stats["avg_loss"],
            epoch_stats["cross_loss"], epoch_stats["anchor_loss"],
            epoch_stats["map10"],
        )

        ckpt_path_ep = run_dir / f"epoch_{epoch}.pt"
        trainable_state = {
            k: v for k, v in model.state_dict().items()
            if any(p in k for p in save_patterns)
        }
        torch.save({
            "epoch": epoch,
            "model_state_dict": trainable_state,
            "optimizer_state_dict": optimizer.state_dict(),
            "stats": epoch_stats,
        }, ckpt_path_ep)
        log.info("Saved: %s", ckpt_path_ep)

        if epoch_stats["map10"] > best_map10:
            best_map10 = epoch_stats["map10"]
            best_path = run_dir / "best.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": trainable_state,
                "stats": epoch_stats,
                "scope": args.training_scope,
                "text_mode": args.text_mode,
            }, best_path)
            log.info("New best MAP@10=%.4f -> %s", best_map10, best_path)

        gc.collect()
        if device.type == "mps":
            torch.mps.empty_cache()

    final_path = run_dir / "final.pt"
    trainable_state = {
        k: v for k, v in model.state_dict().items()
        if any(p in k for p in save_patterns)
    }
    torch.save({
        "model_state_dict": trainable_state,
        "scope": args.training_scope,
        "text_mode": args.text_mode,
        "config": config,
        "history": history,
    }, final_path)

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
    print(f"  Time: {elapsed / 60:.1f} min")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
