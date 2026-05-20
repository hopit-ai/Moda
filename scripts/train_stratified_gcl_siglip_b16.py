"""Train 203M ViT-B-16-SigLIP on the stratified GS + DeepFashion set.

Important constraints:
  - Final model is 203M ViT-B-16-SigLIP only.
  - No fashion200k rows are used for training.
  - Dataset is loaded from disk per batch to avoid keeping 90K images as tensors.
  - Same-query examples inside a batch are treated as positives, not false negatives.

The loss is a SigLIP-style sigmoid loss over the batch similarity matrix:
  - same query => positive label, weighted by each document's rank weight
  - different query => negative label
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset


REPO_ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = REPO_ROOT / "cache"
MODEL_DIR = REPO_ROOT / "models"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

FSL_BASELINE_1K = 0.4902
FSL_BASELINE_5K = 0.3241


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--pairs",
        type=Path,
        default=REPO_ROOT / "data/processed/fashion_stratified_gs_df/pairs.jsonl",
    )
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-6)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-steps", type=int, default=200)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--max-train", type=int, default=0, help="0 means all pairs")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-dir", default="siglip-b16-stratified-gcl")
    p.add_argument(
        "--train-scope",
        choices=["full", "last4"],
        default="last4",
        help="Conservative last4 is safer for the first MPS smoke run.",
    )
    return p.parse_args()


class PairDataset(Dataset):
    def __init__(self, pairs_path: Path, preprocess: Any, max_train: int = 0) -> None:
        rows: list[dict[str, Any]] = []
        with pairs_path.open() as f:
            for line in f:
                row = json.loads(line)
                rows.append(row)
                if max_train and len(rows) >= max_train:
                    break
        self.rows = rows
        self.preprocess = preprocess

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.rows[idx]
        image = Image.open(row["image_path"]).convert("RGB")
        return {
            "image": self.preprocess(image),
            "query": row["query"],
            "weight": float(row.get("weight", 1.0)),
            "source": row.get("source", ""),
            "bucket": row.get("rank_bucket", ""),
        }


def collate_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "images": torch.stack([b["image"] for b in batch]),
        "queries": [b["query"] for b in batch],
        "weights": torch.tensor([b["weight"] for b in batch], dtype=torch.float32),
        "sources": [b["source"] for b in batch],
        "buckets": [b["bucket"] for b in batch],
    }


def same_query_siglip_loss(
    image_features: torch.Tensor,
    text_features: torch.Tensor,
    queries: list[str],
    weights: torch.Tensor,
    logit_scale: torch.Tensor,
    logit_bias: torch.Tensor,
) -> torch.Tensor:
    logits = logit_scale * text_features @ image_features.T + logit_bias
    batch_size = logits.shape[0]

    positive = torch.zeros((batch_size, batch_size), dtype=torch.bool, device=logits.device)
    for i, query_i in enumerate(queries):
        for j, query_j in enumerate(queries):
            if query_i == query_j:
                positive[i, j] = True

    labels = torch.where(
        positive,
        torch.ones_like(logits),
        -torch.ones_like(logits),
    )
    loss_matrix = -F.logsigmoid(labels * logits)

    # Weight same-query positives by the document rank score. Negatives stay at 1.
    doc_weights = weights.to(logits.device).unsqueeze(0).expand_as(logits)
    loss_weights = torch.where(positive, doc_weights, torch.ones_like(logits))
    return (loss_matrix * loss_weights).mean()


def compute_map10(scores: torch.Tensor, query_cats: list[str], cat_to_indices: dict[str, list[int]]) -> float:
    total = 0.0
    for qi, cat in enumerate(query_cats):
        positives = set(cat_to_indices[cat])
        topk = scores[qi].topk(min(10, scores.shape[1])).indices.tolist()
        ap = 0.0
        n_rel = 0
        for rank, idx in enumerate(topk, 1):
            if idx in positives:
                n_rel += 1
                ap += n_rel / rank
        n_pos = min(len(positives), 10)
        if n_pos > 0:
            ap /= n_pos
        total += ap
    return total / max(len(query_cats), 1)


def evaluate_fashion200k(model: Any, preprocess: Any, tokenizer: Any, corpus_size: int, start_offset: int = 0) -> float:
    from datasets import load_dataset

    ds = load_dataset("Marqo/fashion200k", split="data", streaming=True)
    items = []
    for row in ds:
        cat = (row.get("category3") or "").strip()
        if row.get("image") and cat:
            items.append({"category": cat, "image": row["image"]})
        if len(items) >= start_offset + corpus_size:
            break

    items = items[start_offset : start_offset + corpus_size]
    rng = random.Random(42 if start_offset == 0 else 123)
    rng.shuffle(items)

    cat_to_indices: dict[str, list[int]] = defaultdict(list)
    for idx, item in enumerate(items):
        cat_to_indices[item["category"]].append(idx)
    valid_cats = [cat for cat, idxs in cat_to_indices.items() if len(idxs) >= 2]
    rng2 = random.Random(42 if start_offset == 0 else 99)
    rng2.shuffle(valid_cats)
    query_cats = valid_cats[: min(300, len(valid_cats))]

    model.eval()
    image_features = []
    with torch.no_grad():
        for i in range(0, len(items), 32):
            tensors = torch.stack([preprocess(item["image"].convert("RGB")) for item in items[i : i + 32]]).to(DEVICE)
            feats = F.normalize(model.encode_image(tensors), dim=-1)
            image_features.append(feats.cpu())
            del tensors, feats
    image_features_t = torch.cat(image_features, dim=0)

    text_features = []
    with torch.no_grad():
        for i in range(0, len(query_cats), 64):
            tokens = tokenizer(query_cats[i : i + 64]).to(DEVICE)
            feats = F.normalize(model.encode_text(tokens), dim=-1)
            text_features.append(feats.cpu())
            del tokens, feats
    text_features_t = torch.cat(text_features, dim=0)

    scores = text_features_t @ image_features_t.T
    return compute_map10(scores, query_cats, cat_to_indices)


def cosine_schedule(optimizer: torch.optim.Optimizer, warmup_steps: int, total_steps: int) -> torch.optim.lr_scheduler.LambdaLR:
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def set_train_scope(model: Any, scope: str) -> int:
    for param in model.parameters():
        param.requires_grad = scope == "full"

    if scope == "last4":
        keys = [
            "trunk.blocks.8",
            "trunk.blocks.9",
            "trunk.blocks.10",
            "trunk.blocks.11",
            "text.transformer.resblocks.8",
            "text.transformer.resblocks.9",
            "text.transformer.resblocks.10",
            "text.transformer.resblocks.11",
            "ln",
            "norm",
            "proj",
            "logit",
        ]
        for name, param in model.named_parameters():
            if any(key in name for key in keys):
                param.requires_grad = True

    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    import open_clip

    print(f"Device: {DEVICE}")
    print(f"Training data: {args.pairs}")
    print(f"Model: ViT-B-16-SigLIP/webli (203M)")
    print(f"Train scope: {args.train_scope}")

    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-16-SigLIP", pretrained="webli")
    tokenizer = open_clip.get_tokenizer("ViT-B-16-SigLIP")
    model.to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = set_train_scope(model, args.train_scope)
    print(f"Params: {n_params / 1e6:.1f}M | trainable: {n_trainable / 1e6:.1f}M")

    dataset = PairDataset(args.pairs, preprocess, max_train=args.max_train)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_batch,
        drop_last=True,
    )
    print(f"Pairs: {len(dataset)} | batches/epoch: {len(loader)}")

    print("\nInitial eval:")
    init_1k = evaluate_fashion200k(model, preprocess, tokenizer, corpus_size=1000, start_offset=0)
    print(f"  fashion200k 1K MAP@10 = {init_1k:.4f} (FSL {FSL_BASELINE_1K:.4f})")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    total_steps = len(loader) * args.epochs
    scheduler = cosine_schedule(optimizer, args.warmup_steps, total_steps)

    logit_scale = model.logit_scale if hasattr(model, "logit_scale") else torch.tensor(4.765, device=DEVICE)
    logit_bias = model.logit_bias if hasattr(model, "logit_bias") else torch.tensor(-10.0, device=DEVICE)

    best_map = init_1k
    best_state = None
    results_log = []
    step = 0

    print("\nTraining:")
    for epoch in range(args.epochs):
        model.train()
        rolling_loss = 0.0
        rolling_steps = 0
        for batch in loader:
            images = batch["images"].to(DEVICE)
            tokens = tokenizer(batch["queries"]).to(DEVICE)
            weights = batch["weights"].to(DEVICE)

            image_features = F.normalize(model.encode_image(images), dim=-1)
            text_features = F.normalize(model.encode_text(tokens), dim=-1)
            scale = logit_scale.exp() if hasattr(logit_scale, "exp") else torch.tensor(10.0, device=DEVICE)
            loss = same_query_siglip_loss(
                image_features,
                text_features,
                batch["queries"],
                weights,
                scale,
                logit_bias,
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step()
            scheduler.step()

            step += 1
            rolling_loss += loss.item()
            rolling_steps += 1

            if step % 50 == 0:
                print(
                    f"  epoch={epoch + 1} step={step} "
                    f"loss={rolling_loss / rolling_steps:.4f} lr={scheduler.get_last_lr()[0]:.2e}",
                    flush=True,
                )

            if step % args.eval_every == 0:
                map_1k = evaluate_fashion200k(model, preprocess, tokenizer, corpus_size=1000, start_offset=0)
                print(
                    f"\n  >>> eval step={step}: 1K MAP@10={map_1k:.4f} "
                    f"vs init={map_1k - init_1k:+.4f} vs FSL={map_1k - FSL_BASELINE_1K:+.4f}\n",
                    flush=True,
                )
                results_log.append({"step": step, "map10_1k": map_1k, "loss": rolling_loss / rolling_steps})
                if map_1k > best_map:
                    best_map = map_1k
                    best_state = copy.deepcopy(model.state_dict())
                    print("  >>> new best checkpoint in memory", flush=True)
                if map_1k < init_1k * 0.75:
                    print("  >>> early stop: eval dropped too much", flush=True)
                    break
                model.train()

        if results_log and results_log[-1]["map10_1k"] < init_1k * 0.75:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    print("\nFinal eval:")
    final_1k = evaluate_fashion200k(model, preprocess, tokenizer, corpus_size=1000, start_offset=0)
    final_5k = evaluate_fashion200k(model, preprocess, tokenizer, corpus_size=5000, start_offset=1000)
    print(f"  1K MAP@10 = {final_1k:.4f} (FSL {FSL_BASELINE_1K:.4f})")
    print(f"  5K MAP@10 = {final_5k:.4f} (FSL {FSL_BASELINE_5K:.4f})")

    save_dir = MODEL_DIR / args.save_dir
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_dir / "model.pt")

    result = {
        "pairs": str(args.pairs),
        "n_pairs": len(dataset),
        "params_M": n_params / 1e6,
        "trainable_params_M": n_trainable / 1e6,
        "train_scope": args.train_scope,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "init_1k": init_1k,
        "best_1k": best_map,
        "final_1k": final_1k,
        "final_5k": final_5k,
        "fsl_1k": FSL_BASELINE_1K,
        "fsl_5k": FSL_BASELINE_5K,
        "log": results_log,
        "model_path": str(save_dir / "model.pt"),
    }
    out_path = CACHE_DIR / "stratified_gcl_training_results.json"
    with out_path.open("w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved model: {save_dir / 'model.pt'}")
    print(f"Saved results: {out_path}")


if __name__ == "__main__":
    main()
