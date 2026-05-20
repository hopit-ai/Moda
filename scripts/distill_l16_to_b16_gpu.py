"""
Distill a large ViT teacher into a smaller ViT student on GPU.

Self-contained script for running on a cloud GPU.
Downloads models and data from HuggingFace, no local files required.

Training data: Marqo-GS-10M only (no benchmark dataset leakage).
Eval: category-based retrieval on held-out benchmark datasets.

Usage:
    pip install open_clip_torch datasets torch torchvision transformers

    python distill_l16_to_b16_gpu.py --output-dir ./distill_output
    python distill_l16_to_b16_gpu.py --batch-size 128 --max-steps 3000
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import os
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("distill-gpu")


# =============================================================================
# Data loading (downloads from HuggingFace)
# =============================================================================

def load_training_data(n_samples: int = 10000, seed: int = 42) -> list[dict]:
    """Load training pairs ONLY from Marqo-GS-10M (search relevance dataset).

    This dataset is completely separate from the evaluation benchmarks
    (fashion200k, polyvore, atlas, KAGL), so there is zero data leakage.

    GS-10M has query-product pairs with images and product titles.
    We use (title, image) pairs for distillation training.

    Returns list of {"text": str, "image": PIL.Image}.
    """
    from datasets import load_dataset

    log.info("Loading training data from Marqo-GS-10M ONLY (no benchmark data)...")
    pairs = []

    gs_splits = ["in_domain", "novel_document", "novel_query", "zero_shot"]
    per_split = n_samples // len(gs_splits) + 1

    for split in gs_splits:
        log.info("  Loading Marqo/marqo-GS-10M split=%s (target=%d)...", split, per_split)
        try:
            ds = load_dataset("Marqo/marqo-GS-10M", split=split, streaming=True)
            count = 0
            for row in ds:
                if count >= per_split:
                    break
                title = row.get("title", "") or ""
                if row.get("image") and title.strip():
                    pairs.append({"text": title, "image": row["image"]})
                    count += 1
            log.info("    Got %d pairs from GS-10M/%s", count, split)
        except Exception as e:
            log.warning("  Failed to load GS-10M/%s: %s", split, e)

    random.seed(seed)
    random.shuffle(pairs)
    pairs = pairs[:n_samples]
    log.info("Total training pairs: %d (all from GS-10M, zero benchmark overlap)", len(pairs))
    return pairs


def preprocess_all_images(pairs, preprocess_fn, device="cpu"):
    """Pre-convert all PIL images to tensors once to avoid repeated CPU work."""
    import torch
    log.info("Pre-processing %d images to tensors...", len(pairs))
    preprocessed = []
    for i, item in enumerate(pairs):
        try:
            img_tensor = preprocess_fn(item["image"].convert("RGB"))
            preprocessed.append({"text": item["text"], "image_tensor": img_tensor})
        except Exception:
            continue
        if (i + 1) % 1000 == 0:
            log.info("  Pre-processed %d/%d images", i + 1, len(pairs))
    log.info("Pre-processing complete: %d valid pairs", len(preprocessed))
    return preprocessed


# =============================================================================
# Projection head (maps teacher 1024-d → student 768-d)
# =============================================================================

class ProjectionHead(nn.Module):
    def __init__(self, teacher_dim: int = 1024, student_dim: int = 768):
        super().__init__()
        self.proj = nn.Linear(teacher_dim, student_dim, bias=False)

    def forward(self, x):
        return self.proj(x)


# =============================================================================
# Loss functions
# =============================================================================

def relational_kl_loss(
    teacher_img: torch.Tensor, teacher_txt: torch.Tensor,
    student_img: torch.Tensor, student_txt: torch.Tensor,
    tau_t: float = 0.05, tau_s: float = 0.05,
) -> torch.Tensor:
    """KL divergence on score matrices.

    Preserves the relative ranking structure from teacher to student.
    Both score matrices are [B, B] (in-batch cross-modal similarities).
    """
    with torch.no_grad():
        scores_t = teacher_txt @ teacher_img.T  # [B, B]
    scores_s = student_txt @ student_img.T  # [B, B]

    # Row-wise KL (text → image direction)
    log_p_t = F.log_softmax(scores_t / tau_t, dim=-1)
    log_p_s = F.log_softmax(scores_s / tau_s, dim=-1)
    p_t = log_p_t.exp()
    kl_t2i = (p_t * (log_p_t - log_p_s)).sum(dim=-1).mean()

    # Column-wise KL (image → text direction)
    log_p_t_col = F.log_softmax(scores_t.T / tau_t, dim=-1)
    log_p_s_col = F.log_softmax(scores_s.T / tau_s, dim=-1)
    p_t_col = log_p_t_col.exp()
    kl_i2t = (p_t_col * (log_p_t_col - log_p_s_col)).sum(dim=-1).mean()

    return 0.5 * (kl_t2i + kl_i2t)


def feature_mse_loss(
    teacher_emb: torch.Tensor,
    student_emb: torch.Tensor,
    projection: ProjectionHead,
) -> torch.Tensor:
    """MSE between projected teacher embeddings and student embeddings."""
    with torch.no_grad():
        projected = projection(teacher_emb)
        projected = projected / projected.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return F.mse_loss(student_emb, projected)


def symmetric_infonce(
    img_emb: torch.Tensor, txt_emb: torch.Tensor, tau: float = 0.05,
) -> torch.Tensor:
    """Standard symmetric in-batch InfoNCE (CLIP loss)."""
    scores = txt_emb @ img_emb.T / tau  # [B, B]
    labels = torch.arange(scores.shape[0], device=scores.device)
    loss_t2i = F.cross_entropy(scores, labels)
    loss_i2t = F.cross_entropy(scores.T, labels)
    return 0.5 * (loss_t2i + loss_i2t)


# =============================================================================
# Evaluation (5K screener on all 4 datasets)
# =============================================================================

def evaluate_5k_screener(
    model, preprocess, tokenizer, device: str = "cuda",
    datasets: list[str] = None, corpus_size: int = 5000,
) -> dict:
    """Proper retrieval eval on benchmark datasets. No data leakage.

    For each dataset, builds a corpus of images and uses category-based
    retrieval: the query is a category label (e.g. "mini and short dresses"),
    and ground truth positives are all items sharing that same category.
    This is a legitimate zero-shot retrieval task with no training overlap.

    Training uses ONLY Marqo-GS-10M, so eval on fashion200k/polyvore/KAGL
    has zero overlap with training data.

    Returns {dataset: MAP@10, ...} + mean_MAP@10.
    """
    from datasets import load_dataset as ld
    import gc
    from collections import defaultdict

    if datasets is None:
        datasets = ["fashion200k", "polyvore", "KAGL"]

    torch.cuda.empty_cache()
    gc.collect()

    CATEGORY_COLS = {
        "fashion200k": "category3",
        "polyvore": "category",
        "KAGL": "category3",
        "atlas": "category3",
    }

    results = {}
    for ds_name in datasets:
        log.info("  [eval] %s ...", ds_name)
        cat_col = CATEGORY_COLS.get(ds_name, "category")

        try:
            ds_iter = ld(f"Marqo/{ds_name}", split="data", streaming=True)
        except Exception as e:
            log.warning("  [eval] Failed to load %s: %s", ds_name, e)
            continue

        corpus = []
        for row in ds_iter:
            if len(corpus) >= corpus_size:
                break
            cat = row.get(cat_col, "") or ""
            if row.get("image") and cat.strip():
                corpus.append({
                    "image": row["image"],
                    "category": cat.strip(),
                    "item_id": str(row.get("item_ID", len(corpus))),
                })

        if len(corpus) < 100:
            log.warning("  [eval] %s: only %d items, skip", ds_name, len(corpus))
            continue

        rng = random.Random(42)
        rng.shuffle(corpus)
        corpus = corpus[:corpus_size]

        item_ids = [c["item_id"] for c in corpus]
        cat_to_indices = defaultdict(list)
        for idx, c in enumerate(corpus):
            cat_to_indices[c["category"]].append(idx)

        # Select categories with 2+ items as queries (need at least 1 positive
        # that isn't trivially the only item)
        valid_cats = [cat for cat, idxs in cat_to_indices.items() if len(idxs) >= 2]
        rng.shuffle(valid_cats)
        query_cats = valid_cats[:min(300, len(valid_cats))]

        if len(query_cats) < 10:
            log.warning("  [eval] %s: only %d valid categories, skip", ds_name, len(query_cats))
            continue

        # Encode corpus images
        img_feats = []
        model.eval()
        with torch.no_grad():
            for i in range(0, len(corpus), 32):
                batch = corpus[i:i+32]
                imgs = torch.stack([preprocess(it["image"].convert("RGB")) for it in batch]).to(device)
                feat = model.encode_image(imgs)
                feat = feat / feat.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                img_feats.append(feat.cpu())
                del imgs, feat
                torch.cuda.empty_cache()
        img_feats = torch.cat(img_feats, dim=0)

        # Encode category queries
        txt_feats = []
        with torch.no_grad():
            for i in range(0, len(query_cats), 64):
                tokens = tokenizer(query_cats[i:i+64]).to(device)
                feat = model.encode_text(tokens)
                feat = feat / feat.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                txt_feats.append(feat.cpu())
                del tokens, feat
        txt_feats = torch.cat(txt_feats, dim=0)

        # Score and compute MAP@10
        scores = txt_feats @ img_feats.T
        k = 10
        map_at_k = 0.0
        for qi, cat in enumerate(query_cats):
            positive_indices = set(cat_to_indices[cat])
            topk_indices = scores[qi].topk(min(k, scores.shape[1])).indices.tolist()
            ap, n_rel = 0.0, 0
            for rank, idx in enumerate(topk_indices, 1):
                if idx in positive_indices:
                    n_rel += 1
                    ap += n_rel / rank
            n_pos = min(len(positive_indices), k)
            if n_pos > 0:
                ap /= n_pos
            map_at_k += ap
        map_at_k /= max(len(query_cats), 1)
        results[ds_name] = map_at_k
        log.info("    %s MAP@10 = %.4f (%d cat queries, %d corpus)", ds_name, map_at_k, len(query_cats), len(corpus))

        del corpus, img_feats, txt_feats, scores
        gc.collect()
        torch.cuda.empty_cache()

    if results:
        results["mean_MAP@10"] = sum(results.values()) / len(results)
        log.info("  [eval] mean MAP@10 = %.4f", results["mean_MAP@10"])
    return results


# =============================================================================
# Main training loop
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=str, default="./distill_output")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Batch size. 256 for A100 40GB, 128 for 24GB GPUs.")
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--n-train-samples", type=int, default=10000,
                        help="Number of training image-text pairs to load.")
    parser.add_argument("--eval-steps", type=str, default="500,1000,2000,3000,4000,5000",
                        help="Comma-separated steps to run evaluation.")
    parser.add_argument("--eval-datasets", type=str, default="fashion200k,polyvore",
                        help="Datasets for quick eval (full eval at end).")
    parser.add_argument("--w-kl", type=float, default=1.0, help="Relational KL weight")
    parser.add_argument("--w-feat", type=float, default=0.5, help="Feature MSE weight")
    parser.add_argument("--w-infonce", type=float, default=0.3, help="InfoNCE weight")
    parser.add_argument("--tau-teacher", type=float, default=0.05)
    parser.add_argument("--tau-student", type=float, default=0.05)
    parser.add_argument("--tau-infonce", type=float, default=0.05)
    parser.add_argument("--unfreeze-image-step", type=int, default=1000,
                        help="Unfreeze student image tower at this step.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action="store_true", default=True,
                        help="Use mixed precision (default: True).")
    parser.add_argument("--no-fp16", action="store_true", default=False)
    parser.add_argument("--gcs-output", type=str, default="",
                        help="GCS path to upload results (optional).")
    args = parser.parse_args()

    if args.no_fp16:
        args.fp16 = False

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        log.warning("No CUDA device found! This script is designed for GPU training.")
        log.warning("Training will be extremely slow on CPU.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save args
    with open(output_dir / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # =========================================================================
    # Load models
    # =========================================================================
    import open_clip

    log.info("Loading teacher model...")
    teacher, _, teacher_preprocess = open_clip.create_model_and_transforms(
        "ViT-L-16-SigLIP2-384", pretrained="webli"
    )
    teacher.eval().to(device)
    for p in teacher.parameters():
        p.requires_grad = False
    log.info("  Teacher loaded. Params: %d", sum(p.numel() for p in teacher.parameters()))

    log.info("Loading student model...")
    student, _, student_preprocess = open_clip.create_model_and_transforms(
        "ViT-B-16-SigLIP2-384", pretrained="webli"
    )
    student.train().to(device)
    student_tokenizer = open_clip.get_tokenizer("ViT-B-16-SigLIP2-384")
    teacher_tokenizer = open_clip.get_tokenizer("ViT-L-16-SigLIP2-384")
    log.info("  Student loaded. Params: %d", sum(p.numel() for p in student.parameters()))

    # Freeze student image tower initially
    n_frozen = 0
    for name, p in student.named_parameters():
        if name.startswith("visual."):
            p.requires_grad = False
            n_frozen += p.numel()
    log.info("  Image tower frozen: %d params", n_frozen)

    # Projection head (teacher 1024 → student 768)
    teacher_dim = 1024
    student_dim = 768
    projection = ProjectionHead(teacher_dim, student_dim).to(device)

    # =========================================================================
    # Load training data
    # =========================================================================
    raw_data = load_training_data(n_samples=args.n_train_samples, seed=args.seed)
    if len(raw_data) == 0:
        log.error("No training data available! Exiting.")
        return

    train_data = preprocess_all_images(raw_data, student_preprocess)
    del raw_data
    if len(train_data) == 0:
        log.error("No valid preprocessed data! Exiting.")
        return

    # =========================================================================
    # Optimizer and scheduler
    # =========================================================================
    trainable_params = [p for p in student.parameters() if p.requires_grad]
    trainable_params += list(projection.parameters())
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.01)

    def lr_schedule(step):
        if step < args.warmup_steps:
            return step / max(args.warmup_steps, 1)
        progress = (step - args.warmup_steps) / max(args.max_steps - args.warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)
    scaler = GradScaler(enabled=args.fp16)

    eval_steps = set(int(s) for s in args.eval_steps.split(",") if s.strip())
    eval_datasets = [d.strip() for d in args.eval_datasets.split(",")]

    # =========================================================================
    # Training loop
    # =========================================================================
    log.info("=" * 60)
    log.info("Starting distillation: %d steps, batch_size=%d, lr=%g",
             args.max_steps, args.batch_size, args.lr)
    log.info("Loss: KL=%.2f, feat_MSE=%.2f, InfoNCE=%.2f",
             args.w_kl, args.w_feat, args.w_infonce)
    log.info("=" * 60)

    training_log = []
    best_mean_map = 0.0
    best_step = 0
    image_tower_unfrozen = False
    data_idx = 0
    t_start = time.time()

    for step in range(1, args.max_steps + 1):
        student.train()

        # Sample batch (circular over pre-processed data)
        batch = []
        for _ in range(args.batch_size):
            batch.append(train_data[data_idx % len(train_data)])
            data_idx += 1

        try:
            images = torch.stack([item["image_tensor"] for item in batch]).to(device)
            texts = [item["text"] for item in batch]
            tokens_student = student_tokenizer(texts).to(device)
            tokens_teacher = teacher_tokenizer(texts).to(device)
        except Exception as e:
            log.warning("Step %d: batch prep failed (%s), skipping", step, e)
            continue

        # Forward pass
        optimizer.zero_grad()

        with autocast(enabled=args.fp16):
            # Teacher forward (no grad) — same images, same preprocessing
            with torch.no_grad():
                t_img = teacher.encode_image(images)
                t_txt = teacher.encode_text(tokens_teacher)
                t_img = t_img / t_img.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                t_txt = t_txt / t_txt.norm(dim=-1, keepdim=True).clamp_min(1e-12)

            # Student forward
            s_img = student.encode_image(images)
            s_txt = student.encode_text(tokens_student)
            s_img = s_img / s_img.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            s_txt = s_txt / s_txt.norm(dim=-1, keepdim=True).clamp_min(1e-12)

            # Losses
            loss_kl = relational_kl_loss(
                t_img, t_txt, s_img, s_txt,
                tau_t=args.tau_teacher, tau_s=args.tau_student,
            )
            loss_feat = feature_mse_loss(
                torch.cat([t_img, t_txt], dim=0),
                torch.cat([s_img, s_txt], dim=0),
                projection,
            )
            loss_infonce = symmetric_infonce(s_img, s_txt, tau=args.tau_infonce)

            loss = (args.w_kl * loss_kl
                    + args.w_feat * loss_feat
                    + args.w_infonce * loss_infonce)

        # Backward
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        # Unfreeze image tower
        if step == args.unfreeze_image_step and not image_tower_unfrozen:
            log.info("Step %d: unfreezing image tower, halving batch size %d -> %d",
                     step, args.batch_size, args.batch_size // 2)
            args.batch_size = args.batch_size // 2
            for name, p in student.named_parameters():
                if name.startswith("visual."):
                    p.requires_grad = True
            if hasattr(student, 'visual') and hasattr(student.visual, 'set_grad_checkpointing'):
                student.visual.set_grad_checkpointing(True)
                log.info("  Enabled gradient checkpointing on student visual tower")
            trainable_params = [p for p in student.parameters() if p.requires_grad]
            trainable_params += list(projection.parameters())
            optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.01)
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)
            image_tower_unfrozen = True

        # Logging
        if step % 50 == 0 or step == 1:
            elapsed = time.time() - t_start
            steps_per_sec = step / elapsed
            eta_min = (args.max_steps - step) / steps_per_sec / 60
            log.info(
                "Step %d/%d | loss=%.4f (KL=%.4f feat=%.4f NCE=%.4f) | "
                "lr=%.2e | %.2f step/s | ETA %.1f min",
                step, args.max_steps,
                loss.item(), loss_kl.item(), loss_feat.item(), loss_infonce.item(),
                scheduler.get_last_lr()[0],
                steps_per_sec, eta_min,
            )

        training_log.append({
            "step": step,
            "loss": loss.item(),
            "loss_kl": loss_kl.item(),
            "loss_feat": loss_feat.item(),
            "loss_infonce": loss_infonce.item(),
            "lr": scheduler.get_last_lr()[0],
        })

        # Evaluation
        if step in eval_steps:
            log.info("=" * 40)
            log.info("EVALUATION at step %d", step)
            log.info("=" * 40)
            teacher.cpu()
            projection.cpu()
            torch.cuda.empty_cache()
            student.eval()
            metrics = evaluate_5k_screener(
                student, student_preprocess, student_tokenizer,
                device=device, datasets=eval_datasets,
            )
            student.train()
            teacher.to(device)
            projection.to(device)

            training_log[-1]["eval"] = metrics

            # Save checkpoint
            ckpt_dir = output_dir / f"step_{step:05d}"
            ckpt_dir.mkdir(exist_ok=True)
            torch.save(student.state_dict(), ckpt_dir / "student_state_dict.pt")
            torch.save(projection.state_dict(), ckpt_dir / "projection.pt")
            with open(ckpt_dir / "metrics.json", "w") as f:
                json.dump(metrics, f, indent=2)

            mean_map = metrics.get("mean_MAP@10", 0)
            if mean_map > best_mean_map:
                best_mean_map = mean_map
                best_step = step
                best_dir = output_dir / "best"
                best_dir.mkdir(exist_ok=True)
                torch.save(student.state_dict(), best_dir / "student_state_dict.pt")
                torch.save(projection.state_dict(), best_dir / "projection.pt")
                with open(best_dir / "metrics.json", "w") as f:
                    json.dump(metrics, f, indent=2)
                log.info("  NEW BEST: mean MAP@10 = %.4f at step %d", best_mean_map, best_step)

        # Periodic checkpoint
        if step % 1000 == 0:
            with open(output_dir / "training_log.jsonl", "w") as f:
                for entry in training_log:
                    f.write(json.dumps(entry) + "\n")

        del images, tokens_student, tokens_teacher
        del t_img, t_txt, s_img, s_txt, loss, loss_kl, loss_feat, loss_infonce
        torch.cuda.empty_cache()

    # =========================================================================
    # Final evaluation on all 4 datasets
    # =========================================================================
    log.info("=" * 60)
    log.info("FINAL EVALUATION (all 4 datasets)")
    log.info("=" * 60)
    teacher.cpu()
    projection.cpu()
    torch.cuda.empty_cache()
    student.eval()
    final_metrics = evaluate_5k_screener(
        student, student_preprocess, student_tokenizer,
        device=device,
        datasets=["fashion200k", "atlas", "polyvore", "KAGL"],
    )

    # Save final
    with open(output_dir / "training_log.jsonl", "w") as f:
        for entry in training_log:
            f.write(json.dumps(entry) + "\n")

    summary = {
        "best_step": best_step,
        "best_mean_MAP@10": best_mean_map,
        "final_metrics": final_metrics,
        "total_steps": args.max_steps,
        "total_time_min": (time.time() - t_start) / 60,
        "args": vars(args),
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Print final results
    baseline = {"fashion200k": 0.5369, "atlas": 0.3908, "polyvore": 0.6249, "KAGL": 0.4638}
    log.info("")
    log.info("=" * 60)
    log.info("FINAL RESULTS (5K screener MAP@10)")
    log.info("=" * 60)
    log.info(f"{'Dataset':<14} {'Student':<10} {'Baseline':<10} {'Delta':<10} {'Beats?'}")
    log.info("-" * 60)
    beats_all = True
    for ds in ["fashion200k", "atlas", "polyvore", "KAGL"]:
        if ds in final_metrics:
            val = final_metrics[ds]
            bl = baseline[ds]
            delta = (val - bl) / bl * 100
            beats = "YES" if val > bl else "NO"
            if val <= bl:
                beats_all = False
            log.info(f"{ds:<14} {val:<10.4f} {bl:<10.4f} {delta:+.1f}%{'':>4} {beats}")
    log.info("=" * 60)
    log.info("Beats baseline on ALL datasets: %s", "YES" if beats_all else "NO")
    log.info("Best checkpoint at step %d (mean MAP@10 = %.4f)", best_step, best_mean_map)
    log.info("Output dir: %s", output_dir)
    log.info("Total training time: %.1f minutes", (time.time() - t_start) / 60)

    # Upload to GCS if specified
    if args.gcs_output:
        log.info("Uploading to GCS: %s", args.gcs_output)
        os.system(f"gcloud storage cp -r {output_dir}/* {args.gcs_output}/")
        log.info("Upload complete.")


if __name__ == "__main__":
    main()
