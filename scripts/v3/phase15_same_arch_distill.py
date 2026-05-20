"""Phase 15 — Same-Architecture Feature Distillation from B16-256.

Uses the fine-tuned SigLIP-2 B16-256 teacher (768-dim, +17.7% vs FSL) for
DIRECT feature alignment. Key advantage: same embedding dimension means we
can use MSE loss directly on text embeddings — no projector needed.

This is fundamentally different from Phase 12 (SO400M, KL on sim matrices):
  - Same embedding dimension (768 vs 768): direct MSE, not indirect KL
  - Same ViT-B architecture family: feature spaces are structurally compatible
  - Direct feature matching: the student learns WHAT to embed, not just rankings

Combined loss:
  L = α·MSE(student_text, teacher_text) + β·rank_loss + γ·anchor_reg

Usage:
  # Cache B16-256 teacher embeddings (one-time)
  python3 -u scripts/v3/phase15_same_arch_distill.py cache-teacher

  # Train
  python3 -u scripts/v3/phase15_same_arch_distill.py train

  # Both
  python3 -u scripts/v3/phase15_same_arch_distill.py run-all
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
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("phase15")

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data" / "processed" / "v3_phase10_500k"
B16_TEACHER_CACHE = REPO_ROOT / "data" / "processed" / "distillation_cache_b16_256"
SO400M_CACHE = REPO_ROOT / "data" / "processed" / "distillation_cache_so400m"
CHECKPOINT_DIR = REPO_ROOT / "checkpoints" / "v3_phase15"
RESULTS_DIR = REPO_ROOT / "results"

B16_256_CHECKPOINT = REPO_ROOT / "models" / "b16-256-nearmiss" / "model.pt"
STUDENT_MODEL_HF = "hf-hub:Marqo/marqo-fashionSigLIP"

DEVICE = (
    torch.device("mps") if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
    else torch.device("cuda") if torch.cuda.is_available()
    else torch.device("cpu")
)


def quick_eval_map10(model, tokenizer, preprocess, device, corpus_size=3000):
    """Quick MAP@10 evaluation on fashion200k."""
    from datasets import load_dataset

    ds = load_dataset("Marqo/fashion200k", split="data", streaming=True)
    corpus = []
    for row in ds:
        if len(corpus) >= corpus_size:
            break
        cat = row.get("category3", "") or ""
        if row.get("image") and cat.strip():
            corpus.append({"image": row["image"], "category": cat.strip()})

    rng = random.Random(42)
    rng.shuffle(corpus)
    corpus = corpus[:corpus_size]

    cat_to_indices = defaultdict(list)
    for idx, c in enumerate(corpus):
        cat_to_indices[c["category"]].append(idx)
    valid_cats = [cat for cat, idxs in cat_to_indices.items() if len(idxs) >= 2]
    rng.shuffle(valid_cats)
    query_cats = valid_cats[:min(300, len(valid_cats))]

    model.eval()
    img_feats = []
    with torch.no_grad():
        for i in range(0, len(corpus), 32):
            batch = corpus[i:i + 32]
            imgs = torch.stack([preprocess(it["image"].convert("RGB")) for it in batch]).to(device)
            feat = model.encode_image(imgs)
            feat = F.normalize(feat, dim=-1)
            img_feats.append(feat.cpu())
            del imgs, feat
            if device.type == "mps":
                torch.mps.empty_cache()
    img_feats = torch.cat(img_feats, dim=0)

    txt_feats = []
    with torch.no_grad():
        for i in range(0, len(query_cats), 64):
            tokens = tokenizer(query_cats[i:i + 64]).to(device)
            feat = model.encode_text(tokens)
            feat = F.normalize(feat, dim=-1)
            txt_feats.append(feat.cpu())
            del tokens, feat
    txt_feats = torch.cat(txt_feats, dim=0)

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
    return map_at_k


def cache_b16_teacher():
    """Cache fine-tuned B16-256 embeddings on the same pairs as SO400M cache."""
    import open_clip

    B16_TEACHER_CACHE.mkdir(parents=True, exist_ok=True)

    meta_path = B16_TEACHER_CACHE / "meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta.get("n", 0) > 0:
            log.info("B16-256 cache exists with %d pairs. Skipping.", meta["n"])
            return

    with open(SO400M_CACHE / "pairs.json") as f:
        pairs = json.load(f)
    N = len(pairs)

    # Load fine-tuned B16-256
    log.info("Loading fine-tuned B16-256 from %s ...", B16_256_CHECKPOINT)
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-16-SigLIP2-256", pretrained="webli"
    )
    state = torch.load(B16_256_CHECKPOINT, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model = model.eval().to(DEVICE)
    tokenizer = open_clip.get_tokenizer("ViT-B-16-SigLIP2-256")

    n_params = sum(p.numel() for p in model.parameters())
    log.info("  B16-256: %.0fM params", n_params / 1e6)

    images_dir = DATA_DIR / "images"

    # Encode images
    log.info("Encoding %d images...", N)
    img_embs = []
    t0 = time.time()
    failed = 0

    for i in range(0, N, 64):
        batch_pairs = pairs[i:i + 64]
        imgs = []
        for p in batch_pairs:
            img_path = images_dir / Path(p["image_path"]).name
            try:
                img = preprocess(Image.open(img_path).convert("RGB"))
            except Exception:
                img = preprocess(Image.new("RGB", (256, 256)))
                failed += 1
            imgs.append(img)

        img_batch = torch.stack(imgs).to(DEVICE)
        with torch.no_grad():
            emb = model.encode_image(img_batch)
            emb = F.normalize(emb, dim=-1)
        img_embs.append(emb.cpu())
        del img_batch, emb
        if DEVICE.type == "mps":
            torch.mps.empty_cache()

        if i > 0 and (i // 64) % 50 == 0:
            log.info("  Images: %d/%d", i, N)

    img_embs = torch.cat(img_embs, dim=0)
    log.info("  Image encoding: %s (%.0fs, %d failed)", img_embs.shape, time.time() - t0, failed)

    # Encode texts
    log.info("Encoding %d texts...", N)
    txt_embs = []
    texts = [p["text_used"] for p in pairs]
    t0 = time.time()

    for i in range(0, N, 256):
        batch_texts = texts[i:i + 256]
        tokens = tokenizer(batch_texts).to(DEVICE)
        with torch.no_grad():
            emb = model.encode_text(tokens)
            emb = F.normalize(emb, dim=-1)
        txt_embs.append(emb.cpu())
        del tokens, emb
        if DEVICE.type == "mps":
            torch.mps.empty_cache()

    txt_embs = torch.cat(txt_embs, dim=0)
    log.info("  Text encoding: %s (%.0fs)", txt_embs.shape, time.time() - t0)

    # Save
    torch.save(img_embs, B16_TEACHER_CACHE / "teacher_img_embs.pt")
    torch.save(txt_embs, B16_TEACHER_CACHE / "teacher_txt_embs.pt")

    meta = {"n": N, "embed_dim": int(img_embs.shape[1]), "teacher": "B16-256-finetuned", "failed": failed}
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    log.info("Saved B16-256 cache: %s", B16_TEACHER_CACHE)

    del model, img_embs, txt_embs
    gc.collect()
    if DEVICE.type == "mps":
        torch.mps.empty_cache()


def train(
    lr: float = 3e-6,
    mse_weight: float = 1.0,
    rank_weight: float = 0.3,
    anchor_weight: float = 0.2,
    batch_size: int = 64,
    max_epochs: int = 5,
    eval_every: int = 50,
    max_drift: float = 0.04,
    patience: int = 8,
):
    """Train FSL to align text features with B16-256 teacher."""
    import open_clip

    # Load B16-256 teacher cache
    b16_meta = json.loads((B16_TEACHER_CACHE / "meta.json").read_text())
    b16_txt = torch.load(B16_TEACHER_CACHE / "teacher_txt_embs.pt", map_location="cpu", weights_only=True)
    b16_img = torch.load(B16_TEACHER_CACHE / "teacher_img_embs.pt", map_location="cpu", weights_only=True)
    N = b16_meta["n"]
    log.info("B16-256 teacher: %d pairs, dim=%d", N, b16_meta["embed_dim"])

    # Also load SO400M for auxiliary ranking signal
    so400m_txt = torch.load(SO400M_CACHE / "teacher_txt_embs.pt", map_location="cpu", weights_only=True)
    so400m_img = torch.load(SO400M_CACHE / "teacher_img_embs.pt", map_location="cpu", weights_only=True)
    log.info("SO400M auxiliary: loaded for ranking signal")

    with open(SO400M_CACHE / "pairs.json") as f:
        pairs = json.load(f)

    # Load FSL student
    log.info("Loading student: %s ...", STUDENT_MODEL_HF)
    student, _, preprocess = open_clip.create_model_and_transforms(STUDENT_MODEL_HF)
    tokenizer = open_clip.get_tokenizer(STUDENT_MODEL_HF)
    student = student.to(DEVICE)

    # Freeze image tower, unfreeze text blocks 9-11 + head
    for p in student.parameters():
        p.requires_grad = False

    unfrozen = []
    for name, param in student.named_parameters():
        should_unfreeze = False
        if any(k in name for k in ["text_projection", "ln_final", "logit_scale", "logit_bias"]):
            should_unfreeze = True
        if "text" in name and any(f"blocks.{b}" in name or f"resblocks.{b}" in name for b in [9, 10, 11]):
            should_unfreeze = True
        if should_unfreeze:
            param.requires_grad = True
            unfrozen.append(name)

    n_trainable = sum(p.numel() for p in student.parameters() if p.requires_grad)
    log.info("  Trainable: %.1fM params in %d tensors", n_trainable / 1e6, len(unfrozen))

    # Pre-compute student image embeddings (frozen)
    log.info("Pre-computing student image embeddings...")
    images_dir = DATA_DIR / "images"
    student_img_embs = []
    student.eval()
    t0 = time.time()

    for i in range(0, N, 64):
        batch_pairs = pairs[i:i + 64]
        imgs = []
        for p in batch_pairs:
            img_path = images_dir / Path(p["image_path"]).name
            try:
                img = preprocess(Image.open(img_path).convert("RGB"))
            except Exception:
                img = preprocess(Image.new("RGB", (224, 224)))
            imgs.append(img)
        img_batch = torch.stack(imgs).to(DEVICE)
        with torch.no_grad():
            emb = student.encode_image(img_batch)
            emb = F.normalize(emb, dim=-1)
        student_img_embs.append(emb.cpu())
        del img_batch, emb
        if DEVICE.type == "mps":
            torch.mps.empty_cache()
        if i > 0 and (i // 64) % 50 == 0:
            log.info("  Images: %d/%d", i, N)

    student_img_embs = torch.cat(student_img_embs, dim=0)
    log.info("  Student images: %s (%.0fs)", student_img_embs.shape, time.time() - t0)

    # Pre-tokenize
    all_texts = [p["text_used"] for p in pairs]
    all_tokens = tokenizer(all_texts)

    # Anchor embeddings
    anchor_n = min(200, N)
    anchor_texts = all_texts[:anchor_n]
    student.eval()
    with torch.no_grad():
        anchor_tokens = tokenizer(anchor_texts).to(DEVICE)
        anchor_embs = student.encode_text(anchor_tokens)
        anchor_embs = F.normalize(anchor_embs, dim=-1).cpu()

    # Baseline
    init_map10 = quick_eval_map10(student, tokenizer, preprocess, DEVICE)
    log.info("  Student init MAP@10 = %.4f", init_map10)

    # Optimizer with layer-wise LR decay
    param_groups = []
    for name, param in student.named_parameters():
        if not param.requires_grad:
            continue
        layer_lr = lr
        if "blocks.9" in name or "resblocks.9" in name:
            layer_lr = lr * 0.25
        elif "blocks.10" in name or "resblocks.10" in name:
            layer_lr = lr * 0.5
        param_groups.append({"params": [param], "lr": layer_lr})

    optimizer = torch.optim.AdamW(param_groups, weight_decay=0.01, betas=(0.9, 0.98))
    total_steps = (N // batch_size) * max_epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-7)

    save_dir = CHECKPOINT_DIR / "b16_256_feature_distill"
    save_dir.mkdir(parents=True, exist_ok=True)

    log.info("")
    log.info("=" * 70)
    log.info("Phase 15 — Same-Architecture Feature Distillation")
    log.info("=" * 70)
    log.info("  Teacher: B16-256 fine-tuned (768-dim, +17.7%% vs FSL)")
    log.info("  Auxiliary: SO400M (for ranking signal)")
    log.info("  Student: FSL (203M, init MAP@10=%.4f)", init_map10)
    log.info("  Loss: %.1f×MSE + %.1f×rank + %.1f×anchor", mse_weight, rank_weight, anchor_weight)
    log.info("  LR: %.1e (with layer-wise decay), Batch: %d, Epochs: %d", lr, batch_size, max_epochs)
    log.info("  Trainable: blocks 9-11 + head (%.1fM params)", n_trainable / 1e6)
    log.info("=" * 70)

    global_step = 0
    best_map10 = init_map10
    best_state = None
    patience_counter = 0
    eval_log = []
    aborted = False

    for epoch in range(1, max_epochs + 1):
        student.train()
        perm = torch.randperm(N)
        epoch_mse = 0.0
        epoch_rank = 0.0
        epoch_anchor = 0.0
        epoch_steps = 0
        t0 = time.time()

        for batch_start in range(0, N - batch_size + 1, batch_size):
            idx = perm[batch_start:batch_start + batch_size]

            # Student: encode text (live)
            batch_tokens = all_tokens[idx].to(DEVICE)
            s_txt = student.encode_text(batch_tokens)
            s_txt = F.normalize(s_txt, dim=-1)

            # Teacher text embeddings (B16-256, cached, same dim!)
            t_txt = b16_txt[idx].to(DEVICE)

            # Loss 1: Direct MSE on normalized text embeddings
            loss_mse = F.mse_loss(s_txt, t_txt)

            # Loss 2: Ranking alignment with SO400M
            so_txt = so400m_txt[idx].to(DEVICE)
            so_img = so400m_img[idx].to(DEVICE)
            s_img = student_img_embs[idx].to(DEVICE)

            teacher_sims = so_txt @ so_img.T
            student_sims = s_txt @ s_img.T

            t_probs = F.softmax(teacher_sims / 1.0, dim=1)
            s_log_probs = F.log_softmax(student_sims / 1.0, dim=1)
            loss_rank = F.kl_div(s_log_probs, t_probs, reduction="batchmean")

            # Loss 3: Anchor regularization
            anchor_idx = idx[:min(16, batch_size)]
            anchor_ref = anchor_embs[anchor_idx % anchor_n].to(DEVICE)
            anchor_s = s_txt[:min(16, batch_size)]
            loss_anchor = F.mse_loss(anchor_s, anchor_ref)

            loss = mse_weight * loss_mse + rank_weight * loss_rank + anchor_weight * loss_anchor

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in student.parameters() if p.requires_grad], 1.0
            )
            optimizer.step()
            scheduler.step()

            epoch_mse += loss_mse.item()
            epoch_rank += loss_rank.item()
            epoch_anchor += loss_anchor.item()
            epoch_steps += 1
            global_step += 1

            del s_txt, t_txt, loss
            if DEVICE.type == "mps":
                torch.mps.empty_cache()

            if global_step % 50 == 0:
                log.info("  [E%d] step=%d  mse=%.4f  rank=%.4f  anchor=%.6f",
                         epoch, global_step, loss_mse.item(), loss_rank.item(), loss_anchor.item())

            # Eval
            if global_step % eval_every == 0:
                student.eval()
                with torch.no_grad():
                    curr_tokens = tokenizer(anchor_texts).to(DEVICE)
                    curr_embs = student.encode_text(curr_tokens)
                    curr_embs = F.normalize(curr_embs, dim=-1).cpu()
                drift = (1 - (curr_embs * anchor_embs).sum(dim=-1)).mean().item()

                if drift > max_drift:
                    log.warning("  !!! DRIFT EXCEEDED %.3f — aborting", max_drift)
                    aborted = True
                    break

                map10 = quick_eval_map10(student, tokenizer, preprocess, DEVICE)
                delta = map10 - init_map10
                log.info("  >>> EVAL step %d: MAP@10=%.4f (init%+.4f, %+.1f%%) drift=%.5f",
                         global_step, map10, delta, delta / init_map10 * 100, drift)

                eval_log.append({"step": global_step, "epoch": epoch, "map10": map10, "drift": drift})

                if map10 > best_map10:
                    best_map10 = map10
                    best_state = copy.deepcopy(student.state_dict())
                    patience_counter = 0
                    torch.save({
                        "model_state_dict": best_state,
                        "step": global_step, "map10": map10,
                    }, save_dir / "best.pt")
                    log.info("  >>> NEW BEST! Saved to %s", save_dir / "best.pt")
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    log.info("  !!! EARLY STOP — no improvement for %d evals", patience)
                    aborted = True
                    break

                if map10 < init_map10 * 0.7:
                    log.warning("  !!! COLLAPSED — aborting")
                    aborted = True
                    break

                student.train()

        elapsed = time.time() - t0
        log.info("Epoch %d: mse=%.4f rank=%.4f anchor=%.6f steps=%d (%.0fs)",
                 epoch, epoch_mse / max(epoch_steps, 1),
                 epoch_rank / max(epoch_steps, 1),
                 epoch_anchor / max(epoch_steps, 1), epoch_steps, elapsed)

        if aborted:
            break

    # Final
    if best_state:
        student.load_state_dict(best_state)
    final_map10 = quick_eval_map10(student, tokenizer, preprocess, DEVICE)

    log.info("")
    log.info("=" * 70)
    log.info("PHASE 15 COMPLETE")
    log.info("=" * 70)
    log.info("  Init MAP@10:  %.4f", init_map10)
    log.info("  Best MAP@10:  %.4f (%+.1f%%)", best_map10, (best_map10 - init_map10) / init_map10 * 100)
    log.info("  Final MAP@10: %.4f", final_map10)
    log.info("=" * 70)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "phase15_same_arch_distill.json", "w") as f:
        json.dump({
            "init_map10": init_map10, "best_map10": best_map10,
            "final_map10": final_map10, "eval_log": eval_log,
        }, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Phase 15: Same-arch feature distillation")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("cache-teacher", help="Cache B16-256 embeddings")

    p_train = sub.add_parser("train")
    p_train.add_argument("--lr", type=float, default=3e-6)
    p_train.add_argument("--mse-weight", type=float, default=1.0)
    p_train.add_argument("--rank-weight", type=float, default=0.3)
    p_train.add_argument("--anchor-weight", type=float, default=0.2)
    p_train.add_argument("--batch-size", type=int, default=64)
    p_train.add_argument("--max-epochs", type=int, default=5)
    p_train.add_argument("--eval-every", type=int, default=50)
    p_train.add_argument("--max-drift", type=float, default=0.04)
    p_train.add_argument("--patience", type=int, default=8)

    p_all = sub.add_parser("run-all")
    p_all.add_argument("--lr", type=float, default=3e-6)

    args = parser.parse_args()

    if args.command == "cache-teacher":
        cache_b16_teacher()
    elif args.command == "train":
        train(
            lr=args.lr,
            mse_weight=args.mse_weight,
            rank_weight=args.rank_weight,
            anchor_weight=args.anchor_weight,
            batch_size=args.batch_size,
            max_epochs=args.max_epochs,
            eval_every=args.eval_every,
            max_drift=args.max_drift,
            patience=args.patience,
        )
    elif args.command == "run-all":
        cache_b16_teacher()
        train(lr=args.lr)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
