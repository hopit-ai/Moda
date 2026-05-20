"""Phase 12 — SO400M (1.1B) → FSL (203M) ranking distillation.

The strongest teacher we have (SO400M, +77% vs FSL) distills ranking knowledge
into the FSL student (same 203M architecture as FashionSigLIP).

Two-stage pipeline:
  Stage 1: Pre-compute SO400M embeddings on 50K stratified pairs (one-time)
  Stage 2: Train FSL student to match SO400M's similarity rankings via KL divergence

Key design decisions (informed by 15 prior failed experiments):
  - Student starts from FSL checkpoint (not base webli) — preserves fashion knowledge
  - KL divergence on similarity matrices — teaches relative rankings, not absolute scores
  - Text tower blocks 10-11 + ln_final + text_projection trainable (~15M params)
  - Image tower FROZEN — every prior experiment that unfroze it degraded performance
  - Ultra-conservative LR (5e-7) with cosine decay
  - Anchor regularization (λ=0.3) to prevent catastrophic forgetting
  - Drift guard: abort if text embeddings move >0.03 cosine from FSL init
  - Early stopping with patience=3

Usage:
  # Stage 1: Cache SO400M embeddings (~50 min on MPS)
  python3 -u scripts/v3/phase12_so400m_distill.py cache-teacher --n-pairs 50000

  # Stage 2: Train distillation (~30 min on MPS)
  python3 -u scripts/v3/phase12_so400m_distill.py train

  # Both stages sequentially
  python3 -u scripts/v3/phase12_so400m_distill.py run-all
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
log = logging.getLogger("phase12")

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data" / "processed" / "v3_phase10_500k"
CACHE_DIR = REPO_ROOT / "data" / "processed" / "distillation_cache_so400m"
CHECKPOINT_DIR = REPO_ROOT / "checkpoints" / "v3_phase12"
RESULTS_DIR = REPO_ROOT / "results"

TEACHER_MODEL = "ViT-SO400M-14-SigLIP2-378"
STUDENT_MODEL_HF = "hf-hub:Marqo/marqo-fashionSigLIP"

DEVICE = (
    torch.device("mps") if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
    else torch.device("cuda") if torch.cuda.is_available()
    else torch.device("cpu")
)


# ── Stage 1: Cache SO400M teacher embeddings ─────────────────────────────────


def load_training_pairs(n_pairs: int, seed: int = 42) -> list[dict]:
    """Load stratified subset of training pairs from the 500K dataset."""
    pairs_path = DATA_DIR / "pairs.jsonl"
    log.info("Loading pairs from %s ...", pairs_path)

    all_pairs = []
    with open(pairs_path) as f:
        for line in f:
            item = json.loads(line)
            title = item.get("title", "").strip()
            query = item.get("query", "").strip()
            img_path = item.get("image_path", "")
            if (title or query) and img_path:
                all_pairs.append(item)

    log.info("  Total available: %d pairs", len(all_pairs))

    if n_pairs >= len(all_pairs):
        log.info("  Using all %d pairs", len(all_pairs))
        return all_pairs

    by_cat = defaultdict(list)
    for p in all_pairs:
        by_cat[p.get("l1_category", "other")].append(p)

    rng = random.Random(seed)
    sampled = []
    total = len(all_pairs)

    for cat, items in by_cat.items():
        proportion = len(items) / total
        n_take = max(10, int(n_pairs * proportion))
        rng.shuffle(items)
        sampled.extend(items[:n_take])

    rng.shuffle(sampled)
    sampled = sampled[:n_pairs]

    cats = defaultdict(int)
    for p in sampled:
        cats[p.get("l1_category", "other")] += 1
    log.info("  Stratified sample: %d pairs from %d categories", len(sampled), len(cats))
    for cat, cnt in sorted(cats.items(), key=lambda x: -x[1])[:8]:
        log.info("    %s: %d", cat, cnt)

    return sampled


def cache_teacher_embeddings(n_pairs: int = 50000, batch_size: int = 16):
    """Pre-compute SO400M embeddings on training pairs."""
    import open_clip

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    cache_meta_path = CACHE_DIR / "meta.json"
    if cache_meta_path.exists():
        meta = json.loads(cache_meta_path.read_text())
        if meta.get("n", 0) >= n_pairs and meta.get("teacher") == TEACHER_MODEL:
            log.info("Teacher cache already exists with %d pairs. Skipping.", meta["n"])
            return

    pairs = load_training_pairs(n_pairs)
    n = len(pairs)

    log.info("Loading teacher: %s ...", TEACHER_MODEL)
    model, _, preprocess = open_clip.create_model_and_transforms(TEACHER_MODEL, pretrained="webli")
    model = model.eval().to(DEVICE)
    tokenizer = open_clip.get_tokenizer(TEACHER_MODEL)

    n_params = sum(p.numel() for p in model.parameters())
    log.info("  Teacher: %s (%.0fM params) on %s", TEACHER_MODEL, n_params / 1e6, DEVICE)

    images_dir = DATA_DIR / "images"

    # Encode images
    log.info("Encoding %d images ...", n)
    img_embs = []
    t0 = time.time()
    failed = 0

    for i in range(0, n, batch_size):
        batch_pairs = pairs[i:i + batch_size]
        imgs = []
        for p in batch_pairs:
            img_path = images_dir / Path(p["image_path"]).name
            try:
                img = preprocess(Image.open(img_path).convert("RGB"))
            except Exception:
                img = preprocess(Image.new("RGB", (378, 378)))
                failed += 1
            imgs.append(img)

        img_batch = torch.stack(imgs).to(DEVICE)
        with torch.no_grad():
            emb = model.encode_image(img_batch)
            emb = F.normalize(emb, dim=-1)
        img_embs.append(emb.cpu())

        if DEVICE.type == "mps":
            torch.mps.empty_cache()
        elif DEVICE.type == "cuda":
            torch.cuda.empty_cache()

        if (i // batch_size) % 200 == 0 and i > 0:
            elapsed = time.time() - t0
            speed = (i + batch_size) / elapsed
            eta = (n - i - batch_size) / speed
            log.info("  Images: %d/%d (%.0f/s, ETA %.0fs)", i + batch_size, n, speed, eta)

    img_embs = torch.cat(img_embs, dim=0)
    log.info("  Image encoding done: %s (%.0fs, %d failed)", img_embs.shape, time.time() - t0, failed)

    # Encode texts (use title as primary, fallback to query)
    log.info("Encoding %d texts ...", n)
    txt_embs = []
    texts_used = []
    t0 = time.time()

    for i in range(0, n, batch_size * 4):
        batch_pairs = pairs[i:i + batch_size * 4]
        texts = []
        for p in batch_pairs:
            text = p.get("title", "").strip() or p.get("query", "").strip()
            texts.append(text)
            texts_used.append(text)

        tokens = tokenizer(texts).to(DEVICE)
        with torch.no_grad():
            emb = model.encode_text(tokens)
            emb = F.normalize(emb, dim=-1)
        txt_embs.append(emb.cpu())

        if DEVICE.type in ("mps", "cuda"):
            getattr(torch, DEVICE.type).empty_cache()

    txt_embs = torch.cat(txt_embs, dim=0)
    log.info("  Text encoding done: %s (%.0fs)", txt_embs.shape, time.time() - t0)

    # Save
    torch.save(img_embs, CACHE_DIR / "teacher_img_embs.pt")
    torch.save(txt_embs, CACHE_DIR / "teacher_txt_embs.pt")

    pair_meta = []
    for i, p in enumerate(pairs):
        pair_meta.append({
            "idx": i,
            "title": p.get("title", ""),
            "query": p.get("query", ""),
            "image_path": p.get("image_path", ""),
            "l1_category": p.get("l1_category", ""),
            "text_used": texts_used[i],
        })

    with open(CACHE_DIR / "pairs.json", "w") as f:
        json.dump(pair_meta, f)

    meta = {
        "n": n,
        "teacher": TEACHER_MODEL,
        "embed_dim": int(img_embs.shape[1]),
        "n_failed_images": failed,
        "device": str(DEVICE),
    }
    with open(cache_meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    log.info("Teacher cache saved to %s", CACHE_DIR)
    log.info("  Embeddings: img=%s txt=%s (dim=%d)", img_embs.shape, txt_embs.shape, img_embs.shape[1])

    del model, img_embs, txt_embs
    gc.collect()
    if DEVICE.type == "mps":
        torch.mps.empty_cache()


# ── Stage 2: Distillation training ───────────────────────────────────────────


def load_teacher_cache():
    """Load pre-computed teacher embeddings."""
    meta = json.loads((CACHE_DIR / "meta.json").read_text())
    log.info("Loading teacher cache: %d pairs, dim=%d, teacher=%s",
             meta["n"], meta["embed_dim"], meta["teacher"])

    teacher_img = torch.load(CACHE_DIR / "teacher_img_embs.pt", map_location="cpu", weights_only=True)
    teacher_txt = torch.load(CACHE_DIR / "teacher_txt_embs.pt", map_location="cpu", weights_only=True)

    with open(CACHE_DIR / "pairs.json") as f:
        pairs = json.load(f)

    return teacher_img, teacher_txt, pairs, meta


def kl_distill_loss(student_sims, teacher_sims, temperature=2.0):
    """KL divergence between teacher and student similarity distributions.

    Both row-wise (text→image) and column-wise (image→text) for symmetric learning.
    """
    # Row-wise: each text query's distribution over images
    t_row = F.log_softmax(teacher_sims / temperature, dim=1)
    s_row = F.log_softmax(student_sims / temperature, dim=1)
    kl_row = F.kl_div(s_row, t_row, log_target=True, reduction="batchmean")

    # Column-wise: each image's distribution over text queries
    t_col = F.log_softmax(teacher_sims / temperature, dim=0)
    s_col = F.log_softmax(student_sims / temperature, dim=0)
    kl_col = F.kl_div(s_col, t_col, log_target=True, reduction="batchmean")

    return 0.5 * (kl_row + kl_col) * (temperature ** 2)


def compute_drift(model, tokenizer, anchor_texts, anchor_embs, device):
    """Compute mean cosine distance of current text embeddings from anchor."""
    model.eval()
    with torch.no_grad():
        tokens = tokenizer(anchor_texts).to(device)
        current = model.encode_text(tokens)
        current = F.normalize(current, dim=-1).cpu()
    cos_dist = 1 - (current * anchor_embs).sum(dim=-1)
    return cos_dist.mean().item()


def quick_eval_map10(model, tokenizer, preprocess, device, corpus_size=3000):
    """Quick MAP@10 evaluation on fashion200k (category-based retrieval)."""
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

    return map_at_k, len(corpus), len(query_cats)


def train_distillation(
    lr: float = 5e-7,
    temperature: float = 2.0,
    anchor_lambda: float = 0.3,
    batch_size: int = 64,
    max_epochs: int = 2,
    eval_every: int = 200,
    max_drift: float = 0.03,
    patience: int = 3,
):
    """Train FSL student to match SO400M teacher rankings."""
    import open_clip

    # Load teacher cache
    teacher_img, teacher_txt, pairs, teacher_meta = load_teacher_cache()
    N = len(pairs)
    teacher_dim = teacher_meta["embed_dim"]
    log.info("Teacher cache: %d pairs, dim=%d", N, teacher_dim)

    # Load student (FSL — same 203M architecture)
    log.info("Loading student: %s ...", STUDENT_MODEL_HF)
    student, _, preprocess = open_clip.create_model_and_transforms(STUDENT_MODEL_HF)
    tokenizer = open_clip.get_tokenizer(STUDENT_MODEL_HF)
    student = student.to(DEVICE)

    n_total = sum(p.numel() for p in student.parameters())
    log.info("  Student: %.0fM params on %s", n_total / 1e6, DEVICE)

    # Freeze everything first
    for p in student.parameters():
        p.requires_grad = False

    # Unfreeze text tower: blocks 10-11 + ln_final + text_projection
    unfrozen_names = []
    for name, param in student.named_parameters():
        should_unfreeze = False
        # Text transformer blocks 10, 11
        if "text" in name and any(f"blocks.{b}" in name or f"resblocks.{b}" in name
                                   for b in [10, 11]):
            should_unfreeze = True
        # Text layer norm, projection
        if any(k in name for k in ["text.ln_final", "text.text_projection",
                                     "text_projection", "ln_final"]):
            should_unfreeze = True
        # Logit scale and bias
        if name in ("logit_scale", "logit_bias"):
            should_unfreeze = True

        if should_unfreeze:
            param.requires_grad = True
            unfrozen_names.append(name)

    n_trainable = sum(p.numel() for p in student.parameters() if p.requires_grad)
    log.info("  Trainable: %d params (%.1fM) in %d tensors",
             n_trainable, n_trainable / 1e6, len(unfrozen_names))
    for name in unfrozen_names[:5]:
        log.info("    %s", name)
    if len(unfrozen_names) > 5:
        log.info("    ... and %d more", len(unfrozen_names) - 5)

    # Save anchor embeddings for drift monitoring
    anchor_n = min(200, N)
    anchor_texts = [pairs[i]["text_used"] for i in range(anchor_n)]
    student.eval()
    with torch.no_grad():
        anchor_tokens = tokenizer(anchor_texts).to(DEVICE)
        anchor_embs = student.encode_text(anchor_tokens)
        anchor_embs = F.normalize(anchor_embs, dim=-1).cpu()
    log.info("  Anchor embeddings: %d texts cached for drift monitoring", anchor_n)

    # Student baseline MAP@10
    log.info("Computing student baseline MAP@10 ...")
    init_map10, corpus_n, n_queries = quick_eval_map10(student, tokenizer, preprocess, DEVICE)
    log.info("  Student init MAP@10 = %.4f (corpus=%d, queries=%d)", init_map10, corpus_n, n_queries)

    # Pre-compute student image embeddings (frozen tower — compute once)
    log.info("Pre-computing student image embeddings (frozen) ...")
    images_dir = DATA_DIR / "images"
    student_img_embs = []
    t0 = time.time()
    student.eval()

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

        if DEVICE.type == "mps":
            torch.mps.empty_cache()

        if (i // 64) % 100 == 0 and i > 0:
            log.info("  Images: %d/%d", i, N)

    student_img_embs = torch.cat(student_img_embs, dim=0)
    log.info("  Student image embeddings: %s (%.0fs)", student_img_embs.shape, time.time() - t0)

    # Pre-tokenize all texts for student
    log.info("Pre-tokenizing texts for student ...")
    all_texts = [p["text_used"] for p in pairs]
    all_tokens = tokenizer(all_texts)
    log.info("  Tokens: %s", all_tokens.shape)

    # Setup optimizer
    optimizer = torch.optim.AdamW(
        [p for p in student.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=0.01,
        betas=(0.9, 0.98),
        eps=1e-6,
    )

    steps_per_epoch = N // batch_size
    total_steps = steps_per_epoch * max_epochs
    warmup_steps = int(total_steps * 0.05)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Checkpoint dir
    save_dir = CHECKPOINT_DIR / "so400m_distill"
    save_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "teacher": TEACHER_MODEL,
        "student": STUDENT_MODEL_HF,
        "n_train": N,
        "trainable_params": n_trainable,
        "lr": lr,
        "temperature": temperature,
        "anchor_lambda": anchor_lambda,
        "batch_size": batch_size,
        "max_epochs": max_epochs,
        "max_drift": max_drift,
        "loss": "symmetric_kl_divergence + anchor_mse",
        "scope": "text blocks 10-11 + ln_final + text_projection + logit_scale/bias",
    }
    with open(save_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Training loop
    log.info("")
    log.info("=" * 70)
    log.info("Phase 12 — SO400M → FSL Ranking Distillation")
    log.info("=" * 70)
    log.info("  Teacher: %s (%.0fM, MAP@10=0.5696)",
             TEACHER_MODEL, teacher_meta.get("n", 0))
    log.info("  Student: FSL (203M, init MAP@10=%.4f)", init_map10)
    log.info("  Target:  > 0.4902 (FSL baseline on 1K eval)")
    log.info("  Loss:    KL(teacher_sims || student_sims) + %.2f·anchor_MSE", anchor_lambda)
    log.info("  Temp:    %.1f", temperature)
    log.info("  LR:      %.1e → cosine decay", lr)
    log.info("  Batch:   %d, Epochs: %d, Steps: %d", batch_size, max_epochs, total_steps)
    log.info("  Drift:   max %.3f cosine", max_drift)
    log.info("=" * 70)

    global_step = 0
    best_map10 = init_map10
    best_state = None
    eval_log = []
    patience_counter = 0
    aborted = False

    for epoch in range(1, max_epochs + 1):
        student.train()
        perm = torch.randperm(N)
        epoch_kl_loss = 0.0
        epoch_anchor_loss = 0.0
        epoch_steps = 0
        t_epoch = time.time()

        for batch_start in range(0, N - batch_size + 1, batch_size):
            idx = perm[batch_start:batch_start + batch_size]

            # Teacher similarity matrix (from cached embeddings)
            t_txt = teacher_txt[idx]
            t_img = teacher_img[idx]
            teacher_sims = (t_txt @ t_img.T).to(DEVICE)

            # Student: encode texts live (trainable), use cached images (frozen)
            batch_tokens = all_tokens[idx].to(DEVICE)
            s_txt = student.encode_text(batch_tokens)
            s_txt = F.normalize(s_txt, dim=-1)

            s_img = student_img_embs[idx].to(DEVICE)

            # Student similarity matrix
            logit_scale = student.logit_scale.exp().clamp(max=100)
            student_sims = s_txt @ s_img.T * logit_scale

            # Scale teacher sims to comparable range
            teacher_sims_scaled = teacher_sims * logit_scale.detach()

            # KL distillation loss
            loss_kl = kl_distill_loss(student_sims, teacher_sims_scaled, temperature)

            # Anchor regularization: prevent text embeddings from drifting
            anchor_idx = idx[:min(16, batch_size)]
            anchor_ref = anchor_embs[anchor_idx % anchor_n].to(DEVICE)
            anchor_curr_tokens = all_tokens[anchor_idx].to(DEVICE)
            with torch.no_grad():
                pass  # anchor_ref already computed
            anchor_curr = student.encode_text(anchor_curr_tokens)
            anchor_curr = F.normalize(anchor_curr, dim=-1)
            loss_anchor = F.mse_loss(anchor_curr, anchor_ref)

            loss = loss_kl + anchor_lambda * loss_anchor

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in student.parameters() if p.requires_grad], 1.0
            )
            optimizer.step()
            scheduler.step()

            epoch_kl_loss += loss_kl.item()
            epoch_anchor_loss += loss_anchor.item()
            epoch_steps += 1
            global_step += 1

            if global_step % 50 == 0:
                lr_now = scheduler.get_last_lr()[0]
                log.info(
                    "  [E%d] step=%d  kl=%.4f  anchor=%.6f  total=%.4f  lr=%.2e",
                    epoch, global_step, loss_kl.item(), loss_anchor.item(),
                    loss.item(), lr_now,
                )

            del teacher_sims, student_sims, s_txt, s_img, loss
            if DEVICE.type == "mps":
                torch.mps.empty_cache()

            # Periodic evaluation
            if global_step % eval_every == 0:
                # Drift check
                drift = compute_drift(student, tokenizer, anchor_texts, anchor_embs, DEVICE)
                log.info("  Drift check: %.5f (max %.3f)", drift, max_drift)

                if drift > max_drift:
                    log.warning("  !!! DRIFT EXCEEDED %.3f — aborting training", max_drift)
                    aborted = True
                    break

                # MAP@10 eval
                map10, _, _ = quick_eval_map10(student, tokenizer, preprocess, DEVICE)
                delta_init = map10 - init_map10
                delta_fsl = map10 - 0.4902

                status = "BEATS FSL!" if map10 > 0.4902 else "below FSL"
                log.info(
                    "  >>> EVAL step %d: MAP@10=%.4f (init%+.4f, FSL%+.4f) [%s] drift=%.5f",
                    global_step, map10, delta_init, delta_fsl, status, drift,
                )

                eval_log.append({
                    "step": global_step,
                    "epoch": epoch,
                    "map10": map10,
                    "drift": drift,
                    "kl_loss": epoch_kl_loss / max(epoch_steps, 1),
                    "anchor_loss": epoch_anchor_loss / max(epoch_steps, 1),
                })

                if map10 > best_map10:
                    best_map10 = map10
                    best_state = copy.deepcopy(student.state_dict())
                    patience_counter = 0
                    torch.save(
                        {"model_state_dict": best_state, "step": global_step,
                         "map10": map10, "config": config},
                        save_dir / "best.pt",
                    )
                    log.info("  >>> NEW BEST! Saved to %s", save_dir / "best.pt")
                else:
                    patience_counter += 1
                    log.info("  >>> No improvement (patience %d/%d)", patience_counter, patience)

                if patience_counter >= patience:
                    log.info("  !!! EARLY STOP — no improvement for %d evals", patience)
                    aborted = True
                    break

                if map10 < init_map10 * 0.7:
                    log.warning("  !!! COLLAPSED (>30%% below init) — aborting")
                    aborted = True
                    break

                student.train()

        elapsed_epoch = time.time() - t_epoch
        avg_kl = epoch_kl_loss / max(epoch_steps, 1)
        avg_anchor = epoch_anchor_loss / max(epoch_steps, 1)
        log.info(
            "Epoch %d done: avg_kl=%.4f avg_anchor=%.6f steps=%d time=%.0fs",
            epoch, avg_kl, avg_anchor, epoch_steps, elapsed_epoch,
        )

        if aborted:
            break

    # Final evaluation
    if best_state:
        student.load_state_dict(best_state)

    final_map10, _, _ = quick_eval_map10(student, tokenizer, preprocess, DEVICE)

    log.info("")
    log.info("=" * 70)
    log.info("DISTILLATION COMPLETE")
    log.info("=" * 70)
    log.info("  Teacher MAP@10:       0.5696 (SO400M, 1136M)")
    log.info("  Student init MAP@10:  %.4f  (FSL, 203M)", init_map10)
    log.info("  Student best MAP@10:  %.4f  (step %d)",
             best_map10, eval_log[-1]["step"] if eval_log else 0)
    log.info("  Student final MAP@10: %.4f", final_map10)
    log.info("  FSL baseline:         0.4902")
    log.info("  vs init:             %+.4f (%+.1f%%)", best_map10 - init_map10,
             (best_map10 - init_map10) / init_map10 * 100)
    log.info("  vs FSL:              %+.4f (%+.1f%%)", best_map10 - 0.4902,
             (best_map10 - 0.4902) / 0.4902 * 100)
    if best_map10 > 0.4902:
        log.info("  *** BEATS FASHIONSIGLIP! ***")
    log.info("=" * 70)

    # Save results
    results = {
        "teacher": TEACHER_MODEL,
        "student": STUDENT_MODEL_HF,
        "init_map10": init_map10,
        "best_map10": best_map10,
        "final_map10": final_map10,
        "fsl_baseline": 0.4902,
        "beats_fsl": best_map10 > 0.4902,
        "total_steps": global_step,
        "config": config,
        "eval_log": eval_log,
        "aborted": aborted,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "phase12_so400m_distill.json", "w") as f:
        json.dump(results, f, indent=2)
    log.info("Results saved: %s", RESULTS_DIR / "phase12_so400m_distill.json")

    return results


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Phase 12: SO400M → FSL distillation")
    sub = parser.add_subparsers(dest="command")

    # Stage 1
    p_cache = sub.add_parser("cache-teacher", help="Pre-compute SO400M embeddings")
    p_cache.add_argument("--n-pairs", type=int, default=50000)
    p_cache.add_argument("--batch-size", type=int, default=16)

    # Stage 2
    p_train = sub.add_parser("train", help="Run distillation training")
    p_train.add_argument("--lr", type=float, default=5e-7)
    p_train.add_argument("--temperature", type=float, default=2.0)
    p_train.add_argument("--anchor-lambda", type=float, default=0.3)
    p_train.add_argument("--batch-size", type=int, default=64)
    p_train.add_argument("--max-epochs", type=int, default=2)
    p_train.add_argument("--eval-every", type=int, default=200)
    p_train.add_argument("--max-drift", type=float, default=0.03)
    p_train.add_argument("--patience", type=int, default=3)

    # Both stages
    p_all = sub.add_parser("run-all", help="Cache + train sequentially")
    p_all.add_argument("--n-pairs", type=int, default=50000)
    p_all.add_argument("--cache-batch-size", type=int, default=16)
    p_all.add_argument("--lr", type=float, default=5e-7)
    p_all.add_argument("--temperature", type=float, default=2.0)
    p_all.add_argument("--anchor-lambda", type=float, default=0.3)
    p_all.add_argument("--batch-size", type=int, default=64)
    p_all.add_argument("--max-epochs", type=int, default=2)
    p_all.add_argument("--eval-every", type=int, default=200)

    args = parser.parse_args()

    if args.command == "cache-teacher":
        cache_teacher_embeddings(n_pairs=args.n_pairs, batch_size=args.batch_size)

    elif args.command == "train":
        train_distillation(
            lr=args.lr,
            temperature=args.temperature,
            anchor_lambda=args.anchor_lambda,
            batch_size=args.batch_size,
            max_epochs=args.max_epochs,
            eval_every=args.eval_every,
            max_drift=args.max_drift,
            patience=args.patience,
        )

    elif args.command == "run-all":
        cache_teacher_embeddings(n_pairs=args.n_pairs, batch_size=args.cache_batch_size)
        train_distillation(
            lr=args.lr,
            temperature=args.temperature,
            anchor_lambda=args.anchor_lambda,
            batch_size=args.batch_size,
            max_epochs=args.max_epochs,
            eval_every=args.eval_every,
        )

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
