"""Phase 14 — SO400M Teacher-Mined Contrastive Learning.

Core insight: Instead of distilling distributions (KL/MSE which failed 16 times),
use the teacher to CREATE better training data. The teacher's knowledge becomes
encoded in WHICH examples to contrast, not in soft target distributions.

This mirrors what Marqo did: they built FSL by training on 5M curated pairs with GCL.
We can't replicate their data, but we can use SO400M to curate ours.

Pipeline:
  1. Load cached SO400M embeddings (15K pairs)
  2. Build full 15K×15K teacher similarity matrix
  3. For each text: mine top-K positives and hard negatives from teacher rankings
  4. Train FSL student with InfoNCE + ListNet (ranking-aware) loss
  5. Progressive unfreezing: head → block 11 → block 10
  6. Layer-wise LR decay to prevent catastrophic forgetting

Key differences from prior experiments:
  - Teacher knowledge → training data curation, NOT distribution matching
  - ListNet ranking loss (order-preserving) instead of KL divergence (distribution-matching)
  - Progressive unfreezing prevents early catastrophic forgetting
  - Trains from FSL checkpoint, not base webli

Usage:
  python3 -u scripts/v3/phase14_teacher_mined_gcl.py train
  python3 -u scripts/v3/phase14_teacher_mined_gcl.py train --lr 5e-6 --unfreeze-schedule progressive
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
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("phase14")

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data" / "processed" / "v3_phase10_500k"
TEACHER_CACHE_DIR = REPO_ROOT / "data" / "processed" / "distillation_cache_so400m"
CHECKPOINT_DIR = REPO_ROOT / "checkpoints" / "v3_phase14"
RESULTS_DIR = REPO_ROOT / "results"

STUDENT_MODEL_HF = "hf-hub:Marqo/marqo-fashionSigLIP"

DEVICE = (
    torch.device("mps") if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
    else torch.device("cuda") if torch.cuda.is_available()
    else torch.device("cpu")
)


def mine_teacher_triplets(teacher_txt, teacher_img, n_hard_negs=5, n_positives=1):
    """Use SO400M rankings to mine training triplets.

    For each text query i:
      - Positive: image i (matched pair)
      - Hard negatives: images ranked just below the positive by the teacher
        (these are items the teacher thinks are somewhat similar but not the best match)
    """
    log.info("Mining triplets from teacher similarity matrix...")
    N = teacher_txt.shape[0]

    # Compute teacher similarity matrix in chunks to avoid OOM
    chunk_size = 1000
    all_triplets = []

    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        t_txt_chunk = teacher_txt[start:end]
        teacher_sims = t_txt_chunk @ teacher_img.T  # [chunk, N]

        for qi in range(teacher_sims.shape[0]):
            global_qi = start + qi
            scores = teacher_sims[qi].clone()
            scores[global_qi] = -1  # exclude self-match from ranking

            # Top-K ranked images (excluding self)
            topk_vals, topk_idx = scores.topk(n_hard_negs + 5)

            hard_negs = []
            for rank_idx in topk_idx.tolist():
                if rank_idx != global_qi and len(hard_negs) < n_hard_negs:
                    hard_negs.append(rank_idx)

            all_triplets.append({
                "anchor_text_idx": global_qi,
                "positive_img_idx": global_qi,
                "hard_neg_img_indices": hard_negs,
                "teacher_pos_score": teacher_sims[qi, global_qi].item(),
                "teacher_neg_scores": [teacher_sims[qi, ni].item() for ni in hard_negs],
            })

        if start > 0 and start % 5000 == 0:
            log.info("  Mined %d/%d triplets", start, N)

    log.info("  Total triplets: %d (avg hard negs per query: %.1f)",
             len(all_triplets),
             np.mean([len(t["hard_neg_img_indices"]) for t in all_triplets]))

    return all_triplets


def listnet_loss(student_scores, teacher_scores, temperature=1.0):
    """ListNet loss: KL between top-1 probability distributions.

    Unlike full KL on softmax distributions, ListNet focuses on the
    probability of each item being ranked first — more robust for ranking.
    """
    p_teacher = F.softmax(teacher_scores / temperature, dim=-1)
    p_student = F.log_softmax(student_scores / temperature, dim=-1)
    return -(p_teacher * p_student).sum(dim=-1).mean()


def infonce_with_hard_negs(anchor_emb, pos_emb, neg_embs, temperature=0.07):
    """InfoNCE loss with explicit hard negatives."""
    pos_sim = (anchor_emb * pos_emb).sum(dim=-1, keepdim=True) / temperature
    neg_sims = (anchor_emb.unsqueeze(1) @ neg_embs.transpose(-2, -1)).squeeze(1) / temperature
    logits = torch.cat([pos_sim, neg_sims], dim=-1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
    return F.cross_entropy(logits, labels)


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


def train(
    lr: float = 5e-6,
    temperature: float = 0.07,
    listnet_temp: float = 1.0,
    listnet_weight: float = 0.5,
    batch_size: int = 64,
    max_epochs: int = 5,
    eval_every: int = 50,
    max_drift: float = 0.05,
    patience: int = 8,
    n_hard_negs: int = 7,
    unfreeze_schedule: str = "progressive",
):
    """Train FSL student with teacher-mined hard negatives."""
    import open_clip

    # Load teacher cache
    teacher_meta = json.loads((TEACHER_CACHE_DIR / "meta.json").read_text())
    teacher_img = torch.load(TEACHER_CACHE_DIR / "teacher_img_embs.pt", map_location="cpu", weights_only=True)
    teacher_txt = torch.load(TEACHER_CACHE_DIR / "teacher_txt_embs.pt", map_location="cpu", weights_only=True)
    N = teacher_meta["n"]
    log.info("Teacher: %s, %d pairs", teacher_meta["teacher"], N)

    with open(TEACHER_CACHE_DIR / "pairs.json") as f:
        pairs = json.load(f)

    # Mine triplets using teacher rankings
    triplets = mine_teacher_triplets(teacher_txt, teacher_img, n_hard_negs=n_hard_negs)

    # Load student (FSL)
    log.info("Loading student: %s ...", STUDENT_MODEL_HF)
    student, _, preprocess = open_clip.create_model_and_transforms(STUDENT_MODEL_HF)
    tokenizer = open_clip.get_tokenizer(STUDENT_MODEL_HF)
    student = student.to(DEVICE)

    n_total = sum(p.numel() for p in student.parameters())
    log.info("  Student: %.0fM params on %s", n_total / 1e6, DEVICE)

    # Pre-compute student image embeddings (frozen tower)
    log.info("Pre-computing student image embeddings (frozen tower)...")
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
        del img_batch, emb
        if DEVICE.type == "mps":
            torch.mps.empty_cache()

        if i > 0 and (i // 64) % 50 == 0:
            log.info("  Images: %d/%d", i, N)

    student_img_embs = torch.cat(student_img_embs, dim=0)
    log.info("  Student image embeddings: %s (%.0fs)", student_img_embs.shape, time.time() - t0)

    # Pre-tokenize texts
    all_texts = [p["text_used"] for p in pairs]
    all_tokens = tokenizer(all_texts)
    log.info("  Tokens: %s", all_tokens.shape)

    # Setup unfreezing schedule
    for p in student.parameters():
        p.requires_grad = False

    def unfreeze_stage(stage: int):
        """Progressive unfreezing: 0=head only, 1=+block11, 2=+block10, 3=+block9"""
        unfrozen = []
        for name, param in student.named_parameters():
            should_unfreeze = False
            # Always: text_projection, ln_final, logit_scale/bias
            if any(k in name for k in ["text_projection", "ln_final", "logit_scale", "logit_bias"]):
                should_unfreeze = True
            # Stage 1+: block 11
            if stage >= 1 and "text" in name and any(f"blocks.{b}" in name or f"resblocks.{b}" in name for b in [11]):
                should_unfreeze = True
            # Stage 2+: block 10
            if stage >= 2 and "text" in name and any(f"blocks.{b}" in name or f"resblocks.{b}" in name for b in [10]):
                should_unfreeze = True
            # Stage 3+: block 9
            if stage >= 3 and "text" in name and any(f"blocks.{b}" in name or f"resblocks.{b}" in name for b in [9]):
                should_unfreeze = True

            if should_unfreeze:
                param.requires_grad = True
                unfrozen.append(name)

        n_trainable = sum(p.numel() for p in student.parameters() if p.requires_grad)
        log.info("  Stage %d: %d trainable params (%.1fM) in %d tensors",
                 stage, n_trainable, n_trainable / 1e6, len(unfrozen))
        return n_trainable

    if unfreeze_schedule == "progressive":
        current_stage = 0
        stage_steps = {0: 0, 1: 100, 2: 250, 3: 450}
    else:
        current_stage = 2
        stage_steps = {2: 0}

    unfreeze_stage(current_stage)

    # Anchor embeddings for drift monitoring
    anchor_n = min(200, N)
    anchor_texts = [pairs[i]["text_used"] for i in range(anchor_n)]
    student.eval()
    with torch.no_grad():
        anchor_tokens = tokenizer(anchor_texts).to(DEVICE)
        anchor_embs = student.encode_text(anchor_tokens)
        anchor_embs = F.normalize(anchor_embs, dim=-1).cpu()

    # Baseline
    log.info("Computing student baseline MAP@10...")
    init_map10 = quick_eval_map10(student, tokenizer, preprocess, DEVICE)
    log.info("  Student init MAP@10 = %.4f", init_map10)

    # Optimizer (will be rebuilt on unfreeze)
    def make_optimizer():
        params = [p for p in student.parameters() if p.requires_grad]
        return torch.optim.AdamW(params, lr=lr, weight_decay=0.01, betas=(0.9, 0.98))

    optimizer = make_optimizer()
    steps_per_epoch = N // batch_size
    total_steps = steps_per_epoch * max_epochs

    save_dir = CHECKPOINT_DIR / "teacher_mined_gcl"
    save_dir.mkdir(parents=True, exist_ok=True)

    log.info("")
    log.info("=" * 70)
    log.info("Phase 14 — Teacher-Mined Hard Negative GCL")
    log.info("=" * 70)
    log.info("  Teacher: SO400M (for mining only)")
    log.info("  Student: FSL (203M, init MAP@10=%.4f)", init_map10)
    log.info("  Loss: InfoNCE + %.1f×ListNet (temp=%.2f)", listnet_weight, listnet_temp)
    log.info("  Hard negs per query: %d", n_hard_negs)
    log.info("  LR: %.1e, Batch: %d, Epochs: %d", lr, batch_size, max_epochs)
    log.info("  Unfreeze: %s", unfreeze_schedule)
    log.info("=" * 70)

    global_step = 0
    best_map10 = init_map10
    best_state = None
    patience_counter = 0
    eval_log = []

    for epoch in range(1, max_epochs + 1):
        student.train()
        perm = torch.randperm(len(triplets))
        epoch_loss_nce = 0.0
        epoch_loss_ln = 0.0
        epoch_steps = 0
        t_epoch = time.time()

        for batch_start in range(0, len(triplets) - batch_size + 1, batch_size):
            batch_idx = perm[batch_start:batch_start + batch_size]

            # Check progressive unfreezing
            for stage, step_threshold in sorted(stage_steps.items()):
                if global_step == step_threshold and stage > current_stage:
                    current_stage = stage
                    unfreeze_stage(current_stage)
                    optimizer = make_optimizer()
                    log.info("  >>> Unfreezing stage %d at step %d", current_stage, global_step)

            # Build batch
            anchor_text_indices = []
            pos_img_indices = []
            neg_img_indices_flat = []
            n_negs_per = []

            for bi in batch_idx.tolist():
                t = triplets[bi]
                anchor_text_indices.append(t["anchor_text_idx"])
                pos_img_indices.append(t["positive_img_idx"])
                negs = t["hard_neg_img_indices"]
                neg_img_indices_flat.extend(negs)
                n_negs_per.append(len(negs))

            # Encode text (live, trainable)
            batch_tokens = all_tokens[anchor_text_indices].to(DEVICE)
            txt_embs = student.encode_text(batch_tokens)
            txt_embs = F.normalize(txt_embs, dim=-1)

            # Get cached image embeddings
            pos_img_embs = student_img_embs[pos_img_indices].to(DEVICE)
            all_neg_img_embs = student_img_embs[neg_img_indices_flat].to(DEVICE)

            # InfoNCE loss with hard negatives
            max_negs = max(n_negs_per)
            neg_embs_padded = torch.zeros(len(batch_idx), max_negs, txt_embs.shape[-1], device=DEVICE)
            offset = 0
            for i, nn in enumerate(n_negs_per):
                neg_embs_padded[i, :nn] = all_neg_img_embs[offset:offset + nn]
                offset += nn

            loss_nce = infonce_with_hard_negs(txt_embs, pos_img_embs, neg_embs_padded, temperature)

            # ListNet ranking loss: student should reproduce teacher's ranking order
            teacher_t = teacher_txt[anchor_text_indices]
            all_relevant_img_idx = list(set(pos_img_indices + neg_img_indices_flat))
            teacher_i = teacher_img[all_relevant_img_idx]
            student_i = student_img_embs[all_relevant_img_idx].to(DEVICE)

            teacher_scores = (teacher_t @ teacher_i.T).to(DEVICE)
            student_scores = txt_embs @ student_i.T
            loss_ln = listnet_loss(student_scores, teacher_scores, listnet_temp)

            loss = loss_nce + listnet_weight * loss_ln

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in student.parameters() if p.requires_grad], 1.0
            )
            optimizer.step()

            epoch_loss_nce += loss_nce.item()
            epoch_loss_ln += loss_ln.item()
            epoch_steps += 1
            global_step += 1

            del txt_embs, pos_img_embs, loss
            if DEVICE.type == "mps":
                torch.mps.empty_cache()

            if global_step % 50 == 0:
                log.info("  [E%d] step=%d  nce=%.4f  listnet=%.4f  total=%.4f",
                         epoch, global_step, loss_nce.item(), loss_ln.item(),
                         loss_nce.item() + listnet_weight * loss_ln.item())

            # Eval
            if global_step % eval_every == 0:
                # Drift check
                student.eval()
                with torch.no_grad():
                    curr_tokens = tokenizer(anchor_texts).to(DEVICE)
                    curr_embs = student.encode_text(curr_tokens)
                    curr_embs = F.normalize(curr_embs, dim=-1).cpu()
                drift = (1 - (curr_embs * anchor_embs).sum(dim=-1)).mean().item()

                if drift > max_drift:
                    log.warning("  !!! DRIFT EXCEEDED %.3f — aborting", max_drift)
                    break

                map10 = quick_eval_map10(student, tokenizer, preprocess, DEVICE)
                delta = map10 - init_map10
                log.info("  >>> EVAL step %d: MAP@10=%.4f (init%+.4f, %+.1f%%) drift=%.5f",
                         global_step, map10, delta, delta / init_map10 * 100, drift)

                eval_log.append({
                    "step": global_step, "epoch": epoch, "map10": map10,
                    "drift": drift, "stage": current_stage,
                })

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
                    break

                if map10 < init_map10 * 0.7:
                    log.warning("  !!! COLLAPSED — aborting")
                    break

                student.train()

        elapsed = time.time() - t_epoch
        log.info("Epoch %d done: nce=%.4f listnet=%.4f steps=%d (%.0fs)",
                 epoch, epoch_loss_nce / max(epoch_steps, 1),
                 epoch_loss_ln / max(epoch_steps, 1), epoch_steps, elapsed)

        if patience_counter >= patience or drift > max_drift:
            break

    # Final
    if best_state:
        student.load_state_dict(best_state)
    final_map10 = quick_eval_map10(student, tokenizer, preprocess, DEVICE)

    log.info("")
    log.info("=" * 70)
    log.info("PHASE 14 COMPLETE")
    log.info("=" * 70)
    log.info("  Init MAP@10:  %.4f", init_map10)
    log.info("  Best MAP@10:  %.4f (%+.1f%%)", best_map10, (best_map10 - init_map10) / init_map10 * 100)
    log.info("  Final MAP@10: %.4f", final_map10)
    log.info("=" * 70)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "phase14_teacher_mined_gcl.json", "w") as f:
        json.dump({
            "init_map10": init_map10, "best_map10": best_map10,
            "final_map10": final_map10, "eval_log": eval_log,
            "lr": lr, "n_hard_negs": n_hard_negs,
            "listnet_weight": listnet_weight,
        }, f, indent=2)
    log.info("Results saved: %s", RESULTS_DIR / "phase14_teacher_mined_gcl.json")


def main():
    parser = argparse.ArgumentParser(description="Phase 14: Teacher-mined GCL")
    sub = parser.add_subparsers(dest="command")

    p_train = sub.add_parser("train")
    p_train.add_argument("--lr", type=float, default=5e-6)
    p_train.add_argument("--temperature", type=float, default=0.07)
    p_train.add_argument("--listnet-temp", type=float, default=1.0)
    p_train.add_argument("--listnet-weight", type=float, default=0.5)
    p_train.add_argument("--batch-size", type=int, default=64)
    p_train.add_argument("--max-epochs", type=int, default=5)
    p_train.add_argument("--eval-every", type=int, default=50)
    p_train.add_argument("--max-drift", type=float, default=0.05)
    p_train.add_argument("--patience", type=int, default=8)
    p_train.add_argument("--n-hard-negs", type=int, default=7)
    p_train.add_argument("--unfreeze-schedule", type=str, default="progressive",
                         choices=["progressive", "immediate"])

    args = parser.parse_args()

    if args.command == "train":
        train(
            lr=args.lr,
            temperature=args.temperature,
            listnet_temp=args.listnet_temp,
            listnet_weight=args.listnet_weight,
            batch_size=args.batch_size,
            max_epochs=args.max_epochs,
            eval_every=args.eval_every,
            max_drift=args.max_drift,
            patience=args.patience,
            n_hard_negs=args.n_hard_negs,
            unfreeze_schedule=args.unfreeze_schedule,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
