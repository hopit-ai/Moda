"""Distill large ViT teacher → smaller ViT student using DeepFashion data.

Training data: Marqo/deepfashion-inshop + Marqo/deepfashion-multimodal (~95K images)
Eval: Marqo/fashion200k (category-based retrieval, zero overlap with training)

Safety: hard stop if fashion200k MAP@10 drops below 0.3287 (baseline to beat).
"""
import argparse, logging, math, os, random, time, gc
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip
from datasets import load_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

BASELINE_MAP10 = 0.3287  # fashion baseline (category-based) — hard floor


def load_and_preprocess(preprocess_fn, max_per_dataset=15000, seed=42):
    """Load fashion images and preprocess to tensors in one pass. Memory-efficient.
    
    Uses max_per_dataset=15000 per source → 30K total → ~40GB tensor memory (manageable).
    """
    img_tensors = []
    texts = []

    for ds_name in ["Marqo/deepfashion-inshop", "Marqo/deepfashion-multimodal"]:
        log.info("Loading + preprocessing %s ...", ds_name)
        try:
            ds = load_dataset(ds_name, split="data", streaming=True)
            count = 0
            for row in ds:
                if count >= max_per_dataset:
                    break
                if row.get("image"):
                    try:
                        tensor = preprocess_fn(row["image"].convert("RGB"))
                        text = row.get("text", "") or ""
                        if not text.strip():
                            text = row.get("description", "") or ""
                            if isinstance(text, list):
                                text = " ".join(text)
                        img_tensors.append(tensor)
                        texts.append(text.strip())
                        count += 1
                        del row
                    except Exception:
                        continue
                if count % 5000 == 0 and count > 0:
                    log.info("  %d/%d done", count, max_per_dataset)
            log.info("  Got %d images from %s", count, ds_name)
        except Exception as e:
            log.warning("  Failed %s: %s", ds_name, e)

    # Shuffle
    indices = list(range(len(img_tensors)))
    random.seed(seed)
    random.shuffle(indices)
    img_tensors = [img_tensors[i] for i in indices]
    texts = [texts[i] for i in indices]
    del indices
    gc.collect()

    n = len(img_tensors)
    log.info("Total training images: %d (all DeepFashion, zero benchmark overlap)", n)
    log.info("Stacking image tensors (%d x %s)...", n, img_tensors[0].shape)
    stacked = torch.stack(img_tensors)
    del img_tensors
    gc.collect()
    log.info("Image tensor block: %s (%.1f GB)", stacked.shape, stacked.nelement() * 4 / 1e9)
    return stacked, texts


def evaluate_fashion200k(model, preprocess, tokenizer, device="cuda", corpus_size=5000):
    """Category-based retrieval on fashion200k. Identical to baseline eval."""
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

    if len(query_cats) < 10:
        log.warning("Too few categories for eval, skipping")
        return 0.0

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
            torch.cuda.empty_cache()
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

    del corpus, img_feats, txt_feats, scores
    gc.collect()
    torch.cuda.empty_cache()
    return map_at_k


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--freeze-image-steps", type=int, default=500)
    parser.add_argument("--max-images-per-ds", type=int, default=30000)
    parser.add_argument("--save-dir", type=str, default="checkpoints_distill_fashion")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Device: %s", device)
    os.makedirs(args.save_dir, exist_ok=True)

    # --- Load student first (we need its preprocessor for data loading) ---
    log.info("Loading student (smaller ViT)...")
    student, _, student_preprocess = open_clip.create_model_and_transforms(
        "ViT-B-16-SigLIP2-384", pretrained="webli"
    )
    student.cpu()  # keep on CPU for now
    student_dim = 768
    tokenizer = open_clip.get_tokenizer("ViT-B-16-SigLIP2-384")

    # --- Load & preprocess training data (CPU only) ---
    img_tensors, texts = load_and_preprocess(student_preprocess, max_per_dataset=args.max_images_per_ds)
    n_data = len(texts)
    if n_data < 100:
        log.error("Not enough training data! Exiting.")
        return

    # --- Load teacher on GPU, precompute embeddings, then delete ---
    log.info("Loading teacher (large ViT) on GPU...")
    teacher, _, _ = open_clip.create_model_and_transforms(
        "ViT-L-16-SigLIP2-384", pretrained="webli"
    )
    teacher.eval().to(device)
    teacher_dim = 1024

    log.info("Pre-computing teacher image embeddings for %d images...", n_data)
    teacher_embeds = []
    with torch.no_grad():
        for i in range(0, n_data, 32):
            batch = img_tensors[i:i+32].to(device)
            t_feat = teacher.encode_image(batch)
            t_feat = F.normalize(t_feat, dim=-1)
            teacher_embeds.append(t_feat.cpu())
            del batch, t_feat
            torch.cuda.empty_cache()
    teacher_embeds = torch.cat(teacher_embeds, dim=0)
    log.info("Teacher image embeddings: %s", teacher_embeds.shape)

    log.info("Pre-computing teacher text embeddings...")
    teacher_tok = open_clip.get_tokenizer("ViT-L-16-SigLIP2-384")
    teacher_text_embeds = []
    with torch.no_grad():
        for i in range(0, n_data, 64):
            tokens = teacher_tok(texts[i:i+64]).to(device)
            feat = teacher.encode_text(tokens)
            feat = F.normalize(feat, dim=-1)
            teacher_text_embeds.append(feat.cpu())
            del tokens, feat
    teacher_text_embeds = torch.cat(teacher_text_embeds, dim=0)
    log.info("Teacher text embeddings: %s", teacher_text_embeds.shape)

    # Free teacher completely
    del teacher, teacher_tok
    gc.collect()
    torch.cuda.empty_cache()
    log.info("Teacher deleted. Moving student to GPU...")

    # Now move student to GPU
    student.to(device)

    # Projection: student_dim -> teacher_dim
    projection = nn.Linear(student_dim, teacher_dim, bias=False).to(device)
    nn.init.eye_(projection.weight[:student_dim, :student_dim])

    # --- Freeze student image tower initially ---
    for p in student.visual.parameters():
        p.requires_grad = False
    log.info("Student image tower FROZEN for first %d steps", args.freeze_image_steps)

    # --- Optimizer ---
    trainable = list(filter(lambda p: p.requires_grad, student.parameters())) + list(projection.parameters())
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps, eta_min=1e-7)

    # --- Initial eval ---
    log.info("Running initial eval...")
    init_map10 = evaluate_fashion200k(student, student_preprocess, tokenizer, device)
    log.info("INIT fashion200k MAP@10 = %.4f (baseline to beat: %.4f)", init_map10, BASELINE_MAP10)
    best_map10 = init_map10
    best_step = 0

    # --- Training loop ---
    log.info("Starting distillation: %d steps, batch_size=%d, lr=%.1e, %d images",
             args.steps, args.batch_size, args.lr, n_data)

    student.train()
    indices = list(range(n_data))
    step = 0
    t0 = time.time()

    while step < args.steps:
        random.shuffle(indices)
        for batch_start in range(0, n_data, args.batch_size):
            if step >= args.steps:
                break

            # Unfreeze image tower at the right step
            if step == args.freeze_image_steps:
                log.info("Step %d: UNFREEZING student image tower (LR/5)", step)
                for p in student.visual.parameters():
                    p.requires_grad = True
                trainable = list(student.parameters()) + list(projection.parameters())
                optimizer = torch.optim.AdamW([
                    {"params": list(student.text.parameters()) + list(projection.parameters()), "lr": args.lr},
                    {"params": list(student.visual.parameters()), "lr": args.lr / 5},
                ], weight_decay=0.01)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=args.steps - step, eta_min=1e-7
                )
                # Enable gradient checkpointing for memory
                if hasattr(student.visual, 'trunk') and hasattr(student.visual.trunk, 'set_grad_checkpointing'):
                    student.visual.trunk.set_grad_checkpointing(True)
                    log.info("  Gradient checkpointing enabled on student vision")

            batch_idx = indices[batch_start:batch_start + args.batch_size]
            if len(batch_idx) < 16:
                continue

            images = img_tensors[batch_idx].to(device)
            t_img_emb = teacher_embeds[batch_idx].to(device)   # [B, 1024]
            t_txt_emb = teacher_text_embeds[batch_idx].to(device)  # [B, 1024]

            # Student forward
            s_img_emb = student.encode_image(images)           # [B, 768]
            s_img_norm = F.normalize(s_img_emb, dim=-1)

            batch_texts = [texts[i] for i in batch_idx]
            tokens = tokenizer(batch_texts).to(device)
            s_txt_emb = student.encode_text(tokens)            # [B, 768]
            s_txt_norm = F.normalize(s_txt_emb, dim=-1)

            # --- Loss 1: Feature MSE (projected student → teacher space) ---
            s_img_proj = F.normalize(projection(s_img_emb), dim=-1)
            s_txt_proj = F.normalize(projection(s_txt_emb), dim=-1)
            loss_feat_img = F.mse_loss(s_img_proj, t_img_emb)
            loss_feat_txt = F.mse_loss(s_txt_proj, t_txt_emb)
            loss_feat = (loss_feat_img + loss_feat_txt) / 2

            # --- Loss 2: Relational KL (score matrix alignment) ---
            teacher_scores = t_txt_emb @ t_img_emb.T           # [B, B]
            student_scores = s_txt_norm @ s_img_norm.T          # [B, B]
            T = 0.05  # temperature
            teacher_probs = F.softmax(teacher_scores / T, dim=-1)
            student_log_probs = F.log_softmax(student_scores / T, dim=-1)
            loss_kl = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean")

            # --- Loss 3: Light self-supervised InfoNCE ---
            logit_scale = student.logit_scale.exp().clamp(max=100)
            logits = logit_scale * student_scores
            labels = torch.arange(len(batch_idx), device=device)
            loss_nce = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2

            # Combined loss
            loss = 1.0 * loss_feat + 0.5 * loss_kl + 0.1 * loss_nce

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in student.parameters() if p.requires_grad] + list(projection.parameters()),
                max_norm=1.0
            )
            optimizer.step()
            scheduler.step()

            step += 1
            del images, tokens, s_img_emb, s_txt_emb, s_img_proj, s_txt_proj, t_img_emb, t_txt_emb
            torch.cuda.empty_cache()

            if step % 50 == 0 or step == 1:
                elapsed = time.time() - t0
                eta = elapsed / step * (args.steps - step) / 60
                lr_now = scheduler.get_last_lr()[0]
                log.info("Step %d/%d | loss=%.4f (feat=%.4f kl=%.4f nce=%.4f) | lr=%.2e | ETA %.0f min",
                         step, args.steps, loss.item(), loss_feat.item(), loss_kl.item(),
                         loss_nce.item(), lr_now, eta)

            # --- Eval ---
            if step % args.eval_every == 0:
                log.info("=== EVAL at step %d ===", step)
                student.eval()
                map10 = evaluate_fashion200k(student, student_preprocess, tokenizer, device)
                log.info("  fashion200k MAP@10 = %.4f (init=%.4f, best=%.4f, baseline=%.4f)",
                         map10, init_map10, best_map10, BASELINE_MAP10)

                if map10 > best_map10:
                    best_map10 = map10
                    best_step = step
                    ckpt_path = os.path.join(args.save_dir, "best_student.pt")
                    torch.save({
                        "step": step,
                        "map10": map10,
                        "student_state_dict": student.state_dict(),
                        "projection_state_dict": projection.state_dict(),
                    }, ckpt_path)
                    log.info("  NEW BEST! Saved to %s", ckpt_path)

                if map10 < BASELINE_MAP10:
                    log.warning("  BELOW BASELINE (%.4f < %.4f) — HARD STOP", map10, BASELINE_MAP10)
                    log.info("  Best was step %d: %.4f", best_step, best_map10)
                    return

                student.train()
                if step >= args.freeze_image_steps:
                    for p in student.visual.parameters():
                        p.requires_grad = True

    # --- Final eval ---
    log.info("=== FINAL EVAL ===")
    student.eval()
    final_map10 = evaluate_fashion200k(student, student_preprocess, tokenizer, device)
    log.info("  FINAL fashion200k MAP@10 = %.4f", final_map10)
    log.info("  BEST  fashion200k MAP@10 = %.4f at step %d", best_map10, best_step)
    log.info("  INIT  fashion200k MAP@10 = %.4f", init_map10)
    log.info("  BASELINE to beat:          %.4f", BASELINE_MAP10)

    if final_map10 > best_map10:
        ckpt_path = os.path.join(args.save_dir, "best_student.pt")
        torch.save({
            "step": step,
            "map10": final_map10,
            "student_state_dict": student.state_dict(),
            "projection_state_dict": projection.state_dict(),
        }, ckpt_path)
        log.info("  FINAL is new best! Saved.")

    log.info("DONE. Best MAP@10=%.4f (init=%.4f, baseline=%.4f)", 
             max(best_map10, final_map10), init_map10, BASELINE_MAP10)


if __name__ == "__main__":
    main()
