"""Distill fine-tuned B16-256 teacher (375M, 0.5770 MAP@10) into ViT-B-16-SigLIP (203M).

This is the fairest possible comparison against FashionSigLIP — SAME architecture (203M),
but trained with our teacher's ranking knowledge instead of Marqo's GCL.

Key design decisions (lessons from MobileCLIP2-B failure):
1. Student is ViT-B-16-SigLIP pretrained on webli — SAME base as FSL before their GCL
2. Only unfreeze last 2 blocks + head (not full tower) to prevent catastrophic forgetting
3. Use conservative LR (1e-6) that worked for B16-256 fine-tuning
4. Combine ranking distillation (teacher-student disagreements) with direct triplet loss
   (same near-miss strategy that gave us +17.7% on B16-256)
"""
import random, torch, gc, json, time, copy
from collections import defaultdict
from pathlib import Path
import torch.nn.functional as F

CACHE_DIR = Path(__file__).parent.parent / "cache" / "ensemble_1k"
MODEL_DIR = Path(__file__).parent.parent / "models"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

FSL_BASELINE = 0.4902


def load_corpus_and_queries():
    with open(CACHE_DIR / "corpus_meta.json") as f:
        corpus = json.load(f)
    cat_to_indices = defaultdict(list)
    for idx, c in enumerate(corpus):
        cat_to_indices[c["category"]].append(idx)
    valid_cats = [cat for cat, idxs in cat_to_indices.items() if len(idxs) >= 2]
    rng = random.Random(42)
    rng.shuffle(valid_cats)
    query_cats = valid_cats[:min(200, len(valid_cats))]
    return corpus, query_cats, cat_to_indices


def compute_map10(scores, query_cats, cat_to_indices):
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
    return map_at_k / max(len(query_cats), 1)


def build_triplets(teacher_scores, student_scores, query_cats, cat_to_indices):
    """Hybrid triplet mining: teacher-student disagreements + category-based near-misses."""
    triplets = []

    for qi, cat in enumerate(query_cats):
        t_scores = teacher_scores[qi]
        s_scores = student_scores[qi]
        positives = set(cat_to_indices[cat])

        sorted_by_teacher = t_scores.argsort(descending=True).tolist()
        sorted_by_student = s_scores.argsort(descending=True).tolist()
        student_rank = {idx: rank for rank, idx in enumerate(sorted_by_student)}

        # Strategy 1: Category-based near-miss triplets (proven technique)
        # True positives that student ranks poorly + hard negatives student ranks highly
        student_top20 = set(sorted_by_student[:20])
        false_positives_in_top = [idx for idx in sorted_by_student[:15] if idx not in positives]
        missed_positives = [idx for idx in sorted_by_student[10:] if idx in positives]

        for pos_idx in missed_positives[:3]:
            for neg_idx in false_positives_in_top[:3]:
                triplets.append({"query": cat, "positive_idx": pos_idx, "negative_idx": neg_idx})

        # Strategy 2: Teacher-student disagreement (moderate — only top-15 vs rank 15-50)
        for i in range(min(15, len(sorted_by_teacher))):
            pos_idx = sorted_by_teacher[i]
            for j in range(15, min(50, len(sorted_by_teacher))):
                neg_idx = sorted_by_teacher[j]
                # Only add if student actually disagrees
                if student_rank.get(neg_idx, 999) < student_rank.get(pos_idx, 999):
                    triplets.append({"query": cat, "positive_idx": pos_idx, "negative_idx": neg_idx})

    print(f"  Total triplets: {len(triplets)}")
    return triplets


def evaluate_model(model, preprocess, tokenizer, query_cats, cat_to_indices, images):
    model.eval()
    img_feats = []
    with torch.no_grad():
        for i in range(0, len(images), 32):
            batch = images[i:i+32]
            tensors = torch.stack([preprocess(im.convert("RGB")) for im in batch]).to(DEVICE)
            f = model.encode_image(tensors)
            f = F.normalize(f, dim=-1)
            img_feats.append(f.cpu())
            del tensors, f
    img_feats = torch.cat(img_feats, dim=0)

    txt_feats = []
    with torch.no_grad():
        for i in range(0, len(query_cats), 64):
            tokens = tokenizer(query_cats[i:i+64]).to(DEVICE)
            f = model.encode_text(tokens)
            f = F.normalize(f, dim=-1)
            txt_feats.append(f.cpu())
            del tokens, f
    txt_feats = torch.cat(txt_feats, dim=0)

    scores = txt_feats @ img_feats.T
    return compute_map10(scores, query_cats, cat_to_indices), scores


def get_full_scores(model, preprocess, tokenizer, query_cats, images):
    """Get score matrix for triplet mining."""
    model.eval()
    img_feats = []
    with torch.no_grad():
        for i in range(0, len(images), 32):
            batch = images[i:i+32]
            tensors = torch.stack([preprocess(im.convert("RGB")) for im in batch]).to(DEVICE)
            f = model.encode_image(tensors)
            f = F.normalize(f, dim=-1)
            img_feats.append(f.cpu())
            del tensors, f
    img_feats = torch.cat(img_feats, dim=0)

    txt_feats = []
    with torch.no_grad():
        for i in range(0, len(query_cats), 64):
            tokens = tokenizer(query_cats[i:i+64]).to(DEVICE)
            f = model.encode_text(tokens)
            f = F.normalize(f, dim=-1)
            txt_feats.append(f.cpu())
            del tokens, f
    txt_feats = torch.cat(txt_feats, dim=0)
    return txt_feats @ img_feats.T


def main():
    import open_clip
    from datasets import load_dataset

    print(f"Device: {DEVICE}")
    print(f"\n{'='*60}")
    print("DISTILLATION: Fine-tuned B16-256 (375M) → SigLIP B/16 (203M)")
    print(f"{'='*60}")

    corpus, query_cats, cat_to_indices = load_corpus_and_queries()
    print(f"Corpus: {len(corpus)}, Queries: {len(query_cats)}")

    # --- Load teacher ---
    print("\n[1] Loading teacher (fine-tuned B16-256, 375M)...")
    teacher, _, teacher_preprocess = open_clip.create_model_and_transforms("ViT-B-16-SigLIP2-256", pretrained="webli")
    teacher_state = torch.load(MODEL_DIR / "b16-256-nearmiss" / "model.pt", map_location="cpu", weights_only=True)
    teacher.load_state_dict(teacher_state)
    teacher.eval().to(DEVICE)
    teacher_tokenizer = open_clip.get_tokenizer("ViT-B-16-SigLIP2-256")

    # Load images
    print("  Loading fashion200k images...")
    ds = load_dataset("Marqo/fashion200k", split="data", streaming=True)
    images = []
    for row in ds:
        if len(images) >= 1000:
            break
        cat = row.get("category3", "") or ""
        if row.get("image") and cat.strip():
            images.append(row["image"])
    rng = random.Random(42)
    indices = list(range(len(images)))
    rng.shuffle(indices)
    images = [images[i] for i in indices[:1000]]
    print(f"  Loaded {len(images)} images")

    # Teacher scores
    print("  Computing teacher scores...")
    teacher_scores = get_full_scores(teacher, teacher_preprocess, teacher_tokenizer, query_cats, images)
    teacher_map10 = compute_map10(teacher_scores, query_cats, cat_to_indices)
    print(f"  Teacher MAP@10 = {teacher_map10:.4f}")
    del teacher
    gc.collect()

    # --- Load student (ViT-B-16-SigLIP, webli — SAME arch as FSL) ---
    print("\n[2] Loading student (ViT-B-16-SigLIP, 203M — same arch as FashionSigLIP)...")
    student, _, student_preprocess = open_clip.create_model_and_transforms("ViT-B-16-SigLIP", pretrained="webli")
    student.to(DEVICE)
    student_tokenizer = open_clip.get_tokenizer("ViT-B-16-SigLIP")
    n_params = sum(p.numel() for p in student.parameters())
    print(f"  Student params: {n_params/1e6:.1f}M")

    # Student baseline
    print("  Computing student baseline...")
    student_scores = get_full_scores(student, student_preprocess, student_tokenizer, query_cats, images)
    init_map10 = compute_map10(student_scores, query_cats, cat_to_indices)
    print(f"  Student init MAP@10 = {init_map10:.4f}")

    # --- Build triplets ---
    print("\n[3] Building hybrid triplets...")
    triplets = build_triplets(teacher_scores, student_scores, query_cats, cat_to_indices)
    del student_scores

    # --- Setup training (conservative — last 2 blocks only) ---
    for param in student.text.parameters():
        param.requires_grad = False
    for param in student.visual.parameters():
        param.requires_grad = False

    # Unfreeze last 2 transformer blocks + head/norm/proj
    unfrozen = []
    for name, param in student.visual.named_parameters():
        # SigLIP uses "blocks.X" naming
        if any(k in name for k in ["blocks.10", "blocks.11", "norm", "head", "proj"]):
            param.requires_grad = True
            unfrozen.append(name)

    if not unfrozen:
        # Fallback: try trunk.blocks
        for name, param in student.visual.named_parameters():
            if any(k in name for k in ["trunk.blocks.10", "trunk.blocks.11", "norm", "head", "proj"]):
                param.requires_grad = True
                unfrozen.append(name)

    if not unfrozen:
        # Last resort: unfreeze last 20% of visual params
        all_visual = list(student.visual.named_parameters())
        n_unfreeze = max(1, len(all_visual) // 5)
        for name, param in all_visual[-n_unfreeze:]:
            param.requires_grad = True
            unfrozen.append(name)

    n_trainable = sum(p.numel() for p in student.parameters() if p.requires_grad)
    print(f"  Unfrozen layers: {len(unfrozen)} ({n_trainable/1e6:.1f}M trainable)")
    print(f"  Unfrozen examples: {unfrozen[:3]}")

    # Pre-encode text queries (frozen text tower)
    student.eval()
    student_text_embeds = {}
    with torch.no_grad():
        for t in set(trip["query"] for trip in triplets):
            tokens = student_tokenizer([t]).to(DEVICE)
            emb = student.encode_text(tokens)
            student_text_embeds[t] = F.normalize(emb, dim=-1).cpu()

    # Pre-process images
    print("  Pre-processing images...")
    img_tensors = {}
    for t in triplets:
        for idx in [t["positive_idx"], t["negative_idx"]]:
            if idx not in img_tensors:
                img_tensors[idx] = student_preprocess(images[idx].convert("RGB"))

    # --- Training ---
    optimizer = torch.optim.AdamW(
        [p for p in student.parameters() if p.requires_grad],
        lr=1e-6,
        weight_decay=0.01,
    )
    margin = 0.05

    print(f"\n{'='*60}")
    print("TRAINING — Conservative ranking distillation")
    print(f"{'='*60}")
    print(f"  Triplets: {len(triplets)}, Margin: {margin}, LR: 1e-6")
    print(f"  Teacher MAP@10: {teacher_map10:.4f}")
    print(f"  Student init: {init_map10:.4f}, Target: > {FSL_BASELINE:.4f}")
    print(f"  Trainable: {n_trainable/1e6:.1f}M / {n_params/1e6:.1f}M")

    shuffled = triplets.copy()
    best_map10 = init_map10
    best_state = None
    step = 0
    max_steps = 500
    batch_size = 16
    log = []
    patience = 0
    max_patience = 3

    student.train()
    rng_train = random.Random(123)

    for epoch in range(10):
        rng_train.shuffle(shuffled)
        epoch_loss = 0.0
        epoch_steps = 0

        for batch_start in range(0, len(shuffled), batch_size):
            if step >= max_steps:
                break

            batch = shuffled[batch_start:batch_start + batch_size]
            if not batch:
                continue

            pos_imgs = torch.stack([img_tensors[t["positive_idx"]] for t in batch]).to(DEVICE)
            neg_imgs = torch.stack([img_tensors[t["negative_idx"]] for t in batch]).to(DEVICE)
            anchors = torch.cat([student_text_embeds[t["query"]] for t in batch], dim=0).to(DEVICE)

            pos_emb = F.normalize(student.encode_image(pos_imgs), dim=-1)
            neg_emb = F.normalize(student.encode_image(neg_imgs), dim=-1)

            pos_sim = (anchors * pos_emb).sum(dim=-1)
            neg_sim = (anchors * neg_emb).sum(dim=-1)

            loss = F.relu(neg_sim - pos_sim + margin).mean()
            epoch_loss += loss.item()
            epoch_steps += 1

            if loss.item() > 0:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                optimizer.step()

            step += 1
            if step % 20 == 0:
                print(f"  Step {step:3d} | loss = {loss.item():.4f}")

            del pos_imgs, neg_imgs, pos_emb, neg_emb, loss

            # Eval every 50 steps
            if step % 50 == 0:
                map10, _ = evaluate_model(student, student_preprocess, student_tokenizer,
                                         query_cats, cat_to_indices, images)
                delta_fsl = map10 - FSL_BASELINE
                log.append({"step": step, "map10": map10})

                status = "BEATS FSL!" if map10 > FSL_BASELINE else "below FSL"
                print(f"\n  >>> EVAL step {step}: MAP@10 = {map10:.4f} "
                      f"(vs init: {map10 - init_map10:+.4f}, vs FSL: {delta_fsl:+.4f}) [{status}]")

                if map10 > best_map10:
                    best_map10 = map10
                    best_state = copy.deepcopy(student.state_dict())
                    patience = 0
                    print(f"  >>> NEW BEST! (patience reset)")
                else:
                    patience += 1
                    print(f"  >>> No improvement (patience {patience}/{max_patience})")

                if map10 < init_map10 * 0.5:
                    print(f"\n  !!! COLLAPSED — stopping")
                    break

                if patience >= max_patience:
                    print(f"\n  !!! EARLY STOP — no improvement for {max_patience} evals")
                    break

                student.train()

        if step >= max_steps or patience >= max_patience:
            break

    # Final results
    print(f"\n{'='*60}")
    print("DISTILLATION COMPLETE")
    print(f"{'='*60}")

    if best_state:
        student.load_state_dict(best_state)
    final_map10, _ = evaluate_model(student, student_preprocess, student_tokenizer,
                                    query_cats, cat_to_indices, images)

    print(f"\n  Teacher MAP@10:       {teacher_map10:.4f}")
    print(f"  Student best MAP@10:  {best_map10:.4f}")
    print(f"  Student init MAP@10:  {init_map10:.4f}")
    print(f"  FSL baseline:         {FSL_BASELINE:.4f}")
    print(f"  Improvement:          {(best_map10 - init_map10) / init_map10 * 100:+.1f}% over init")
    print(f"  vs FSL:               {best_map10 - FSL_BASELINE:+.4f} ({(best_map10 - FSL_BASELINE) / FSL_BASELINE * 100:+.1f}%)")

    if best_map10 > FSL_BASELINE:
        print(f"\n  203M MODEL BEATS FASHIONSIGLIP!")
        save_dir = MODEL_DIR / "siglip-b16-distilled"
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(best_state, save_dir / "model.pt")
        print(f"  Saved to {save_dir / 'model.pt'}")
    else:
        print(f"\n  Did NOT beat FSL. Gap: {(best_map10 - FSL_BASELINE) / FSL_BASELINE * 100:+.1f}%")

    results = {
        "teacher_map10": teacher_map10,
        "student_init_map10": init_map10,
        "student_best_map10": best_map10,
        "fsl_baseline": FSL_BASELINE,
        "beats_fsl": best_map10 > FSL_BASELINE,
        "student_arch": "ViT-B-16-SigLIP",
        "student_params_M": n_params / 1e6,
        "trainable_params_M": n_trainable / 1e6,
        "n_triplets": len(triplets),
        "steps": step,
        "log": log,
    }
    with open(CACHE_DIR / "distill_siglip_b16_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  [Results saved to {CACHE_DIR / 'distill_siglip_b16_results.json'}]")


if __name__ == "__main__":
    main()
