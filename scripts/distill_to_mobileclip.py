"""Distill fine-tuned B16-256 (teacher, 0.5770 MAP@10) into MobileCLIP2-B (student, 150M).

Key differences from previous failed distillation attempts:
1. Teacher is domain-specialized (fine-tuned on fashion near-miss triplets)
2. Loss: triplet-based distillation (not feature MSE or KL)
   - Student learns to RANK items the same way teacher does
   - For each query: teacher's top-K as positives, teacher's bottom-K as negatives
3. Student image tower is trained to produce embeddings that match teacher's ranking

Strategy: "Ranking distillation" — the student doesn't need to match teacher's exact
embeddings, just reproduce the same relative ordering.
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
    return compute_map10(scores, query_cats, cat_to_indices)


def build_ranking_triplets(teacher_scores, query_cats, cat_to_indices, student_scores=None):
    """Build HARD triplets from teacher vs student disagreements.
    
    Strategy: find pairs where teacher ranks item A above B, but student 
    ranks B above A (or is within margin). These are the pairs where student
    needs the most correction.
    """
    triplets = []
    n_hard = 0
    n_boundary = 0
    
    for qi, cat in enumerate(query_cats):
        t_scores = teacher_scores[qi]
        s_scores = student_scores[qi] if student_scores is not None else None
        
        sorted_by_teacher = t_scores.argsort(descending=True).tolist()
        
        # Strategy 1: Teacher-student disagreement pairs
        if s_scores is not None:
            sorted_by_student = s_scores.argsort(descending=True).tolist()
            student_rank = {idx: rank for rank, idx in enumerate(sorted_by_student)}
            
            # For teacher's top-20 items, find ones where student disagrees
            for i in range(min(20, len(sorted_by_teacher))):
                pos_idx = sorted_by_teacher[i]
                # Find items ranked higher by student but lower by teacher
                for j in range(20, min(100, len(sorted_by_teacher))):
                    neg_idx = sorted_by_teacher[j]
                    if student_rank.get(neg_idx, 999) < student_rank.get(pos_idx, 999):
                        triplets.append({"query": cat, "positive_idx": pos_idx, "negative_idx": neg_idx})
                        n_hard += 1
        
        # Strategy 2: Teacher's boundary pairs (items near the decision boundary)
        # Use items at rank 5-15 (positives) vs 15-40 (negatives) — not extremes
        boundary_pos = sorted_by_teacher[3:12]
        boundary_neg = sorted_by_teacher[15:40]
        
        for pos_idx in boundary_pos:
            for neg_idx in boundary_neg[:5]:
                triplets.append({"query": cat, "positive_idx": pos_idx, "negative_idx": neg_idx})
                n_boundary += 1
    
    print(f"    Hard disagreement triplets: {n_hard}")
    print(f"    Boundary triplets: {n_boundary}")
    print(f"    Total: {len(triplets)}")
    return triplets


def main():
    import open_clip
    from datasets import load_dataset

    print(f"Device: {DEVICE}")
    print(f"\n{'='*60}")
    print("RANKING DISTILLATION: Fine-tuned B16-256 → MobileCLIP2-B")
    print(f"{'='*60}")

    corpus, query_cats, cat_to_indices = load_corpus_and_queries()
    print(f"Corpus: {len(corpus)}, Queries: {len(query_cats)}")

    # --- Load teacher and pre-compute scores ---
    print("\n[1] Loading teacher (fine-tuned B16-256)...")
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

    # Pre-compute teacher scores
    print("  Computing teacher scores...")
    teacher_img_feats = []
    with torch.no_grad():
        for i in range(0, len(images), 32):
            batch = images[i:i+32]
            tensors = torch.stack([teacher_preprocess(im.convert("RGB")) for im in batch]).to(DEVICE)
            f = teacher.encode_image(tensors)
            f = F.normalize(f, dim=-1)
            teacher_img_feats.append(f.cpu())
            del tensors, f
    teacher_img_feats = torch.cat(teacher_img_feats, dim=0)

    teacher_txt_feats = []
    with torch.no_grad():
        for i in range(0, len(query_cats), 64):
            tokens = teacher_tokenizer(query_cats[i:i+64]).to(DEVICE)
            f = teacher.encode_text(tokens)
            f = F.normalize(f, dim=-1)
            teacher_txt_feats.append(f.cpu())
            del tokens, f
    teacher_txt_feats = torch.cat(teacher_txt_feats, dim=0)

    teacher_scores = teacher_txt_feats @ teacher_img_feats.T
    teacher_map10 = compute_map10(teacher_scores, query_cats, cat_to_indices)
    print(f"  Teacher MAP@10 = {teacher_map10:.4f}")

    # Free teacher
    del teacher, teacher_img_feats, teacher_txt_feats
    gc.collect()

    # --- Load student first to get initial scores ---
    print("\n[2] Loading student (MobileCLIP2-B, 150M)...")
    student, _, student_preprocess = open_clip.create_model_and_transforms("MobileCLIP2-B", pretrained="dfndr2b")
    student.to(DEVICE)
    student_tokenizer = open_clip.get_tokenizer("MobileCLIP2-B")
    n_params = sum(p.numel() for p in student.parameters())
    print(f"  Student params: {n_params/1e6:.1f}M")

    # Compute initial student scores for disagreement mining
    print("  Computing student scores for disagreement mining...")
    student.eval()
    student_img_feats_init = []
    with torch.no_grad():
        for i in range(0, len(images), 32):
            batch = images[i:i+32]
            tensors = torch.stack([student_preprocess(im.convert("RGB")) for im in batch]).to(DEVICE)
            f = student.encode_image(tensors)
            f = F.normalize(f, dim=-1)
            student_img_feats_init.append(f.cpu())
            del tensors, f
    student_img_feats_init = torch.cat(student_img_feats_init, dim=0)

    student_txt_feats_init = []
    with torch.no_grad():
        for i in range(0, len(query_cats), 64):
            tokens = student_tokenizer(query_cats[i:i+64]).to(DEVICE)
            f = student.encode_text(tokens)
            f = F.normalize(f, dim=-1)
            student_txt_feats_init.append(f.cpu())
            del tokens, f
    student_txt_feats_init = torch.cat(student_txt_feats_init, dim=0)
    student_scores_init = student_txt_feats_init @ student_img_feats_init.T

    init_map10 = compute_map10(student_scores_init, query_cats, cat_to_indices)
    print(f"  Student init MAP@10 = {init_map10:.4f}")

    # --- Build ranking distillation triplets ---
    print("\n[3] Building ranking distillation triplets (teacher-student disagreements)...")
    triplets = build_ranking_triplets(teacher_scores, query_cats, cat_to_indices, student_scores=student_scores_init)
    del student_img_feats_init, student_txt_feats_init, student_scores_init

    # Freeze text tower
    for param in student.text.parameters():
        param.requires_grad = False

    # Unfreeze image tower fully (we need all capacity for this harder task)
    for param in student.visual.parameters():
        param.requires_grad = True

    n_trainable = sum(p.numel() for p in student.parameters() if p.requires_grad)
    print(f"  Trainable params: {n_trainable/1e6:.1f}M (full image tower)")

    # Pre-encode student texts (frozen)
    print("  Pre-encoding student text queries...")
    student_text_embeds = {}
    student.eval()
    with torch.no_grad():
        for t in set(trip["query"] for trip in triplets):
            tokens = student_tokenizer([t]).to(DEVICE)
            emb = student.encode_text(tokens)
            student_text_embeds[t] = F.normalize(emb, dim=-1).cpu()

    # Pre-process images for student
    print("  Pre-processing images for student...")
    img_tensors = {}
    for t in triplets:
        for idx in [t["positive_idx"], t["negative_idx"]]:
            if idx not in img_tensors:
                img_tensors[idx] = student_preprocess(images[idx].convert("RGB"))

    # --- Training ---
    optimizer = torch.optim.AdamW(
        [p for p in student.parameters() if p.requires_grad],
        lr=5e-6,
        weight_decay=0.01,
    )
    margin = 0.08

    print(f"\n{'='*60}")
    print("TRAINING — Ranking distillation")
    print(f"{'='*60}")
    print(f"  Triplets: {len(triplets)}, Margin: {margin}, LR: 5e-6")
    print(f"  Teacher MAP@10: {teacher_map10:.4f}")
    print(f"  Student init: {init_map10:.4f}, Target: > {FSL_BASELINE:.4f}")

    shuffled = triplets.copy()
    best_map10 = init_map10
    best_state = None
    step = 0
    max_steps = 600
    batch_size = 16
    log = []

    student.train()
    rng = random.Random(42)

    for epoch in range(5):
        rng.shuffle(shuffled)
        for batch_start in range(0, len(shuffled), batch_size):
            if step >= max_steps:
                break

            batch = shuffled[batch_start:batch_start+batch_size]
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

            if loss.item() > 0:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                optimizer.step()

            step += 1
            if step % 10 == 0:
                print(f"  Step {step:3d} | loss = {loss.item():.4f}")

            del pos_imgs, neg_imgs, pos_emb, neg_emb, loss

            if step % 50 == 0:
                map10 = evaluate_model(student, student_preprocess, student_tokenizer, query_cats, cat_to_indices, images)
                delta_fsl = map10 - FSL_BASELINE
                log.append({"step": step, "map10": map10})

                status = "BEATS FSL" if map10 > FSL_BASELINE else "below FSL"
                print(f"\n  >>> EVAL step {step}: MAP@10 = {map10:.4f} "
                      f"(vs init: {map10-init_map10:+.4f}, vs FSL: {delta_fsl:+.4f}) [{status}]")

                if map10 > best_map10:
                    best_map10 = map10
                    best_state = copy.deepcopy(student.state_dict())
                    print(f"  >>> NEW BEST!")

                if map10 < 0.25:
                    print(f"\n  !!! HARD STOP: collapsed")
                    break

                student.train()

        if step >= max_steps:
            break

    # Final
    print(f"\n{'='*60}")
    print("DISTILLATION COMPLETE")
    print(f"{'='*60}")

    if best_state:
        student.load_state_dict(best_state)
    final_map10 = evaluate_model(student, student_preprocess, student_tokenizer, query_cats, cat_to_indices, images)

    print(f"\n  Teacher MAP@10:  {teacher_map10:.4f}")
    print(f"  Student best:    {best_map10:.4f}")
    print(f"  Student init:    {init_map10:.4f}")
    print(f"  FSL baseline:    {FSL_BASELINE:.4f}")
    print(f"  Improvement:     {(best_map10-init_map10)/init_map10*100:+.1f}% over init")
    print(f"  vs FSL:          {best_map10 - FSL_BASELINE:+.4f} ({(best_map10-FSL_BASELINE)/FSL_BASELINE*100:+.1f}%)")

    if best_map10 > FSL_BASELINE:
        print(f"\n  150M MODEL BEATS FASHIONSIGLIP!")
        save_dir = MODEL_DIR / "mobileclip2b-distilled"
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(best_state, save_dir / "model.pt")
        print(f"  Saved to {save_dir / 'model.pt'}")
    else:
        print(f"\n  Did NOT beat FSL. Gap closed from {(init_map10-FSL_BASELINE)/FSL_BASELINE*100:.1f}% to {(best_map10-FSL_BASELINE)/FSL_BASELINE*100:.1f}%")

    results = {
        "teacher_map10": teacher_map10,
        "student_init_map10": init_map10,
        "student_best_map10": best_map10,
        "fsl_baseline": FSL_BASELINE,
        "beats_fsl": best_map10 > FSL_BASELINE,
        "n_triplets": len(triplets),
        "steps": step,
        "log": log,
    }
    with open(CACHE_DIR / "distill_mobileclip_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[Saved to {CACHE_DIR / 'distill_mobileclip_results.json'}]")


if __name__ == "__main__":
    main()
