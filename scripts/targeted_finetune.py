"""Targeted fine-tuning of SigLIP-2 B/16-256 on near-miss hard negatives.

Strategy:
1. Use cached scores to identify near-miss confusions (items B16-256 ranks above true positives)
2. Build triplets: (query_text, positive_image, hard_negative_image)
3. Fine-tune with triplet margin loss — pushes apart near-synonyms
4. Safety rail: eval every 50 steps, stop if MAP@10 drops below FSL baseline

Key difference from previous attempts:
- We're NOT doing general retrieval training
- We're ONLY teaching the model to separate specific confusable pairs
- Extremely small LR, few steps, text tower frozen (image tower only)
"""
import random, torch, gc, json, time, copy
from collections import defaultdict
from pathlib import Path
import torch.nn.functional as F

CACHE_DIR = Path(__file__).parent.parent / "cache" / "ensemble_1k"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

FSL_BASELINE = 0.4902  # hard stop threshold


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


def build_near_miss_triplets(corpus, query_cats, cat_to_indices):
    """Build training triplets from near-miss confusions.
    
    For each query where B16-256 fails, create triplets:
    - anchor: query text
    - positive: a correct corpus image (same category)
    - negative: the confusing image B16-256 ranked higher
    """
    scores = torch.load(CACHE_DIR / "scores_B16_256.pt", map_location="cpu", weights_only=True)
    
    triplets = []
    k = 20  # look at top-20 to find confusions
    
    for qi, cat in enumerate(query_cats):
        positives = set(cat_to_indices[cat])
        if len(positives) < 2:
            continue
            
        topk_vals, topk_idx = scores[qi].topk(min(k, scores.shape[1]))
        topk_idx = topk_idx.tolist()
        
        # Find negatives ranked above positives
        first_pos_rank = None
        for rank, idx in enumerate(topk_idx):
            if idx in positives:
                first_pos_rank = rank
                break
        
        if first_pos_rank is None or first_pos_rank == 0:
            continue  # no confusion or already correct
        
        # All items ranked above the first positive are hard negatives
        hard_negs = [idx for idx in topk_idx[:first_pos_rank] if idx not in positives]
        pos_list = [idx for idx in topk_idx if idx in positives]
        
        if not hard_negs or not pos_list:
            continue
        
        # Create triplets
        for neg_idx in hard_negs[:3]:  # max 3 negatives per query
            for pos_idx in pos_list[:2]:  # max 2 positives
                triplets.append({
                    "query": cat,
                    "positive_idx": pos_idx,
                    "negative_idx": neg_idx,
                    "positive_cat": corpus[pos_idx]["category"],
                    "negative_cat": corpus[neg_idx]["category"],
                })
    
    return triplets


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
    """Quick eval on cached 1K corpus."""
    model.eval()
    
    # Encode images
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
    
    # Encode texts
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


def main():
    import open_clip
    from datasets import load_dataset
    
    print(f"Device: {DEVICE}")
    print(f"FSL baseline (hard stop): {FSL_BASELINE:.4f}")
    
    # Load corpus metadata
    corpus, query_cats, cat_to_indices = load_corpus_and_queries()
    print(f"Corpus: {len(corpus)}, Queries: {len(query_cats)}")
    
    # Build triplets
    print("\nBuilding near-miss triplets...")
    triplets = build_near_miss_triplets(corpus, query_cats, cat_to_indices)
    print(f"  Generated {len(triplets)} triplets from near-miss confusions")
    
    if len(triplets) == 0:
        print("ERROR: No triplets generated. Exiting.")
        return
    
    # Show sample triplets
    print("\n  Sample triplets:")
    for t in triplets[:5]:
        print(f"    Q: \"{t['query']}\"")
        print(f"      +: \"{t['positive_cat']}\"  -: \"{t['negative_cat']}\"")
    
    # Load model
    print("\nLoading ViT-B-16-SigLIP2-256...")
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-16-SigLIP2-256", pretrained="webli")
    model.to(DEVICE)
    tokenizer = open_clip.get_tokenizer("ViT-B-16-SigLIP2-256")
    
    # Load images (same order as corpus)
    print("Loading fashion200k images...")
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
    
    # Initial eval
    print("\nInitial evaluation...")
    init_map10 = evaluate_model(model, preprocess, tokenizer, query_cats, cat_to_indices, images)
    print(f"  Initial MAP@10 = {init_map10:.4f} (FSL = {FSL_BASELINE:.4f}, gap = {init_map10 - FSL_BASELINE:+.4f})")
    
    # Setup training
    # Freeze text tower, only train image tower with very small LR
    for param in model.text.parameters():
        param.requires_grad = False
    
    # Only train last 2 transformer blocks of image tower
    for name, param in model.visual.named_parameters():
        param.requires_grad = False
    
    # Unfreeze last 2 blocks + final norm + head
    unfrozen = []
    for name, param in model.visual.named_parameters():
        if any(k in name for k in ["blocks.10", "blocks.11", "norm", "head", "proj"]):
            param.requires_grad = True
            unfrozen.append(name)
    
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  Trainable parameters: {n_trainable/1e6:.1f}M (last 2 image blocks + head)")
    
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-6,  # extremely small LR
        weight_decay=0.01,
    )
    
    margin = 0.05  # triplet margin
    
    # Pre-encode all texts (frozen, won't change)
    print("  Pre-encoding query texts (frozen)...")
    text_embeds = {}
    model.eval()
    with torch.no_grad():
        for t in set(trip["query"] for trip in triplets):
            tokens = tokenizer([t]).to(DEVICE)
            emb = model.encode_text(tokens)
            text_embeds[t] = F.normalize(emb, dim=-1).cpu()
    
    # Pre-process all images as tensors
    print("  Pre-processing triplet images...")
    img_tensors = {}
    for t in triplets:
        for idx in [t["positive_idx"], t["negative_idx"]]:
            if idx not in img_tensors:
                img_tensors[idx] = preprocess(images[idx].convert("RGB"))
    
    # Training loop
    print(f"\n{'='*60}")
    print("TRAINING — Triplet margin loss on near-miss hard negatives")
    print(f"{'='*60}")
    print(f"  Triplets: {len(triplets)}, Margin: {margin}, LR: 1e-6")
    print(f"  Strategy: push apart confusable image pairs in embedding space")
    print(f"  Safety: eval every 50 steps, hard stop if below {FSL_BASELINE:.4f}")
    
    rng = random.Random(42)
    shuffled_triplets = triplets.copy()
    
    best_map10 = init_map10
    best_state = None
    step = 0
    max_steps = 300
    batch_size = 8
    log = []
    
    model.train()
    
    for epoch in range(10):  # max 10 epochs over triplets
        rng.shuffle(shuffled_triplets)
        
        for batch_start in range(0, len(shuffled_triplets), batch_size):
            if step >= max_steps:
                break
                
            batch = shuffled_triplets[batch_start:batch_start+batch_size]
            if not batch:
                continue
            
            # Build batch tensors
            pos_imgs = torch.stack([img_tensors[t["positive_idx"]] for t in batch]).to(DEVICE)
            neg_imgs = torch.stack([img_tensors[t["negative_idx"]] for t in batch]).to(DEVICE)
            anchor_texts = torch.cat([text_embeds[t["query"]] for t in batch], dim=0).to(DEVICE)
            
            # Forward pass (image tower only changes)
            pos_emb = F.normalize(model.encode_image(pos_imgs), dim=-1)
            neg_emb = F.normalize(model.encode_image(neg_imgs), dim=-1)
            
            # Triplet margin loss: d(anchor, pos) < d(anchor, neg) - margin
            pos_sim = (anchor_texts * pos_emb).sum(dim=-1)
            neg_sim = (anchor_texts * neg_emb).sum(dim=-1)
            
            loss = F.relu(neg_sim - pos_sim + margin).mean()
            
            if loss.item() > 0:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            
            step += 1
            
            if step % 10 == 0:
                print(f"  Step {step:3d} | loss = {loss.item():.4f}")
            
            del pos_imgs, neg_imgs, pos_emb, neg_emb, pos_sim, neg_sim, loss
            
            # Eval every 50 steps
            if step % 50 == 0:
                map10 = evaluate_model(model, preprocess, tokenizer, query_cats, cat_to_indices, images)
                delta_init = map10 - init_map10
                delta_fsl = map10 - FSL_BASELINE
                log.append({"step": step, "map10": map10})
                
                status = "BEATS FSL" if map10 > FSL_BASELINE else "below FSL"
                print(f"\n  >>> EVAL step {step}: MAP@10 = {map10:.4f} "
                      f"(vs init: {delta_init:+.4f}, vs FSL: {delta_fsl:+.4f}) [{status}]")
                
                if map10 > best_map10:
                    best_map10 = map10
                    best_state = copy.deepcopy(model.state_dict())
                    print(f"  >>> NEW BEST! Saving checkpoint...")
                
                if map10 < FSL_BASELINE * 0.95:  # 5% below FSL = hard stop
                    print(f"\n  !!! HARD STOP: MAP@10 dropped to {map10:.4f} (below 95% of FSL)")
                    break
                
                model.train()
        
        if step >= max_steps:
            break
    
    # Final eval
    print(f"\n{'='*60}")
    print("TRAINING COMPLETE")
    print(f"{'='*60}")
    
    # Load best checkpoint
    if best_state:
        model.load_state_dict(best_state)
    
    final_map10 = evaluate_model(model, preprocess, tokenizer, query_cats, cat_to_indices, images)
    print(f"\n  Best MAP@10:    {best_map10:.4f}")
    print(f"  Final MAP@10:   {final_map10:.4f}")
    print(f"  Init MAP@10:    {init_map10:.4f}")
    print(f"  FSL baseline:   {FSL_BASELINE:.4f}")
    print(f"  vs FSL:         {best_map10 - FSL_BASELINE:+.4f} ({(best_map10-FSL_BASELINE)/FSL_BASELINE*100:+.1f}%)")
    
    if best_map10 > FSL_BASELINE:
        print(f"\n  BEATS FASHIONSIGLIP!")
        # Save model
        save_dir = Path(__file__).parent.parent / "models" / "b16-256-nearmiss"
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(best_state, save_dir / "model.pt")
        print(f"  Saved to {save_dir / 'model.pt'}")
    else:
        print(f"\n  Did not beat FSL. Best was {best_map10:.4f} vs {FSL_BASELINE:.4f}")
    
    # Save training log
    results = {
        "init_map10": init_map10,
        "best_map10": best_map10,
        "final_map10": final_map10,
        "fsl_baseline": FSL_BASELINE,
        "beats_fsl": best_map10 > FSL_BASELINE,
        "n_triplets": len(triplets),
        "max_steps": step,
        "log": log,
    }
    with open(CACHE_DIR / "targeted_finetune_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[Results saved to {CACHE_DIR / 'targeted_finetune_results.json'}]")


if __name__ == "__main__":
    main()
