"""Train ViT-B-16-SigLIP (203M) on GSFashion-5M with GCL-style weighted sigmoid loss.

This replicates Marqo's recipe:
- Same base model (ViT-B-16-SigLIP, webli)
- Same data (Marqo/marqo-GS-10M, in_domain split)
- GCL-style weighted contrastive loss with graded relevance scores
- Full model fine-tuning with proper LR schedule

Key difference from our failed attempts: SCALE (50K-5M pairs vs 1K images)
and proper contrastive loss (sigmoid) instead of triplet margin.
"""
import argparse, random, time, copy, json, gc, math
import torch
import torch.nn.functional as F
from pathlib import Path
from collections import defaultdict

CACHE_DIR = Path(__file__).parent.parent / "cache"
MODEL_DIR = Path(__file__).parent.parent / "models"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

FSL_BASELINE_1K = 0.4902
FSL_BASELINE_5K = 0.3241
FSL_BASELINE_10K = 0.2463


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-train", type=int, default=50000, help="Number of training pairs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--lr-vision", type=float, default=1e-5, help="Vision tower LR")
    parser.add_argument("--lr-text", type=float, default=1e-5, help="Text tower LR")
    parser.add_argument("--lr-logit", type=float, default=1e-3, help="Logit scale/bias LR")
    parser.add_argument("--warmup-steps", type=int, default=500, help="Warmup steps")
    parser.add_argument("--weight-decay", type=float, default=0.1, help="Weight decay")
    parser.add_argument("--gcl-weight", action="store_true", default=True, help="Use GCL score weighting")
    parser.add_argument("--eval-every", type=int, default=500, help="Eval every N steps")
    parser.add_argument("--save-dir", type=str, default="siglip-b16-gcl", help="Model save directory name")
    return parser.parse_args()


def load_and_preprocess_training_data(n_pairs, preprocess):
    """Stream n_pairs from GSFashion-5M and preprocess images on-the-fly.
    
    Returns tensors directly to avoid OOM from storing PIL images.
    """
    from datasets import load_dataset
    
    print(f"  Loading {n_pairs} pairs from Marqo/marqo-GS-10M (in_domain)...")
    ds = load_dataset("Marqo/marqo-GS-10M", split="in_domain", streaming=True)
    
    img_tensors = []
    texts = []
    scores = []
    skipped = 0
    
    for row in ds:
        if len(img_tensors) >= n_pairs:
            break
        if row.get("image") and row.get("query") and row.get("score_linear") is not None:
            try:
                t = preprocess(row["image"].convert("RGB"))
                img_tensors.append(t)
                texts.append(row["query"])
                scores.append(row["score_linear"] / 100.0)
            except Exception:
                skipped += 1
                continue
        
        if len(img_tensors) % 5000 == 0 and len(img_tensors) > 0:
            print(f"    Processed {len(img_tensors)}/{n_pairs}...")
    
    print(f"  Loaded {len(img_tensors)} pairs (skipped {skipped})")
    queries = set(texts)
    print(f"  Unique queries: {len(queries)}")
    print(f"  Score range: [{min(scores):.2f}, {max(scores):.2f}]")
    return img_tensors, texts, scores


def sigmoid_contrastive_loss(image_features, text_features, logit_scale, logit_bias, weights=None):
    """SigLIP-style sigmoid contrastive loss with optional GCL weighting.
    
    For each (image_i, text_j) pair:
    - if i==j (matching pair): target = 1
    - if i!=j (non-matching): target = -1
    
    Loss = -log(sigmoid(target * (logit_scale * sim + logit_bias)))
    
    With GCL weighting: loss_ij is scaled by weight_i for positive pairs.
    """
    logits = logit_scale * image_features @ text_features.T + logit_bias
    n = logits.shape[0]
    
    # Target matrix: +1 on diagonal, -1 off-diagonal
    labels = 2 * torch.eye(n, device=logits.device) - 1
    
    # Sigmoid loss
    loss_matrix = -F.logsigmoid(labels * logits)
    
    if weights is not None:
        # GCL: weight positive pairs by relevance score
        # Higher score = more important to get right
        weight_matrix = torch.ones_like(loss_matrix)
        for i in range(n):
            weight_matrix[i, i] = weights[i]  # weight diagonal by relevance
        loss_matrix = loss_matrix * weight_matrix
    
    return loss_matrix.mean()


def compute_map10(scores, query_cats, cat_to_indices):
    k = 10
    total = 0.0
    for qi, cat in enumerate(query_cats):
        positives = set(cat_to_indices[cat])
        topk = scores[qi].topk(min(k, scores.shape[1])).indices.tolist()
        ap, n_rel = 0.0, 0
        for rank, idx in enumerate(topk, 1):
            if idx in positives:
                n_rel += 1
                ap += n_rel / rank
        n_pos = min(len(positives), k)
        if n_pos > 0:
            ap /= n_pos
        total += ap
    return total / max(len(query_cats), 1)


def evaluate_on_fashion200k(model, preprocess, tokenizer, corpus_size=1000, start_offset=0):
    """Evaluate on fashion200k with category-based retrieval."""
    from datasets import load_dataset
    
    ds = load_dataset("Marqo/fashion200k", split="data", streaming=True)
    items = []
    for row in ds:
        cat = row.get("category3", "") or ""
        if row.get("image") and cat.strip():
            items.append({"category": cat.strip(), "image": row["image"]})
        if len(items) >= start_offset + corpus_size:
            break
    
    items = items[start_offset:start_offset + corpus_size]
    rng = random.Random(42 if start_offset == 0 else 123)
    rng.shuffle(items)
    items = items[:corpus_size]
    
    cat_to_indices = defaultdict(list)
    for idx, item in enumerate(items):
        cat_to_indices[item["category"]].append(idx)
    valid_cats = [c for c, idxs in cat_to_indices.items() if len(idxs) >= 2]
    rng2 = random.Random(42 if start_offset == 0 else 99)
    rng2.shuffle(valid_cats)
    query_cats = valid_cats[:min(300, len(valid_cats))]
    
    model.eval()
    images = [item["image"] for item in items]
    
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
    map10 = compute_map10(scores, query_cats, cat_to_indices)
    return map10


def get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps):
    """Cosine LR schedule with linear warmup."""
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def main():
    args = parse_args()
    
    import open_clip
    from datasets import load_dataset
    
    print(f"Device: {DEVICE}")
    print(f"\n{'='*70}")
    print(f"GCL TRAINING: ViT-B-16-SigLIP (203M) on GSFashion-5M")
    print(f"{'='*70}")
    print(f"  Training pairs: {args.n_train}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Epochs: {args.epochs}")
    print(f"  LR (vision/text/logit): {args.lr_vision}/{args.lr_text}/{args.lr_logit}")
    print(f"  Warmup: {args.warmup_steps} steps")
    print(f"  GCL weighting: {args.gcl_weight}")
    
    # Load model
    print(f"\n[1] Loading ViT-B-16-SigLIP (203M, webli)...")
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-16-SigLIP", pretrained="webli")
    model.to(DEVICE)
    tokenizer = open_clip.get_tokenizer("ViT-B-16-SigLIP")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Params: {n_params/1e6:.1f}M")
    
    # Initial eval
    print(f"\n[2] Initial evaluation on fashion200k (1K corpus)...")
    init_map10 = evaluate_on_fashion200k(model, preprocess, tokenizer, corpus_size=1000)
    print(f"  Init MAP@10 = {init_map10:.4f} (FSL = {FSL_BASELINE_1K:.4f})")
    
    # Load and preprocess training data (stream + convert to tensors immediately)
    print(f"\n[3] Loading and preprocessing training data...")
    train_images, train_texts, train_scores = load_and_preprocess_training_data(args.n_train, preprocess)
    gc.collect()
    
    # Setup optimizer with parameter groups
    # Separate LR for vision, text, and logit scale/bias
    vision_params = []
    text_params = []
    logit_params = []
    
    for name, param in model.named_parameters():
        param.requires_grad = True
        if "logit" in name:
            logit_params.append(param)
        elif "visual" in name or "trunk" in name:
            vision_params.append(param)
        else:
            text_params.append(param)
    
    optimizer = torch.optim.AdamW([
        {"params": vision_params, "lr": args.lr_vision, "weight_decay": args.weight_decay},
        {"params": text_params, "lr": args.lr_text, "weight_decay": args.weight_decay},
        {"params": logit_params, "lr": args.lr_logit, "weight_decay": 0.0},
    ])
    
    steps_per_epoch = len(train_images) // args.batch_size
    total_steps = steps_per_epoch * args.epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, args.warmup_steps, total_steps)
    
    print(f"\n[5] Training setup:")
    print(f"  Steps per epoch: {steps_per_epoch}")
    print(f"  Total steps: {total_steps}")
    print(f"  Vision params: {sum(p.numel() for p in vision_params)/1e6:.1f}M")
    print(f"  Text params: {sum(p.numel() for p in text_params)/1e6:.1f}M")
    print(f"  Logit params: {sum(p.numel() for p in logit_params)}")
    
    # Get logit scale and bias from model
    # SigLIP uses learnable logit_bias (init=-10) and logit_scale
    logit_scale = model.logit_scale if hasattr(model, 'logit_scale') else None
    logit_bias = model.logit_bias if hasattr(model, 'logit_bias') else None
    
    if logit_scale is not None:
        print(f"  Initial logit_scale: {logit_scale.item():.4f}")
    if logit_bias is not None:
        print(f"  Initial logit_bias: {logit_bias.item():.4f}")
    
    # Training loop
    print(f"\n{'='*70}")
    print("TRAINING")
    print(f"{'='*70}")
    
    best_map10 = init_map10
    best_state = None
    step = 0
    log = []
    indices = list(range(len(train_images)))
    
    for epoch in range(args.epochs):
        random.shuffle(indices)
        epoch_loss = 0.0
        epoch_steps = 0
        
        model.train()
        
        for batch_start in range(0, len(indices), args.batch_size):
            batch_idx = indices[batch_start:batch_start + args.batch_size]
            if len(batch_idx) < 4:
                continue
            
            # Prepare batch
            img_batch = torch.stack([train_images[i] for i in batch_idx]).to(DEVICE)
            txt_batch = tokenizer([train_texts[i] for i in batch_idx]).to(DEVICE)
            score_batch = torch.tensor([train_scores[i] for i in batch_idx], device=DEVICE)
            
            # Forward
            img_features = F.normalize(model.encode_image(img_batch), dim=-1)
            txt_features = F.normalize(model.encode_text(txt_batch), dim=-1)
            
            # Get logit scale/bias
            ls = logit_scale.exp() if logit_scale is not None else torch.tensor(10.0, device=DEVICE)
            lb = logit_bias if logit_bias is not None else torch.tensor(-10.0, device=DEVICE)
            
            # Compute GCL-weighted sigmoid loss
            weights = score_batch if args.gcl_weight else None
            loss = sigmoid_contrastive_loss(img_features, txt_features, ls, lb, weights=weights)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            epoch_loss += loss.item()
            epoch_steps += 1
            step += 1
            
            del img_batch, txt_batch, img_features, txt_features, loss
            
            if step % 50 == 0:
                avg_loss = epoch_loss / epoch_steps
                lr_now = scheduler.get_last_lr()[0]
                print(f"  Epoch {epoch+1} Step {step:5d} | loss={avg_loss:.4f} | lr={lr_now:.2e}")
            
            if step % args.eval_every == 0:
                map10 = evaluate_on_fashion200k(model, preprocess, tokenizer, corpus_size=1000)
                delta = map10 - FSL_BASELINE_1K
                status = "BEATS FSL!" if map10 > FSL_BASELINE_1K else "below FSL"
                print(f"\n  >>> EVAL step {step}: MAP@10 = {map10:.4f} "
                      f"(vs init: {map10-init_map10:+.4f}, vs FSL: {delta:+.4f}) [{status}]")
                log.append({"step": step, "epoch": epoch+1, "map10": map10, "loss": epoch_loss/epoch_steps})
                
                if map10 > best_map10:
                    best_map10 = map10
                    best_state = copy.deepcopy(model.state_dict())
                    print(f"  >>> NEW BEST!")
                
                if map10 < init_map10 * 0.5:
                    print(f"  >>> COLLAPSED — stopping")
                    break
                
                model.train()
        
        # End of epoch eval
        map10 = evaluate_on_fashion200k(model, preprocess, tokenizer, corpus_size=1000)
        delta = map10 - FSL_BASELINE_1K
        status = "BEATS FSL!" if map10 > FSL_BASELINE_1K else "below FSL"
        print(f"\n  === END EPOCH {epoch+1}: MAP@10 = {map10:.4f} (vs FSL: {delta:+.4f}) [{status}]")
        log.append({"step": step, "epoch": epoch+1, "map10": map10, "loss": epoch_loss/max(epoch_steps,1), "end_epoch": True})
        
        if map10 > best_map10:
            best_map10 = map10
            best_state = copy.deepcopy(model.state_dict())
            print(f"  === NEW BEST!")
    
    # Final results
    print(f"\n{'='*70}")
    print("TRAINING COMPLETE")
    print(f"{'='*70}")
    
    if best_state:
        model.load_state_dict(best_state)
    
    # Eval on 1K
    final_1k = evaluate_on_fashion200k(model, preprocess, tokenizer, corpus_size=1000)
    print(f"\n  Best MAP@10 (1K):     {best_map10:.4f}")
    print(f"  Init MAP@10:          {init_map10:.4f}")
    print(f"  FSL baseline (1K):    {FSL_BASELINE_1K:.4f}")
    print(f"  Improvement vs init:  {(best_map10-init_map10)/init_map10*100:+.1f}%")
    print(f"  vs FSL:               {best_map10-FSL_BASELINE_1K:+.4f} ({(best_map10-FSL_BASELINE_1K)/FSL_BASELINE_1K*100:+.1f}%)")
    
    # Eval on 5K holdout
    print(f"\n  Evaluating on 5K holdout...")
    map10_5k = evaluate_on_fashion200k(model, preprocess, tokenizer, corpus_size=5000, start_offset=1000)
    print(f"  MAP@10 (5K holdout):  {map10_5k:.4f}")
    print(f"  FSL baseline (5K):    {FSL_BASELINE_5K:.4f}")
    print(f"  vs FSL (5K):          {map10_5k-FSL_BASELINE_5K:+.4f} ({(map10_5k-FSL_BASELINE_5K)/FSL_BASELINE_5K*100:+.1f}%)")
    
    beats_1k = best_map10 > FSL_BASELINE_1K
    beats_5k = map10_5k > FSL_BASELINE_5K
    
    if beats_1k and beats_5k:
        print(f"\n  203M MODEL BEATS FASHIONSIGLIP ON BOTH 1K AND 5K!")
    elif beats_1k:
        print(f"\n  Beats on 1K but not 5K holdout")
    else:
        print(f"\n  Does not beat FSL")
    
    # Save
    save_dir = MODEL_DIR / args.save_dir
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(best_state if best_state else model.state_dict(), save_dir / "model.pt")
    print(f"\n  Saved to {save_dir / 'model.pt'}")
    
    results = {
        "init_map10_1k": init_map10,
        "best_map10_1k": best_map10,
        "map10_5k_holdout": map10_5k,
        "fsl_1k": FSL_BASELINE_1K,
        "fsl_5k": FSL_BASELINE_5K,
        "beats_fsl_1k": beats_1k,
        "beats_fsl_5k": beats_5k,
        "n_train": args.n_train,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "total_steps": step,
        "log": log,
    }
    with open(CACHE_DIR / "gcl_training_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"  [Results: {CACHE_DIR / 'gcl_training_results.json'}]")


if __name__ == "__main__":
    main()
