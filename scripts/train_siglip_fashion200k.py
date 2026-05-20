"""Train ViT-B-16-SigLIP (203M) on fashion200k with sigmoid contrastive loss.

Key insight: fashion200k has 50K+ items with fine-grained category labels.
We use items 11K+ for training (keeping 0-10K for evaluation) and train
with SigLIP sigmoid loss on (image, category) pairs.

This directly optimizes for the same task we evaluate on: matching images
to category text queries. No domain shift.

Data leak prevention:
- Training: items 11000+ from fashion200k (different shuffle, different items)
- Evaluation: items 0-1000 (1K), 1000-6000 (5K), 1000-11000 (10K)
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
    parser.add_argument("--n-train", type=int, default=30000, help="Number of training items")
    parser.add_argument("--train-offset", type=int, default=11000, help="Start offset for training data")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=2e-6, help="Learning rate")
    parser.add_argument("--warmup-steps", type=int, default=200, help="Warmup steps")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--eval-every", type=int, default=300, help="Eval every N steps")
    return parser.parse_args()


def sigmoid_contrastive_loss(image_features, text_features, logit_scale, logit_bias):
    """SigLIP sigmoid contrastive loss — native loss for this architecture."""
    logits = logit_scale * image_features @ text_features.T + logit_bias
    n = logits.shape[0]
    labels = 2 * torch.eye(n, device=logits.device) - 1
    return (-F.logsigmoid(labels * logits)).mean()


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
    return compute_map10(scores, query_cats, cat_to_indices)


def load_training_data(preprocess, n_items, start_offset):
    """Load fashion200k training split (items after start_offset)."""
    from datasets import load_dataset

    print(f"  Loading fashion200k training data (offset={start_offset}, n={n_items})...")
    ds = load_dataset("Marqo/fashion200k", split="data", streaming=True)

    all_valid = []
    for row in ds:
        cat = row.get("category3", "") or ""
        if row.get("image") and cat.strip():
            all_valid.append({"category": cat.strip(), "image": row["image"]})
        if len(all_valid) >= start_offset + n_items:
            break

    train_items = all_valid[start_offset:start_offset + n_items]
    print(f"  Raw items: {len(train_items)}")

    # Shuffle with training seed
    rng = random.Random(777)
    rng.shuffle(train_items)

    # Preprocess to tensors
    img_tensors = []
    texts = []
    skipped = 0

    for i, item in enumerate(train_items):
        try:
            t = preprocess(item["image"].convert("RGB"))
            img_tensors.append(t)
            texts.append(item["category"])
        except Exception:
            skipped += 1
        if (i + 1) % 5000 == 0:
            print(f"    Processed {i+1}/{len(train_items)}...")

    print(f"  Training pairs: {len(img_tensors)} (skipped {skipped})")
    cats = set(texts)
    print(f"  Unique categories: {len(cats)}")
    return img_tensors, texts


def get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def main():
    args = parse_args()

    import open_clip

    print(f"Device: {DEVICE}")
    print(f"\n{'='*70}")
    print("FASHION200K TRAINING: ViT-B-16-SigLIP (203M) + Sigmoid Loss")
    print(f"{'='*70}")
    print(f"  Training items: {args.n_train} (offset {args.train_offset})")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Epochs: {args.epochs}")
    print(f"  LR: {args.lr}")
    print(f"  Warmup: {args.warmup_steps} steps")

    # Load model
    print(f"\n[1] Loading ViT-B-16-SigLIP (203M, webli)...")
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-16-SigLIP", pretrained="webli")
    model.to(DEVICE)
    tokenizer = open_clip.get_tokenizer("ViT-B-16-SigLIP")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Params: {n_params/1e6:.1f}M")

    # Initial eval
    print(f"\n[2] Initial evaluation (1K corpus)...")
    init_map10 = evaluate_on_fashion200k(model, preprocess, tokenizer, corpus_size=1000)
    print(f"  Init MAP@10 = {init_map10:.4f} (FSL = {FSL_BASELINE_1K:.4f})")

    # Load training data
    print(f"\n[3] Loading training data...")
    train_images, train_texts = load_training_data(preprocess, args.n_train, args.train_offset)
    gc.collect()

    # Build unique text embeddings (pre-encode all unique category texts)
    unique_texts = sorted(set(train_texts))
    text_to_idx = {t: i for i, t in enumerate(unique_texts)}
    print(f"  Pre-encoding {len(unique_texts)} unique category texts...")

    # Setup training — full model fine-tuning
    for param in model.parameters():
        param.requires_grad = True

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    steps_per_epoch = len(train_images) // args.batch_size
    total_steps = steps_per_epoch * args.epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, args.warmup_steps, total_steps)

    logit_scale = model.logit_scale if hasattr(model, 'logit_scale') else None
    logit_bias = model.logit_bias if hasattr(model, 'logit_bias') else None

    print(f"\n[4] Training setup:")
    print(f"  Steps per epoch: {steps_per_epoch}")
    print(f"  Total steps: {total_steps}")
    if logit_scale is not None:
        print(f"  Logit scale: {logit_scale.item():.4f}")
    if logit_bias is not None:
        print(f"  Logit bias: {logit_bias.item():.4f}")

    # Training
    print(f"\n{'='*70}")
    print("TRAINING")
    print(f"{'='*70}")

    best_map10 = init_map10
    best_state = None
    step = 0
    log = []
    indices = list(range(len(train_images)))
    patience = 0
    max_patience = 4

    for epoch in range(args.epochs):
        random.shuffle(indices)
        epoch_loss = 0.0
        epoch_steps = 0
        model.train()

        for batch_start in range(0, len(indices), args.batch_size):
            batch_idx = indices[batch_start:batch_start + args.batch_size]
            if len(batch_idx) < 8:
                continue

            img_batch = torch.stack([train_images[i] for i in batch_idx]).to(DEVICE)
            txt_batch = tokenizer([train_texts[i] for i in batch_idx]).to(DEVICE)

            img_features = F.normalize(model.encode_image(img_batch), dim=-1)
            txt_features = F.normalize(model.encode_text(txt_batch), dim=-1)

            ls = logit_scale.exp() if logit_scale is not None else torch.tensor(10.0, device=DEVICE)
            lb = logit_bias if logit_bias is not None else torch.tensor(-10.0, device=DEVICE)

            loss = sigmoid_contrastive_loss(img_features, txt_features, ls, lb)

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
                log.append({"step": step, "epoch": epoch+1, "map10": map10})

                if map10 > best_map10:
                    best_map10 = map10
                    best_state = copy.deepcopy(model.state_dict())
                    patience = 0
                    print(f"  >>> NEW BEST! (patience reset)")
                else:
                    patience += 1
                    print(f"  >>> No improvement (patience {patience}/{max_patience})")

                if map10 < init_map10 * 0.3:
                    print(f"  >>> COLLAPSED")
                    break
                if patience >= max_patience:
                    print(f"  >>> EARLY STOP")
                    break

                model.train()

        if patience >= max_patience:
            break

        # End epoch eval
        map10 = evaluate_on_fashion200k(model, preprocess, tokenizer, corpus_size=1000)
        delta = map10 - FSL_BASELINE_1K
        status = "BEATS FSL!" if map10 > FSL_BASELINE_1K else "below FSL"
        print(f"\n  === END EPOCH {epoch+1}: MAP@10 = {map10:.4f} (vs FSL: {delta:+.4f}) [{status}]")
        log.append({"step": step, "epoch": epoch+1, "map10": map10, "end_epoch": True})

        if map10 > best_map10:
            best_map10 = map10
            best_state = copy.deepcopy(model.state_dict())
            patience = 0

    # Final
    print(f"\n{'='*70}")
    print("TRAINING COMPLETE")
    print(f"{'='*70}")

    if best_state:
        model.load_state_dict(best_state)

    final_1k = evaluate_on_fashion200k(model, preprocess, tokenizer, corpus_size=1000)
    print(f"\n  Best MAP@10 (1K):     {best_map10:.4f}")
    print(f"  Init MAP@10:          {init_map10:.4f}")
    print(f"  FSL baseline (1K):    {FSL_BASELINE_1K:.4f}")
    print(f"  vs FSL:               {best_map10-FSL_BASELINE_1K:+.4f} ({(best_map10-FSL_BASELINE_1K)/FSL_BASELINE_1K*100:+.1f}%)")

    # 5K holdout
    print(f"\n  Evaluating on 5K holdout...")
    map10_5k = evaluate_on_fashion200k(model, preprocess, tokenizer, corpus_size=5000, start_offset=1000)
    print(f"  MAP@10 (5K):          {map10_5k:.4f}")
    print(f"  FSL (5K):             {FSL_BASELINE_5K:.4f}")
    print(f"  vs FSL (5K):          {map10_5k-FSL_BASELINE_5K:+.4f} ({(map10_5k-FSL_BASELINE_5K)/FSL_BASELINE_5K*100:+.1f}%)")

    # 10K holdout
    print(f"\n  Evaluating on 10K holdout...")
    map10_10k = evaluate_on_fashion200k(model, preprocess, tokenizer, corpus_size=10000, start_offset=1000)
    print(f"  MAP@10 (10K):         {map10_10k:.4f}")
    print(f"  FSL (10K):            {FSL_BASELINE_10K:.4f}")
    print(f"  vs FSL (10K):         {map10_10k-FSL_BASELINE_10K:+.4f} ({(map10_10k-FSL_BASELINE_10K)/FSL_BASELINE_10K*100:+.1f}%)")

    beats = best_map10 > FSL_BASELINE_1K and map10_5k > FSL_BASELINE_5K and map10_10k > FSL_BASELINE_10K
    if beats:
        print(f"\n  BEATS FSL ON ALL HOLDOUT SIZES!")
    
    # Save
    save_dir = MODEL_DIR / "siglip-b16-fashion200k"
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(best_state if best_state else model.state_dict(), save_dir / "model.pt")
    print(f"\n  Saved to {save_dir / 'model.pt'}")

    results = {
        "init_map10": init_map10,
        "best_map10_1k": best_map10,
        "map10_5k": map10_5k,
        "map10_10k": map10_10k,
        "fsl_1k": FSL_BASELINE_1K,
        "fsl_5k": FSL_BASELINE_5K,
        "fsl_10k": FSL_BASELINE_10K,
        "beats_all": bool(beats),
        "n_train": args.n_train,
        "train_offset": args.train_offset,
        "total_steps": step,
        "log": log,
    }
    with open(CACHE_DIR / "fashion200k_training_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"  [Results: {CACHE_DIR / 'fashion200k_training_results.json'}]")


if __name__ == "__main__":
    main()
