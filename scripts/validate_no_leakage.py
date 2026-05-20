"""Validate the fine-tuned B16-256 model on a HELD-OUT corpus.

Critical test: the model was trained on triplets mined from the first 1K items.
Now we evaluate on a DIFFERENT 5K slice (items 1001-6000) to confirm no overfitting.

If performance holds on unseen data, the improvement is real.
If it drops back to baseline, we overfit to the training corpus.
"""
import random, torch, gc, json, time
from collections import defaultdict
from pathlib import Path
import torch.nn.functional as F

CACHE_DIR = Path(__file__).parent.parent / "cache" / "validation_5k"
MODEL_DIR = Path(__file__).parent.parent / "models" / "b16-256-nearmiss"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

HOLDOUT_START = 1000  # skip first 1K (used for training)
HOLDOUT_SIZE = 5000


def load_holdout_corpus():
    """Load items 1001-6000 from fashion200k (completely unseen during training)."""
    cache_path = CACHE_DIR / "holdout_corpus_meta.json"
    if cache_path.exists():
        with open(cache_path) as f:
            return json.load(f), None

    from datasets import load_dataset
    print("  Streaming fashion200k (skipping first 1K, loading next 5K)...")
    ds = load_dataset("Marqo/fashion200k", split="data", streaming=True)

    all_valid = []
    images = []
    for row in ds:
        cat = row.get("category3", "") or ""
        if row.get("image") and cat.strip():
            all_valid.append({"category": cat.strip(), "image": row["image"]})

        if len(all_valid) >= HOLDOUT_START + HOLDOUT_SIZE:
            break

    # Take items after the training corpus
    holdout = all_valid[HOLDOUT_START:HOLDOUT_START + HOLDOUT_SIZE]

    # Deterministic shuffle with different seed than training
    rng = random.Random(123)
    rng.shuffle(holdout)
    holdout = holdout[:HOLDOUT_SIZE]

    corpus_meta = [{"category": h["category"], "idx": i} for i, h in enumerate(holdout)]
    images = [h["image"] for h in holdout]

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(corpus_meta, f)

    return corpus_meta, images


def load_holdout_images():
    """Re-load images for the holdout set (can't cache PIL images)."""
    from datasets import load_dataset
    print("  Re-streaming fashion200k for holdout images...")
    ds = load_dataset("Marqo/fashion200k", split="data", streaming=True)

    all_valid = []
    for row in ds:
        cat = row.get("category3", "") or ""
        if row.get("image") and cat.strip():
            all_valid.append(row["image"])
        if len(all_valid) >= HOLDOUT_START + HOLDOUT_SIZE:
            break

    holdout_imgs = all_valid[HOLDOUT_START:HOLDOUT_START + HOLDOUT_SIZE]
    rng = random.Random(123)
    indices = list(range(len(holdout_imgs)))
    rng.shuffle(indices)
    return [holdout_imgs[i] for i in indices[:HOLDOUT_SIZE]]


def build_queries(corpus):
    cat_to_indices = defaultdict(list)
    for idx, c in enumerate(corpus):
        cat_to_indices[c["category"]].append(idx)
    valid_cats = [cat for cat, idxs in cat_to_indices.items() if len(idxs) >= 2]
    rng = random.Random(123)  # different seed
    rng.shuffle(valid_cats)
    return valid_cats[:min(300, len(valid_cats))], cat_to_indices


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


def encode_and_eval(model, preprocess, tokenizer, images, query_cats, cat_to_indices, label):
    """Encode corpus + queries, compute MAP@10."""
    model.eval()

    print(f"    Encoding {len(images)} images...")
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

    print(f"    Encoding {len(query_cats)} text queries...")
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

    # Cache scores
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(scores, CACHE_DIR / f"scores_{label}.pt")

    return map10


def main():
    import open_clip

    print(f"Device: {DEVICE}")
    print(f"\n{'='*60}")
    print("HELD-OUT VALIDATION (no data leakage)")
    print(f"{'='*60}")
    print(f"\n  Training corpus: items 0-999 (1K, seed=42)")
    print(f"  Validation corpus: items 1000-5999 (5K, seed=123)")
    print(f"  Zero overlap between train and eval data.\n")

    # Load holdout corpus
    corpus_meta, images = load_holdout_corpus()
    if images is None:
        images = load_holdout_images()

    query_cats, cat_to_indices = build_queries(corpus_meta)
    print(f"  Holdout corpus: {len(corpus_meta)} items")
    print(f"  Holdout queries: {len(query_cats)} categories")

    # Load base model (zero-shot B16-256)
    print(f"\n--- Evaluating ZERO-SHOT B16-256 (baseline) ---")
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-16-SigLIP2-256", pretrained="webli")
    model.eval().to(DEVICE)
    tokenizer = open_clip.get_tokenizer("ViT-B-16-SigLIP2-256")

    map10_zeroshot = encode_and_eval(model, preprocess, tokenizer, images, query_cats, cat_to_indices, "B16_256_zeroshot")
    print(f"    Zero-shot MAP@10 = {map10_zeroshot:.4f}")

    # Load fine-tuned model
    print(f"\n--- Evaluating FINE-TUNED B16-256 (near-miss triplets) ---")
    ckpt_path = MODEL_DIR / "model.pt"
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval().to(DEVICE)

    map10_finetuned = encode_and_eval(model, preprocess, tokenizer, images, query_cats, cat_to_indices, "B16_256_finetuned")
    print(f"    Fine-tuned MAP@10 = {map10_finetuned:.4f}")

    del model
    gc.collect()

    # Load FashionSigLIP for comparison
    print(f"\n--- Evaluating FASHIONSIGLIP (target to beat) ---")
    from huggingface_hub import hf_hub_download
    model_fsl, _, preprocess_fsl = open_clip.create_model_and_transforms("ViT-B-16-SigLIP", pretrained="webli")
    ckpt = hf_hub_download("Marqo/marqo-fashionSigLIP", filename="open_clip_pytorch_model.bin")
    fsl_state = torch.load(ckpt, map_location="cpu", weights_only=True)
    model_fsl.load_state_dict(fsl_state)
    model_fsl.eval().to(DEVICE)
    tokenizer_fsl = open_clip.get_tokenizer("ViT-B-16-SigLIP")

    map10_fsl = encode_and_eval(model_fsl, preprocess_fsl, tokenizer_fsl, images, query_cats, cat_to_indices, "FSL")
    print(f"    FSL MAP@10 = {map10_fsl:.4f}")

    del model_fsl
    gc.collect()

    # Final verdict
    print(f"\n{'='*60}")
    print("VALIDATION RESULTS — Held-out 5K corpus (zero overlap with training)")
    print(f"{'='*60}")
    print(f"\n  {'Model':<40} {'MAP@10':>8} {'vs FSL':>10}")
    print(f"  {'-'*60}")
    print(f"  {'FashionSigLIP (203M, 224px)':<40} {map10_fsl:>8.4f} {'—':>10}")
    print(f"  {'B16-256 zero-shot (375M)':<40} {map10_zeroshot:>8.4f} {(map10_zeroshot-map10_fsl)/map10_fsl*100:>+9.1f}%")
    print(f"  {'B16-256 fine-tuned (375M)':<40} {map10_finetuned:>8.4f} {(map10_finetuned-map10_fsl)/map10_fsl*100:>+9.1f}%")

    delta_ft_vs_zs = (map10_finetuned - map10_zeroshot) / map10_zeroshot * 100
    print(f"\n  Fine-tune improvement over zero-shot: {delta_ft_vs_zs:+.1f}%")

    if map10_finetuned > map10_fsl:
        print(f"\n  VERDICT: Fine-tuned model BEATS FashionSigLIP on held-out data!")
        print(f"  The improvement generalizes — NOT overfitting.")
    elif map10_finetuned > map10_zeroshot:
        print(f"\n  VERDICT: Fine-tuning helps (+{delta_ft_vs_zs:.1f}%) but still below FSL on held-out.")
        print(f"  Partial generalization — some overfitting to training corpus.")
    else:
        print(f"\n  VERDICT: Fine-tuning does NOT generalize to held-out data.")
        print(f"  The 1K corpus improvement was overfitting.")

    # Save
    results = {
        "holdout_size": len(corpus_meta),
        "holdout_queries": len(query_cats),
        "map10_zeroshot": map10_zeroshot,
        "map10_finetuned": map10_finetuned,
        "map10_fsl": map10_fsl,
        "beats_fsl": map10_finetuned > map10_fsl,
        "generalizes": map10_finetuned > map10_zeroshot,
    }
    with open(CACHE_DIR / "validation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[Results saved to {CACHE_DIR / 'validation_results.json'}]")


if __name__ == "__main__":
    main()
