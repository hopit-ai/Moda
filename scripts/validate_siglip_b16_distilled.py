"""Validate distilled SigLIP B/16 (203M) on held-out 5K corpus.

Confirms the +2.4% over FSL generalizes to unseen data (no overfitting to training 1K).
"""
import random, torch, gc, json, time
from collections import defaultdict
from pathlib import Path
import torch.nn.functional as F

CACHE_DIR = Path(__file__).parent.parent / "cache" / "validation_5k"
MODEL_DIR = Path(__file__).parent.parent / "models"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

HOLDOUT_START = 1000
HOLDOUT_SIZE = 5000


def load_holdout_data():
    """Load held-out corpus (items 1001-6000), reusing cache if available."""
    cache_meta = CACHE_DIR / "holdout_corpus_meta.json"
    if cache_meta.exists():
        with open(cache_meta) as f:
            corpus = json.load(f)
        return corpus, None

    from datasets import load_dataset
    print("  Streaming fashion200k holdout slice...")
    ds = load_dataset("Marqo/fashion200k", split="data", streaming=True)

    all_valid = []
    for row in ds:
        cat = row.get("category3", "") or ""
        if row.get("image") and cat.strip():
            all_valid.append({"category": cat.strip(), "image": row["image"]})
        if len(all_valid) >= HOLDOUT_START + HOLDOUT_SIZE:
            break

    holdout = all_valid[HOLDOUT_START:HOLDOUT_START + HOLDOUT_SIZE]
    rng = random.Random(123)
    rng.shuffle(holdout)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    meta = [{"category": h["category"]} for h in holdout]
    with open(cache_meta, "w") as f:
        json.dump(meta, f)

    images = [h["image"] for h in holdout]
    return meta, images


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


def encode_and_eval(model, preprocess, tokenizer, images, query_cats, cat_to_indices):
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
            if i % 160 == 0:
                print(f"    images: {i+32}/{len(images)}")
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


def main():
    import open_clip
    from datasets import load_dataset

    print(f"Device: {DEVICE}")
    print(f"\n{'='*60}")
    print("VALIDATION: Distilled SigLIP B/16 (203M) on held-out 5K")
    print(f"{'='*60}")

    # Load holdout
    corpus_meta, images_cache = load_holdout_data()
    print(f"  Holdout corpus: {len(corpus_meta)} items")

    cat_to_indices = defaultdict(list)
    for idx, c in enumerate(corpus_meta):
        cat_to_indices[c["category"]].append(idx)
    valid_cats = [cat for cat, idxs in cat_to_indices.items() if len(idxs) >= 2]
    rng = random.Random(99)
    rng.shuffle(valid_cats)
    query_cats = valid_cats[:min(300, len(valid_cats))]
    print(f"  Queries: {len(query_cats)}")

    # Load images if not cached
    if images_cache is None:
        print("  Loading holdout images from dataset...")
        ds = load_dataset("Marqo/fashion200k", split="data", streaming=True)
        all_valid = []
        for row in ds:
            cat = row.get("category3", "") or ""
            if row.get("image") and cat.strip():
                all_valid.append(row["image"])
            if len(all_valid) >= HOLDOUT_START + HOLDOUT_SIZE:
                break
        holdout_imgs = all_valid[HOLDOUT_START:HOLDOUT_START + HOLDOUT_SIZE]
        rng2 = random.Random(123)
        indices = list(range(len(holdout_imgs)))
        rng2.shuffle(indices)
        images = [holdout_imgs[i] for i in indices]
    else:
        images = images_cache

    # --- Evaluate FashionSigLIP ---
    print("\n[1] Evaluating FashionSigLIP (baseline, 203M)...")
    fsl, _, fsl_pre = open_clip.create_model_and_transforms(
        "ViT-B-16-SigLIP", pretrained="webli"
    )
    fsl_state = torch.load(MODEL_DIR / "marqo-fashionSigLIP" / "open_clip_pytorch_model.bin",
                           map_location="cpu", weights_only=True)
    fsl.load_state_dict(fsl_state)
    fsl.eval().to(DEVICE)
    fsl_tok = open_clip.get_tokenizer("ViT-B-16-SigLIP")
    map10_fsl = encode_and_eval(fsl, fsl_pre, fsl_tok, images, query_cats, cat_to_indices)
    print(f"  FSL MAP@10 = {map10_fsl:.4f}")
    del fsl
    gc.collect()

    # --- Evaluate zero-shot SigLIP B/16 ---
    print("\n[2] Evaluating SigLIP B/16 zero-shot (203M)...")
    zs, _, zs_pre = open_clip.create_model_and_transforms("ViT-B-16-SigLIP", pretrained="webli")
    zs.eval().to(DEVICE)
    zs_tok = open_clip.get_tokenizer("ViT-B-16-SigLIP")
    map10_zs = encode_and_eval(zs, zs_pre, zs_tok, images, query_cats, cat_to_indices)
    print(f"  Zero-shot MAP@10 = {map10_zs:.4f}")
    del zs
    gc.collect()

    # --- Evaluate distilled model ---
    print("\n[3] Evaluating distilled SigLIP B/16 (203M)...")
    dist, _, dist_pre = open_clip.create_model_and_transforms("ViT-B-16-SigLIP", pretrained="webli")
    dist_state = torch.load(MODEL_DIR / "siglip-b16-distilled" / "model.pt",
                            map_location="cpu", weights_only=True)
    dist.load_state_dict(dist_state)
    dist.eval().to(DEVICE)
    dist_tok = open_clip.get_tokenizer("ViT-B-16-SigLIP")
    map10_dist = encode_and_eval(dist, dist_pre, dist_tok, images, query_cats, cat_to_indices)
    print(f"  Distilled MAP@10 = {map10_dist:.4f}")
    del dist
    gc.collect()

    # --- Summary ---
    print(f"\n{'='*60}")
    print("HELD-OUT VALIDATION RESULTS (5K corpus)")
    print(f"{'='*60}")
    print(f"\n  {'Model':<40} {'MAP@10':>8} {'vs FSL':>10}")
    print(f"  {'-'*60}")
    print(f"  {'FashionSigLIP (203M)':<40} {map10_fsl:>8.4f} {'—':>10}")
    print(f"  {'SigLIP B/16 zero-shot (203M)':<40} {map10_zs:>8.4f} {(map10_zs-map10_fsl)/map10_fsl*100:>+9.1f}%")
    print(f"  {'SigLIP B/16 distilled (203M)':<40} {map10_dist:>8.4f} {(map10_dist-map10_fsl)/map10_fsl*100:>+9.1f}%")

    beats = map10_dist > map10_fsl
    print(f"\n  {'BEATS FSL ON HELD-OUT DATA!' if beats else 'Does NOT beat FSL on held-out data.'}")

    results = {
        "map10_fsl": map10_fsl,
        "map10_zeroshot": map10_zs,
        "map10_distilled": map10_dist,
        "beats_fsl": beats,
        "delta_vs_fsl_pct": (map10_dist - map10_fsl) / map10_fsl * 100,
        "n_corpus": len(corpus_meta),
        "n_queries": len(query_cats),
    }
    with open(CACHE_DIR / "distilled_b16_validation.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  [Saved to {CACHE_DIR / 'distilled_b16_validation.json'}]")


if __name__ == "__main__":
    main()
