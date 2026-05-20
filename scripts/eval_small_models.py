"""Evaluate small SigLIP-2 models (B/16 at 224px and 256px) on fashion200k.

Uses same 1K corpus and category-based eval as ensemble experiments.
Caches all embeddings for reuse.
"""
import random, torch, gc, json, time
from collections import defaultdict
from pathlib import Path
import torch.nn.functional as F

CACHE_DIR = Path(__file__).parent.parent / "cache" / "ensemble_1k"

MODELS = {
    "B16_224": ("ViT-B-16-SigLIP2", "webli"),
    "B16_256": ("ViT-B-16-SigLIP2-256", "webli"),
    "B32_256": ("ViT-B-32-SigLIP2-256", "webli"),
}

CORPUS_SIZE = 1000


def load_corpus_meta():
    with open(CACHE_DIR / "corpus_meta.json") as f:
        return json.load(f)


def build_queries(corpus):
    cat_to_indices = defaultdict(list)
    for idx, c in enumerate(corpus):
        cat_to_indices[c["category"]].append(idx)
    valid_cats = [cat for cat, idxs in cat_to_indices.items() if len(idxs) >= 2]
    rng = random.Random(42)
    rng.shuffle(valid_cats)
    return valid_cats[:min(200, len(valid_cats))], cat_to_indices


def get_cached_scores(model_key):
    path = CACHE_DIR / f"scores_{model_key}.pt"
    if path.exists():
        return torch.load(path, map_location="cpu", weights_only=True)
    return None


def encode_and_cache(model_key, query_cats, device):
    """Encode images and texts, compute scores, cache to disk."""
    import open_clip
    from datasets import load_dataset

    arch, pretrained = MODELS[model_key]
    print(f"    Loading model {arch} (pretrained={pretrained})...")
    model, _, preprocess = open_clip.create_model_and_transforms(arch, pretrained=pretrained)
    model.eval().to(device)
    tokenizer = open_clip.get_tokenizer(arch)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"    Parameters: {n_params/1e6:.1f}M")

    # Load images (same order as corpus_meta.json)
    print(f"    Loading fashion200k images (streaming)...")
    ds = load_dataset("Marqo/fashion200k", split="data", streaming=True)
    images = []
    for row in ds:
        if len(images) >= CORPUS_SIZE:
            break
        cat = row.get("category3", "") or ""
        if row.get("image") and cat.strip():
            images.append(row["image"])
    rng = random.Random(42)
    indices = list(range(len(images)))
    rng.shuffle(indices)
    images = [images[i] for i in indices[:CORPUS_SIZE]]

    # Encode images
    print(f"    Encoding {len(images)} images...")
    img_feats = []
    with torch.no_grad():
        for i in range(0, len(images), 32):
            batch_imgs = images[i:i+32]
            tensors = torch.stack([preprocess(im.convert("RGB")) for im in batch_imgs]).to(device)
            f = model.encode_image(tensors)
            f = F.normalize(f, dim=-1)
            img_feats.append(f.cpu())
            del tensors, f
            if device == "cuda":
                torch.cuda.empty_cache()
    img_feats = torch.cat(img_feats, dim=0)

    # Encode texts (raw category names)
    print(f"    Encoding {len(query_cats)} text queries...")
    txt_feats = []
    with torch.no_grad():
        for i in range(0, len(query_cats), 64):
            tokens = tokenizer(query_cats[i:i+64]).to(device)
            f = model.encode_text(tokens)
            f = F.normalize(f, dim=-1)
            txt_feats.append(f.cpu())
            del tokens, f
    txt_feats = torch.cat(txt_feats, dim=0)

    scores = txt_feats @ img_feats.T

    # Cache
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(scores, CACHE_DIR / f"scores_{model_key}.pt")

    # Also save param count
    meta_path = CACHE_DIR / f"meta_{model_key}.json"
    with open(meta_path, "w") as f:
        json.dump({"arch": arch, "pretrained": pretrained, "n_params": n_params}, f)

    del model, tokenizer, img_feats, txt_feats, images
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    return scores, n_params


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


def main():
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")

    corpus = load_corpus_meta()
    query_cats, cat_to_indices = build_queries(corpus)
    print(f"Corpus: {len(corpus)}, Queries: {len(query_cats)}")

    # Also load existing cached baselines for comparison
    baselines = {}
    for bk in ["B16", "L16", "SO400M", "FSL"]:
        cached = get_cached_scores(bk)
        if cached is not None:
            baselines[bk] = compute_map10(cached, query_cats, cat_to_indices)

    print(f"\n--- Baselines (from cache) ---")
    baseline_names = {"B16": "SigLIP-2 B/16-384 (375M)", "L16": "SigLIP-2 L/16-384 (882M)",
                      "SO400M": "SO400M-14-SigLIP2-378 (1136M)", "FSL": "FashionSigLIP (203M)"}
    for bk, score in sorted(baselines.items(), key=lambda x: x[1], reverse=True):
        print(f"  {baseline_names.get(bk, bk):40s} MAP@10 = {score:.4f}")

    fsl_score = baselines.get("FSL", 0.3222)

    # Evaluate new models
    print(f"\n{'='*60}")
    print("EVALUATING SMALL MODELS")
    print(f"{'='*60}")

    results = {}
    for key in MODELS:
        print(f"\n  [{key}]")
        cached = get_cached_scores(key)
        if cached is not None:
            print(f"    Loaded from cache")
            scores = cached
            meta_path = CACHE_DIR / f"meta_{key}.json"
            if meta_path.exists():
                with open(meta_path) as f:
                    meta = json.load(f)
                n_params = meta["n_params"]
            else:
                n_params = None
        else:
            t0 = time.time()
            scores, n_params = encode_and_cache(key, query_cats, device)
            print(f"    Encoded + cached in {time.time()-t0:.1f}s")

        map10 = compute_map10(scores, query_cats, cat_to_indices)
        delta_vs_fsl = (map10 - fsl_score) / fsl_score * 100
        results[key] = {"map10": map10, "n_params": n_params, "delta_vs_fsl": delta_vs_fsl}
        
        beats = "BEATS FSL" if map10 > fsl_score else "LOSES TO FSL"
        print(f"    MAP@10 = {map10:.4f} | vs FSL: {delta_vs_fsl:+.1f}% | {beats}")

    # Final summary
    print(f"\n{'='*60}")
    print("FINAL RESULTS — Small models vs FashionSigLIP")
    print(f"{'='*60}")
    print(f"\n  {'Model':<45} {'Params':>8} {'MAP@10':>8} {'vs FSL':>8} {'Verdict'}")
    print(f"  {'-'*90}")
    print(f"  {'FashionSigLIP (baseline)':<45} {'203M':>8} {fsl_score:>8.4f} {'—':>8} {'—'}")
    
    for key, r in sorted(results.items(), key=lambda x: x[1]["map10"], reverse=True):
        arch = MODELS[key][0]
        params_str = f"{r['n_params']/1e6:.0f}M" if r['n_params'] else "?"
        verdict = "BEATS" if r["map10"] > fsl_score else "LOSES"
        print(f"  {arch:<45} {params_str:>8} {r['map10']:>8.4f} {r['delta_vs_fsl']:>+7.1f}% {verdict}")

    # Comparison with B16-384
    b16_384 = baselines.get("B16")
    if b16_384:
        print(f"\n  {'SigLIP-2 B/16-384 (reference)':<45} {'375M':>8} {b16_384:>8.4f} {(b16_384-fsl_score)/fsl_score*100:>+7.1f}%")

    # Save results
    output = {"baselines": baselines, "small_models": results}
    with open(CACHE_DIR / "small_models_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n[Results cached to {CACHE_DIR / 'small_models_results.json'}]")


if __name__ == "__main__":
    main()
