"""Per-query failure analysis across 4 models on fashion200k 1K corpus.

First run: encodes all models and caches embeddings to disk.
Subsequent runs: loads from cache instantly.

Output: For each query, shows which models failed and the confidence
(score gap between top retrieved item and best positive item).
"""
import os, random, torch, gc, json
from collections import defaultdict
from pathlib import Path
import torch.nn.functional as F

CACHE_DIR = Path(__file__).parent.parent / "cache" / "ensemble_1k"

MODELS = {
    "B16": ("ViT-B-16-SigLIP2-384", "webli", None),
    "L16": ("ViT-L-16-SigLIP2-384", "webli", None),
    "SO400M": ("ViT-SO400M-14-SigLIP2-378", "webli", None),
    "FSL": ("ViT-B-16-SigLIP", "webli", "Marqo/marqo-fashionSigLIP"),
}

CORPUS_SIZE = 1000


def load_corpus():
    """Load 1K corpus (deterministic, same as ensemble_4model_eval.py)."""
    cache_meta = CACHE_DIR / "corpus_meta.json"
    if cache_meta.exists():
        with open(cache_meta) as f:
            return json.load(f)

    from datasets import load_dataset
    ds = load_dataset("Marqo/fashion200k", split="data", streaming=True)
    corpus = []
    for row in ds:
        if len(corpus) >= CORPUS_SIZE:
            break
        cat = row.get("category3", "") or ""
        if row.get("image") and cat.strip():
            corpus.append({"category": cat.strip(), "idx": len(corpus)})

    rng = random.Random(42)
    rng.shuffle(corpus)
    corpus = corpus[:CORPUS_SIZE]

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(cache_meta, "w") as f:
        json.dump(corpus, f)
    return corpus


def build_queries(corpus):
    cat_to_indices = defaultdict(list)
    for idx, c in enumerate(corpus):
        cat_to_indices[c["category"]].append(idx)
    valid_cats = [cat for cat, idxs in cat_to_indices.items() if len(idxs) >= 2]
    rng = random.Random(42)
    rng.shuffle(valid_cats)
    return valid_cats[:min(200, len(valid_cats))], cat_to_indices


def get_cached_scores(model_key):
    """Return score matrix [n_queries x n_docs] from cache, or None."""
    path = CACHE_DIR / f"scores_{model_key}.pt"
    if path.exists():
        return torch.load(path, map_location="cpu", weights_only=True)
    return None


def encode_and_cache(model_key, query_cats, corpus, device):
    """Encode images and texts, compute scores, cache to disk."""
    import open_clip
    from huggingface_hub import hf_hub_download

    arch, pretrained, hf_ckpt = MODELS[model_key]
    model, _, preprocess = open_clip.create_model_and_transforms(arch, pretrained=pretrained)
    if hf_ckpt:
        ckpt = hf_hub_download(hf_ckpt, filename="open_clip_pytorch_model.bin")
        state = torch.load(ckpt, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
    model.eval().to(device)
    tokenizer = open_clip.get_tokenizer(arch)

    # Need actual images for encoding
    from datasets import load_dataset
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

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(scores, CACHE_DIR / f"scores_{model_key}.pt")

    del model, tokenizer, img_feats, txt_feats, images
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    return scores


def per_query_analysis(all_scores, query_cats, cat_to_indices):
    """
    For each query, compute per-model:
    - AP@10
    - confidence = avg score of top-10 retrieved items
    - failure_gap = score(top-1 retrieved) - score(best positive in top-10)
    
    Sort queries by "hardness" (how many models fail on it).
    """
    k = 10
    results = []

    for qi, cat in enumerate(query_cats):
        positives = set(cat_to_indices[cat])
        n_pos = min(len(positives), k)
        query_result = {
            "query": cat,
            "n_positives_in_corpus": len(positives),
            "models": {},
        }

        for model_key, scores in all_scores.items():
            topk_vals, topk_idx = scores[qi].topk(min(k, scores.shape[1]))
            topk_idx = topk_idx.tolist()
            topk_vals = topk_vals.tolist()

            # AP@10
            ap, n_rel = 0.0, 0
            for rank, idx in enumerate(topk_idx, 1):
                if idx in positives:
                    n_rel += 1
                    ap += n_rel / rank
            if n_pos > 0:
                ap /= n_pos

            # Precision@10
            hits = sum(1 for idx in topk_idx if idx in positives)
            p_at_10 = hits / k

            # Confidence: score of rank-1 item
            top1_score = topk_vals[0]

            # Score of best positive in top-10 (0 if none found)
            best_pos_score = 0.0
            for rank, idx in enumerate(topk_idx):
                if idx in positives:
                    best_pos_score = topk_vals[rank]
                    break

            # Rank of first positive
            first_pos_rank = None
            for rank, idx in enumerate(topk_idx, 1):
                if idx in positives:
                    first_pos_rank = rank
                    break

            query_result["models"][model_key] = {
                "ap10": round(ap, 4),
                "p10": round(p_at_10, 4),
                "top1_score": round(top1_score, 4),
                "best_pos_score_in_top10": round(best_pos_score, 4),
                "first_pos_rank": first_pos_rank,
                "hits_in_top10": hits,
            }

        # Aggregate: how many models get AP@10 = 0?
        n_models_failed = sum(
            1 for m in query_result["models"].values() if m["ap10"] == 0
        )
        avg_ap = sum(m["ap10"] for m in query_result["models"].values()) / len(all_scores)
        query_result["n_models_failed"] = n_models_failed
        query_result["avg_ap10"] = round(avg_ap, 4)
        results.append(query_result)

    return results


def print_report(results, all_scores, query_cats, cat_to_indices):
    # Sort by hardness (most models failed, then lowest avg AP)
    results.sort(key=lambda r: (-r["n_models_failed"], r["avg_ap10"]))

    model_keys = list(all_scores.keys())

    # Global stats
    print("=" * 80)
    print("FAILURE ANALYSIS — fashion200k 1K corpus, category-based retrieval")
    print("=" * 80)

    # Per-model MAP@10
    print("\n--- Per-model MAP@10 ---")
    for mk in model_keys:
        map10 = sum(r["models"][mk]["ap10"] for r in results) / len(results)
        print(f"  {mk:10s}: {map10:.4f}")

    # Failure distribution
    print("\n--- Query failure distribution ---")
    for n_fail in range(len(model_keys) + 1):
        count = sum(1 for r in results if r["n_models_failed"] == n_fail)
        if count > 0:
            pct = count / len(results) * 100
            print(f"  {n_fail} models failed: {count:3d} queries ({pct:.1f}%)")

    # Hardest queries (all models fail)
    print("\n--- HARDEST QUERIES (all 4 models get AP@10=0) ---")
    all_fail = [r for r in results if r["n_models_failed"] == len(model_keys)]
    for r in all_fail[:20]:
        conf = max(r["models"][mk]["top1_score"] for mk in model_keys)
        print(f"  \"{r['query']}\" (n_pos={r['n_positives_in_corpus']}, "
              f"best_top1_conf={conf:.3f})")

    # Queries where ONLY FSL fails but SigLIP-2 models succeed
    print("\n--- QUERIES WHERE ONLY FSL FAILS (SigLIP-2 models succeed) ---")
    fsl_only_fail = [
        r for r in results
        if r["models"]["FSL"]["ap10"] == 0
        and all(r["models"][mk]["ap10"] > 0 for mk in ["B16", "L16", "SO400M"])
    ]
    fsl_only_fail.sort(key=lambda r: -r["models"]["FSL"]["top1_score"])
    for r in fsl_only_fail[:15]:
        print(f"  \"{r['query']}\" | FSL conf={r['models']['FSL']['top1_score']:.3f} "
              f"| B16 AP={r['models']['B16']['ap10']:.2f} "
              f"| L16 AP={r['models']['L16']['ap10']:.2f} "
              f"| SO400M AP={r['models']['SO400M']['ap10']:.2f}")

    # Queries where ONLY SigLIP-2 B16 fails
    print("\n--- QUERIES WHERE ONLY B16 FAILS (others succeed) ---")
    b16_only_fail = [
        r for r in results
        if r["models"]["B16"]["ap10"] == 0
        and r["models"]["L16"]["ap10"] > 0
        and r["models"]["SO400M"]["ap10"] > 0
    ]
    b16_only_fail.sort(key=lambda r: -r["models"]["B16"]["top1_score"])
    for r in b16_only_fail[:15]:
        print(f"  \"{r['query']}\" | B16 conf={r['models']['B16']['top1_score']:.3f} "
              f"| L16 AP={r['models']['L16']['ap10']:.2f} "
              f"| SO400M AP={r['models']['SO400M']['ap10']:.2f} "
              f"| FSL AP={r['models']['FSL']['ap10']:.2f}")

    # High-confidence failures (model is confident but wrong)
    print("\n--- HIGH-CONFIDENCE FAILURES (per model, sorted by top1_score desc) ---")
    for mk in model_keys:
        failures = [
            r for r in results if r["models"][mk]["ap10"] == 0
        ]
        failures.sort(key=lambda r: -r["models"][mk]["top1_score"])
        print(f"\n  {mk} — {len(failures)} total failures:")
        for r in failures[:10]:
            m = r["models"][mk]
            print(f"    \"{r['query']}\" | top1_conf={m['top1_score']:.3f} "
                  f"| n_pos={r['n_positives_in_corpus']}")

    # Easy queries (all models get perfect AP@10)
    perfect = [r for r in results if all(r["models"][mk]["ap10"] >= 0.99 for mk in model_keys)]
    print(f"\n--- EASY QUERIES (all models AP@10 ≥ 0.99): {len(perfect)} ---")
    for r in perfect[:10]:
        print(f"  \"{r['query']}\" (n_pos={r['n_positives_in_corpus']})")

    # Save full results to JSON
    output_path = CACHE_DIR / "failure_analysis.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[Full results saved to {output_path}]")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")

    # Load corpus metadata
    corpus_meta = load_corpus()
    query_cats, cat_to_indices = build_queries(corpus_meta)
    print(f"Corpus: {len(corpus_meta)}, Queries: {len(query_cats)}")

    # Load or compute scores for each model
    all_scores = {}
    for key in MODELS:
        cached = get_cached_scores(key)
        if cached is not None:
            print(f"  {key}: loaded from cache")
            all_scores[key] = cached
        else:
            print(f"  {key}: encoding (will cache for next time)...")
            all_scores[key] = encode_and_cache(key, query_cats, corpus_meta, device)
            print(f"  {key}: cached to {CACHE_DIR / f'scores_{key}.pt'}")

    # Run analysis
    results = per_query_analysis(all_scores, query_cats, cat_to_indices)
    print_report(results, all_scores, query_cats, cat_to_indices)
