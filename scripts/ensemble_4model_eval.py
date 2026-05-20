"""4-model ensemble eval on fashion200k (1K corpus for speed).

M1: SigLIP-2 B/16-384 (375M)
M2: SigLIP-2 L/16-384 (882M)
M3: SO400M-14-SigLIP2-378 (1136M)
M4: FashionSigLIP (203M)

Ensemble strategies:
- Individual baselines
- Score-mean (all 4, all 3 SigLIP2, best pairs)
- Reciprocal Rank Fusion (RRF)
- Weighted score-mean (weight by individual MAP@10)
"""
import random, torch, gc, itertools
from collections import defaultdict
import torch.nn.functional as F

MODELS = {
    "M1_B16": ("ViT-B-16-SigLIP2-384", "webli", None),
    "M2_L16": ("ViT-L-16-SigLIP2-384", "webli", None),
    "M3_SO400M": ("ViT-SO400M-14-SigLIP2-378", "webli", None),
    "M4_FSL": ("ViT-B-16-SigLIP", "webli", "Marqo/marqo-fashionSigLIP"),
}


def load_model(key, device="cuda"):
    import open_clip
    arch, pretrained, hf_ckpt = MODELS[key]
    model, _, preprocess = open_clip.create_model_and_transforms(arch, pretrained=pretrained)
    if hf_ckpt:
        from huggingface_hub import hf_hub_download
        ckpt = hf_hub_download(hf_ckpt, filename="open_clip_pytorch_model.bin")
        state = torch.load(ckpt, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
    model.eval().to(device)
    tokenizer = open_clip.get_tokenizer(arch)
    n_params = sum(p.numel() for p in model.parameters())
    return model, preprocess, tokenizer, n_params


def load_corpus(corpus_size=1000):
    from datasets import load_dataset
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
    return corpus[:corpus_size]


def build_queries(corpus):
    cat_to_indices = defaultdict(list)
    for idx, c in enumerate(corpus):
        cat_to_indices[c["category"]].append(idx)
    valid_cats = [cat for cat, idxs in cat_to_indices.items() if len(idxs) >= 2]
    rng = random.Random(42)
    rng.shuffle(valid_cats)
    return valid_cats[:min(200, len(valid_cats))], cat_to_indices


def encode_images(model, preprocess, corpus, device):
    feats = []
    with torch.no_grad():
        for i in range(0, len(corpus), 32):
            batch = corpus[i:i+32]
            imgs = torch.stack([preprocess(it["image"].convert("RGB")) for it in batch]).to(device)
            f = model.encode_image(imgs)
            f = F.normalize(f, dim=-1)
            feats.append(f.cpu())
            del imgs, f
            torch.cuda.empty_cache()
    return torch.cat(feats, dim=0)


def encode_texts(model, tokenizer, categories, device):
    feats = []
    with torch.no_grad():
        for i in range(0, len(categories), 64):
            tokens = tokenizer(categories[i:i+64]).to(device)
            f = model.encode_text(tokens)
            f = F.normalize(f, dim=-1)
            feats.append(f.cpu())
            del tokens, f
    return torch.cat(feats, dim=0)


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


def rrf_scores(score_matrices, k_param=60):
    """Reciprocal Rank Fusion across multiple score matrices."""
    n_queries, n_docs = score_matrices[0].shape
    fused = torch.zeros(n_queries, n_docs)
    for scores in score_matrices:
        ranks = scores.argsort(dim=-1, descending=True).argsort(dim=-1) + 1  # 1-indexed ranks
        fused += 1.0 / (k_param + ranks.float())
    return fused


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")

    # Load corpus once
    print("Loading fashion200k (1K corpus)...")
    corpus = load_corpus(1000)
    query_cats, cat_to_indices = build_queries(corpus)
    print(f"  {len(corpus)} corpus, {len(query_cats)} queries")

    # Encode with each model
    all_scores = {}
    individual_map = {}

    for key in MODELS:
        print(f"\nLoading {key}...")
        model, preprocess, tokenizer, n_params = load_model(key, device)
        print(f"  {n_params/1e6:.0f}M params")

        img_feats = encode_images(model, preprocess, corpus, device)
        txt_feats = encode_texts(model, tokenizer, query_cats, device)
        scores = txt_feats @ img_feats.T
        all_scores[key] = scores

        map10 = compute_map10(scores, query_cats, cat_to_indices)
        individual_map[key] = map10
        print(f"  {key}: MAP@10 = {map10:.4f}")

        del model, preprocess, tokenizer, img_feats, txt_feats
        gc.collect()
        torch.cuda.empty_cache()

    # --- Ensemble experiments ---
    print(f"\n{'='*60}")
    print("ENSEMBLE EXPERIMENTS")
    print(f"{'='*60}")

    results = {}
    for k, v in individual_map.items():
        results[k] = v

    # All pairs
    keys = list(MODELS.keys())
    for i, j in itertools.combinations(range(len(keys)), 2):
        k1, k2 = keys[i], keys[j]
        name = f"mean({k1}+{k2})"
        fused = (all_scores[k1] + all_scores[k2]) / 2
        results[name] = compute_map10(fused, query_cats, cat_to_indices)

    # All triples
    for i, j, k in itertools.combinations(range(len(keys)), 3):
        k1, k2, k3 = keys[i], keys[j], keys[k]
        name = f"mean({k1}+{k2}+{k3})"
        fused = (all_scores[k1] + all_scores[k2] + all_scores[k3]) / 3
        results[name] = compute_map10(fused, query_cats, cat_to_indices)

    # All 4
    name = "mean(ALL4)"
    fused = sum(all_scores[k] for k in keys) / 4
    results[name] = compute_map10(fused, query_cats, cat_to_indices)

    # Weighted mean (weight by individual MAP@10)
    total_w = sum(individual_map.values())
    fused_w = sum(all_scores[k] * (individual_map[k] / total_w) for k in keys)
    results["weighted_mean(ALL4)"] = compute_map10(fused_w, query_cats, cat_to_indices)

    # SigLIP-2 only (no FSL)
    sl2_keys = ["M1_B16", "M2_L16", "M3_SO400M"]
    fused_sl2 = sum(all_scores[k] for k in sl2_keys) / 3
    results["mean(SigLIP2_only)"] = compute_map10(fused_sl2, query_cats, cat_to_indices)

    # RRF variants
    rrf_all = rrf_scores([all_scores[k] for k in keys])
    results["RRF(ALL4)"] = compute_map10(rrf_all, query_cats, cat_to_indices)

    rrf_sl2 = rrf_scores([all_scores[k] for k in sl2_keys])
    results["RRF(SigLIP2_only)"] = compute_map10(rrf_sl2, query_cats, cat_to_indices)

    # Best pair + RRF
    for i, j in itertools.combinations(range(len(keys)), 2):
        k1, k2 = keys[i], keys[j]
        name = f"RRF({k1}+{k2})"
        fused = rrf_scores([all_scores[k1], all_scores[k2]])
        results[name] = compute_map10(fused, query_cats, cat_to_indices)

    # Print sorted results
    print(f"\n{'='*60}")
    print("ALL RESULTS — fashion200k MAP@10 (1K corpus)")
    print(f"{'='*60}")
    for name, score in sorted(results.items(), key=lambda x: x[1], reverse=True):
        delta = (score - individual_map["M4_FSL"]) / individual_map["M4_FSL"] * 100
        marker = " <-- BEST" if score == max(results.values()) else ""
        print(f"  {name:45s} | MAP@10 = {score:.4f} | vs FSL: {delta:+.1f}%{marker}")
