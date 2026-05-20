"""Honest apples-to-apples comparison: all models on fashion200k.
Category-based retrieval, 5K corpus, 300 queries. Identical methodology.
"""
import random, torch, gc
from collections import defaultdict

MODELS = [
    # (display_name, open_clip_name, pretrained, param_count_approx)
    ("FashionSigLIP (baseline)", "ViT-B-16-SigLIP", None, "203M"),
    ("SigLIP-2 B/16-384 (student candidate)", "ViT-B-16-SigLIP2-384", "webli", "375M"),
    ("SigLIP-2 L/16-384 (teacher)", "ViT-L-16-SigLIP2-384", "webli", "882M"),
]

def category_map10(model, preprocess, tokenizer, device="cuda", corpus_size=5000):
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
    corpus = corpus[:corpus_size]

    cat_to_indices = defaultdict(list)
    for idx, c in enumerate(corpus):
        cat_to_indices[c["category"]].append(idx)

    valid_cats = [cat for cat, idxs in cat_to_indices.items() if len(idxs) >= 2]
    rng.shuffle(valid_cats)
    query_cats = valid_cats[:min(300, len(valid_cats))]

    img_feats = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(corpus), 32):
            batch = corpus[i:i+32]
            imgs = torch.stack([preprocess(it["image"].convert("RGB")) for it in batch]).to(device)
            feat = model.encode_image(imgs)
            feat = feat / feat.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            img_feats.append(feat.cpu())
            del imgs, feat
            torch.cuda.empty_cache()
    img_feats = torch.cat(img_feats, dim=0)

    txt_feats = []
    with torch.no_grad():
        for i in range(0, len(query_cats), 64):
            tokens = tokenizer(query_cats[i:i+64]).to(device)
            feat = model.encode_text(tokens)
            feat = feat / feat.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            txt_feats.append(feat.cpu())
            del tokens, feat
    txt_feats = torch.cat(txt_feats, dim=0)

    scores = txt_feats @ img_feats.T
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
    map_at_k /= max(len(query_cats), 1)
    return map_at_k, len(corpus), len(query_cats)


if __name__ == "__main__":
    import open_clip

    results = {}
    for display, model_name, pretrained, params in MODELS:
        print(f"\n{'='*50}")
        print(f"Loading: {display} ({params})")
        print(f"{'='*50}")

        if pretrained is None:
            from huggingface_hub import hf_hub_download
            model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained="webli")
            ckpt = hf_hub_download("Marqo/marqo-fashionSigLIP", filename="open_clip_pytorch_model.bin")
            state = torch.load(ckpt, map_location="cpu", weights_only=True)
            model.load_state_dict(state)
        else:
            model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)

        model.eval().cuda()
        tokenizer = open_clip.get_tokenizer(model_name)

        map10, n_corpus, n_queries = category_map10(model, preprocess, tokenizer)
        results[display] = (map10, params)
        print(f"  -> MAP@10 = {map10:.4f} ({n_corpus} corpus, {n_queries} queries)")

        del model
        gc.collect()
        torch.cuda.empty_cache()

    print(f"\n{'='*60}")
    print(f"FINAL COMPARISON — fashion200k category-based MAP@10")
    print(f"{'='*60}")
    for name, (score, params) in results.items():
        print(f"  {name:45s} | {params:>5s} | MAP@10 = {score:.4f}")
    print(f"{'='*60}")
