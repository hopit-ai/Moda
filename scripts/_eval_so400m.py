"""Quick eval of SO400M SigLIP2 variants on fashion200k.
Category-based retrieval, identical methodology.
"""
import random, torch, gc
from collections import defaultdict

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

    candidates = [
        "ViT-SO400M-16-SigLIP2-384",
        "ViT-SO400M-14-SigLIP2-378",
        "ViT-SO400M-14-SigLIP2",
    ]
    results = {}
    for name in candidates:
        print(f"\nLoading {name}...")
        model, _, preprocess = open_clip.create_model_and_transforms(name, pretrained="webli")
        model.eval().cuda()
        tokenizer = open_clip.get_tokenizer(name)
        n_params = sum(p.numel() for p in model.parameters())
        map10, nc, nq = category_map10(model, preprocess, tokenizer)
        results[name] = (map10, n_params)
        print(f"  {name}: {n_params/1e6:.0f}M params, MAP@10 = {map10:.4f}")
        del model
        gc.collect()
        torch.cuda.empty_cache()

    print(f"\n{'='*60}")
    print("RESULTS — fashion200k category-based MAP@10")
    print(f"{'='*60}")
    print(f"  FashionSigLIP (baseline):     203M | MAP@10 = 0.5369")
    print(f"  SigLIP-2 B/16-384:            375M | MAP@10 = 0.3343")
    print(f"  SigLIP-2 L/16-384 (teacher):  882M | MAP@10 = 0.3773")
    for name, (score, params) in results.items():
        marker = " <-- BEATS BASELINE" if score > 0.5369 else ""
        print(f"  {name}: {params/1e6:.0f}M | MAP@10 = {score:.4f}{marker}")
