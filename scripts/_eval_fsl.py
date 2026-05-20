"""Zero-shot baseline eval for the fashion-specific baseline model on fashion200k.
Category-based retrieval, identical methodology to _eval_zeroshot.py.
"""
import random, time, torch, gc
from collections import defaultdict

def eval_model(device="cuda", corpus_size=5000):
    import open_clip
    from huggingface_hub import hf_hub_download

    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-L-16-SigLIP-384", pretrained="webli"
    )
    ckpt_path = hf_hub_download("Marqo/marqo-fashionSigLIP", filename="open_clip_pytorch_model.bin")
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval().to(device)
    tokenizer = open_clip.get_tokenizer("ViT-L-16-SigLIP-384")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nEvaluating fashion baseline: {n_params/1e6:.0f}M params")

    from datasets import load_dataset
    ds = load_dataset("Marqo/fashion200k", split="data", streaming=True)

    corpus = []
    for row in ds:
        if len(corpus) >= corpus_size:
            break
        cat = row.get("category3", "") or ""
        if row.get("image") and cat.strip():
            corpus.append({"image": row["image"], "category": cat.strip(),
                           "item_id": str(row.get("item_ID", len(corpus)))})

    rng = random.Random(42)
    rng.shuffle(corpus)
    corpus = corpus[:corpus_size]

    cat_to_indices = defaultdict(list)
    for idx, c in enumerate(corpus):
        cat_to_indices[c["category"]].append(idx)

    valid_cats = [cat for cat, idxs in cat_to_indices.items() if len(idxs) >= 2]
    rng.shuffle(valid_cats)
    query_cats = valid_cats[:min(300, len(valid_cats))]
    print(f"  {len(corpus)} corpus, {len(query_cats)} category queries")

    img_feats = []
    with torch.no_grad():
        for i in range(0, len(corpus), 32):
            batch = corpus[i:i+32]
            imgs = torch.stack([preprocess(it["image"].convert("RGB")) for it in batch]).to(device)
            feat = model.encode_image(imgs)
            feat = feat / feat.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            img_feats.append(feat.cpu())
            del imgs, feat
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
    print(f"  fashion200k MAP@10 = {map_at_k:.4f}")
    return map_at_k

if __name__ == "__main__":
    v = eval_model()
    print(f"\n  Fashion baseline: {v:.4f}")
