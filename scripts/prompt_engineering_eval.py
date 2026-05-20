"""Prompt engineering eval: test various text prompt templates on fashion200k.

Category-based retrieval with prompt-wrapped queries.
Tests individual templates + ensemble averaging.
Supports SigLIP-2 B/16, L/16, and FashionSigLIP.
"""
import argparse, random, torch, gc, json, time
from collections import defaultdict
import torch.nn.functional as F

TEMPLATES = [
    "{}",
    "a photo of {}",
    "a fashion product photo of {}",
    "a photo of {} clothing",
    "a product image showing {}",
    "an image of {}",
    "a fashion photo of {}",
    "a close-up photo of {}",
    "{} fashion item",
    "online shopping photo of {}",
]

TEMPLATE_NAMES = [
    "raw",
    "a_photo_of",
    "fashion_product_photo",
    "photo_clothing",
    "product_image_showing",
    "an_image_of",
    "fashion_photo",
    "closeup_photo",
    "fashion_item",
    "online_shopping",
]


def load_corpus(corpus_size=5000):
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
    return corpus


def build_queries(corpus):
    cat_to_indices = defaultdict(list)
    for idx, c in enumerate(corpus):
        cat_to_indices[c["category"]].append(idx)
    valid_cats = [cat for cat, idxs in cat_to_indices.items() if len(idxs) >= 2]
    rng = random.Random(42)
    rng.shuffle(valid_cats)
    query_cats = valid_cats[:min(300, len(valid_cats))]
    return query_cats, cat_to_indices


def encode_images(model, preprocess, corpus, device):
    img_feats = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(corpus), 32):
            batch = corpus[i:i + 32]
            imgs = torch.stack([preprocess(it["image"].convert("RGB")) for it in batch]).to(device)
            feat = model.encode_image(imgs)
            feat = F.normalize(feat, dim=-1)
            img_feats.append(feat.cpu())
            del imgs, feat
            torch.cuda.empty_cache()
    return torch.cat(img_feats, dim=0)


def encode_texts_with_template(model, tokenizer, categories, template, device):
    prompts = [template.format(cat) for cat in categories]
    txt_feats = []
    with torch.no_grad():
        for i in range(0, len(prompts), 64):
            tokens = tokenizer(prompts[i:i + 64]).to(device)
            feat = model.encode_text(tokens)
            feat = F.normalize(feat, dim=-1)
            txt_feats.append(feat.cpu())
            del tokens, feat
    return torch.cat(txt_feats, dim=0)


def compute_map10(txt_feats, img_feats, query_cats, cat_to_indices):
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
    return map_at_k


def eval_model(model_key, device="cuda", corpus_size=5000):
    import open_clip

    if model_key == "b16":
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-16-SigLIP2-384", pretrained="webli")
        tokenizer = open_clip.get_tokenizer("ViT-B-16-SigLIP2-384")
        display = "SigLIP-2 B/16-384 (375M)"
    elif model_key == "l16":
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-L-16-SigLIP2-384", pretrained="webli")
        tokenizer = open_clip.get_tokenizer("ViT-L-16-SigLIP2-384")
        display = "SigLIP-2 L/16-384 (882M)"
    elif model_key == "fsl":
        from huggingface_hub import hf_hub_download
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-16-SigLIP", pretrained="webli")
        ckpt = hf_hub_download("Marqo/marqo-fashionSigLIP",
                               filename="open_clip_pytorch_model.bin")
        state = torch.load(ckpt, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        tokenizer = open_clip.get_tokenizer("ViT-B-16-SigLIP")
        display = "FashionSigLIP (203M)"
    else:
        raise ValueError(f"Unknown model: {model_key}")

    model.eval().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n{'='*60}")
    print(f"Model: {display} ({n_params/1e6:.0f}M params)")
    print(f"{'='*60}")

    # Load corpus and encode images (once per model)
    print("Loading fashion200k corpus...")
    corpus = load_corpus(corpus_size)
    query_cats, cat_to_indices = build_queries(corpus)
    print(f"  {len(corpus)} corpus images, {len(query_cats)} category queries")

    print("Encoding images...")
    img_feats = encode_images(model, preprocess, corpus, device)

    # Test each template individually
    results = {}
    print("\n--- Single template sweep ---")
    for tmpl, name in zip(TEMPLATES, TEMPLATE_NAMES):
        txt_feats = encode_texts_with_template(model, tokenizer, query_cats, tmpl, device)
        map10 = compute_map10(txt_feats, img_feats, query_cats, cat_to_indices)
        results[name] = map10
        marker = " ***" if map10 == max(results.values()) else ""
        print(f"  {name:30s} | MAP@10 = {map10:.4f}{marker}")
        del txt_feats

    # Ensemble: average embeddings from all templates
    print("\n--- Ensemble (all templates) ---")
    ensemble_feats = torch.zeros(len(query_cats), img_feats.shape[1])
    for tmpl in TEMPLATES:
        feat = encode_texts_with_template(model, tokenizer, query_cats, tmpl, device)
        ensemble_feats += feat
        del feat
    ensemble_feats = F.normalize(ensemble_feats, dim=-1)
    map10_all = compute_map10(ensemble_feats, img_feats, query_cats, cat_to_indices)
    results["ensemble_all"] = map10_all
    print(f"  {'ensemble_all':30s} | MAP@10 = {map10_all:.4f}")

    # Ensemble: top-5 templates
    sorted_templates = sorted(
        zip(TEMPLATES, TEMPLATE_NAMES, [results[n] for n in TEMPLATE_NAMES]),
        key=lambda x: x[2], reverse=True
    )
    top5 = sorted_templates[:5]
    print(f"\n--- Ensemble (top-5: {[t[1] for t in top5]}) ---")
    ensemble5_feats = torch.zeros(len(query_cats), img_feats.shape[1])
    for tmpl, name, _ in top5:
        feat = encode_texts_with_template(model, tokenizer, query_cats, tmpl, device)
        ensemble5_feats += feat
        del feat
    ensemble5_feats = F.normalize(ensemble5_feats, dim=-1)
    map10_top5 = compute_map10(ensemble5_feats, img_feats, query_cats, cat_to_indices)
    results["ensemble_top5"] = map10_top5
    print(f"  {'ensemble_top5':30s} | MAP@10 = {map10_top5:.4f}")

    # Ensemble: top-3
    top3 = sorted_templates[:3]
    print(f"\n--- Ensemble (top-3: {[t[1] for t in top3]}) ---")
    ensemble3_feats = torch.zeros(len(query_cats), img_feats.shape[1])
    for tmpl, name, _ in top3:
        feat = encode_texts_with_template(model, tokenizer, query_cats, tmpl, device)
        ensemble3_feats += feat
        del feat
    ensemble3_feats = F.normalize(ensemble3_feats, dim=-1)
    map10_top3 = compute_map10(ensemble3_feats, img_feats, query_cats, cat_to_indices)
    results["ensemble_top3"] = map10_top3
    print(f"  {'ensemble_top3':30s} | MAP@10 = {map10_top3:.4f}")

    # Summary
    best_name = max(results, key=results.get)
    best_score = results[best_name]
    raw_score = results["raw"]
    print(f"\n--- SUMMARY for {display} ---")
    print(f"  Raw (no prompt):  {raw_score:.4f}")
    print(f"  Best template:    {best_name} = {best_score:.4f} ({(best_score-raw_score)/raw_score*100:+.1f}%)")

    del model, img_feats, corpus
    gc.collect()
    torch.cuda.empty_cache()
    return results, display


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=str, default="b16",
                        help="Comma-separated: b16,l16,fsl")
    parser.add_argument("--corpus-size", type=int, default=5000)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")

    all_results = {}
    for model_key in args.models.split(","):
        model_key = model_key.strip()
        results, display = eval_model(model_key, device, args.corpus_size)
        all_results[model_key] = {"display": display, "results": results}

    # Final comparison
    print(f"\n{'='*70}")
    print("FINAL COMPARISON — fashion200k category-based MAP@10")
    print(f"{'='*70}")
    header = f"{'Template':30s}"
    for mk in all_results:
        header += f" | {all_results[mk]['display']:>20s}"
    print(header)
    print("-" * len(header))

    all_template_names = TEMPLATE_NAMES + ["ensemble_all", "ensemble_top5", "ensemble_top3"]
    for tn in all_template_names:
        row = f"{tn:30s}"
        for mk in all_results:
            score = all_results[mk]["results"].get(tn, 0)
            row += f" | {score:20.4f}"
        print(row)

    # Save to JSON
    with open("prompt_engineering_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("\nResults saved to prompt_engineering_results.json")
