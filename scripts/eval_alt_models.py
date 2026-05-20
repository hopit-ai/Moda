"""Evaluate alternative CLIP families (~150M params) on fashion200k.

Tests: EVA02-B-16, DFN2B-B-16, MetaCLIP-B-16, MobileCLIP2 variants.
Uses same 1K corpus, caches everything.
"""
import random, torch, gc, json, time
from collections import defaultdict
from pathlib import Path
import torch.nn.functional as F

CACHE_DIR = Path(__file__).parent.parent / "cache" / "ensemble_1k"

MODELS = {
    "EVA02_B16": ("EVA02-B-16", "merged2b_s8b_b131k"),
    "DFN2B_B16": ("ViT-B-16", "dfn2b"),
    "MetaCLIP_B16": ("ViT-B-16", "metaclip_fullcc"),
    "MetaCLIP_400m_B16": ("ViT-B-16", "metaclip_400m"),
    "OpenAI_B16": ("ViT-B-16", "openai"),
    "DFN2B_L14": ("ViT-L-14", "dfn2b"),
    "MobileCLIP2_B": ("MobileCLIP2-B", "dfndr2b"),
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
    import open_clip
    from datasets import load_dataset

    arch, pretrained = MODELS[model_key]
    print(f"    Loading model {arch} (pretrained={pretrained})...")
    model, _, preprocess = open_clip.create_model_and_transforms(arch, pretrained=pretrained)
    model.eval().to(device)
    tokenizer = open_clip.get_tokenizer(arch)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"    Parameters: {n_params/1e6:.1f}M")

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

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(scores, CACHE_DIR / f"scores_{model_key}.pt")

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

    # Load FSL baseline from cache
    fsl_cached = get_cached_scores("FSL")
    fsl_score = compute_map10(fsl_cached, query_cats, cat_to_indices) if fsl_cached is not None else 0.4902
    print(f"\nFashionSigLIP baseline: MAP@10 = {fsl_score:.4f}")

    print(f"\n{'='*60}")
    print("EVALUATING ALTERNATIVE MODELS")
    print(f"{'='*60}")

    results = {}
    for key in MODELS:
        print(f"\n  [{key}]")
        cached = get_cached_scores(key)
        if cached is not None:
            print(f"    Loaded from cache")
            scores = cached
            meta_path = CACHE_DIR / f"meta_{key}.json"
            n_params = None
            if meta_path.exists():
                with open(meta_path) as f:
                    meta = json.load(f)
                n_params = meta["n_params"]
        else:
            t0 = time.time()
            try:
                scores, n_params = encode_and_cache(key, query_cats, device)
                print(f"    Encoded + cached in {time.time()-t0:.1f}s")
            except Exception as e:
                print(f"    FAILED: {e}")
                continue

        map10 = compute_map10(scores, query_cats, cat_to_indices)
        delta_vs_fsl = (map10 - fsl_score) / fsl_score * 100
        results[key] = {"map10": map10, "n_params": n_params, "delta_vs_fsl": delta_vs_fsl}

        beats = "BEATS FSL" if map10 > fsl_score else "LOSES TO FSL"
        print(f"    MAP@10 = {map10:.4f} | vs FSL: {delta_vs_fsl:+.1f}% | {beats}")

    # Final summary
    print(f"\n{'='*60}")
    print("FINAL RESULTS — Alternative models vs FashionSigLIP (203M)")
    print(f"{'='*60}")
    print(f"\n  {'Model':<45} {'Params':>8} {'MAP@10':>8} {'vs FSL':>8} {'Verdict'}")
    print(f"  {'-'*90}")
    print(f"  {'FashionSigLIP (baseline)':<45} {'203M':>8} {fsl_score:>8.4f} {'—':>8} {'—'}")

    for key, r in sorted(results.items(), key=lambda x: x[1]["map10"], reverse=True):
        arch, pretrained = MODELS[key]
        params_str = f"{r['n_params']/1e6:.0f}M" if r['n_params'] else "?"
        verdict = "BEATS" if r["map10"] > fsl_score else "LOSES"
        print(f"  {arch} ({pretrained}){' '*(35-len(arch)-len(pretrained))} {params_str:>8} {r['map10']:>8.4f} {r['delta_vs_fsl']:>+7.1f}% {verdict}")

    # Save
    output = {"fsl_baseline": fsl_score, "alt_models": results}
    with open(CACHE_DIR / "alt_models_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n[Results cached to {CACHE_DIR / 'alt_models_results.json'}]")


if __name__ == "__main__":
    main()
