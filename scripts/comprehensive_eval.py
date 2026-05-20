"""Comprehensive evaluation of distilled SigLIP B/16 (203M) vs FashionSigLIP.

Tests:
1. Bootstrap significance test on 5K holdout (is +1.4% statistically significant?)
2. Larger 10K holdout evaluation 
3. Cross-benchmark: DeepFashion (category retrieval)
4. Cross-benchmark: FashionIQ (category retrieval)

Goal: confirm the win is real and generalizes beyond fashion200k.
"""
import random, torch, gc, json, time, numpy as np
from collections import defaultdict
from pathlib import Path
import torch.nn.functional as F

CACHE_DIR = Path(__file__).parent.parent / "cache"
MODEL_DIR = Path(__file__).parent.parent / "models"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def compute_map10(scores, query_cats, cat_to_indices):
    k = 10
    aps = []
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
        aps.append(ap)
    return aps


def encode_images(model, preprocess, images, batch_size=32):
    model.eval()
    feats = []
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size]
            tensors = torch.stack([preprocess(im.convert("RGB")) for im in batch]).to(DEVICE)
            f = model.encode_image(tensors)
            f = F.normalize(f, dim=-1)
            feats.append(f.cpu())
            del tensors, f
    return torch.cat(feats, dim=0)


def encode_texts(model, tokenizer, texts, batch_size=64):
    model.eval()
    feats = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            tokens = tokenizer(texts[i:i+batch_size]).to(DEVICE)
            f = model.encode_text(tokens)
            f = F.normalize(f, dim=-1)
            feats.append(f.cpu())
            del tokens, f
    return torch.cat(feats, dim=0)


def bootstrap_test(aps_a, aps_b, n_bootstrap=10000):
    """Bootstrap paired test: is model A significantly better than model B?"""
    diffs = np.array(aps_a) - np.array(aps_b)
    observed_diff = np.mean(diffs)
    
    rng = np.random.default_rng(42)
    boot_diffs = []
    n = len(diffs)
    for _ in range(n_bootstrap):
        sample = rng.choice(diffs, size=n, replace=True)
        boot_diffs.append(np.mean(sample))
    
    boot_diffs = np.array(boot_diffs)
    p_value = np.mean(boot_diffs <= 0)  # P(our model is NOT better)
    ci_lower = np.percentile(boot_diffs, 2.5)
    ci_upper = np.percentile(boot_diffs, 97.5)
    
    return {
        "observed_diff": float(observed_diff),
        "p_value": float(p_value),
        "ci_95_lower": float(ci_lower),
        "ci_95_upper": float(ci_upper),
        "significant_at_05": p_value < 0.05,
        "significant_at_01": p_value < 0.01,
    }


def load_fashion200k_holdout(n_items, start_offset=1000):
    """Load fashion200k items starting from offset (skip training data)."""
    from datasets import load_dataset
    print(f"  Loading fashion200k (skip {start_offset}, take {n_items})...")
    ds = load_dataset("Marqo/fashion200k", split="data", streaming=True)
    
    all_valid = []
    for row in ds:
        cat = row.get("category3", "") or ""
        if row.get("image") and cat.strip():
            all_valid.append({"category": cat.strip(), "image": row["image"]})
        if len(all_valid) >= start_offset + n_items:
            break
    
    holdout = all_valid[start_offset:start_offset + n_items]
    rng = random.Random(123)
    rng.shuffle(holdout)
    return holdout


def load_deepfashion():
    """Load DeepFashion dataset for category-based retrieval."""
    from datasets import load_dataset
    print("  Loading DeepFashion...")
    try:
        ds = load_dataset("lirus18/deepfashion", split="train", streaming=True)
        items = []
        for row in ds:
            if len(items) >= 5000:
                break
            # Try different possible column names
            cat = None
            for col in ["category", "label", "class", "category_name"]:
                if col in row and row[col]:
                    cat = str(row[col]).strip()
                    break
            
            img = None
            for col in ["image", "img", "photo"]:
                if col in row and row[col]:
                    img = row[col]
                    break
            
            if cat and img:
                items.append({"category": cat, "image": img})
        
        if len(items) < 100:
            print(f"  WARNING: only {len(items)} items found")
            return None
        
        print(f"  Loaded {len(items)} items")
        return items
    except Exception as e:
        print(f"  ERROR loading DeepFashion: {e}")
        return None


def load_fashioniq():
    """Load FashionIQ dataset for category-based retrieval."""
    from datasets import load_dataset
    print("  Loading FashionIQ...")
    try:
        ds = load_dataset("Plachta/FashionIQ", split="train", streaming=True)
        items = []
        for row in ds:
            if len(items) >= 5000:
                break
            cat = None
            for col in ["category", "dress_type", "label", "class"]:
                if col in row and row[col]:
                    cat = str(row[col]).strip()
                    break
            
            img = None
            for col in ["image", "target_image", "candidate_image", "img"]:
                if col in row and row[col]:
                    img = row[col]
                    break
            
            if cat and img:
                items.append({"category": cat, "image": img})
        
        if len(items) < 100:
            print(f"  WARNING: only {len(items)} items found")
            return None
        
        print(f"  Loaded {len(items)} items")
        return items
    except Exception as e:
        print(f"  ERROR loading FashionIQ: {e}")
        return None


def eval_on_dataset(items, model, preprocess, tokenizer, dataset_name):
    """Evaluate MAP@10 on a dataset with category-based retrieval."""
    cat_to_indices = defaultdict(list)
    for idx, item in enumerate(items):
        cat_to_indices[item["category"]].append(idx)
    
    valid_cats = [cat for cat, idxs in cat_to_indices.items() if len(idxs) >= 2]
    rng = random.Random(77)
    rng.shuffle(valid_cats)
    query_cats = valid_cats[:min(300, len(valid_cats))]
    
    if len(query_cats) < 10:
        print(f"  {dataset_name}: too few valid categories ({len(valid_cats)})")
        return None, None
    
    images = [item["image"] for item in items]
    print(f"  {dataset_name}: {len(items)} items, {len(query_cats)} queries, {len(valid_cats)} categories")
    
    img_feats = encode_images(model, preprocess, images)
    txt_feats = encode_texts(model, tokenizer, query_cats)
    scores = txt_feats @ img_feats.T
    
    aps = compute_map10(scores, query_cats, cat_to_indices)
    map10 = np.mean(aps)
    return float(map10), aps


def main():
    import open_clip
    from datasets import load_dataset

    print(f"Device: {DEVICE}")
    print(f"\n{'='*70}")
    print("COMPREHENSIVE EVALUATION: Distilled SigLIP B/16 (203M) vs FSL")
    print(f"{'='*70}")

    results = {}

    # =========================================================
    # TEST 1: Bootstrap significance on 5K holdout
    # =========================================================
    print(f"\n{'─'*70}")
    print("TEST 1: Bootstrap significance test (5K holdout, 300 queries)")
    print(f"{'─'*70}")

    holdout_5k = load_fashion200k_holdout(5000, start_offset=1000)
    images_5k = [h["image"] for h in holdout_5k]
    
    cat_to_indices_5k = defaultdict(list)
    for idx, h in enumerate(holdout_5k):
        cat_to_indices_5k[h["category"]].append(idx)
    valid_cats_5k = [c for c, idxs in cat_to_indices_5k.items() if len(idxs) >= 2]
    rng = random.Random(99)
    rng.shuffle(valid_cats_5k)
    query_cats_5k = valid_cats_5k[:min(300, len(valid_cats_5k))]
    print(f"  {len(holdout_5k)} items, {len(query_cats_5k)} queries")

    # Load FSL
    print("\n  Loading FashionSigLIP...")
    fsl, _, fsl_pre = open_clip.create_model_and_transforms("ViT-B-16-SigLIP", pretrained="webli")
    fsl_state = torch.load(MODEL_DIR / "marqo-fashionSigLIP" / "open_clip_pytorch_model.bin",
                           map_location="cpu", weights_only=True)
    fsl.load_state_dict(fsl_state)
    fsl.eval().to(DEVICE)
    fsl_tok = open_clip.get_tokenizer("ViT-B-16-SigLIP")

    print("  Encoding with FSL...")
    fsl_img = encode_images(fsl, fsl_pre, images_5k)
    fsl_txt = encode_texts(fsl, fsl_tok, query_cats_5k)
    fsl_scores_5k = fsl_txt @ fsl_img.T
    fsl_aps_5k = compute_map10(fsl_scores_5k, query_cats_5k, cat_to_indices_5k)
    fsl_map10_5k = np.mean(fsl_aps_5k)
    print(f"  FSL MAP@10 = {fsl_map10_5k:.4f}")
    del fsl_img, fsl_txt, fsl_scores_5k

    # Load distilled
    print("\n  Loading distilled model...")
    dist, _, dist_pre = open_clip.create_model_and_transforms("ViT-B-16-SigLIP", pretrained="webli")
    dist_state = torch.load(MODEL_DIR / "siglip-b16-distilled" / "model.pt",
                            map_location="cpu", weights_only=True)
    dist.load_state_dict(dist_state)
    dist.eval().to(DEVICE)
    dist_tok = open_clip.get_tokenizer("ViT-B-16-SigLIP")

    print("  Encoding with distilled...")
    dist_img = encode_images(dist, dist_pre, images_5k)
    dist_txt = encode_texts(dist, dist_tok, query_cats_5k)
    dist_scores_5k = dist_txt @ dist_img.T
    dist_aps_5k = compute_map10(dist_scores_5k, query_cats_5k, cat_to_indices_5k)
    dist_map10_5k = np.mean(dist_aps_5k)
    print(f"  Distilled MAP@10 = {dist_map10_5k:.4f}")
    del dist_img, dist_txt, dist_scores_5k

    # Bootstrap
    print("\n  Running bootstrap (10,000 resamples)...")
    boot = bootstrap_test(dist_aps_5k, fsl_aps_5k)
    print(f"  Observed diff: {boot['observed_diff']:+.4f}")
    print(f"  95% CI: [{boot['ci_95_lower']:+.4f}, {boot['ci_95_upper']:+.4f}]")
    print(f"  p-value: {boot['p_value']:.4f}")
    print(f"  Significant at 0.05? {'YES' if boot['significant_at_05'] else 'NO'}")
    print(f"  Significant at 0.01? {'YES' if boot['significant_at_01'] else 'NO'}")
    
    results["test1_bootstrap_5k"] = {
        "fsl_map10": float(fsl_map10_5k),
        "dist_map10": float(dist_map10_5k),
        "bootstrap": boot,
    }

    # Free FSL for now, reload later
    del fsl
    gc.collect()

    # =========================================================
    # TEST 2: Larger 10K holdout
    # =========================================================
    print(f"\n{'─'*70}")
    print("TEST 2: Larger holdout (10K corpus)")
    print(f"{'─'*70}")

    holdout_10k = load_fashion200k_holdout(10000, start_offset=1000)
    if len(holdout_10k) >= 8000:
        images_10k = [h["image"] for h in holdout_10k]
        
        cat_to_indices_10k = defaultdict(list)
        for idx, h in enumerate(holdout_10k):
            cat_to_indices_10k[h["category"]].append(idx)
        valid_cats_10k = [c for c, idxs in cat_to_indices_10k.items() if len(idxs) >= 2]
        rng10 = random.Random(77)
        rng10.shuffle(valid_cats_10k)
        query_cats_10k = valid_cats_10k[:min(400, len(valid_cats_10k))]
        print(f"  {len(holdout_10k)} items, {len(query_cats_10k)} queries")

        # FSL on 10K
        print("\n  Loading FSL...")
        fsl, _, fsl_pre = open_clip.create_model_and_transforms("ViT-B-16-SigLIP", pretrained="webli")
        fsl_state = torch.load(MODEL_DIR / "marqo-fashionSigLIP" / "open_clip_pytorch_model.bin",
                               map_location="cpu", weights_only=True)
        fsl.load_state_dict(fsl_state)
        fsl.eval().to(DEVICE)
        fsl_tok = open_clip.get_tokenizer("ViT-B-16-SigLIP")

        print("  Encoding FSL on 10K...")
        fsl_img_10k = encode_images(fsl, fsl_pre, images_10k)
        fsl_txt_10k = encode_texts(fsl, fsl_tok, query_cats_10k)
        fsl_scores_10k = fsl_txt_10k @ fsl_img_10k.T
        fsl_aps_10k = compute_map10(fsl_scores_10k, query_cats_10k, cat_to_indices_10k)
        fsl_map10_10k = np.mean(fsl_aps_10k)
        print(f"  FSL MAP@10 (10K) = {fsl_map10_10k:.4f}")
        del fsl, fsl_img_10k, fsl_txt_10k, fsl_scores_10k
        gc.collect()

        # Distilled on 10K
        print("  Encoding distilled on 10K...")
        dist_img_10k = encode_images(dist, dist_pre, images_10k)
        dist_txt_10k = encode_texts(dist, dist_tok, query_cats_10k)
        dist_scores_10k = dist_txt_10k @ dist_img_10k.T
        dist_aps_10k = compute_map10(dist_scores_10k, query_cats_10k, cat_to_indices_10k)
        dist_map10_10k = np.mean(dist_aps_10k)
        print(f"  Distilled MAP@10 (10K) = {dist_map10_10k:.4f}")
        del dist_img_10k, dist_txt_10k, dist_scores_10k

        # Bootstrap on 10K
        print("  Bootstrap on 10K...")
        boot_10k = bootstrap_test(dist_aps_10k, fsl_aps_10k)
        print(f"  Observed diff: {boot_10k['observed_diff']:+.4f}")
        print(f"  95% CI: [{boot_10k['ci_95_lower']:+.4f}, {boot_10k['ci_95_upper']:+.4f}]")
        print(f"  p-value: {boot_10k['p_value']:.4f}")
        print(f"  Significant at 0.05? {'YES' if boot_10k['significant_at_05'] else 'NO'}")

        results["test2_10k_holdout"] = {
            "fsl_map10": float(fsl_map10_10k),
            "dist_map10": float(dist_map10_10k),
            "n_items": len(holdout_10k),
            "n_queries": len(query_cats_10k),
            "bootstrap": boot_10k,
        }
        del images_10k
    else:
        print(f"  Only got {len(holdout_10k)} items, skipping 10K test")
        results["test2_10k_holdout"] = {"skipped": True}

    gc.collect()

    # =========================================================
    # TEST 3: DeepFashion cross-benchmark
    # =========================================================
    print(f"\n{'─'*70}")
    print("TEST 3: Cross-benchmark — DeepFashion")
    print(f"{'─'*70}")

    df_items = load_deepfashion()
    if df_items:
        # FSL on DeepFashion
        print("\n  Loading FSL for DeepFashion...")
        fsl, _, fsl_pre = open_clip.create_model_and_transforms("ViT-B-16-SigLIP", pretrained="webli")
        fsl_state = torch.load(MODEL_DIR / "marqo-fashionSigLIP" / "open_clip_pytorch_model.bin",
                               map_location="cpu", weights_only=True)
        fsl.load_state_dict(fsl_state)
        fsl.eval().to(DEVICE)
        fsl_tok = open_clip.get_tokenizer("ViT-B-16-SigLIP")

        fsl_map_df, fsl_aps_df = eval_on_dataset(df_items, fsl, fsl_pre, fsl_tok, "DeepFashion-FSL")
        print(f"  FSL MAP@10 = {fsl_map_df:.4f}" if fsl_map_df else "  FSL: FAILED")
        del fsl
        gc.collect()

        # Distilled on DeepFashion
        dist_map_df, dist_aps_df = eval_on_dataset(df_items, dist, dist_pre, dist_tok, "DeepFashion-Dist")
        print(f"  Distilled MAP@10 = {dist_map_df:.4f}" if dist_map_df else "  Distilled: FAILED")

        if fsl_map_df and dist_map_df:
            delta = (dist_map_df - fsl_map_df) / fsl_map_df * 100
            print(f"  Delta: {delta:+.1f}%")
            boot_df = bootstrap_test(dist_aps_df, fsl_aps_df)
            print(f"  Bootstrap p-value: {boot_df['p_value']:.4f}")
            results["test3_deepfashion"] = {
                "fsl_map10": fsl_map_df,
                "dist_map10": dist_map_df,
                "delta_pct": delta,
                "bootstrap": boot_df,
            }
        else:
            results["test3_deepfashion"] = {"failed": True}
    else:
        results["test3_deepfashion"] = {"skipped": True, "reason": "could not load"}

    gc.collect()

    # =========================================================
    # TEST 4: FashionIQ cross-benchmark
    # =========================================================
    print(f"\n{'─'*70}")
    print("TEST 4: Cross-benchmark — FashionIQ")
    print(f"{'─'*70}")

    fiq_items = load_fashioniq()
    if fiq_items:
        # FSL on FashionIQ
        print("\n  Loading FSL for FashionIQ...")
        fsl, _, fsl_pre = open_clip.create_model_and_transforms("ViT-B-16-SigLIP", pretrained="webli")
        fsl_state = torch.load(MODEL_DIR / "marqo-fashionSigLIP" / "open_clip_pytorch_model.bin",
                               map_location="cpu", weights_only=True)
        fsl.load_state_dict(fsl_state)
        fsl.eval().to(DEVICE)
        fsl_tok = open_clip.get_tokenizer("ViT-B-16-SigLIP")

        fsl_map_fiq, fsl_aps_fiq = eval_on_dataset(fiq_items, fsl, fsl_pre, fsl_tok, "FashionIQ-FSL")
        print(f"  FSL MAP@10 = {fsl_map_fiq:.4f}" if fsl_map_fiq else "  FSL: FAILED")
        del fsl
        gc.collect()

        # Distilled on FashionIQ
        dist_map_fiq, dist_aps_fiq = eval_on_dataset(fiq_items, dist, dist_pre, dist_tok, "FashionIQ-Dist")
        print(f"  Distilled MAP@10 = {dist_map_fiq:.4f}" if dist_map_fiq else "  Distilled: FAILED")

        if fsl_map_fiq and dist_map_fiq:
            delta = (dist_map_fiq - fsl_map_fiq) / fsl_map_fiq * 100
            print(f"  Delta: {delta:+.1f}%")
            boot_fiq = bootstrap_test(dist_aps_fiq, fsl_aps_fiq)
            print(f"  Bootstrap p-value: {boot_fiq['p_value']:.4f}")
            results["test4_fashioniq"] = {
                "fsl_map10": fsl_map_fiq,
                "dist_map10": dist_map_fiq,
                "delta_pct": delta,
                "bootstrap": boot_fiq,
            }
        else:
            results["test4_fashioniq"] = {"failed": True}
    else:
        results["test4_fashioniq"] = {"skipped": True, "reason": "could not load"}

    del dist
    gc.collect()

    # =========================================================
    # SUMMARY
    # =========================================================
    print(f"\n{'='*70}")
    print("COMPREHENSIVE EVALUATION SUMMARY")
    print(f"{'='*70}")
    
    print(f"\n  {'Test':<35} {'FSL':>8} {'Ours':>8} {'Delta':>8} {'p-val':>8} {'Sig?':>6}")
    print(f"  {'─'*75}")
    
    if "test1_bootstrap_5k" in results:
        r = results["test1_bootstrap_5k"]
        sig = "YES" if r["bootstrap"]["significant_at_05"] else "NO"
        print(f"  {'fashion200k 5K holdout':<35} {r['fsl_map10']:>8.4f} {r['dist_map10']:>8.4f} "
              f"{r['bootstrap']['observed_diff']:>+8.4f} {r['bootstrap']['p_value']:>8.4f} {sig:>6}")
    
    if "test2_10k_holdout" in results and not results["test2_10k_holdout"].get("skipped"):
        r = results["test2_10k_holdout"]
        sig = "YES" if r["bootstrap"]["significant_at_05"] else "NO"
        print(f"  {'fashion200k 10K holdout':<35} {r['fsl_map10']:>8.4f} {r['dist_map10']:>8.4f} "
              f"{r['bootstrap']['observed_diff']:>+8.4f} {r['bootstrap']['p_value']:>8.4f} {sig:>6}")
    
    if "test3_deepfashion" in results and not results["test3_deepfashion"].get("skipped") and not results["test3_deepfashion"].get("failed"):
        r = results["test3_deepfashion"]
        sig = "YES" if r["bootstrap"]["significant_at_05"] else "NO"
        print(f"  {'DeepFashion':<35} {r['fsl_map10']:>8.4f} {r['dist_map10']:>8.4f} "
              f"{r['bootstrap']['observed_diff']:>+8.4f} {r['bootstrap']['p_value']:>8.4f} {sig:>6}")
    
    if "test4_fashioniq" in results and not results["test4_fashioniq"].get("skipped") and not results["test4_fashioniq"].get("failed"):
        r = results["test4_fashioniq"]
        sig = "YES" if r["bootstrap"]["significant_at_05"] else "NO"
        print(f"  {'FashionIQ':<35} {r['fsl_map10']:>8.4f} {r['dist_map10']:>8.4f} "
              f"{r['bootstrap']['observed_diff']:>+8.4f} {r['bootstrap']['p_value']:>8.4f} {sig:>6}")

    # Save
    save_path = CACHE_DIR / "comprehensive_eval_results.json"
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  [Saved to {save_path}]")


if __name__ == "__main__":
    main()
