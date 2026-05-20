"""Error analysis for MobileCLIP2-B (150M) vs FashionSigLIP (203M).

Uses cached scores from both models. Identifies:
1. Where MobileCLIP2-B fails vs FSL (gap queries)
2. Near-miss pattern analysis
3. Score gap to first positive
4. Then builds triplets for targeted fine-tuning
"""
import random, torch, json
from collections import defaultdict
from pathlib import Path

CACHE_DIR = Path(__file__).parent.parent / "cache" / "ensemble_1k"
MODEL_KEYS = ["MobileCLIP2_B", "FSL"]


def load_data():
    with open(CACHE_DIR / "corpus_meta.json") as f:
        corpus = json.load(f)
    scores = {}
    for key in MODEL_KEYS:
        scores[key] = torch.load(CACHE_DIR / f"scores_{key}.pt", map_location="cpu", weights_only=True)
    return corpus, scores


def build_queries(corpus):
    cat_to_indices = defaultdict(list)
    for idx, c in enumerate(corpus):
        cat_to_indices[c["category"]].append(idx)
    valid_cats = [cat for cat, idxs in cat_to_indices.items() if len(idxs) >= 2]
    rng = random.Random(42)
    rng.shuffle(valid_cats)
    return valid_cats[:min(200, len(valid_cats))], cat_to_indices


def compute_ap10(scores_tensor, qi, cat_to_indices, query_cat, k=10):
    positives = set(cat_to_indices[query_cat])
    topk_idx = scores_tensor[qi].topk(min(k, scores_tensor.shape[1])).indices.tolist()
    ap, n_rel = 0.0, 0
    for rank, idx in enumerate(topk_idx, 1):
        if idx in positives:
            n_rel += 1
            ap += n_rel / rank
    n_pos = min(len(positives), k)
    return ap / n_pos if n_pos > 0 else 0.0


def analyze_near_miss(topk_idx, topk_vals, query_cat, corpus, positives):
    """Classify what the model retrieves when it fails."""
    garment_types = {"dress", "shirt", "blouse", "skirt", "pants", "jeans", "jacket",
                     "coat", "sweater", "top", "shorts", "cardigan", "tunic", "jumpsuit",
                     "romper", "blazer", "vest", "hoodie", "tee", "tank"}
    colors = {"black", "white", "blue", "red", "green", "yellow", "purple",
              "pink", "orange", "brown", "grey", "gray", "beige", "navy",
              "multicolor", "silver", "gold"}

    query_words = set(query_cat.lower().split())
    query_garment = query_words & garment_types
    query_color = query_words & colors

    near_miss = 0
    wrong = 0
    for idx in topk_idx[:10]:
        if idx in positives:
            continue
        item_words = set(corpus[idx]["category"].lower().split())
        item_garment = item_words & garment_types
        item_color = item_words & colors
        if (query_garment & item_garment) or (query_color & item_color):
            near_miss += 1
        else:
            wrong += 1

    total = near_miss + wrong
    if total == 0:
        return "all_correct"
    ratio = near_miss / total
    if ratio >= 0.7:
        return "near_miss"
    elif ratio >= 0.3:
        return "mixed"
    return "totally_wrong"


def main():
    print("Loading cached data...")
    corpus, scores = load_data()
    query_cats, cat_to_indices = build_queries(corpus)
    print(f"  Corpus: {len(corpus)}, Queries: {len(query_cats)}")

    # Compute per-query AP for both models
    mc_failures = 0
    fsl_failures = 0
    mc_only_failures = []  # MC fails, FSL succeeds
    both_fail = []
    mc_wins = []  # MC succeeds, FSL fails

    mc_map = 0
    fsl_map = 0

    for qi, cat in enumerate(query_cats):
        mc_ap = compute_ap10(scores["MobileCLIP2_B"], qi, cat_to_indices, cat)
        fsl_ap = compute_ap10(scores["FSL"], qi, cat_to_indices, cat)
        mc_map += mc_ap
        fsl_map += fsl_ap

        if mc_ap == 0:
            mc_failures += 1
        if fsl_ap == 0:
            fsl_failures += 1

        if mc_ap == 0 and fsl_ap > 0:
            # Find what MC retrieves
            positives = set(cat_to_indices[cat])
            topk_vals, topk_idx = scores["MobileCLIP2_B"][qi].topk(20)
            pattern = analyze_near_miss(topk_idx.tolist(), topk_vals.tolist(), cat, corpus, positives)

            # Score gap
            best_pos_score = None
            best_pos_rank = None
            sorted_scores = scores["MobileCLIP2_B"][qi].sort(descending=True)
            for r, (sv, si) in enumerate(zip(sorted_scores.values.tolist(), sorted_scores.indices.tolist()), 1):
                if si in positives:
                    best_pos_score = sv
                    best_pos_rank = r
                    break

            mc_only_failures.append({
                "query": cat,
                "n_pos": len(positives),
                "fsl_ap": round(fsl_ap, 4),
                "pattern": pattern,
                "top5_retrieved": [corpus[i]["category"] for i in topk_idx.tolist()[:5]],
                "top1_score": round(topk_vals[0].item(), 4),
                "best_pos_score": round(best_pos_score, 4) if best_pos_score else None,
                "best_pos_rank": best_pos_rank,
                "gap": round(topk_vals[0].item() - best_pos_score, 4) if best_pos_score else None,
            })
        elif mc_ap == 0 and fsl_ap == 0:
            both_fail.append(cat)
        elif mc_ap > 0 and fsl_ap == 0:
            mc_wins.append({"query": cat, "mc_ap": round(mc_ap, 4)})

    mc_map /= len(query_cats)
    fsl_map /= len(query_cats)

    # Print results
    print(f"\n{'='*70}")
    print("ERROR ANALYSIS: MobileCLIP2-B (150M) vs FashionSigLIP (203M)")
    print(f"{'='*70}")
    print(f"\n  MobileCLIP2-B MAP@10: {mc_map:.4f}")
    print(f"  FashionSigLIP MAP@10: {fsl_map:.4f}")
    print(f"  Gap: {(mc_map - fsl_map)/fsl_map*100:+.1f}%")

    print(f"\n--- Failure distribution ---")
    print(f"  MobileCLIP2-B failures: {mc_failures}/{len(query_cats)} ({mc_failures/len(query_cats)*100:.1f}%)")
    print(f"  FashionSigLIP failures: {fsl_failures}/{len(query_cats)} ({fsl_failures/len(query_cats)*100:.1f}%)")
    print(f"  MC-only failures (FSL succeeds): {len(mc_only_failures)}")
    print(f"  Both fail: {len(both_fail)}")
    print(f"  MC wins (FSL fails): {len(mc_wins)}")

    # Pattern analysis
    patterns = defaultdict(int)
    for f in mc_only_failures:
        patterns[f["pattern"]] += 1

    print(f"\n--- MC-only failure patterns ---")
    for p, count in sorted(patterns.items(), key=lambda x: -x[1]):
        print(f"  {p}: {count} ({count/max(len(mc_only_failures),1)*100:.0f}%)")

    # Score gaps
    gaps = [f["gap"] for f in mc_only_failures if f["gap"] is not None]
    ranks = [f["best_pos_rank"] for f in mc_only_failures if f["best_pos_rank"] is not None]
    if gaps:
        print(f"\n--- Score gap analysis (MC-only failures) ---")
        print(f"  Avg gap (top1 - best_pos): {sum(gaps)/len(gaps):.4f}")
        print(f"  Avg rank of first positive: {sum(ranks)/len(ranks):.0f}")
        print(f"  Min gap: {min(gaps):.4f}, Max gap: {max(gaps):.4f}")

    # Show worst failures
    mc_only_failures.sort(key=lambda x: -(x["gap"] or 0))
    print(f"\n--- MC-only failures (sorted by gap, hardest first) ---")
    for f in mc_only_failures[:15]:
        print(f"  \"{f['query']}\" (n_pos={f['n_pos']}, FSL_AP={f['fsl_ap']:.2f})")
        print(f"    pattern={f['pattern']}, gap={f['gap']}, pos_rank={f['best_pos_rank']}")
        print(f"    top3: {f['top5_retrieved'][:3]}")

    # Show where MC wins over FSL
    if mc_wins:
        print(f"\n--- Queries where MC BEATS FSL ({len(mc_wins)}) ---")
        for w in mc_wins[:10]:
            print(f"  \"{w['query']}\" (MC AP={w['mc_ap']:.2f})")

    # Save
    results = {
        "mc_map10": mc_map,
        "fsl_map10": fsl_map,
        "mc_failures": mc_failures,
        "fsl_failures": fsl_failures,
        "mc_only_failure_count": len(mc_only_failures),
        "both_fail_count": len(both_fail),
        "mc_wins_count": len(mc_wins),
        "patterns": dict(patterns),
        "avg_gap": sum(gaps)/len(gaps) if gaps else None,
        "avg_pos_rank": sum(ranks)/len(ranks) if ranks else None,
        "mc_only_failures": mc_only_failures,
        "mc_wins": mc_wins,
    }
    with open(CACHE_DIR / "mobileclip_error_analysis.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[Results saved to {CACHE_DIR / 'mobileclip_error_analysis.json'}]")


if __name__ == "__main__":
    main()
