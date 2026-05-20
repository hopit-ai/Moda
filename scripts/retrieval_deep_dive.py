"""Retrieval deep-dive using cached score matrices.

No model loading — pure tensor ops on cached scores.
Produces a JSON report showing what each model retrieves for failed queries.
"""
import json, random, torch
from collections import defaultdict
from pathlib import Path

CACHE_DIR = Path(__file__).parent.parent / "cache" / "ensemble_1k"
OUTPUT_PATH = CACHE_DIR / "retrieval_deep_dive.json"
MODEL_KEYS = ["B16", "L16", "SO400M", "FSL"]


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


def classify_query_failure(query_cat, scores_dict, cat_to_indices, k=10):
    """Compute AP@10 per model and classify the failure type."""
    positives = set(cat_to_indices[query_cat])
    qi = None  # will be set by caller
    return positives


def get_top_k_details(scores_tensor, qi, corpus, cat_to_indices, query_cat, k=10):
    """Get top-k retrieved items with categories and whether they're correct."""
    positives = set(cat_to_indices[query_cat])
    topk_vals, topk_idx = scores_tensor[qi].topk(min(k, scores_tensor.shape[1]))

    retrieved = []
    for rank, (idx, score) in enumerate(zip(topk_idx.tolist(), topk_vals.tolist()), 1):
        item_cat = corpus[idx]["category"]
        is_correct = idx in positives
        retrieved.append({
            "rank": rank,
            "corpus_idx": idx,
            "category": item_cat,
            "score": round(score, 4),
            "correct": is_correct,
        })
    return retrieved


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


def analyze_near_miss(retrieved, query_cat):
    """Determine if wrong retrievals are 'near misses' or 'totally wrong'.
    
    Near miss: shares key words with query (color, garment type).
    Totally wrong: different garment family entirely.
    """
    query_words = set(query_cat.lower().split())
    garment_types = {
        "dress", "shirt", "blouse", "skirt", "pants", "jeans", "jacket",
        "coat", "sweater", "top", "shorts", "cardigan", "tunic", "jumpsuit",
        "romper", "blazer", "vest", "hoodie", "tee", "tank",
    }
    colors = {
        "black", "white", "blue", "red", "green", "yellow", "purple",
        "pink", "orange", "brown", "grey", "gray", "beige", "navy",
        "multicolor", "silver", "gold",
    }

    query_garment = query_words & garment_types
    query_color = query_words & colors

    near_miss_count = 0
    wrong_count = 0

    for item in retrieved:
        if item["correct"]:
            continue
        item_words = set(item["category"].lower().split())
        item_garment = item_words & garment_types
        item_color = item_words & colors

        garment_match = bool(query_garment & item_garment)
        color_match = bool(query_color & item_color)

        if garment_match and color_match:
            near_miss_count += 1
        elif garment_match or color_match:
            near_miss_count += 1
        else:
            wrong_count += 1

    total_wrong = near_miss_count + wrong_count
    if total_wrong == 0:
        return "all_correct"
    near_miss_ratio = near_miss_count / total_wrong
    if near_miss_ratio >= 0.7:
        return "near_miss"
    elif near_miss_ratio >= 0.3:
        return "mixed"
    else:
        return "totally_wrong"


def main():
    print("Loading cached data...")
    corpus, scores = load_data()
    query_cats, cat_to_indices = build_queries(corpus)
    print(f"  Corpus: {len(corpus)}, Queries: {len(query_cats)}")

    results = {
        "summary": {},
        "b16_only_failures": [],
        "fsl_only_failures": [],
        "all_fail": [],
        "partial_failures": [],
    }

    # Classify each query
    b16_failures = 0
    l16_failures = 0
    so400m_failures = 0
    fsl_failures = 0

    for qi, cat in enumerate(query_cats):
        ap = {}
        for mk in MODEL_KEYS:
            ap[mk] = compute_ap10(scores[mk], qi, cat_to_indices, cat)

        if ap["B16"] == 0:
            b16_failures += 1
        if ap["L16"] == 0:
            l16_failures += 1
        if ap["SO400M"] == 0:
            so400m_failures += 1
        if ap["FSL"] == 0:
            fsl_failures += 1

        # Classify failure type
        all_zero = all(ap[mk] == 0 for mk in MODEL_KEYS)
        b16_only = ap["B16"] == 0 and ap["L16"] > 0 and ap["SO400M"] > 0
        fsl_only = ap["FSL"] == 0 and all(ap[mk] > 0 for mk in ["B16", "L16", "SO400M"])

        if all_zero or b16_only or fsl_only or ap["B16"] == 0:
            # Get top-10 per model
            per_model = {}
            for mk in MODEL_KEYS:
                retrieved = get_top_k_details(scores[mk], qi, corpus, cat_to_indices, cat)
                near_miss_type = analyze_near_miss(retrieved, cat)
                per_model[mk] = {
                    "ap10": round(ap[mk], 4),
                    "retrieved": retrieved,
                    "failure_type": near_miss_type,
                }

            entry = {
                "query": cat,
                "n_positives": len(cat_to_indices[cat]),
                "models": per_model,
            }

            if all_zero:
                results["all_fail"].append(entry)
            elif b16_only:
                results["b16_only_failures"].append(entry)
            elif fsl_only:
                results["fsl_only_failures"].append(entry)
            elif ap["B16"] == 0:
                results["partial_failures"].append(entry)

    # Summary stats
    results["summary"] = {
        "total_queries": len(query_cats),
        "b16_failures": b16_failures,
        "l16_failures": l16_failures,
        "so400m_failures": so400m_failures,
        "fsl_failures": fsl_failures,
        "all_fail_count": len(results["all_fail"]),
        "b16_only_count": len(results["b16_only_failures"]),
        "fsl_only_count": len(results["fsl_only_failures"]),
        "partial_count": len(results["partial_failures"]),
    }

    # Analyze near-miss patterns for B16 failures
    b16_near_miss = 0
    b16_mixed = 0
    b16_totally_wrong = 0
    for entry in results["b16_only_failures"] + results["partial_failures"]:
        ft = entry["models"]["B16"]["failure_type"]
        if ft == "near_miss":
            b16_near_miss += 1
        elif ft == "mixed":
            b16_mixed += 1
        elif ft == "totally_wrong":
            b16_totally_wrong += 1

    results["summary"]["b16_near_miss_pattern"] = {
        "near_miss": b16_near_miss,
        "mixed": b16_mixed,
        "totally_wrong": b16_totally_wrong,
    }

    # Same for all-fail
    all_fail_near_miss = 0
    all_fail_mixed = 0
    all_fail_wrong = 0
    for entry in results["all_fail"]:
        ft = entry["models"]["SO400M"]["failure_type"]
        if ft == "near_miss":
            all_fail_near_miss += 1
        elif ft == "mixed":
            all_fail_mixed += 1
        elif ft == "totally_wrong":
            all_fail_wrong += 1

    results["summary"]["all_fail_pattern"] = {
        "near_miss": all_fail_near_miss,
        "mixed": all_fail_mixed,
        "totally_wrong": all_fail_wrong,
    }

    # Score gap analysis for B16 failures
    score_gaps = []
    for entry in results["b16_only_failures"]:
        b16_data = entry["models"]["B16"]
        top1_score = b16_data["retrieved"][0]["score"]
        # Find best positive score in full ranking
        qi_idx = query_cats.index(entry["query"])
        positives = set(cat_to_indices[entry["query"]])
        all_scores_sorted = scores["B16"][qi_idx].sort(descending=True)
        best_pos_score = None
        best_pos_rank = None
        for rank_idx, (score_val, doc_idx) in enumerate(
            zip(all_scores_sorted.values.tolist(), all_scores_sorted.indices.tolist()), 1
        ):
            if doc_idx in positives:
                best_pos_score = score_val
                best_pos_rank = rank_idx
                break

        score_gaps.append({
            "query": entry["query"],
            "top1_score": top1_score,
            "best_pos_score": round(best_pos_score, 4) if best_pos_score else None,
            "best_pos_rank": best_pos_rank,
            "gap": round(top1_score - best_pos_score, 4) if best_pos_score else None,
        })

    results["score_gaps_b16"] = score_gaps

    # Write output
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)

    # Print summary
    print(f"\n{'='*70}")
    print("RETRIEVAL DEEP DIVE RESULTS")
    print(f"{'='*70}")
    print(f"\nTotal queries: {results['summary']['total_queries']}")
    print(f"  All-fail (impossible):    {results['summary']['all_fail_count']}")
    print(f"  B16-only failures:        {results['summary']['b16_only_count']}")
    print(f"  FSL-only failures:        {results['summary']['fsl_only_count']}")
    print(f"  Other B16 failures:       {results['summary']['partial_count']}")

    print(f"\n--- B16 failure pattern (is it near-miss or totally wrong?) ---")
    p = results["summary"]["b16_near_miss_pattern"]
    print(f"  Near-miss (same garment family): {p['near_miss']}")
    print(f"  Mixed:                           {p['mixed']}")
    print(f"  Totally wrong:                   {p['totally_wrong']}")

    print(f"\n--- All-fail pattern (best model SO400M still retrieves...) ---")
    p = results["summary"]["all_fail_pattern"]
    print(f"  Near-miss (same garment family): {p['near_miss']}")
    print(f"  Mixed:                           {p['mixed']}")
    print(f"  Totally wrong:                   {p['totally_wrong']}")

    print(f"\n--- B16-only failures: score gap to first positive ---")
    for sg in score_gaps:
        print(f"  \"{sg['query']}\"")
        print(f"    top1={sg['top1_score']:.4f}, best_pos={sg['best_pos_score']}, "
              f"rank_of_pos={sg['best_pos_rank']}, gap={sg['gap']}")

    print(f"\n--- Sample retrievals for B16-only failures ---")
    for entry in results["b16_only_failures"][:5]:
        print(f"\n  Query: \"{entry['query']}\" (n_pos={entry['n_positives']})")
        print(f"  {'Model':<8} {'AP@10':>6} {'Type':<13} Top-5 retrieved categories")
        print(f"  {'-'*75}")
        for mk in MODEL_KEYS:
            m = entry["models"][mk]
            top5 = [f"{'✓' if r['correct'] else '✗'}{r['category'][:30]}" for r in m["retrieved"][:5]]
            print(f"  {mk:<8} {m['ap10']:>6.3f} {m['failure_type']:<13} {' | '.join(top5)}")

    print(f"\n--- Sample retrievals for ALL-FAIL queries ---")
    for entry in results["all_fail"][:5]:
        print(f"\n  Query: \"{entry['query']}\" (n_pos={entry['n_positives']})")
        print(f"  {'Model':<8} {'Type':<13} Top-3 retrieved categories")
        print(f"  {'-'*75}")
        for mk in MODEL_KEYS:
            m = entry["models"][mk]
            top3 = [f"{r['category'][:35]}" for r in m["retrieved"][:3]]
            print(f"  {mk:<8} {m['failure_type']:<13} {' | '.join(top3)}")

    print(f"\n\n[Full results saved to {OUTPUT_PATH}]")


if __name__ == "__main__":
    main()
