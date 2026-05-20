"""
Disagreement analysis — Where does the fusion win, where does it stall?

For each query, we compute the rank of the gold doc under three retrieval
runs:
    A: FashionSigLIP solo
    B: Google SigLIP-2 B/16/384 solo
    F: Fusion (score-mean of A and B)

We then bucket queries into:
    BOTH_HIT_TOP10  — A and B both rank gold ≤10 (fusion is along for the ride)
    A_HIT_ONLY      — A ≤10 but B >10 (fusion *should* save these)
    B_HIT_ONLY      — B ≤10 but A >10 (fusion *should* save these)
    FUSION_RECOVERS — neither A nor B has gold ≤10, but fusion does (the magic)
    FUSION_HURTS    — A or B had gold ≤10, but fusion pushed it >10 (the cost)
    BOTH_MISS_100   — neither A nor B has gold in top-100 (RECALL CLIFF — no
                       fusion or fine-tune of these two encoders can help;
                       only data work or a 3rd encoder with different geometry
                       can save these queries)
    BOTH_MISS_10    — both miss top-10 but at least one has it in [11..100]
                       (RANKING GAP — re-ranker / fine-tune territory)

The point of this is to estimate the *ceiling* on further model work:

    - Sum of FUSION_HURTS = preventable losses (better fusion calibration)
    - Size of BOTH_MISS_10 \\ BOTH_MISS_100 = how much room a re-ranker has
    - Size of BOTH_MISS_100 = how much we *cannot* fix without a 3rd tower
       or data work (this is the recall cliff)

Reads cached score matrices from data/processed/fusion_cache/scores_*_FULL.pt
and ground truth from repos/marqo-FashionCLIP/data/<DS>/gt_query_doc/.

Outputs:
    results/disagreement/<dataset>_per_query.csv
    results/disagreement/<dataset>_buckets.json
    results/disagreement/summary.md
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "repos" / "marqo-FashionCLIP"))
sys.path.insert(0, str(REPO / "benchmark"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("disagree")

from fuse_and_eval import reconstruct_corpus_full  # noqa: E402

CACHE_DIR = REPO / "data/processed/fusion_cache"
OUT_DIR = REPO / "results/disagreement"

# How deep to look. Anything ranked >MAX_RANK is treated as "outside top-K".
MAX_RANK = 100


def gold_rank(scores_row: torch.Tensor, gold_doc_idx_set: set[int]) -> int:
    """Return min rank (1-indexed) of any gold doc under this query's scores.

    Returns MAX_RANK+1 if no gold appears in top-MAX_RANK.
    """
    # argsort descending; only take top MAX_RANK to keep it cheap on full corpus
    top_inds = torch.topk(scores_row, k=min(MAX_RANK, scores_row.shape[0]), largest=True).indices.tolist()
    for r, idx in enumerate(top_inds, 1):
        if idx in gold_doc_idx_set:
            return r
    return MAX_RANK + 1


def analyse_dataset(dataset: str) -> dict:
    log.info("==== %s ====", dataset)

    # 1. Reconstruct the corpus + queries + GT in the same order as the eval.
    item_IDs, queries, gt = reconstruct_corpus_full(dataset)
    item_id_to_idx = {iid: i for i, iid in enumerate(item_IDs)}

    # GT we got back is keyed by query string; pre-resolve to corpus indices.
    gold_per_query: list[set[int]] = []
    valid_query_mask: list[bool] = []
    for q in queries:
        gold_docs = {item_id_to_idx[d] for d in gt.get(q, {}).keys() if d in item_id_to_idx}
        gold_per_query.append(gold_docs)
        valid_query_mask.append(len(gold_docs) > 0)
    log.info("  queries=%d  with gold in corpus=%d", len(queries), sum(valid_query_mask))

    # 2. Load cached score matrices (these are float32 [N_q, N_doc] cosine scores)
    s_a = torch.load(CACHE_DIR / f"scores_fashion-siglip_{dataset}_FULL.pt", map_location="cpu", weights_only=False)
    s_b = torch.load(CACHE_DIR / f"scores_google-siglip2-b16-384_{dataset}_FULL.pt", map_location="cpu", weights_only=False)
    log.info("  s_a=%s  s_b=%s", tuple(s_a.shape), tuple(s_b.shape))
    assert s_a.shape == s_b.shape == (len(queries), len(item_IDs)), \
        f"score matrix shape mismatch: a={s_a.shape}, b={s_b.shape}, expected=({len(queries)}, {len(item_IDs)})"

    # 3. Build the fusion (score-mean) matrix on-the-fly.
    s_f = 0.5 * (s_a + s_b)

    # 4. Per-query rank of gold under A, B, F.
    rows: list[dict] = []
    for qi, q in enumerate(queries):
        if not valid_query_mask[qi]:
            continue
        gold = gold_per_query[qi]
        rank_a = gold_rank(s_a[qi], gold)
        rank_b = gold_rank(s_b[qi], gold)
        rank_f = gold_rank(s_f[qi], gold)
        rows.append({"qi": qi, "query": q, "rank_a": rank_a, "rank_b": rank_b, "rank_f": rank_f})

        if (qi + 1) % 500 == 0:
            log.info("    %d/%d", qi + 1, len(queries))

    # 5. Bucketise.
    def hit10(r: int) -> bool: return r <= 10
    def hit100(r: int) -> bool: return r <= 100

    buckets: dict[str, int] = {
        "BOTH_HIT_TOP10": 0,
        "A_HIT_ONLY_TOP10": 0,
        "B_HIT_ONLY_TOP10": 0,
        "FUSION_RECOVERS_TOP10": 0,    # neither A nor B in top-10, fusion in top-10
        "FUSION_HURTS_TOP10": 0,       # A or B in top-10, fusion NOT in top-10
        "BOTH_MISS_TOP10": 0,
        "BOTH_MISS_TOP100": 0,         # subset of BOTH_MISS_TOP10 — recall cliff
        "RERANKER_REACHABLE": 0,       # both miss top-10 but at least one in [11..100]
    }

    for r in rows:
        a, b, f = r["rank_a"], r["rank_b"], r["rank_f"]
        if hit10(a) and hit10(b):
            buckets["BOTH_HIT_TOP10"] += 1
        elif hit10(a) and not hit10(b):
            buckets["A_HIT_ONLY_TOP10"] += 1
        elif not hit10(a) and hit10(b):
            buckets["B_HIT_ONLY_TOP10"] += 1

        if not hit10(a) and not hit10(b):
            buckets["BOTH_MISS_TOP10"] += 1
            if not hit100(a) and not hit100(b):
                buckets["BOTH_MISS_TOP100"] += 1
            else:
                buckets["RERANKER_REACHABLE"] += 1
            if hit10(f):
                buckets["FUSION_RECOVERS_TOP10"] += 1

        if (hit10(a) or hit10(b)) and not hit10(f):
            buckets["FUSION_HURTS_TOP10"] += 1

    n = len(rows)
    log.info("  buckets:")
    for k, v in buckets.items():
        log.info("    %-25s %5d  (%5.1f%%)", k, v, 100.0 * v / max(n, 1))

    # 6. Compute model-marginal hit@10 and ceiling.
    hit_a = sum(1 for r in rows if hit10(r["rank_a"]))
    hit_b = sum(1 for r in rows if hit10(r["rank_b"]))
    hit_f = sum(1 for r in rows if hit10(r["rank_f"]))
    # Oracle = perfect router: take the better of A and B per query
    hit_oracle = sum(1 for r in rows if hit10(r["rank_a"]) or hit10(r["rank_b"]))
    log.info("  hit@10  A=%.4f  B=%.4f  Fusion=%.4f  Oracle=%.4f",
             hit_a / n, hit_b / n, hit_f / n, hit_oracle / n)

    return {
        "dataset": dataset,
        "n_queries_with_gold": n,
        "hit10_A": hit_a / n,
        "hit10_B": hit_b / n,
        "hit10_Fusion": hit_f / n,
        "hit10_Oracle_AB": hit_oracle / n,
        "buckets_count": buckets,
        "buckets_pct": {k: 100.0 * v / max(n, 1) for k, v in buckets.items()},
        "rows": rows,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--datasets", nargs="+", default=["fashion200k", "atlas", "polyvore", "KAGL"])
    p.add_argument("--out-dir", default=str(OUT_DIR))
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results: list[dict] = []
    for ds in args.datasets:
        res = analyse_dataset(ds)
        all_results.append(res)

        # Per-dataset outputs
        with open(out_dir / f"{ds}_per_query.csv", "w") as f:
            w = csv.DictWriter(f, fieldnames=["qi", "query", "rank_a", "rank_b", "rank_f"])
            w.writeheader()
            for r in res["rows"]:
                w.writerow(r)
        with open(out_dir / f"{ds}_buckets.json", "w") as f:
            slim = {k: v for k, v in res.items() if k != "rows"}
            json.dump(slim, f, indent=2)

    # Cross-dataset summary
    md = ["# Disagreement analysis — FashionSigLIP ⊕ SigLIP-2 B/16/384 (full corpus)",
          "",
          "Per-query bucketisation of the rank of the gold doc under each model.",
          "",
          "Notation: A = Marqo-FashionSigLIP, B = Google SigLIP-2 B/16/384, F = score-mean fusion.",
          "",
          "## 1. Hit@10 and the perfect-router ceiling",
          "",
          "| Dataset | A hit@10 | B hit@10 | Fusion hit@10 | Oracle (best of A,B) | Headroom for fusion |",
          "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for r in all_results:
        head = r["hit10_Oracle_AB"] - r["hit10_Fusion"]
        md.append(
            f"| {r['dataset']} | {r['hit10_A']:.4f} | {r['hit10_B']:.4f} | "
            f"**{r['hit10_Fusion']:.4f}** | {r['hit10_Oracle_AB']:.4f} | "
            f"{head:+.4f} |"
        )

    md += ["",
           "**Reading:** `Oracle - Fusion` is the residual headroom available to a *better* fusion of the same two models. ",
           "If it's small, score-mean is already near-optimal. If it's large, a learned router could help.",
           "",
           "## 2. Failure mode buckets",
           "",
           "| Dataset | BOTH_HIT@10 | A_only@10 | B_only@10 | BOTH_MISS@10 | …of which RECALL_CLIFF (miss@100) | …of which RERANKER_REACHABLE | FUSION_RECOVERS@10 | FUSION_HURTS@10 |",
           "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for r in all_results:
        b = r["buckets_pct"]
        md.append(
            f"| {r['dataset']} | {b['BOTH_HIT_TOP10']:.1f}% | {b['A_HIT_ONLY_TOP10']:.1f}% | "
            f"{b['B_HIT_ONLY_TOP10']:.1f}% | {b['BOTH_MISS_TOP10']:.1f}% | "
            f"**{b['BOTH_MISS_TOP100']:.1f}%** | {b['RERANKER_REACHABLE']:.1f}% | "
            f"{b['FUSION_RECOVERS_TOP10']:.1f}% | {b['FUSION_HURTS_TOP10']:.1f}% |"
        )

    md += ["",
           "**How to read this table:**",
           "",
           "- **BOTH_HIT@10**: easy queries; fusion neither helps nor hurts.",
           "- **A_only@10 / B_only@10**: complementary wins. These are the slices where each model already captures something the other misses; fusion preserves them.",
           "- **BOTH_MISS@10**: the population we still need to fix.",
           "  - **RECALL_CLIFF (miss@100)**: gold doc is not in either model's top-100. **A re-ranker cannot fix these.** This is the *upper bound* on the value of any rank-stage improvement on these two encoders. To shrink this we need either (a) a third tower with different geometry (e.g. DINOv2), (b) data work (synthetic captions, query expansion), or (c) a sparse retriever (BM25/SPLADE) added to the recall pool.",
           "  - **RERANKER_REACHABLE**: both miss top-10 but at least one has gold in top-100. A learned re-ranker over the union of top-100s *could* recover these.",
           "- **FUSION_RECOVERS@10**: queries where *neither* solo model put gold in top-10 but the fusion did. This is the slice that justifies score-mean fusion existing at all.",
           "- **FUSION_HURTS@10**: queries where one of the solo models had gold in top-10 but fusion pushed it out. The cost of averaging.",
           "",
           "## 3. What this means for the next step",
           "",
           "**If RECALL_CLIFF is large (e.g. >10%)**: model-side work has a hard ceiling. Highest-EV moves are data-side or 3rd-tower (different geometry).",
           "",
           "**If RERANKER_REACHABLE is large**: a small cross-encoder or a learned re-ranker on the fusion's top-100 could capture meaningful headroom.",
           "",
           "**If FUSION_HURTS is large relative to FUSION_RECOVERS**: the score-mean fusion is destabilising the strong-model wins; a calibrated or learned router beats vanilla mean.",
           ""]

    (out_dir / "summary.md").write_text("\n".join(md))
    log.info("wrote %s", out_dir / "summary.md")
    log.info("DONE")


if __name__ == "__main__":
    main()
