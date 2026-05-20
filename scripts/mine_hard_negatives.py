"""
Mine hard negatives for Path 2 distillation.

Inputs (already on disk):
  - data/processed/marqo_gs_wfash_subset/triplets.jsonl  (5000 rows, 421 queries)
  - data/processed/distillation_cache_fusion/fashion_siglip_embeddings.pt
        L2-normed FSL image+text embeddings, in row order matching triplets.jsonl

For each unique query q with positive set P_q (= the rows where (query==q, score_linear>0)):
  1. Pull q's text embedding (any row with that query has the same text emb under FSL).
  2. Score q against the full image pool (all 5000 image embeddings, L2-normed).
  3. Take the top-100 by FSL cosine similarity.
  4. Drop any candidate that is in P_q (gold positives).
  5. Drop any candidate whose FSL score > 0.95 * (FSL score on q's strongest gold)
     -> false-negative filter, Tamber 2025 §3.3.4.
  6. Bias the remaining candidates toward FSL cos(p+, p_k) ≈ 0.85
     (within-family but not gold-equivalent) -- the diagnostic-driven sampling.
     Sample weight w_k = exp(-((cos_FSL(p_strongest_gold, p_k) - 0.85)**2) / (0.05**2)).
  7. Sample K=15 without replacement using w_k.

For each (q, hard_neg_k) we ALSO store FSL cos(q, hard_neg_k) so the trainer can
build the listwise teacher distribution without re-encoding.

Output: data/processed/path2/hardnegs.jsonl
  one JSON line per query:
    {"query": str,
     "positives": [{"image_path": str, "score_linear": float, "fsl_text_img_score": float}, ...],
     "hard_negatives": [{"image_path": str, "fsl_text_img_score": float,
                          "fsl_img_to_strongest_gold": float}, ...]}

Sanity checks done at end:
  - Coverage: how many queries got K=15 negs vs. fewer.
  - Score gap: median FSL score of mined hard negs - median FSL score of golds (should be NEGATIVE small).
  - Within-family check: sample 5 queries, print 3 negs each for visual inspection
    via the printed image_path.

Usage:
  .venv/bin/python scripts/mine_hard_negatives.py
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("mine-hardnegs")


def load_inputs(triplets_path: Path, fsl_cache_path: Path):
    rows: list[dict] = []
    with open(triplets_path) as f:
        for line in f:
            rows.append(json.loads(line))
    log.info("loaded %d triplet rows from %s", len(rows), triplets_path)

    fsl = torch.load(fsl_cache_path, map_location="cpu", weights_only=False)
    img_emb: torch.Tensor = fsl["image"]
    txt_emb: torch.Tensor = fsl["text"]
    cache_queries: list[str] = fsl["queries"]
    cache_paths: list[str] = fsl["image_paths"]

    assert len(rows) == img_emb.shape[0] == len(cache_queries) == len(cache_paths), (
        f"row/cache size mismatch: triplets={len(rows)}, "
        f"img_emb={img_emb.shape[0]}, queries={len(cache_queries)}, paths={len(cache_paths)}"
    )
    for i, r in enumerate(rows):
        if r["query"] != cache_queries[i] or r["image_path"] != cache_paths[i]:
            raise RuntimeError(
                f"row order mismatch at i={i}: triplet=({r['query'][:30]!r}, "
                f"{r['image_path'][-20:]!r}) vs cache=({cache_queries[i][:30]!r}, "
                f"{cache_paths[i][-20:]!r})"
            )
    log.info("FSL cache aligned: %d rows, embed_dim=%d", img_emb.shape[0], img_emb.shape[1])
    return rows, img_emb, txt_emb


def build_image_pool(rows: list[dict], img_emb: torch.Tensor):
    """Deduplicate the image pool by image_path (some images repeat across queries)."""
    path_to_emb_idx: dict[str, int] = {}
    for i, r in enumerate(rows):
        if r["image_path"] not in path_to_emb_idx:
            path_to_emb_idx[r["image_path"]] = i
    pool_paths = list(path_to_emb_idx.keys())
    pool_idx = torch.tensor([path_to_emb_idx[p] for p in pool_paths], dtype=torch.long)
    pool_emb = img_emb.index_select(0, pool_idx)  # [P, D] L2-normed
    log.info("image pool: %d unique images (from %d rows)", pool_emb.shape[0], len(rows))
    return pool_paths, pool_emb


def group_by_query(rows: list[dict]):
    """Returns {query: [(row_idx, image_path, score_linear), ...]}."""
    out: dict[str, list[tuple[int, str, float]]] = defaultdict(list)
    for i, r in enumerate(rows):
        out[r["query"]].append((i, r["image_path"], float(r.get("score_linear", 0.0))))
    log.info("unique queries: %d", len(out))
    return out


def sample_weights(scores_to_strongest_gold: torch.Tensor, peak: float = 0.75, sigma: float = 0.10) -> torch.Tensor:
    """Gaussian-shaped sampling weight peaked at FSL cos(p+, p_k) == peak.

    Given a 1D tensor of cos(strongest_gold, candidate) scores in [-1, 1],
    return a 1D tensor of (unnormalized) sampling weights of the same shape.

    Peak default 0.75 (not 0.85): empirically the Marqo-GS 5K image pool's
    actual within-family similarity distribution peaks at cos≈0.70 (median).
    Setting peak=0.85 with σ=0.05 sampled into the tail; peak=0.75 with σ=0.10
    matches the realisable pool while still preferring same-family candidates.
    """
    return torch.exp(-((scores_to_strongest_gold - peak) ** 2) / (sigma ** 2))


def weighted_sample_no_replace(weights: torch.Tensor, k: int, rng: random.Random) -> list[int]:
    """Weighted sample without replacement using the standard Efraimidis-Spirakis trick:
    key_i = u_i^(1/w_i) with u_i ~ Uniform(0,1); take top-k by key.
    """
    n = weights.shape[0]
    if n == 0 or k <= 0:
        return []
    k = min(k, n)
    # Use python-side rng for determinism
    keys = []
    for i in range(n):
        w = float(weights[i])
        if w <= 0:
            keys.append(-math.inf)
            continue
        u = rng.random()
        # log key for numerical stability: log(u)/w
        keys.append(math.log(max(u, 1e-300)) / w)
    keys_t = torch.tensor(keys)
    top = torch.topk(keys_t, k=k).indices.tolist()
    return top


def mine_one_query(
    q: str,
    members: list[tuple[int, str, float]],
    txt_emb_q: torch.Tensor,
    pool_paths: list[str],
    pool_emb: torch.Tensor,
    pool_path_to_idx: dict[str, int],
    K: int,
    top_n: int,
    fn_filter_factor: float,
    rng: random.Random,
):
    # FSL text->image scores against the full pool.
    scores = (pool_emb @ txt_emb_q).flatten()  # [P]

    pos_paths = {ip for _, ip, sl in members if sl > 0}
    if not pos_paths:
        return None
    strongest_idx_in_members = max(range(len(members)), key=lambda j: members[j][2])
    strongest_path = members[strongest_idx_in_members][1]
    strongest_pool_idx = pool_path_to_idx[strongest_path]
    strongest_fsl_text_score = float(scores[strongest_pool_idx])

    # Top-N candidates by FSL text->image score
    top_vals, top_idx = torch.topk(scores, k=min(top_n, scores.shape[0]))
    cand_idx = top_idx.tolist()
    cand_vals = top_vals.tolist()

    # Drop golds + apply false-negative filter (FSL_text_score > factor * strongest_pos_text_score)
    fn_threshold = fn_filter_factor * strongest_fsl_text_score
    surviving_pool_idx: list[int] = []
    surviving_text_scores: list[float] = []
    for ci, cv in zip(cand_idx, cand_vals):
        path = pool_paths[ci]
        if path in pos_paths:
            continue
        if cv > fn_threshold:
            continue
        surviving_pool_idx.append(ci)
        surviving_text_scores.append(cv)

    if not surviving_pool_idx:
        return {
            "positives": [{"image_path": p, "score_linear": sl,
                            "fsl_text_img_score": float(scores[pool_path_to_idx[p]])}
                           for _, p, sl in members],
            "hard_negatives": [],
            "stats": {"n_candidates_after_filter": 0,
                      "strongest_pos_text_score": strongest_fsl_text_score,
                      "fn_threshold": fn_threshold},
        }

    # Image-image scores from each survivor to the strongest gold image (NOT to query)
    survivor_emb = pool_emb.index_select(0, torch.tensor(surviving_pool_idx))   # [S, D]
    strongest_emb = pool_emb[strongest_pool_idx]                                 # [D]
    img_img_scores = survivor_emb @ strongest_emb                                # [S]
    weights = sample_weights(img_img_scores, peak=0.85, sigma=0.05)

    chosen_local = weighted_sample_no_replace(weights, k=K, rng=rng)
    chosen = []
    for li in chosen_local:
        pi = surviving_pool_idx[li]
        chosen.append({
            "image_path": pool_paths[pi],
            "fsl_text_img_score": float(surviving_text_scores[li]),
            "fsl_img_to_strongest_gold": float(img_img_scores[li]),
        })

    positives = [{
        "image_path": p,
        "score_linear": sl,
        "fsl_text_img_score": float(scores[pool_path_to_idx[p]]),
    } for _, p, sl in members]
    positives.sort(key=lambda d: -d["score_linear"])

    return {
        "positives": positives,
        "hard_negatives": chosen,
        "stats": {
            "n_candidates_in_top_n": len(cand_idx),
            "n_candidates_after_filter": len(surviving_pool_idx),
            "strongest_pos_text_score": strongest_fsl_text_score,
            "fn_threshold": fn_threshold,
        },
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--triplets", default=str(REPO / "data/processed/marqo_gs_wfash_subset/triplets.jsonl"))
    p.add_argument("--fsl-cache", default=str(REPO / "data/processed/distillation_cache_fusion/fashion_siglip_embeddings.pt"))
    p.add_argument("--out", default=str(REPO / "data/processed/path2/hardnegs.jsonl"))
    p.add_argument("-K", "--K", type=int, default=15, help="hard negatives per query")
    p.add_argument("--top-n", type=int, default=100)
    p.add_argument("--fn-filter-factor", type=float, default=0.95,
                   help="exclude candidates whose FSL text score exceeds factor*strongest_pos_score")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows, img_emb, txt_emb = load_inputs(Path(args.triplets), Path(args.fsl_cache))
    pool_paths, pool_emb = build_image_pool(rows, img_emb)
    pool_path_to_idx = {p: i for i, p in enumerate(pool_paths)}
    by_query = group_by_query(rows)

    # We need a single text-embedding per unique query; pick the first row's index.
    query_to_first_row = {}
    for i, r in enumerate(rows):
        if r["query"] not in query_to_first_row:
            query_to_first_row[r["query"]] = i

    rng = random.Random(args.seed)
    t0 = time.time()
    out_lines: list[str] = []
    cov_full = 0
    cov_partial = 0
    cov_zero = 0
    score_gaps: list[float] = []
    img_img_means: list[float] = []

    for j, (q, members) in enumerate(by_query.items()):
        txt_q = txt_emb[query_to_first_row[q]]
        result = mine_one_query(
            q, members, txt_q, pool_paths, pool_emb, pool_path_to_idx,
            K=args.K, top_n=args.top_n, fn_filter_factor=args.fn_filter_factor, rng=rng,
        )
        if result is None:
            cov_zero += 1
            continue

        n_neg = len(result["hard_negatives"])
        if n_neg >= args.K:
            cov_full += 1
        elif n_neg > 0:
            cov_partial += 1
        else:
            cov_zero += 1

        # stats: (mean FSL text score on golds) vs (mean FSL text score on negs)
        if result["positives"] and result["hard_negatives"]:
            pos_mean = sum(p["fsl_text_img_score"] for p in result["positives"]) / len(result["positives"])
            neg_mean = sum(n["fsl_text_img_score"] for n in result["hard_negatives"]) / len(result["hard_negatives"])
            score_gaps.append(neg_mean - pos_mean)
            img_img_means.append(
                sum(n["fsl_img_to_strongest_gold"] for n in result["hard_negatives"]) /
                len(result["hard_negatives"])
            )

        out_lines.append(json.dumps({"query": q, **result}))

        if (j + 1) % 50 == 0:
            log.info("  mined %d/%d  rate=%.1f q/s  cov_full=%d  cov_partial=%d",
                     j + 1, len(by_query), (j + 1) / max(time.time() - t0, 1e-6),
                     cov_full, cov_partial)

    out_path.write_text("\n".join(out_lines) + "\n")
    log.info("wrote %d query records to %s in %.1fs", len(out_lines), out_path, time.time() - t0)

    log.info("=== coverage ===")
    log.info("  full (K=%d):      %d", args.K, cov_full)
    log.info("  partial (1..K-1): %d", cov_partial)
    log.info("  zero negs:        %d", cov_zero)

    if score_gaps:
        score_gaps_t = torch.tensor(score_gaps)
        img_img_t = torch.tensor(img_img_means)
        log.info("=== sanity: FSL text-score gap (neg_mean - pos_mean) per query ===")
        log.info("  median = %.4f  (negative => negs are weaker than positives, expected)",
                 float(score_gaps_t.median()))
        log.info("  p10    = %.4f", float(torch.quantile(score_gaps_t, 0.10)))
        log.info("  p90    = %.4f", float(torch.quantile(score_gaps_t, 0.90)))
        log.info("=== sanity: FSL img-img cos(strongest_gold, neg) per query (should peak ~0.85) ===")
        log.info("  median = %.4f", float(img_img_t.median()))
        log.info("  p10    = %.4f", float(torch.quantile(img_img_t, 0.10)))
        log.info("  p90    = %.4f", float(torch.quantile(img_img_t, 0.90)))

    log.info("=== sample queries (for visual inspection) ===")
    sample_qs = list(by_query.keys())[:5]
    for q in sample_qs:
        ln = next((l for l in out_lines if json.loads(l)["query"] == q), None)
        if not ln:
            continue
        rec = json.loads(ln)
        log.info("  query: %s", q)
        log.info("    positives (top 2 by score_linear):")
        for pp in rec["positives"][:2]:
            log.info("      score=%.0f  fsl=%.3f  %s", pp["score_linear"],
                     pp["fsl_text_img_score"], pp["image_path"])
        log.info("    hard negatives (first 3):")
        for nn in rec["hard_negatives"][:3]:
            log.info("      fsl_txt=%.3f  fsl_img2gold=%.3f  %s",
                     nn["fsl_text_img_score"], nn["fsl_img_to_strongest_gold"], nn["image_path"])

    meta_path = out_path.parent / "hardnegs_meta.json"
    meta_path.write_text(json.dumps({
        "n_queries": len(out_lines),
        "K": args.K,
        "top_n": args.top_n,
        "fn_filter_factor": args.fn_filter_factor,
        "image_pool_size": len(pool_paths),
        "coverage": {"full": cov_full, "partial": cov_partial, "zero": cov_zero},
        "fsl_score_gap_median": float(torch.tensor(score_gaps).median()) if score_gaps else None,
        "fsl_img_img_to_strongest_gold_median": float(torch.tensor(img_img_means).median()) if img_img_means else None,
    }, indent=2))
    log.info("wrote %s", meta_path)


if __name__ == "__main__":
    main()
