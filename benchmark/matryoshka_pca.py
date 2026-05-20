"""
Phase 5 — Matryoshka via PCA (diagnostic + production).

Big insight: SigLIP-family embeddings often have most of their semantic mass
on a low-rank subspace, so a *fixed-basis* PCA projection might preserve
retrieval quality at 64/128/256-d without any training.

If PCA preserves >=95% of full-dim MAP@10, we ship PCA-truncated fusion
embeddings for storage/serve cost reductions — zero training risk, zero
collapse risk, fully reproducible.

If PCA loses too much, we'll fall back to a learned linear Matryoshka head.

This script:
  1. Loads cached image+text embeddings for FashionSigLIP (and optionally
     SigLIP-2 B/16/384) on the 10K subsample of each Marqo dataset.
  2. Fits PCA on the image embeddings of one chosen "fit dataset" (default:
     fashion200k, our hardest target). Same basis is used for all evals
     (out-of-domain generalisation check).
  3. Truncates image+text embeddings to {64, 128, 256, 384, 512, 768} and runs
     the SAME BEIR retrieval metric as our other evals.
  4. Optionally fuses two PCA-truncated models (score-mean) for the
     fusion-Matryoshka Pareto curve.

Outputs:
  results/matryoshka_pca/<dataset>/pareto.json  (per-dim metrics)
  results/matryoshka_pca/pareto_summary.md      (cross-dataset table)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "repos" / "marqo-FashionCLIP"))
sys.path.insert(0, str(REPO / "benchmark"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("mat-pca")


# Reuse fuse_and_eval helpers (they already handle subsample reconstruction +
# query encoding + BEIR scoring identical to the existing eval harness).
from fuse_and_eval import (  # noqa: E402
    MODEL_REGISTRY,
    DATASET_CONFIGS,
    reconstruct_corpus_and_queries,
    reconstruct_corpus_full,
    load_image_embeddings,
    encode_queries,
    topk_dict,
    evaluate_with_beir,
)


def fit_pca(X: torch.Tensor, max_dim: int = 768) -> tuple[torch.Tensor, torch.Tensor]:
    """Fit PCA on X [N, D]. Returns (mean[D], components[max_dim, D]).

    Uses torch.linalg.svd on the centered matrix. Cheap for D=768 even with
    200K rows (1.5 GB float32, fits in RAM; SVD on the D×D covariance is
    instant).
    """
    log.info("fitting PCA on %s ...", tuple(X.shape))
    X = X.float()
    mean = X.mean(dim=0)
    Xc = X - mean
    # Covariance route (cheap: D×D = 768×768 SVD instead of N×D)
    cov = (Xc.T @ Xc) / max(X.shape[0] - 1, 1)
    U, S, _ = torch.linalg.svd(cov, full_matrices=False)
    # Components are the principal directions, sorted by variance
    components = U.T  # [D, D]
    components = components[:max_dim]
    explained = (S / S.sum()).tolist()
    log.info("  variance explained: top1=%.3f  top10=%.3f  top64=%.3f  top128=%.3f  top256=%.3f  top512=%.3f",
             explained[0], sum(explained[:10]), sum(explained[:64]),
             sum(explained[:128]), sum(explained[:256]), sum(explained[:512]))
    return mean, components, S


def project_pca(
    X: torch.Tensor, mean: torch.Tensor, components: torch.Tensor, dim: int,
    center: bool = False,
) -> torch.Tensor:
    """Project X to top-`dim` PCA components. Returns L2-normalized [N, dim].

    NOTE: For cosine retrieval, we *do not* center by default. Centering
    distorts the cosine geometry — at full rank the rotation alone is an
    isometry, but mean-subtraction shifts every vector by the same fixed
    offset, which is fine for ranking *within* one query (constant offset
    cancels) but breaks cross-query consistency in BEIR's evaluation.
    """
    if center:
        Xc = X.float() - mean
    else:
        Xc = X.float()
    Y = Xc @ components[:dim].T  # [N, dim]
    Y = Y / Y.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return Y


def evaluate_dim(
    img_pca: torch.Tensor, txt_pca: torch.Tensor,
    item_IDs: list[str], queries: list[str], gt: dict, k_eval: int = 100,
) -> dict:
    """Score [N_q, dim] @ [N_d, dim].T -> top-K -> BEIR metrics."""
    scores = txt_pca @ img_pca.T  # cosine since both L2-normed
    retrieved = topk_dict(scores, item_IDs, queries, k_eval)
    return evaluate_with_beir(retrieved, gt)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--models", nargs="+", required=True,
        help="One or two model keys. If two, also runs score-mean fusion at each truncation dim.",
    )
    p.add_argument(
        "--datasets", nargs="+", default=list(DATASET_CONFIGS.keys()),
        help="Datasets to evaluate.",
    )
    p.add_argument(
        "--fit-dataset", default="fashion200k",
        choices=list(DATASET_CONFIGS.keys()) + ["joint"],
        help="Which dataset's image embeddings the PCA basis is fit on. "
             "Use 'joint' to fit on the concatenation of all eval datasets' "
             "image embeddings (best for cross-domain robustness).",
    )
    p.add_argument(
        "--dims", nargs="+", type=int, default=[64, 128, 256, 384, 512, 768],
        help="Truncation dimensions to evaluate.",
    )
    p.add_argument("--corpus-size", type=int, default=10000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--full-corpus", action="store_true",
                   help="Use full corpus + full GT instead of 10K subsample.")
    p.add_argument("--out-dir", default=str(REPO / "results/matryoshka_pca"))
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if len(args.models) > 2:
        raise ValueError("at most two models supported")

    # ---------------------------------------------------------------
    # 1. Fit PCA basis per model (on the chosen fit-dataset's images)
    # ---------------------------------------------------------------
    pca_basis: dict[str, dict] = {}
    for m in args.models:
        log.info("---- fitting PCA for %s on %s images ----", m, args.fit_dataset)
        if args.fit_dataset == "joint":
            chunks = []
            for ds in args.datasets:
                ch = load_image_embeddings(m, ds, args.corpus_size, args.seed,
                                            full_corpus=args.full_corpus)
                chunks.append(ch)
            img = torch.cat(chunks, dim=0)
            log.info("  joint corpus shape=%s", tuple(img.shape))
            del chunks
        else:
            img = load_image_embeddings(m, args.fit_dataset, args.corpus_size, args.seed,
                                         full_corpus=args.full_corpus)
        # Important: fit on raw embeddings BEFORE any L2-norm. SigLIP embeddings
        # already have similar norms but PCA cares about absolute geometry.
        mean, comps, sing = fit_pca(img)
        pca_basis[m] = {"mean": mean, "components": comps, "singvals": sing}
        del img

    # ---------------------------------------------------------------
    # 2. For each dataset: load embeddings, encode queries, evaluate
    # ---------------------------------------------------------------
    pareto: dict[str, dict] = {}
    for ds in args.datasets:
        log.info("==== evaluating %s ====", ds)
        if args.full_corpus:
            item_IDs, queries, gt = reconstruct_corpus_full(ds)
        else:
            item_IDs, queries, gt = reconstruct_corpus_and_queries(
                ds, args.corpus_size, args.seed,
            )

        per_model_runs: dict[str, dict] = {}
        for m in args.models:
            log.info("---- model %s ----", m)
            img = load_image_embeddings(m, ds, args.corpus_size, args.seed,
                                         full_corpus=args.full_corpus)
            txt = encode_queries(m, queries)  # already L2-normed [N_q, D]
            # Re-derive raw txt geometry by re-projecting through the SAME PCA mean/components.
            # encode_queries returned L2-normed; we want pre-norm geometry. Instead, treat
            # txt as the raw direction (unit vector); project that. PCA mean is subtracted,
            # which slightly shifts unit vectors -- this is the standard approach for text
            # embeddings (image and text live in the same geometry of SigLIP).
            mean = pca_basis[m]["mean"]
            comps = pca_basis[m]["components"]

            results = {}
            for d in args.dims:
                img_pca = project_pca(img, mean, comps, d)
                txt_pca = project_pca(txt, mean, comps, d)
                metrics = evaluate_dim(img_pca, txt_pca, item_IDs, queries, gt)
                results[d] = metrics
                log.info("  %s @ %d-d  MAP@10=%.4f  Recall@10=%.4f  NDCG@10=%.4f",
                         m, d, metrics.get("MAP@10", 0), metrics.get("Recall@10", 0),
                         metrics.get("NDCG@10", 0))
            per_model_runs[m] = results
            del img, txt

        # ---------------------------------------------------------------
        # 3. If two models: score-mean fusion at each dim
        # ---------------------------------------------------------------
        fusion_runs: dict[int, dict] = {}
        if len(args.models) == 2:
            mA, mB = args.models
            log.info("---- fusion %s ⊕ %s ----", mA, mB)
            imgA = load_image_embeddings(mA, ds, args.corpus_size, args.seed,
                                          full_corpus=args.full_corpus)
            txtA = encode_queries(mA, queries)
            imgB = load_image_embeddings(mB, ds, args.corpus_size, args.seed,
                                          full_corpus=args.full_corpus)
            txtB = encode_queries(mB, queries)
            mean_A, comps_A = pca_basis[mA]["mean"], pca_basis[mA]["components"]
            mean_B, comps_B = pca_basis[mB]["mean"], pca_basis[mB]["components"]

            for d in args.dims:
                iA = project_pca(imgA, mean_A, comps_A, d)
                tA = project_pca(txtA, mean_A, comps_A, d)
                iB = project_pca(imgB, mean_B, comps_B, d)
                tB = project_pca(txtB, mean_B, comps_B, d)
                sA = tA @ iA.T
                sB = tB @ iB.T
                s = 0.5 * (sA + sB)
                retrieved = topk_dict(s, item_IDs, queries, 100)
                metrics = evaluate_with_beir(retrieved, gt)
                fusion_runs[d] = metrics
                log.info("  fusion @ %d-d  MAP@10=%.4f  Recall@10=%.4f  NDCG@10=%.4f",
                         d, metrics.get("MAP@10", 0), metrics.get("Recall@10", 0),
                         metrics.get("NDCG@10", 0))
            del imgA, imgB, txtA, txtB

        pareto[ds] = {
            "per_model": per_model_runs,
            "fusion_score_mean": fusion_runs,
        }

        # Per-dataset save
        ds_path = out_dir / f"{ds}_pareto.json"
        ds_path.write_text(json.dumps(pareto[ds], indent=2, default=lambda x: float(x) if isinstance(x, torch.Tensor) else str(x)))
        log.info("wrote %s", ds_path)

    # ---------------------------------------------------------------
    # 4. Cross-dataset summary markdown
    # ---------------------------------------------------------------
    md = ["# Matryoshka via PCA — Phase 5",
          "",
          f"Fit basis: **{args.fit_dataset}** image embeddings, applied to all eval datasets (true OOD).",
          f"Models: {args.models}",
          f"Mode: {'full corpus' if args.full_corpus else f'10K subsample seed={args.seed}'}",
          f"Dims evaluated: {args.dims}",
          ""]

    for ds in args.datasets:
        md.append(f"## {ds}")
        md.append("")
        md.append("| Dim | " + " | ".join(args.models) +
                  (" | Fusion (score-mean) |" if len(args.models) == 2 else " |"))
        md.append("| ---: |" + " ---: |" * (len(args.models) + (1 if len(args.models) == 2 else 0)))
        for d in args.dims:
            row = [str(d)]
            for m in args.models:
                v = pareto[ds]["per_model"][m].get(d, {}).get("MAP@10", 0)
                row.append(f"{v:.4f}")
            if len(args.models) == 2:
                v = pareto[ds]["fusion_score_mean"].get(d, {}).get("MAP@10", 0)
                row.append(f"**{v:.4f}**")
            md.append("| " + " | ".join(row) + " |")
        md.append("")

    summary_path = out_dir / "pareto_summary.md"
    summary_path.write_text("\n".join(md))
    log.info("wrote %s", summary_path)
    log.info("DONE")


if __name__ == "__main__":
    main()
