"""
Fuse two models' retrievals on a Marqo benchmark + score with the SAME metric
code as the existing eval (BEIR EvaluateRetrieval), so results are directly
comparable to runs in repos/marqo-FashionCLIP/results/.

Why this exists:
  Error analysis showed FashionSigLIP and Google-SigLIP-2 B/16/384 fail on
  DIFFERENT queries (13% disagreement). Ensemble upper bound on hit@10 is
  +5.5pts absolute. Fusion is the highest-EV move we have, with zero training
  and zero collapse risk.

Approach:
  1. Reconstruct the same 10K subsample (seed=42) used by the existing runs.
  2. Load each model's image embeddings (10000, 768) — already cached on disk.
  3. Encode all 2000 fashion200k queries with EACH model's text tower, get a
     full [2000, 10000] cosine score matrix per model.
  4. Apply 3 fusion strategies and produce {query: {item_ID: score}} top-K dicts:
       - score_mean: simple mean of L2-normalized scores from both models
       - score_weighted: w*A + (1-w)*B with sweep over w in {0.3, 0.5, 0.6, 0.7}
       - rrf: rank-based Reciprocal Rank Fusion (k=60 default)
  5. Score each via the same BEIR EvaluateRetrieval code as the eval harness.
  6. Save run dirs under repos/marqo-FashionCLIP/results/<dataset>/<fusion_name>_subsample10000_seed42/
     so they're discovered by every existing eval / analysis script.

Usage:
  .venv/bin/python benchmark/fuse_and_eval.py \
      --models fashion-siglip google-siglip2-b16-384 \
      --dataset fashion200k \
      --corpus-size 10000 --seed 42

What this does NOT do:
  - Re-encode images (those are already cached; we reuse).
  - Re-run any retrieval that's already on disk for the individual models.
  - Train anything. Zero gradients, zero LoRA, zero collapse risk.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "repos" / "marqo-FashionCLIP"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("fuse")

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

# Map our short model keys -> (open_clip identifier, pretrained tag, run_name prefix used for caches)
MODEL_REGISTRY: dict[str, dict] = {
    "fashion-siglip": {
        "model_name": "hf-hub:Marqo/marqo-fashionSigLIP",
        "pretrained": None,
        "run_prefix": "Marqo-FashionSigLIP_subsample{N}_seed{S}",
        "full_run_name": "Marqo-FashionSigLIP",
    },
    "google-siglip2-b16-384": {
        "model_name": "ViT-B-16-SigLIP2-384",
        "pretrained": "webli",
        "run_prefix": "Google-SigLIP2-B16-384_subsample{N}_seed{S}",
        "full_run_name": "Google-SigLIP2-B16-384",
    },
    "google-siglip-b16-224": {
        "model_name": "ViT-B-16-SigLIP",
        "pretrained": "webli",
        "run_prefix": "Google-SigLIP-B16-224_subsample{N}_seed{S}",
        "full_run_name": "Google-SigLIP-B16-224",
    },
    "google-siglip-l16-384": {
        "model_name": "ViT-L-16-SigLIP-384",
        "pretrained": "webli",
        "run_prefix": "Google-SigLIP-L16-384_subsample{N}_seed{S}",
        "full_run_name": "Google-SigLIP-L16-384",
    },
}

DATASET_CONFIGS = {
    "fashion200k": REPO / "repos/marqo-FashionCLIP/configs/fashion200k.json",
    "atlas":       REPO / "repos/marqo-FashionCLIP/configs/atlas.json",
    "polyvore":    REPO / "repos/marqo-FashionCLIP/configs/polyvore.json",
    "KAGL":        REPO / "repos/marqo-FashionCLIP/configs/KAGL.json",
}

GT_PATHS = {
    "fashion200k": REPO / "repos/marqo-FashionCLIP/data/Fashion200k/gt_query_doc/ground_truth_text-image.json",
    "atlas":       REPO / "repos/marqo-FashionCLIP/data/Atlas/gt_query_doc/ground_truth_text-image.json",
    "polyvore":    REPO / "repos/marqo-FashionCLIP/data/Polyvore/gt_query_doc/ground_truth_text-image.json",
    "KAGL":        REPO / "repos/marqo-FashionCLIP/data/KAGL/gt_query_doc/ground_truth_text-image.json",
}


def _import_stratified_subsample():
    """eval_subsample.py runs argparse at import-time, which crashes when our
    argv doesn't match. Pull the function out by AST/source instead."""
    import importlib.util
    src_path = REPO / "repos/marqo-FashionCLIP/eval_subsample.py"
    src = src_path.read_text()
    # Extract the def block — the function is self-contained except for `random`
    # and `logging` which we already have imported.
    import ast
    mod = ast.parse(src)
    func_node = next(n for n in mod.body
                     if isinstance(n, ast.FunctionDef) and n.name == "stratified_subsample")
    # Compile just the function in a fresh namespace.
    import random as _random  # noqa: F401
    ns: dict = {"random": _random, "logging": logging}
    exec(compile(ast.Module(body=[func_node], type_ignores=[]), str(src_path), "exec"), ns)
    return ns["stratified_subsample"]


def reconstruct_corpus_full(dataset: str) -> tuple[list[str], list[str], dict]:
    """Full-corpus version: no subsampling.

    Returns (item_IDs, test_queries, gt) where item_IDs is the FULL corpus order
    used by repos/marqo-FashionCLIP/eval.py (the same order get_dataset()
    returns). The eval harness uses all GT queries on the full corpus, so we do
    the same.
    """
    from data.utils import get_dataset                  # type: ignore

    cfg_path = DATASET_CONFIGS[dataset]
    with open(cfg_path) as f:
        cfg = json.load(f)

    class _Args:
        dataset_config = cfg
        data_dir = str(REPO / "repos/marqo-FashionCLIP/data")
        cache_dir = str(REPO / "data/hf_cache")

    args = _Args()
    log.info("loading %s ... (FULL CORPUS)", cfg["hf_dataset"])
    doc_dataset, item_ID = get_dataset(args, tokenizer=None, preprocess=None)
    log.info("  full corpus: %d", len(item_ID))

    # Load GT for the text-to-image anchor task
    gt_per_task = {}
    for task in cfg["tasks"]:
        for query_col in task["query_col"]:
            gt_path = REPO / "repos/marqo-FashionCLIP/data" / cfg["name"] / "gt_query_doc" / \
                f"ground_truth_{query_col}-{'+'.join(task['doc_col'])}.json"
            if gt_path.exists():
                gt_per_task[f"{task['name']}::{query_col}"] = json.load(open(gt_path))

    anchor = next((v for k, v in gt_per_task.items() if "text-to-image" in k), None)
    if anchor is None:
        raise RuntimeError(f"no text-to-image task found in {dataset}")

    # eval.py uses ALL keys in the GT file (already ~2000 queries capped by Marqo data prep)
    test_queries = list(anchor.keys())
    log.info("  test queries (full GT): %d", len(test_queries))
    return item_ID, test_queries, anchor


def reconstruct_corpus_and_queries(
    dataset: str, corpus_size: int, seed: int
) -> tuple[list[str], list[str], dict]:
    """Return (item_IDs in corpus order, query strings, gt_filtered) using the SAME
    subsample logic as repos/marqo-FashionCLIP/eval_subsample.py so embeddings
    line up with rows.

    We re-import the eval's own get_dataset + stratified_subsample functions so
    we cannot accidentally diverge from the eval's row order.
    """
    from data.utils import get_dataset                  # type: ignore
    # Note: eval_subsample.py has top-level argparse so we can't import from it.
    # Re-extract the function via importlib without executing the module body.
    stratified_subsample = _import_stratified_subsample()

    cfg_path = DATASET_CONFIGS[dataset]
    with open(cfg_path) as f:
        cfg = json.load(f)

    # get_dataset() reads args.dataset_config as a DICT (eval_subsample reassigns
    # it after json.load) and uses args.data_dir / args.cache_dir.
    class _Args:
        dataset_config = cfg
        data_dir = str(REPO / "repos/marqo-FashionCLIP/data")
        cache_dir = str(REPO / "data/hf_cache")

    args = _Args()
    log.info("loading %s ...", cfg["hf_dataset"])
    doc_dataset, item_ID = get_dataset(args, tokenizer=None, preprocess=None)
    log.info("  full corpus: %d", len(doc_dataset))

    # Load ground truth and pick the text-to-image anchor task (same logic as eval).
    gt_per_task = {}
    for task in cfg["tasks"]:
        for query_col in task["query_col"]:
            gt_path = REPO / "repos/marqo-FashionCLIP/data" / cfg["name"] / "gt_query_doc" / \
                f"ground_truth_{query_col}-{'+'.join(task['doc_col'])}.json"
            if not gt_path.exists():
                continue
            with open(gt_path) as f:
                gt_per_task[f"{task['name']}::{query_col}"] = json.load(f)

    anchor_tasks = {k: v for k, v in gt_per_task.items() if "text-to-image" in k}
    if not anchor_tasks:
        raise RuntimeError(f"no text-to-image task found in {dataset}")

    kept_indices, kept_id_set, _ = stratified_subsample(
        item_ID, anchor_tasks, corpus_size, seed,
    )
    item_ID = [item_ID[i] for i in kept_indices]
    log.info("  subsampled corpus: %d", len(item_ID))

    # Build the test_query list = same as eval (gt filtered to kept docs)
    task_key = next(iter(anchor_tasks))
    gt_full = anchor_tasks[task_key]
    gt_filtered = {q: {d: r for d, r in v.items() if d in kept_id_set}
                   for q, v in gt_full.items()}
    gt_filtered = {q: v for q, v in gt_filtered.items() if v}
    test_queries = list(gt_filtered.keys())
    log.info("  queries with positives in subsample: %d", len(test_queries))

    return item_ID, test_queries, gt_filtered


def load_image_embeddings(
    model_key: str, dataset: str, corpus_size: int, seed: int,
    full_corpus: bool = False,
) -> torch.Tensor:
    info = MODEL_REGISTRY[model_key]
    if full_corpus:
        run_dir = info["full_run_name"]
    else:
        run_dir = info["run_prefix"].format(N=corpus_size, S=seed)
    emb_path = REPO / f"repos/marqo-FashionCLIP/results/{dataset}/{run_dir}/embeddings.pt"
    if not emb_path.exists():
        raise FileNotFoundError(f"image embeddings missing for {model_key} on {dataset}: {emb_path}")
    log.info("loading %s image embeddings: %s", model_key, emb_path)
    payload = torch.load(emb_path, map_location="cpu", weights_only=False)
    img = payload["image"]
    log.info("  shape=%s, dtype=%s, mean_norm=%.3f", tuple(img.shape), img.dtype, img.norm(dim=-1).mean().item())
    return img.float()


def encode_queries(model_key: str, queries: list[str], batch_size: int = 64) -> torch.Tensor:
    """Encode all queries with the model's text tower; return [N, D] float32 (L2-normalized)."""
    import open_clip
    info = MODEL_REGISTRY[model_key]
    log.info("loading %s text tower (%s, pretrained=%s)", model_key, info["model_name"], info["pretrained"])
    model, _, _ = open_clip.create_model_and_transforms(
        info["model_name"], pretrained=info["pretrained"],
        cache_dir=str(REPO / "data/hf_cache"),
    )
    tokenizer = open_clip.get_tokenizer(info["model_name"])
    model.eval().to(DEVICE)
    for p in model.parameters():
        p.requires_grad = False

    feats: list[torch.Tensor] = []
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, len(queries), batch_size):
            chunk = queries[i:i + batch_size]
            tokens = tokenizer(chunk).to(DEVICE)
            f = model.encode_text(tokens)
            f = f / f.norm(dim=-1, keepdim=True)
            feats.append(f.float().cpu())
            if (i // batch_size) % 5 == 0:
                rate = (i + len(chunk)) / max(time.time() - t0, 1e-6)
                log.info("  encoded %d/%d (%.1f q/s)", i + len(chunk), len(queries), rate)
    out = torch.cat(feats, dim=0)
    log.info("  done: %s in %.1fs", tuple(out.shape), time.time() - t0)
    # Free the model from memory
    del model
    if DEVICE == "mps":
        torch.mps.empty_cache()
    return out


def get_score_matrix(
    model_key: str, dataset: str, corpus_size: int, seed: int,
    queries: list[str], cache_dir: Path,
    full_corpus: bool = False,
) -> torch.Tensor:
    """Compute (or load from cache) full [N_q, N_doc] cosine score matrix."""
    if full_corpus:
        cache_path = cache_dir / f"scores_{model_key}_{dataset}_FULL.pt"
    else:
        cache_path = cache_dir / f"scores_{model_key}_{dataset}_size{corpus_size}_seed{seed}.pt"
    if cache_path.exists():
        log.info("loading cached score matrix: %s", cache_path)
        return torch.load(cache_path, map_location="cpu", weights_only=False)

    img = load_image_embeddings(model_key, dataset, corpus_size, seed, full_corpus=full_corpus)
    img = img / img.norm(dim=-1, keepdim=True)
    txt = encode_queries(model_key, queries)
    log.info("computing cosine score matrix [%d, %d] ...", txt.shape[0], img.shape[0])
    scores = txt @ img.T  # [N_q, N_doc] cosine since both are L2-normed

    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp = cache_path.with_suffix(".pt.tmp")
    torch.save(scores, tmp)
    os.replace(tmp, cache_path)
    log.info("saved %s (%.1f MB)", cache_path, cache_path.stat().st_size / 1e6)
    return scores


def topk_dict(scores: torch.Tensor, item_IDs: list[str], queries: list[str], k: int) -> dict:
    """Convert score matrix [N_q, N_doc] -> {query: {doc_id: score}} with top-K."""
    top_scores, top_inds = torch.topk(scores, k=min(k, scores.shape[1]), dim=1)
    out: dict[str, dict[str, float]] = {}
    top_inds = top_inds.tolist()
    top_scores = top_scores.tolist()
    for qi, q in enumerate(queries):
        out[q] = {item_IDs[d]: float(s) for d, s in zip(top_inds[qi], top_scores[qi])}
    return out


def fuse_score_mean(scores_a: torch.Tensor, scores_b: torch.Tensor) -> torch.Tensor:
    """Mean of two cosine score matrices. Both are already in [-1, 1] range."""
    return 0.5 * (scores_a + scores_b)


def fuse_score_weighted(scores_a: torch.Tensor, scores_b: torch.Tensor, w_a: float) -> torch.Tensor:
    return w_a * scores_a + (1.0 - w_a) * scores_b


def fuse_rrf(scores_a: torch.Tensor, scores_b: torch.Tensor, k_const: float = 60.0,
             K_topk: int = 200) -> torch.Tensor:
    """Reciprocal Rank Fusion.

    For each query and each model, take the top-K_topk candidates and assign
    each candidate a contribution 1/(k_const + rank). Sum across models.
    Docs not in either model's top-K_topk get score 0. We use K_topk=200 to
    cover essentially all queries' useful candidates while staying tractable.

    Output is a [N_q, N_doc] dense matrix where most entries are 0; topk on
    that gives the fused ranking.
    """
    n_q, n_doc = scores_a.shape
    out = torch.zeros_like(scores_a)
    for scores in (scores_a, scores_b):
        # Get top-K_topk indices per query
        K = min(K_topk, n_doc)
        _, top_idx = torch.topk(scores, k=K, dim=1)  # [N_q, K]
        # Ranks 1..K
        ranks = torch.arange(1, K + 1, dtype=torch.float32).unsqueeze(0).expand(n_q, K)
        contrib = 1.0 / (k_const + ranks)
        # Scatter-add contributions into out
        out.scatter_add_(dim=1, index=top_idx, src=contrib)
    return out


def evaluate_with_beir(retrieved: dict, gt: dict, ks=(1, 5, 10, 100)) -> dict:
    """Use the SAME metric code as the eval harness."""
    from beir.retrieval.evaluation import EvaluateRetrieval
    evaluator = EvaluateRetrieval()
    ndcg, _map, recall, precision = evaluator.evaluate(gt, retrieved, list(ks))
    out = {"MAP": _map, "NDCG": ndcg, "Recall": recall, "Precision": precision}
    # Compute MRR (use full retrieved dict)
    mrr = 0.0
    for q, scores in retrieved.items():
        if q not in gt:
            continue
        positives = {d for d, r in gt[q].items() if r > 0}
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        for r_i, (doc, _) in enumerate(ranked, 1):
            if doc in positives:
                mrr += 1.0 / r_i
                break
    out["MRR"] = mrr / max(len(retrieved), 1)
    # Flatten the standard metrics
    flat = {}
    for prefix, d in [("MAP", _map), ("NDCG", ndcg), ("Recall", recall), ("Precision", precision)]:
        for k, v in d.items():
            flat[k] = v   # BEIR keys are like "MAP@1", "MAP@10"...
    flat["MRR"] = out["MRR"]
    return flat


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", nargs=2, required=True,
                   help="Two model keys to fuse (e.g. fashion-siglip google-siglip2-b16-384)")
    p.add_argument("--dataset", default="fashion200k", choices=list(DATASET_CONFIGS.keys()))
    p.add_argument("--corpus-size", type=int, default=10000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--k-eval", type=int, default=100, help="Top-K saved in the retrieval json + used for metrics")
    p.add_argument("--rrf-k", type=float, default=60.0)
    p.add_argument("--rrf-topk", type=int, default=200, help="Per-model top-K considered for RRF")
    p.add_argument(
        "--weight-sweep", type=float, nargs="*", default=[0.3, 0.5, 0.6, 0.7],
        help="Weights to sweep for w*A + (1-w)*B (A=first model)",
    )
    p.add_argument("--cache-dir", default=str(REPO / "data/processed/fusion_cache"))
    p.add_argument("--results-root", default=str(REPO / "repos/marqo-FashionCLIP/results"))
    p.add_argument(
        "--summary-out",
        default=str(REPO / "results/fusion/fusion_summary.md"),
    )
    p.add_argument("--full-corpus", action="store_true",
                   help="Use full corpus + full GT (no subsample). Requires "
                        "full-corpus image embeddings cached on disk for both models.")
    args = p.parse_args()

    cache_dir = Path(args.cache_dir)
    summary_path = Path(args.summary_out)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    mode_tag = "FULL CORPUS" if args.full_corpus else f"subsample{args.corpus_size}_seed{args.seed}"
    log.info("==== fusion eval: %s ⊕ %s on %s [%s] ====",
             args.models[0], args.models[1], args.dataset, mode_tag)

    if args.full_corpus:
        item_IDs, queries, gt = reconstruct_corpus_full(args.dataset)
    else:
        item_IDs, queries, gt = reconstruct_corpus_and_queries(args.dataset, args.corpus_size, args.seed)

    log.info("---- model A: %s ----", args.models[0])
    s_a = get_score_matrix(args.models[0], args.dataset, args.corpus_size, args.seed,
                            queries, cache_dir, full_corpus=args.full_corpus)
    log.info("---- model B: %s ----", args.models[1])
    s_b = get_score_matrix(args.models[1], args.dataset, args.corpus_size, args.seed,
                            queries, cache_dir, full_corpus=args.full_corpus)
    assert s_a.shape == s_b.shape == (len(queries), len(item_IDs)), \
        f"score matrix shape mismatch: A={s_a.shape}, B={s_b.shape}, expected=({len(queries)},{len(item_IDs)})"

    # ----- score baselines + fusions -----
    runs: list[tuple[str, torch.Tensor]] = []
    runs.append((f"{args.models[0]}_solo",  s_a))
    runs.append((f"{args.models[1]}_solo",  s_b))
    runs.append(("fusion_score_mean",       fuse_score_mean(s_a, s_b)))
    for w in args.weight_sweep:
        runs.append((f"fusion_weighted_w{w:.2f}", fuse_score_weighted(s_a, s_b, w)))
    runs.append((f"fusion_rrf_k{int(args.rrf_k)}", fuse_rrf(s_a, s_b, args.rrf_k, args.rrf_topk)))

    rows = []
    for name, scores in runs:
        retrieved = topk_dict(scores, item_IDs, queries, args.k_eval)
        metrics = evaluate_with_beir(retrieved, gt)
        log.info(
            "%-32s  MAP@10=%.4f  Recall@10=%.4f  NDCG@10=%.4f  MRR=%.4f",
            name, metrics.get("MAP@10", -1), metrics.get("Recall@10", -1),
            metrics.get("NDCG@10", -1), metrics.get("MRR", -1),
        )
        rows.append((name, metrics, retrieved))

    # ----- emit run dirs in marqo-FashionCLIP/results/<dataset>/ for each fusion variant -----
    results_root = Path(args.results_root) / args.dataset
    results_root.mkdir(parents=True, exist_ok=True)
    suffix = "fullcorpus" if args.full_corpus else f"subsample{args.corpus_size}_seed{args.seed}"
    for name, metrics, retrieved in rows:
        if not name.startswith("fusion"):
            continue
        run_dir = results_root / f"Fusion-{args.models[0]}+{args.models[1]}-{name}_{suffix}"
        task_dir = run_dir / "text-to-image"
        task_dir.mkdir(parents=True, exist_ok=True)
        with open(task_dir / "retrieved_text-image.json", "w") as f:
            json.dump(retrieved, f)
        with open(task_dir / "result_text-image.json", "w") as f:
            json.dump(metrics, f, indent=2)
        with open(run_dir / "args.json", "w") as f:
            json.dump({"fusion": name, "models": args.models, "dataset": args.dataset,
                       "full_corpus": args.full_corpus,
                       "corpus_size": None if args.full_corpus else args.corpus_size,
                       "seed": None if args.full_corpus else args.seed}, f, indent=2)

    # ----- summary markdown -----
    md = [f"# Fusion eval — {args.dataset} {suffix}",
          "",
          f"Fusing **{args.models[0]}** + **{args.models[1]}**",
          "",
          "| Run | MAP@10 | Recall@10 | NDCG@10 | MRR | Δ MAP@10 vs A | Δ MAP@10 vs B |",
          "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    base_a = rows[0][1].get("MAP@10", 0)
    base_b = rows[1][1].get("MAP@10", 0)
    for name, metrics, _ in rows:
        m10 = metrics.get("MAP@10", 0)
        md.append(
            f"| `{name}` | {m10:.4f} | {metrics.get('Recall@10',0):.4f} | "
            f"{metrics.get('NDCG@10',0):.4f} | {metrics.get('MRR',0):.4f} | "
            f"{m10-base_a:+.4f} | {m10-base_b:+.4f} |"
        )
    summary_path.write_text("\n".join(md))
    log.info("wrote %s", summary_path)
    log.info("DONE")


if __name__ == "__main__":
    main()
