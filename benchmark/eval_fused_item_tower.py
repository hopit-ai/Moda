"""
MODA Phase 3 — Fused Item Tower Evaluation (Path A)

Rebuilds the article embedding index using the trained Fused Item Tower,
then evaluates dense retrieval on the test split. Compares:

  1. Dense-only baseline (FashionCLIP 512d)
  2. Fused Item Tower (256d, with color/category/group metadata)
  3. Original MoE (672d, Phase 3.8) if available

Usage:
  python benchmark/eval_fused_item_tower.py
  python benchmark/eval_fused_item_tower.py --n_queries 2000
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from benchmark.models import load_clip_model, encode_texts_clip
from benchmark.embed_hnm import build_article_text
from benchmark.eval_full_pipeline import (
    evaluate,
    RESULTS_DIR,
)
from benchmark.train_fused_item_tower import (
    FusedRetriever,
    FUSED_DIM,
    FIELD_COLS,
    build_vocabularies,
    OUTPUT_DIR as FUSED_MODEL_DIR,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

HNM_DIR = _REPO_ROOT / "data" / "raw" / "hnm_real"
PROCESSED_DIR = _REPO_ROOT / "data" / "processed"
SPLIT_PATH = PROCESSED_DIR / "query_splits.json"
EMB_DIR = PROCESSED_DIR / "embeddings"
EMB_DIR.mkdir(parents=True, exist_ok=True)

TOP_K = 100


def load_fused_model(
    vocabs: dict[str, dict[str, int]],
    device: str,
    use_best: bool = True,
) -> FusedRetriever:
    """Load trained Fused Item Tower model."""
    ckpt_dir = FUSED_MODEL_DIR / "best" if use_best else FUSED_MODEL_DIR
    state_path = ckpt_dir / "model_state_dict.pt"

    if not state_path.exists():
        raise FileNotFoundError(f"Model not found at {state_path}")

    model = FusedRetriever(
        clip_dim=512,
        color_vocab_size=len(vocabs["color"]),
        category_vocab_size=len(vocabs["category"]),
        group_vocab_size=len(vocabs["group"]),
        out_dim=FUSED_DIM,
    )
    sd = torch.load(state_path, map_location="cpu", weights_only=True)
    model.load_state_dict(sd)
    model = model.to(device).eval()
    log.info("Loaded fused model from %s", ckpt_dir)
    return model


def build_fused_index(
    model: FusedRetriever,
    articles_df: pd.DataFrame,
    vocabs: dict[str, dict[str, int]],
    clip_embs: np.ndarray,
    article_ids: list[str],
    device: str,
    batch_size: int = 1024,
) -> np.ndarray:
    """Build fused 256d embeddings for all articles using the Item Tower."""
    art_map = {str(r["article_id"]).strip(): r.to_dict()
               for _, r in articles_df.iterrows()}

    color_indices = []
    cat_indices = []
    group_indices = []
    for aid in article_ids:
        row = art_map.get(aid, {})
        color_indices.append(vocabs["color"].get(
            str(row.get(FIELD_COLS["color"], "")).strip(), 0))
        cat_indices.append(vocabs["category"].get(
            str(row.get(FIELD_COLS["category"], "")).strip(), 0))
        group_indices.append(vocabs["group"].get(
            str(row.get(FIELD_COLS["group"], "")).strip(), 0))

    n = len(article_ids)
    fused_embs = np.zeros((n, FUSED_DIM), dtype=np.float32)

    log.info("Building fused embeddings for %d articles...", n)
    with torch.no_grad():
        for start in tqdm(range(0, n, batch_size), desc="Fused embed", ncols=80):
            end = min(start + batch_size, n)
            clip_batch = torch.from_numpy(clip_embs[start:end]).to(device)
            color_batch = torch.tensor(color_indices[start:end], device=device)
            cat_batch = torch.tensor(cat_indices[start:end], device=device)
            group_batch = torch.tensor(group_indices[start:end], device=device)

            fused = model.encode_item(clip_batch, color_batch, cat_batch, group_batch)
            fused_embs[start:end] = fused.cpu().numpy()

    log.info("Fused embeddings shape: %s", fused_embs.shape)
    return fused_embs


def main():
    p = argparse.ArgumentParser(description="Evaluate Fused Item Tower (Path A)")
    p.add_argument("--n_queries", type=int, default=0,
                   help="Sample N test queries (0 = all)")
    args = p.parse_args()

    t_start = time.time()
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    # Load test split
    splits = json.loads(SPLIT_PATH.read_text())
    test_qids = set(splits["test"])
    log.info("Test queries: %d", len(test_qids))

    # Load articles
    articles_df = pd.read_csv(HNM_DIR / "articles.csv", dtype=str).fillna("")
    vocabs = build_vocabularies(articles_df)

    # Load pre-computed FashionCLIP embeddings
    ft_path = _REPO_ROOT / "models" / "moda-fashionclip-finetuned" / "best"
    use_ft = (ft_path / "model_state_dict.pt").exists()
    tag = "finetuned" if use_ft else "baseline"
    safe_name = f"fashion-clip_{tag}"

    emb_path = EMB_DIR / f"{safe_name}_embeddings.npy"
    ids_path = EMB_DIR / f"{safe_name}_article_ids.json"

    if not emb_path.exists():
        log.error("Pre-computed embeddings not found at %s", emb_path)
        log.info("Run embed_hnm.py or train_moe_encoders.py --eval first")
        return

    clip_embs = np.load(str(emb_path)).astype(np.float32)
    with open(ids_path) as f:
        article_ids = json.load(f)
    log.info("Loaded CLIP embeddings: %s (%s)", clip_embs.shape, tag)

    norms = np.linalg.norm(clip_embs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    clip_embs /= norms

    # Load test queries
    queries_df = pd.read_csv(HNM_DIR / "queries.csv", dtype=str)
    queries = [
        (str(r["query_id"]), str(r["query_text"]))
        for _, r in queries_df.iterrows()
        if str(r["query_id"]) in test_qids
    ]

    if args.n_queries > 0 and len(queries) > args.n_queries:
        rng = random.Random(42)
        queries = rng.sample(queries, args.n_queries)
    log.info("Evaluating on %d test queries", len(queries))

    # Build qrels
    qrels_raw = pd.read_csv(HNM_DIR / "qrels.csv", dtype=str)
    qrels: dict[str, dict[str, int]] = {}
    for _, r in qrels_raw.iterrows():
        qid = str(r["query_id"])
        pos_ids = str(r.get("positive_ids", "")).split()
        qrels[qid] = {aid: 1 for aid in pos_ids}

    # Encode queries with FashionCLIP
    log.info("Encoding queries with FashionCLIP%s...", " (fine-tuned)" if use_ft else "")
    model_clip, _, tokenizer = load_clip_model("fashion-clip", device=device)
    if use_ft:
        sd = torch.load(ft_path / "model_state_dict.pt", map_location="cpu")
        model_clip.load_state_dict(sd, strict=False)
        model_clip = model_clip.to(device).eval()

    query_texts = [q[1] for q in queries]
    query_clip_embs = encode_texts_clip(
        query_texts, model_clip, tokenizer, device, batch_size=128)
    qnorms = np.linalg.norm(query_clip_embs, axis=1, keepdims=True)
    qnorms[qnorms == 0] = 1.0
    query_clip_embs /= qnorms

    del model_clip, tokenizer
    if device == "mps":
        torch.mps.empty_cache()

    # ── 1. Dense-only baseline ────────────────────────────────────────────
    log.info("Running dense-only baseline...")
    dense_results: dict[str, list[str]] = {}
    for i, (qid, _) in enumerate(tqdm(queries, desc="Dense search", ncols=80)):
        scores = clip_embs @ query_clip_embs[i]
        top_idx = np.argpartition(-scores, TOP_K)[:TOP_K]
        top_idx = top_idx[np.argsort(-scores[top_idx])]
        dense_results[qid] = [article_ids[j] for j in top_idx]

    res_dense = evaluate(dense_results, qrels, label="Dense_only")

    # ── 2. Fused Item Tower ───────────────────────────────────────────────
    fused_ckpt = FUSED_MODEL_DIR / "best" / "model_state_dict.pt"
    if not fused_ckpt.exists():
        fused_ckpt = FUSED_MODEL_DIR / "model_state_dict.pt"

    res_fused = None
    res_rerank = None
    res_interp = None

    if fused_ckpt.exists():
        fused_model = load_fused_model(vocabs, device,
                                        use_best=(FUSED_MODEL_DIR / "best" / "model_state_dict.pt").exists())

        fused_embs = build_fused_index(
            fused_model, articles_df, vocabs, clip_embs, article_ids, device)

        # Diagnostic: check for representation collapse
        sample_fused = fused_embs[:500]
        pairwise = sample_fused @ sample_fused.T
        np.fill_diagonal(pairwise, 0)
        avg_sim = pairwise.sum() / (500 * 499)
        log.info("Item tower avg pairwise cos-sim: %.4f (collapse if >0.8)", avg_sim)

        # Encode queries through the query tower
        log.info("Encoding queries through query tower...")
        query_fused = np.zeros((len(queries), FUSED_DIM), dtype=np.float32)
        with torch.no_grad():
            for start in range(0, len(queries), 512):
                end = min(start + 512, len(queries))
                q_clip = torch.from_numpy(query_clip_embs[start:end]).to(device)
                q_vec = fused_model.encode_query(q_clip)
                query_fused[start:end] = q_vec.cpu().numpy()

        # 2a. Pure fused retrieval
        log.info("Running fused retrieval...")
        fused_results: dict[str, list[str]] = {}
        for i, (qid, _) in enumerate(tqdm(queries, desc="Fused search", ncols=80)):
            scores = fused_embs @ query_fused[i]
            top_idx = np.argpartition(-scores, TOP_K)[:TOP_K]
            top_idx = top_idx[np.argsort(-scores[top_idx])]
            fused_results[qid] = [article_ids[j] for j in top_idx]

        res_fused = evaluate(fused_results, qrels, label="Fused_ItemTower")

        # 2b. Fused reranking: retrieve top-200 with CLIP, rerank with fused
        RERANK_K = 200
        log.info("Running fused reranking (top-%d CLIP → rerank)...", RERANK_K)
        rerank_results: dict[str, list[str]] = {}
        for i, (qid, _) in enumerate(tqdm(queries, desc="Fused rerank", ncols=80)):
            clip_scores = clip_embs @ query_clip_embs[i]
            cand_idx = np.argpartition(-clip_scores, RERANK_K)[:RERANK_K]
            fused_scores = fused_embs[cand_idx] @ query_fused[i]
            reranked = cand_idx[np.argsort(-fused_scores)][:TOP_K]
            rerank_results[qid] = [article_ids[j] for j in reranked]

        res_rerank = evaluate(rerank_results, qrels, label="Fused_Rerank@200")

        # 2c. Score interpolation: alpha * dense + (1-alpha) * fused
        best_alpha = 0.60
        log.info("Running score interpolation (alpha=%.2f dense + %.2f fused)...",
                 best_alpha, 1 - best_alpha)
        interp_results: dict[str, list[str]] = {}
        for i, (qid, _) in enumerate(tqdm(queries, desc="Interp search", ncols=80)):
            dense_s = clip_embs @ query_clip_embs[i]
            fused_s = fused_embs @ query_fused[i]
            d_norm = (dense_s - dense_s.mean()) / (dense_s.std() + 1e-8)
            f_norm = (fused_s - fused_s.mean()) / (fused_s.std() + 1e-8)
            combined = best_alpha * d_norm + (1 - best_alpha) * f_norm
            top_idx = np.argpartition(-combined, TOP_K)[:TOP_K]
            top_idx = top_idx[np.argsort(-combined[top_idx])]
            interp_results[qid] = [article_ids[j] for j in top_idx]

        res_interp = evaluate(interp_results, qrels, label="Interp_0.6_dense")
    else:
        log.warning("No fused model found at %s — skipping", FUSED_MODEL_DIR)

    # ── Print comparison ──────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("FUSED ITEM TOWER (Path A) — Evaluation Results")
    print(f"  {len(queries):,} test queries | Fused dim: {FUSED_DIM}d")
    print("=" * 80)

    all_results = {"Dense_only": res_dense}
    result_list = [
        ("Dense_only", res_dense),
        ("Fused_ItemTower", res_fused),
        ("Fused_Rerank@200", res_rerank),
        ("Interp_0.6_dense", res_interp),
    ]

    for name, res in result_list:
        if res is None:
            continue
        m = res["metrics"]
        delta = ""
        if name != "Dense_only":
            d = (m["ndcg@10"] / res_dense["metrics"]["ndcg@10"] - 1) * 100
            delta = f"  ({'+' if d >= 0 else ''}{d:.1f}% vs baseline)"
        print(f"  {name:<25} nDCG@10={m['ndcg@10']:.4f}  MRR={m['mrr']:.4f}  "
              f"R@10={m['recall@10']:.4f}{delta}")
        all_results[name] = res

    print("=" * 80)

    # Save results
    out_path = RESULTS_DIR / "phase3_fused_item_tower_eval.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    log.info("Results saved → %s", out_path)
    log.info("Total elapsed: %.1f min", (time.time() - t_start) / 60)


if __name__ == "__main__":
    main()
