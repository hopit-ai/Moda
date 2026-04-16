"""
MODA Phase 3.9 — Comprehensive Evaluation of New Components

Tests all combinations of:
  - Retrieval: Dense-only vs Fused-Interpolated (Path A)
  - Reranking: None / Off-shelf CE / LLM-trained CE / Attr-conditioned CE (Path B)

Matrix (8 configs total):
  Dense   → None / OffshelfCE / LLMCE / AttrCE
  Fused   → None / OffshelfCE / LLMCE / AttrCE

Works without OpenSearch — uses FashionCLIP dense retrieval only.
When OpenSearch is available, use eval_finetuned_ce.py for full pipeline.

Usage:
  python benchmark/eval_phase3_9.py
  python benchmark/eval_phase3_9.py --n_queries 2000
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
from sentence_transformers import CrossEncoder
from tqdm import tqdm

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from benchmark.article_text import build_article_text as _build_article_text
from benchmark.models import load_clip_model, encode_texts_clip
from benchmark.eval_full_pipeline import evaluate, RESULTS_DIR
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

OFFSHELF_CE = "cross-encoder/ms-marco-MiniLM-L-6-v2"
LLM_CE = str(_REPO_ROOT / "models" / "moda-fashion-ce-llm-best")
ATTR_CE = str(_REPO_ROOT / "models" / "moda-fashion-ce-attr-best")

TOP_K_RETRIEVAL = 100
TOP_K_FINAL = 50
FUSED_ALPHA = 0.60


def _build_tagged_article_text(row: dict) -> str:
    parts = []
    name = str(row.get("prod_name", "")).strip()
    if name:
        parts.append(name)
    desc = str(row.get("detail_desc", "")).strip()
    if desc:
        parts.append(desc[:150])
    text = " | ".join(parts)
    for tag, field in [
        ("[COLOR]", "colour_group_name"),
        ("[TYPE]", "product_type_name"),
        ("[SEC]", "section_name"),
        ("[GROUP]", "product_group_name"),
    ]:
        val = str(row.get(field, "")).strip()
        if val:
            text += f" {tag} {val}"
    return text


def ce_rerank(
    queries: list[str],
    candidate_lists: list[list[str]],
    articles: dict[str, dict],
    model_name: str,
    text_fn=_build_article_text,
    batch_size: int = 64,
    top_k: int = TOP_K_FINAL,
) -> list[list[str]]:
    ce = CrossEncoder(model_name, max_length=512)
    text_cache: dict[str, str] = {}

    def get_text(aid: str) -> str:
        if aid not in text_cache:
            text_cache[aid] = text_fn(articles.get(aid, {}))
        return text_cache[aid]

    results = []
    for query, candidates in tqdm(
        zip(queries, candidate_lists),
        total=len(queries),
        desc=f"CE({Path(model_name).stem[:20]})",
        ncols=80,
    ):
        if not candidates:
            results.append([])
            continue
        pairs = [(query, get_text(cid)) for cid in candidates]
        scores = ce.predict(pairs, batch_size=batch_size, show_progress_bar=False)
        ranked = sorted(zip(candidates, scores), key=lambda x: -x[1])
        results.append([cid for cid, _ in ranked[:top_k]])
    return results


def main():
    p = argparse.ArgumentParser(description="Phase 3.9 — comprehensive evaluation")
    p.add_argument("--n_queries", type=int, default=2000)
    args = p.parse_args()

    t_start = time.time()
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    # ── Load data ─────────────────────────────────────────────────────────
    splits = json.loads(SPLIT_PATH.read_text())
    test_qids = set(splits["test"])

    articles_df = pd.read_csv(HNM_DIR / "articles.csv", dtype=str).fillna("")
    articles = {str(r["article_id"]).strip(): r.to_dict()
                for _, r in articles_df.iterrows()}

    clip_embs = np.load(
        str(EMB_DIR / "fashion-clip_finetuned_embeddings.npy")).astype(np.float32)
    with open(EMB_DIR / "fashion-clip_finetuned_article_ids.json") as f:
        article_ids = json.load(f)
    norms = np.linalg.norm(clip_embs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    clip_embs /= norms

    queries_df = pd.read_csv(HNM_DIR / "queries.csv", dtype=str)
    queries = [
        (str(r["query_id"]), str(r["query_text"]))
        for _, r in queries_df.iterrows()
        if str(r["query_id"]) in test_qids
    ]
    if args.n_queries > 0 and len(queries) > args.n_queries:
        queries = random.Random(42).sample(queries, args.n_queries)
    log.info("Test queries: %d", len(queries))

    qrels_raw = pd.read_csv(HNM_DIR / "qrels.csv", dtype=str)
    qrels: dict[str, dict[str, int]] = {}
    for _, r in qrels_raw.iterrows():
        qid = str(r["query_id"])
        qrels[qid] = {aid: 1 for aid in str(r.get("positive_ids", "")).split()}

    # ── Encode queries ────────────────────────────────────────────────────
    model_clip, _, tokenizer = load_clip_model("fashion-clip", device=device)
    ft_path = _REPO_ROOT / "models" / "moda-fashionclip-finetuned" / "best"
    if (ft_path / "model_state_dict.pt").exists():
        sd = torch.load(ft_path / "model_state_dict.pt", map_location="cpu")
        model_clip.load_state_dict(sd, strict=False)
        model_clip = model_clip.to(device).eval()
        log.info("Using fine-tuned FashionCLIP")

    query_texts = [q[1] for q in queries]
    query_clip_embs = encode_texts_clip(
        query_texts, model_clip, tokenizer, device, batch_size=128)
    qn = np.linalg.norm(query_clip_embs, axis=1, keepdims=True)
    qn[qn == 0] = 1.0
    query_clip_embs /= qn
    del model_clip, tokenizer
    if device == "mps":
        torch.mps.empty_cache()

    # ── Dense-only retrieval ──────────────────────────────────────────────
    log.info("Dense-only retrieval...")
    dense_results: dict[str, list[str]] = {}
    for i, (qid, _) in enumerate(tqdm(queries, desc="Dense", ncols=80)):
        scores = clip_embs @ query_clip_embs[i]
        top_idx = np.argpartition(-scores, TOP_K_RETRIEVAL)[:TOP_K_RETRIEVAL]
        top_idx = top_idx[np.argsort(-scores[top_idx])]
        dense_results[qid] = [article_ids[j] for j in top_idx]

    # ── Fused-interpolated retrieval ──────────────────────────────────────
    fused_ckpt = FUSED_MODEL_DIR / "best" / "model_state_dict.pt"
    has_fused = fused_ckpt.exists()

    fused_results: dict[str, list[str]] = {}
    if has_fused:
        log.info("Building fused embeddings...")
        vocabs = build_vocabularies(articles_df)
        fused_model = FusedRetriever(
            clip_dim=512,
            color_vocab_size=len(vocabs["color"]),
            category_vocab_size=len(vocabs["category"]),
            group_vocab_size=len(vocabs["group"]),
            out_dim=FUSED_DIM,
        )
        sd = torch.load(fused_ckpt, map_location="cpu", weights_only=True)
        fused_model.load_state_dict(sd)
        fused_model = fused_model.to(device).eval()

        art_map = articles
        color_idx = [vocabs["color"].get(
            str(art_map.get(a, {}).get(FIELD_COLS["color"], "")).strip(), 0)
            for a in article_ids]
        cat_idx = [vocabs["category"].get(
            str(art_map.get(a, {}).get(FIELD_COLS["category"], "")).strip(), 0)
            for a in article_ids]
        grp_idx = [vocabs["group"].get(
            str(art_map.get(a, {}).get(FIELD_COLS["group"], "")).strip(), 0)
            for a in article_ids]

        n = len(article_ids)
        fused_embs = np.zeros((n, FUSED_DIM), dtype=np.float32)
        with torch.no_grad():
            for s in tqdm(range(0, n, 1024), desc="Fused embed", ncols=80):
                e = min(s + 1024, n)
                fused_embs[s:e] = fused_model.encode_item(
                    torch.from_numpy(clip_embs[s:e]).to(device),
                    torch.tensor(color_idx[s:e], device=device),
                    torch.tensor(cat_idx[s:e], device=device),
                    torch.tensor(grp_idx[s:e], device=device),
                ).cpu().numpy()

        query_fused = np.zeros((len(queries), FUSED_DIM), dtype=np.float32)
        with torch.no_grad():
            for s in range(0, len(queries), 512):
                e = min(s + 512, len(queries))
                query_fused[s:e] = fused_model.encode_query(
                    torch.from_numpy(query_clip_embs[s:e]).to(device)
                ).cpu().numpy()

        del fused_model
        if device == "mps":
            torch.mps.empty_cache()

        log.info("Fused-interpolated retrieval (alpha=%.2f)...", FUSED_ALPHA)
        for i, (qid, _) in enumerate(tqdm(queries, desc="Fused-interp", ncols=80)):
            ds = clip_embs @ query_clip_embs[i]
            fs = fused_embs @ query_fused[i]
            dn = (ds - ds.mean()) / (ds.std() + 1e-8)
            fn = (fs - fs.mean()) / (fs.std() + 1e-8)
            combined = FUSED_ALPHA * dn + (1 - FUSED_ALPHA) * fn
            top_idx = np.argpartition(-combined, TOP_K_RETRIEVAL)[:TOP_K_RETRIEVAL]
            top_idx = top_idx[np.argsort(-combined[top_idx])]
            fused_results[qid] = [article_ids[j] for j in top_idx]
    else:
        log.warning("No fused model found — skipping fused retrieval")

    # ── CE models to test ─────────────────────────────────────────────────
    ce_configs = [("NoRerank", None, None)]
    ce_configs.append(("OffshelfCE", OFFSHELF_CE, _build_article_text))
    if Path(LLM_CE).exists():
        ce_configs.append(("LLM_CE", LLM_CE, _build_article_text))
    if Path(ATTR_CE).exists():
        ce_configs.append(("AttrCE", ATTR_CE, _build_tagged_article_text))

    # ── Evaluate all combinations ─────────────────────────────────────────
    retrieval_sets = [("Dense", dense_results)]
    if has_fused:
        retrieval_sets.append(("Fused", fused_results))

    all_results = {}
    for ret_name, ret_results in retrieval_sets:
        ret_lists = [ret_results.get(qid, []) for qid, _ in queries]

        for ce_name, ce_model, text_fn in ce_configs:
            config_name = f"{ret_name}_{ce_name}"

            if ce_model is None:
                final_results = ret_results
            else:
                log.info("Reranking: %s + %s", ret_name, ce_name)
                reranked = ce_rerank(
                    query_texts, ret_lists, articles,
                    model_name=ce_model, text_fn=text_fn)
                final_results = {
                    qid: lst for (qid, _), lst in zip(queries, reranked)}

            res = evaluate(final_results, qrels, label=config_name)
            all_results[config_name] = res

    # ── Print results ─────────────────────────────────────────────────────
    baseline_ndcg = all_results["Dense_NoRerank"]["metrics"]["ndcg@10"]

    print("\n" + "=" * 90)
    print("PHASE 3.9 — Comprehensive Evaluation (Path A + Path B)")
    print(f"  {len(queries):,} test queries | Retrieval: top-{TOP_K_RETRIEVAL}"
          f" → Rerank: top-{TOP_K_FINAL}")
    if has_fused:
        print(f"  Fused alpha: {FUSED_ALPHA} (={FUSED_ALPHA:.0%} dense"
              f" + {1-FUSED_ALPHA:.0%} fused)")
    print("=" * 90)
    print(f"  {'Config':<35} {'nDCG@10':>9} {'MRR':>9} {'R@10':>9}"
          f"  {'vs Dense_NoRerank':>16}")
    print("-" * 90)

    for name, res in all_results.items():
        m = res["metrics"]
        ndcg = m["ndcg@10"]
        delta = (ndcg / baseline_ndcg - 1) * 100 if baseline_ndcg > 0 else 0
        sign = "+" if delta >= 0 else ""
        print(f"  {name:<35} {ndcg:>9.4f} {m['mrr']:>9.4f}"
              f" {m['recall@10']:>9.4f}  {sign}{delta:>6.1f}%")

    print("=" * 90)

    if has_fused:
        dense_best = max(
            (r for k, r in all_results.items() if k.startswith("Dense_")),
            key=lambda r: r["metrics"]["ndcg@10"],
        )
        fused_best = max(
            (r for k, r in all_results.items() if k.startswith("Fused_")),
            key=lambda r: r["metrics"]["ndcg@10"],
        )
        d_ndcg = dense_best["metrics"]["ndcg@10"]
        f_ndcg = fused_best["metrics"]["ndcg@10"]
        delta = (f_ndcg / d_ndcg - 1) * 100
        print(f"\n  Best Dense config:  {dense_best['config']:<30} nDCG@10={d_ndcg:.4f}")
        print(f"  Best Fused config:  {fused_best['config']:<30} nDCG@10={f_ndcg:.4f}"
              f"  ({'+' if delta >= 0 else ''}{delta:.1f}%)")

    print(f"\n  Total elapsed: {(time.time() - t_start) / 60:.1f} min\n")

    out_path = RESULTS_DIR / "phase3_9_comprehensive_eval.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    log.info("Results saved → %s", out_path)


if __name__ == "__main__":
    main()
