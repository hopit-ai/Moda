"""
MODA Phase 3 — Combined Pipeline Evaluation (Apples-to-Apples)

Full factorial comparison of retriever × reranker on the EXACT SAME
22,855 held-out test queries.  Every config shares the same BM25-NER
stage, the same articles, and the same qrels.

Retriever variants:
  A. Baseline FashionCLIP (pre-trained Marqo/marqo-fashionCLIP)
  B. Fine-tuned FashionCLIP (Phase 3C — LLM-judged hard negatives)

Reranker variants:
  0. No reranker (hybrid retrieval only)
  1. Off-the-shelf CE (cross-encoder/ms-marco-MiniLM-L-6-v2)
  2. LLM-trained CE (Phase 3B — moda-fashion-ce-llm-best)
  3. Attr-conditioned CE (Phase 3.8 Path B — moda-fashion-ce-attr-best)

That gives 2×4 = 8 configs, all on 22,855 test queries.

Usage:
  python benchmark/eval_combined_pipeline.py
  python benchmark/eval_combined_pipeline.py --n_queries 5000  # quick test
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
from opensearchpy import OpenSearch
from tqdm import tqdm

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from sentence_transformers import CrossEncoder

from benchmark.eval_full_pipeline import (
    load_benchmark,
    load_articles,
    load_or_compute_ner,
    bm25_ner_search,
    dense_search_batch,
    rrf_fusion,
    evaluate,
    ce_rerank_batch,
    _build_article_text,
    RESULTS_DIR,
    DENSE_MODEL,
    BM25_WEIGHT,
    DENSE_WEIGHT,
    TOP_K_RERANK,
    TOP_K_FINAL,
)
from benchmark.eval_finetuned_biencoder import (
    dense_search_finetuned,
    load_test_data as load_test_data_bienc,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

SPLIT_PATH = _REPO_ROOT / "data" / "processed" / "query_splits.json"
MODEL_DIR = _REPO_ROOT / "models"
FINETUNED_BIENC = MODEL_DIR / "moda-fashionclip-finetuned" / "best"
OFFSHELF_CE = "cross-encoder/ms-marco-MiniLM-L-6-v2"
LLM_TRAINED_CE = str(MODEL_DIR / "moda-fashion-ce-llm-best")
ATTR_CE = str(MODEL_DIR / "moda-fashion-ce-attr-best")


def _build_tagged_article_text(row: dict) -> str:
    """Attribute-tagged product text for the Attr-conditioned CE."""
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


def ce_rerank_batch_custom(
    queries: list[str],
    candidate_lists: list[list[str]],
    articles: dict[str, dict],
    model_name: str,
    text_fn=_build_article_text,
    batch_size: int = 64,
    top_k: int = TOP_K_FINAL,
) -> list[list[str]]:
    """CE reranking with a pluggable article text builder."""
    log.info("Loading cross-encoder: %s...", model_name)
    ce = CrossEncoder(model_name, max_length=512)
    text_cache: dict[str, str] = {}

    def get_text(aid: str) -> str:
        if aid not in text_cache:
            text_cache[aid] = text_fn(articles.get(aid, {}))
        return text_cache[aid]

    results = []
    label = Path(model_name).stem[:25]
    for query, candidates in tqdm(
        zip(queries, candidate_lists),
        total=len(queries), desc=f"CE({label})", ncols=80,
    ):
        if not candidates:
            results.append([])
            continue
        pairs = [(query, get_text(cid)) for cid in candidates]
        scores = ce.predict(pairs, batch_size=batch_size, show_progress_bar=False)
        ranked = sorted(zip(candidates, scores), key=lambda x: -x[1])
        results.append([cid for cid, _ in ranked[:top_k]])
    return results


def load_test_query_ids() -> set[str]:
    splits = json.loads(SPLIT_PATH.read_text())
    test_qids = set(splits["test"])
    train_qids = set(splits["train"])
    val_qids = set(splits["val"])
    overlap = test_qids & (train_qids | val_qids)
    assert len(overlap) == 0, f"LEAK: {len(overlap)} test qids in train/val!"
    log.info("Test split: %d query IDs (zero overlap verified)", len(test_qids))
    return test_qids


def main():
    p = argparse.ArgumentParser(
        description="MODA Phase 3 — Combined pipeline evaluation (2×4 factorial)")
    p.add_argument("--n_queries", type=int, default=0,
                   help="Sample N test queries (0 = use all ~22K)")
    p.add_argument("--ner-model", type=str, default=None,
                   help="Path to fine-tuned NER model (GLiNER2 adapter dir)")
    args = p.parse_args()

    t_start = time.time()
    log.info("=" * 80)
    log.info("MODA — Combined Pipeline Evaluation (Apples-to-Apples)")
    log.info("=" * 80)

    # ── 1. Load test split ────────────────────────────────────────────────────
    test_qids = load_test_query_ids()

    all_queries, all_qrels = load_benchmark(n_queries=None)

    queries = [(qid, qt) for qid, qt in all_queries if qid in test_qids]
    qrels = {qid: v for qid, v in all_qrels.items() if qid in test_qids}
    log.info("Filtered to test split: %d → %d queries", len(all_queries), len(queries))

    if args.n_queries > 0 and len(queries) > args.n_queries:
        import random
        rng = random.Random(42)
        queries = rng.sample(queries, args.n_queries)
        log.info("Sampled %d test queries (seed=42)", args.n_queries)

    articles = load_articles()
    qids = [q[0] for q in queries]
    texts = [q[1] for q in queries]

    # ── 2. NER (shared across all configs) ────────────────────────────────────
    ner_cache = load_or_compute_ner(queries, ner_model=args.ner_model)

    # ── 3. BM25-NER retrieval (shared) ────────────────────────────────────────
    client = OpenSearch(
        hosts=[{"host": "localhost", "port": 9200}],
        http_compress=True, timeout=30,
    )
    log.info("Running NER-boosted BM25 retrieval...")
    bm25_results: dict[str, list[str]] = {}
    for qid, text in tqdm(queries, desc="BM25+NER", ncols=80):
        ner_entities = ner_cache.get(qid, {})
        bm25_results[qid] = bm25_ner_search(
            client, text, ner_entities, top_k=TOP_K_RERANK)

    bm25_lists = [bm25_results.get(qid, []) for qid in qids]

    # ── 4a. Dense retrieval — BASELINE FashionCLIP ────────────────────────────
    log.info("Dense retrieval with BASELINE FashionCLIP...")
    dense_baseline = dense_search_batch(
        queries, model_name=DENSE_MODEL, top_k=TOP_K_RERANK)
    dense_baseline_lists = [dense_baseline.get(qid, []) for qid in qids]

    # ── 4b. Dense retrieval — FINE-TUNED FashionCLIP ──────────────────────────
    if not (FINETUNED_BIENC / "model_state_dict.pt").exists():
        log.error("Fine-tuned bi-encoder not found at %s", FINETUNED_BIENC)
        return
    log.info("Dense retrieval with FINE-TUNED FashionCLIP (Phase 3C)...")
    dense_finetuned = dense_search_finetuned(
        queries, FINETUNED_BIENC, top_k=TOP_K_RERANK)
    dense_finetuned_lists = [dense_finetuned.get(qid, []) for qid in qids]

    # ── 5. Hybrid fusion — two pools ──────────────────────────────────────────
    log.info("RRF hybrid fusion (BM25=%.1f, Dense=%.1f)...", BM25_WEIGHT, DENSE_WEIGHT)

    hybrid_baseline_lists = rrf_fusion(bm25_lists, dense_baseline_lists)
    hybrid_baseline = {qid: lst for qid, lst in zip(qids, hybrid_baseline_lists)}

    hybrid_finetuned_lists = rrf_fusion(bm25_lists, dense_finetuned_lists)
    hybrid_finetuned = {qid: lst for qid, lst in zip(qids, hybrid_finetuned_lists)}

    # ── 6. Evaluate retrieval-only (no reranker) ──────────────────────────────
    all_results = {}

    log.info("\n--- Retriever A (baseline) × No rerank ---")
    all_results["A0_Baseline_Hybrid_NoRerank"] = evaluate(
        hybrid_baseline, qrels, label="A0_Baseline_Hybrid_NoRerank")

    log.info("\n--- Retriever B (fine-tuned) × No rerank ---")
    all_results["B0_FineTuned_Hybrid_NoRerank"] = evaluate(
        hybrid_finetuned, qrels, label="B0_FineTuned_Hybrid_NoRerank")

    # ── 7. CE reranking — off-the-shelf ───────────────────────────────────────
    log.info("\n--- Retriever A × Off-shelf CE ---")
    a1_lists = ce_rerank_batch(
        texts, hybrid_baseline_lists, articles, model_name=OFFSHELF_CE)
    a1_results = {qid: lst for qid, lst in zip(qids, a1_lists)}
    all_results["A1_Baseline_OffshelfCE"] = evaluate(
        a1_results, qrels, label="A1_Baseline_OffshelfCE")

    log.info("\n--- Retriever B × Off-shelf CE ---")
    b1_lists = ce_rerank_batch(
        texts, hybrid_finetuned_lists, articles, model_name=OFFSHELF_CE)
    b1_results = {qid: lst for qid, lst in zip(qids, b1_lists)}
    all_results["B1_FineTuned_OffshelfCE"] = evaluate(
        b1_results, qrels, label="B1_FineTuned_OffshelfCE")

    # ── 8. CE reranking — LLM-trained (Phase 3B) ─────────────────────────────
    llm_ce_path = Path(LLM_TRAINED_CE)
    if llm_ce_path.exists():
        log.info("\n--- Retriever A × LLM-trained CE ---")
        a2_lists = ce_rerank_batch(
            texts, hybrid_baseline_lists, articles, model_name=LLM_TRAINED_CE)
        a2_results = {qid: lst for qid, lst in zip(qids, a2_lists)}
        all_results["A2_Baseline_LLMtrainedCE"] = evaluate(
            a2_results, qrels, label="A2_Baseline_LLMtrainedCE")

        log.info("\n--- Retriever B × LLM-trained CE ---")
        b2_lists = ce_rerank_batch(
            texts, hybrid_finetuned_lists, articles, model_name=LLM_TRAINED_CE)
        b2_results = {qid: lst for qid, lst in zip(qids, b2_lists)}
        all_results["B2_FineTuned_LLMtrainedCE"] = evaluate(
            b2_results, qrels, label="B2_FineTuned_LLMtrainedCE")
    else:
        log.warning("LLM-trained CE not found at %s — skipping", LLM_TRAINED_CE)

    # ── 8b. CE reranking — Attr-conditioned (Phase 3.8 Path B) ───────────────
    attr_ce_path = Path(ATTR_CE)
    if attr_ce_path.exists():
        log.info("\n--- Retriever A × Attr-conditioned CE ---")
        a3_lists = ce_rerank_batch_custom(
            texts, hybrid_baseline_lists, articles,
            model_name=ATTR_CE, text_fn=_build_tagged_article_text)
        a3_results = {qid: lst for qid, lst in zip(qids, a3_lists)}
        all_results["A3_Baseline_AttrCE"] = evaluate(
            a3_results, qrels, label="A3_Baseline_AttrCE")

        log.info("\n--- Retriever B × Attr-conditioned CE ---")
        b3_lists = ce_rerank_batch_custom(
            texts, hybrid_finetuned_lists, articles,
            model_name=ATTR_CE, text_fn=_build_tagged_article_text)
        b3_results = {qid: lst for qid, lst in zip(qids, b3_lists)}
        all_results["B3_FineTuned_AttrCE"] = evaluate(
            b3_results, qrels, label="B3_FineTuned_AttrCE")
    else:
        log.warning("Attr-conditioned CE not found at %s — skipping", ATTR_CE)

    # ── 9. Save results ──────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    output = {
        "configs": all_results,
        "settings": {
            "n_queries": len(queries),
            "n_queries_with_qrels": sum(1 for q in qids if q in qrels),
            "split": "test_only",
            "test_qids_total": len(test_qids),
            "bm25_weight": BM25_WEIGHT,
            "dense_weight": DENSE_WEIGHT,
            "pool_size": TOP_K_RERANK,
            "rerank_top_k": TOP_K_FINAL,
            "offshelf_ce": OFFSHELF_CE,
            "llm_trained_ce": LLM_TRAINED_CE,
            "attr_ce": ATTR_CE,
            "finetuned_bienc": str(FINETUNED_BIENC),
            "elapsed_min": round(elapsed / 60, 1),
        },
    }
    out_path = RESULTS_DIR / "phase3_combined_eval.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    log.info("Results saved → %s", out_path)

    # ── 10. Print comparison matrix ──────────────────────────────────────────
    print("\n" + "=" * 100)
    print("MODA — Combined Pipeline Evaluation (TEST SPLIT — Apples-to-Apples)")
    print(f"  {len(queries):,} test queries  |  Pool: {TOP_K_RERANK} → Top-{TOP_K_FINAL}"
          f"  |  BM25×{BM25_WEIGHT} + Dense×{DENSE_WEIGHT}")
    print("=" * 100)
    print(f"{'Config':<45} {'nDCG@10':>9} {'MRR':>9} {'R@10':>9} {'R@50':>9}")
    print("-" * 100)

    for name, res in all_results.items():
        m = res["metrics"]
        print(
            f"  {name:<43} {m['ndcg@10']:>9.4f} {m['mrr']:>9.4f}"
            f" {m['recall@10']:>9.4f} {m.get('recall@50', 0):>9.4f}"
        )
    print("-" * 100)

    # Show the 2×4 matrix
    print("\n  RETRIEVER × RERANKER MATRIX (nDCG@10):")
    print(f"  {'':30} {'No Rerank':>12} {'Off-shelf CE':>14}"
          f" {'LLM-trained CE':>16} {'Attr-cond CE':>14}")
    print(f"  {'-'*88}")

    def get_ndcg(key):
        r = all_results.get(key)
        return r["metrics"]["ndcg@10"] if r else None

    baseline_vals = [get_ndcg("A0_Baseline_Hybrid_NoRerank"),
                     get_ndcg("A1_Baseline_OffshelfCE"),
                     get_ndcg("A2_Baseline_LLMtrainedCE"),
                     get_ndcg("A3_Baseline_AttrCE")]
    finetuned_vals = [get_ndcg("B0_FineTuned_Hybrid_NoRerank"),
                      get_ndcg("B1_FineTuned_OffshelfCE"),
                      get_ndcg("B2_FineTuned_LLMtrainedCE"),
                      get_ndcg("B3_FineTuned_AttrCE")]

    def fmt(v):
        return f"{v:.4f}" if v is not None else "  N/A"

    print(f"  {'Baseline FashionCLIP':<30} {fmt(baseline_vals[0]):>12}"
          f" {fmt(baseline_vals[1]):>14} {fmt(baseline_vals[2]):>16}"
          f" {fmt(baseline_vals[3]):>14}")
    print(f"  {'Fine-tuned FashionCLIP (3C)':<30} {fmt(finetuned_vals[0]):>12}"
          f" {fmt(finetuned_vals[1]):>14} {fmt(finetuned_vals[2]):>16}"
          f" {fmt(finetuned_vals[3]):>14}")

    # Deltas
    print(f"\n  {'Retriever lift (B vs A)':<30}", end="")
    for b, a in zip(finetuned_vals, baseline_vals):
        if b is not None and a is not None and a > 0:
            d = (b / a - 1) * 100
            print(f" {d:>+11.1f}%", end="")
        else:
            print(f" {'N/A':>12}", end="")
    print()

    # Best config
    best_key = max(all_results, key=lambda k: all_results[k]["metrics"]["ndcg@10"])
    best = all_results[best_key]
    bm = best["metrics"]
    print(f"\n  BEST CONFIG: {best_key}")
    print(f"    nDCG@10={bm['ndcg@10']:.4f}  MRR={bm['mrr']:.4f}"
          f"  Recall@10={bm['recall@10']:.4f}")

    # vs previous best (Phase 3B = 0.0747)
    prev_best = 0.0747
    delta = (bm["ndcg@10"] / prev_best - 1) * 100
    print(f"    vs Phase 3B SOTA (0.0747): {delta:+.1f}%")

    print("=" * 100)
    print(f"\nTotal elapsed: {elapsed / 60:.1f} min\n")

    return all_results


if __name__ == "__main__":
    main()
