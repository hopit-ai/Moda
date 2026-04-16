"""
MODA — Evaluate Fine-Tuned SPLADE Retriever

Re-encodes articles with the fine-tuned SPLADE model, then runs the same
evaluation pipelines as eval_phase1_2_splade.py but using the new model.

DATA LEAKAGE SAFEGUARDS:
  - Evaluation uses ONLY test-split queries (verified at startup)
  - The fine-tuned model was trained on train-split only
  - Article embeddings are re-computed with the fine-tuned model (separate
    cache from off-shelf model)
  - Comparison: off-shelf → fine-tuned on identical test queries

Pipelines evaluated:
  1. Fine-tuned SPLADE standalone
  2. Fine-tuned SPLADE + FashionCLIP RRF (weight sweep)
  3. Fine-tuned SPLADE + off-shelf CE rerank
  4. Fine-tuned SPLADE + LLM-trained CE rerank
  5. Best hybrid + CE rerank

Usage:
  python -m benchmark.eval_finetuned_splade
  python -m benchmark.eval_finetuned_splade --compare  # side-by-side with off-shelf
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from benchmark.eval_full_pipeline import (
    RESULTS_DIR,
    TOP_K_RERANK,
    ce_rerank_batch,
    evaluate,
    load_articles,
    load_benchmark,
    rrf_fusion,
)
from benchmark.eval_splade_pipeline import (
    LLM_TRAINED_CE,
    RRF_K,
    dense_retrieval_fashionclip,
)
from benchmark.eval_three_tower import load_article_ids_and_texts, load_test_query_ids
from benchmark.splade_retriever import SpladeRetriever

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

MODEL_DIR = _REPO_ROOT / "models"
FINETUNED_MODEL = str(MODEL_DIR / "moda-splade-finetuned")
OFF_SHELF_CE = "cross-encoder/ms-marco-MiniLM-L-6-v2"
SPLIT_PATH = _REPO_ROOT / "data" / "processed" / "query_splits.json"

PHASE2_RRF_WEIGHTS = [
    (0.2, 0.8, "FT_SPL02_DN08"),
    (0.3, 0.7, "FT_SPL03_DN07"),
    (0.4, 0.6, "FT_SPL04_DN06"),
    (0.5, 0.5, "FT_SPL05_DN05"),
]


def verify_test_only(test_qids: set[str]) -> None:
    """Verify loaded test query IDs are disjoint from train and val."""
    splits = json.loads(SPLIT_PATH.read_text())
    train = set(splits["train"])
    val = set(splits["val"])
    test_from_file = set(splits["test"])

    assert test_qids == test_from_file, (
        f"Test QIDs mismatch: got {len(test_qids)}, expected {len(test_from_file)}"
    )
    assert len(test_qids & train) == 0, "LEAKAGE: test ∩ train!"
    assert len(test_qids & val) == 0, "LEAKAGE: test ∩ val!"
    log.info("Test split verified: %d queries, disjoint from train/val", len(test_qids))


def main() -> dict:
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned SPLADE")
    parser.add_argument("--n_queries", type=int, default=0)
    parser.add_argument("--top_k_rerank", type=int, default=TOP_K_RERANK)
    parser.add_argument("--splade_batch_size", type=int, default=32)
    parser.add_argument("--splade_query_chunk", type=int, default=1000)
    parser.add_argument("--faiss_query_chunk", type=int, default=512)
    parser.add_argument("--no_offshelf_ce", action="store_true")
    parser.add_argument("--no_llm_ce", action="store_true")
    parser.add_argument("--skip_dense", action="store_true")
    parser.add_argument("--skip_hybrid", action="store_true")
    parser.add_argument("--compare", action="store_true",
                        help="Also run off-shelf SPLADE for side-by-side comparison")
    parser.add_argument("--model_path", type=str, default=FINETUNED_MODEL,
                        help="Path to fine-tuned SPLADE model")
    args = parser.parse_args()

    t0 = time.time()
    all_results: dict[str, dict] = {}

    # ── 1. Verify splits (leakage check) ─────────────────────────────────
    from benchmark.leakage_guard import run_all_checks
    run_all_checks(split_path=SPLIT_PATH)

    test_qids = load_test_query_ids()
    verify_test_only(test_qids)

    all_queries, all_qrels = load_benchmark(n_queries=None)
    queries = [(qid, qt) for qid, qt in all_queries if qid in test_qids]
    qrels = {qid: v for qid, v in all_qrels.items() if qid in test_qids}

    if args.n_queries > 0 and len(queries) > args.n_queries:
        import random
        queries = random.Random(42).sample(queries, args.n_queries)

    qids = [q[0] for q in queries]
    texts = [q[1] for q in queries]
    log.info("Test queries: %d", len(queries))

    article_ids, article_texts = load_article_ids_and_texts()
    articles_for_ce = load_articles()

    # ── 2. Fine-tuned SPLADE retrieval ───────────────────────────────────
    if not Path(args.model_path).exists():
        log.error("Fine-tuned model not found at %s — run train_splade.py first",
                  args.model_path)
        sys.exit(1)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    log.info("Loading fine-tuned SPLADE from %s", args.model_path)
    ft_splade = SpladeRetriever(
        model_name=args.model_path, device=device,
    )

    # Force re-encode articles with fine-tuned model (separate cache)
    ft_splade.encode_articles(
        article_ids, article_texts,
        batch_size=args.splade_batch_size,
        force=True,
    )

    ft_splade_lists = ft_splade.search_batch(
        texts, top_k=args.top_k_rerank,
        query_chunk=args.splade_query_chunk,
    )
    ft_splade.free_model()

    ft_dict = {qid: lst for qid, lst in zip(qids, ft_splade_lists)}
    all_results["FT_SPLADE_only"] = evaluate(ft_dict, qrels, label="FT_SPLADE_only")
    log.info("FT SPLADE standalone: nDCG@10=%.4f",
             all_results["FT_SPLADE_only"]["metrics"]["ndcg@10"])

    # ── 3. Optional: off-shelf comparison ────────────────────────────────
    offshelf_lists = None
    if args.compare:
        log.info("Running off-shelf SPLADE for comparison...")
        os_splade = SpladeRetriever(device=device)
        os_splade.encode_articles(article_ids, article_texts,
                                  batch_size=args.splade_batch_size, force=False)
        offshelf_lists = os_splade.search_batch(
            texts, top_k=args.top_k_rerank,
            query_chunk=args.splade_query_chunk,
        )
        os_splade.free_model()

        os_dict = {qid: lst for qid, lst in zip(qids, offshelf_lists)}
        all_results["OffShelf_SPLADE_only"] = evaluate(os_dict, qrels,
                                                        label="OffShelf_SPLADE_only")
        os_ndcg = all_results["OffShelf_SPLADE_only"]["metrics"]["ndcg@10"]
        ft_ndcg = all_results["FT_SPLADE_only"]["metrics"]["ndcg@10"]
        delta = ft_ndcg - os_ndcg
        log.info("Off-shelf: nDCG@10=%.4f | Fine-tuned: %.4f | Delta: %+.4f",
                 os_ndcg, ft_ndcg, delta)

    # ── 4. FashionCLIP dense ─────────────────────────────────────────────
    dense_lists = None
    if not args.skip_dense:
        log.info("Running FashionCLIP dense retrieval...")
        dense_lists = dense_retrieval_fashionclip(
            queries, article_ids,
            top_k=args.top_k_rerank,
            faiss_query_chunk=args.faiss_query_chunk,
        )
        dense_dict = {qid: lst for qid, lst in zip(qids, dense_lists)}
        all_results["FashionCLIP_dense"] = evaluate(
            dense_dict, qrels, label="FashionCLIP_dense",
        )

    # ── 5. FT-SPLADE + FashionCLIP RRF (weight sweep) ───────────────────
    best_hybrid_name = None
    best_hybrid_ndcg = -1.0
    best_hybrid_lists = None

    if not args.skip_hybrid and dense_lists is not None:
        log.info("FT-SPLADE + FashionCLIP RRF weight sweep...")
        for w_sp, w_dn, label in PHASE2_RRF_WEIGHTS:
            hybrid = rrf_fusion(
                ft_splade_lists, dense_lists,
                k=RRF_K, bm25_weight=w_sp, dense_weight=w_dn,
                top_k=args.top_k_rerank,
            )
            h_dict = {qid: lst for qid, lst in zip(qids, hybrid)}
            res = evaluate(h_dict, qrels, label=label)
            all_results[label] = res
            ndcg = res["metrics"]["ndcg@10"]
            if ndcg > best_hybrid_ndcg:
                best_hybrid_ndcg = ndcg
                best_hybrid_name = label
                best_hybrid_lists = hybrid

        log.info("Best RRF fusion: %s (nDCG@10=%.4f)", best_hybrid_name, best_hybrid_ndcg)

    # ── 6. CE reranking ──────────────────────────────────────────────────
    if not args.no_offshelf_ce:
        log.info("FT-SPLADE + off-shelf CE (ms-marco L6)...")
        ft_ce_off = ce_rerank_batch(
            texts, ft_splade_lists, articles_for_ce,
            model_name=OFF_SHELF_CE, top_k=10,
        )
        all_results["FT_SPLADE_OffshelfCE"] = evaluate(
            {qid: lst for qid, lst in zip(qids, ft_ce_off)},
            qrels, label="FT_SPLADE_OffshelfCE",
        )

    if not args.no_llm_ce and Path(LLM_TRAINED_CE).exists():
        log.info("FT-SPLADE + LLM-trained CE...")
        ft_ce_llm = ce_rerank_batch(
            texts, ft_splade_lists, articles_for_ce,
            model_name=LLM_TRAINED_CE, top_k=10,
        )
        all_results["FT_SPLADE_LLMCE"] = evaluate(
            {qid: lst for qid, lst in zip(qids, ft_ce_llm)},
            qrels, label="FT_SPLADE_LLMCE",
        )
    elif not args.no_llm_ce:
        log.warning("LLM CE not found at %s — skip", LLM_TRAINED_CE)

    if not args.no_offshelf_ce and best_hybrid_lists is not None:
        log.info("Best hybrid + off-shelf CE...")
        h_ce_off = ce_rerank_batch(
            texts, best_hybrid_lists, articles_for_ce,
            model_name=OFF_SHELF_CE, top_k=10,
        )
        all_results["FT_BestHybrid_OffshelfCE"] = evaluate(
            {qid: lst for qid, lst in zip(qids, h_ce_off)},
            qrels, label="FT_BestHybrid_OffshelfCE",
        )

    if (
        not args.no_llm_ce
        and Path(LLM_TRAINED_CE).exists()
        and best_hybrid_lists is not None
    ):
        log.info("Best hybrid + LLM-trained CE...")
        h_ce_llm = ce_rerank_batch(
            texts, best_hybrid_lists, articles_for_ce,
            model_name=LLM_TRAINED_CE, top_k=10,
        )
        all_results["FT_BestHybrid_LLMCE"] = evaluate(
            {qid: lst for qid, lst in zip(qids, h_ce_llm)},
            qrels, label="FT_BestHybrid_LLMCE",
        )

    # ── 7. Summary ───────────────────────────────────────────────────────
    out_path = RESULTS_DIR / "finetuned_splade_eval.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)

    elapsed = time.time() - t0
    log.info("=" * 60)
    log.info("FINE-TUNED SPLADE EVALUATION COMPLETE (%.1f min)", elapsed / 60)
    log.info("-" * 60)
    for label, res in sorted(all_results.items()):
        m = res["metrics"]
        log.info("  %-35s nDCG@10=%.4f  MRR=%.4f  R@10=%.4f",
                 label, m["ndcg@10"], m.get("mrr", 0), m.get("recall@10", 0))
    log.info("=" * 60)
    log.info("Results saved to %s", out_path)
    return all_results


if __name__ == "__main__":
    main()
