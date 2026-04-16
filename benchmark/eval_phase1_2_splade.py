"""
Phase 1 & 2 (H&M Tier-2 style) with SPLADE replacing BM25.

Maps original EXPERIMENT_LOG phases to sparse=SPLADE:

  Phase 1 (retrieval baselines on test split)
    - BM25-only           → SPLADE-only
    - FashionCLIP dense   → unchanged (same Marqo FashionCLIP FAISS path)

  Phase 2 (hybrid + CE)
    - BM25 + FashionCLIP RRF (weight sweep A–D) → SPLADE + FashionCLIP RRF
      Same weights as Phase 2 table: (0.2,0.8), (0.3,0.7), (0.4,0.6), (0.5,0.5)
    - NER-boosted BM25 + hybrid → optional: SPLADE query expanded with GLiNER
      entities (text append) + same RRF vs dense; dense still uses raw query text.
    - CE ablations (cf. Phase 2): off-shelf ms-marco L6 + LLM-trained CE on
      SPLADE-only pool and on best Phase-2 RRF fusion.

Usage:
  python -m benchmark.eval_phase1_2_splade
  python -m benchmark.eval_phase1_2_splade --n_queries 500 --no_llm_ce
  python -m benchmark.eval_phase1_2_splade --ner   # needs NER cache covering queries

Output: results/real/phase1_2_splade_eval.json
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
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
    load_or_compute_ner,
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

OFF_SHELF_CE = "cross-encoder/ms-marco-MiniLM-L-6-v2"

OVERNIGHT_CACHE = RESULTS_DIR / "overnight_cache"
PHASE1_2_PART1_PKL = OVERNIGHT_CACHE / "phase1_2_part1.pkl"

# Same weight grid as Phase 2 hybrid (BM25 / dense) → (SPLADE / dense)
PHASE2_RRF_WEIGHTS = [
    (0.2, 0.8, "Phase2_A_SPL02_DN08"),
    (0.3, 0.7, "Phase2_B_SPL03_DN07"),
    (0.4, 0.6, "Phase2_C_SPL04_DN06"),
    (0.5, 0.5, "Phase2_D_SPL05_DN05"),
]


def _ner_expand_text(qid: str, base: str, ner_cache: dict) -> str:
    ent = ner_cache.get(qid, {})
    extra: list[str] = []
    for _label, vals in ent.items():
        extra.extend(vals)
    if not extra:
        return base
    dedup = list(dict.fromkeys(extra))
    return f"{base} {' '.join(dedup)}"


def main() -> dict:
    parser = argparse.ArgumentParser(description="Phase 1–2 eval with SPLADE (not BM25)")
    parser.add_argument("--n_queries", type=int, default=0, help="0 = all test queries")
    parser.add_argument("--top_k_rerank", type=int, default=TOP_K_RERANK)
    parser.add_argument("--splade_batch_size", type=int, default=32)
    parser.add_argument("--splade_query_chunk", type=int, default=1000,
                        help="Queries per chunk for sparse dot product (lower = less RAM)")
    parser.add_argument("--force_encode", action="store_true")
    parser.add_argument("--faiss_query_chunk", type=int, default=512)
    parser.add_argument("--skip_dense", action="store_true", help="Skip FashionCLIP dense (Phase 1 dense)")
    parser.add_argument("--skip_hybrid", action="store_true", help="Skip SPLADE+dense RRF sweep")
    parser.add_argument("--no_offshelf_ce", action="store_true")
    parser.add_argument("--no_llm_ce", action="store_true")
    parser.add_argument(
        "--skip_known",
        action="store_true",
        help="Skip experiments already logged (SPLADE-only, FashionCLIP dense, SPLADE+LLM CE)",
    )
    parser.add_argument(
        "--ner",
        action="store_true",
        help="Run NER-expanded SPLADE + hybrid (GLiNER v1 cache; may recompute if missing)",
    )
    parser.add_argument(
        "--mode",
        choices=("full", "part1", "part2"),
        default="full",
        help="full=all in one process; part1=retrieval+dense then checkpoint; part2=CE+hybrid from checkpoint",
    )
    args = parser.parse_args()

    t0 = time.time()
    all_results: dict[str, dict] = {}

    if args.mode == "part2":
        if not PHASE1_2_PART1_PKL.exists():
            log.error("Missing checkpoint %s — run --mode part1 first", PHASE1_2_PART1_PKL)
            sys.exit(1)
        with open(PHASE1_2_PART1_PKL, "rb") as f:
            ckpt = pickle.load(f)
        queries = ckpt["queries"]
        qrels = ckpt["qrels"]
        qids = ckpt["qids"]
        texts = ckpt["texts"]
        splade_lists = ckpt["splade_lists"]
        dense_lists = ckpt.get("dense_lists")
        all_results = ckpt["all_results"]
        article_ids = ckpt["article_ids"]
        article_texts = ckpt["article_texts"]
        log.info("Loaded part1 checkpoint (%d queries)", len(queries))
        articles_for_ce = load_articles()
        splade = SpladeRetriever(device="mps" if torch.backends.mps.is_available() else "cpu")
        splade.encode_articles(
            article_ids,
            article_texts,
            batch_size=args.splade_batch_size,
            force=False,
        )
        # Fall through to CE / hybrid block only (skip retrieval above)
        goto_ce_hybrid = True
    else:
        goto_ce_hybrid = False

    if not goto_ce_hybrid:
        test_qids = load_test_query_ids()
        all_queries, all_qrels = load_benchmark(n_queries=None)
        queries = [(qid, qt) for qid, qt in all_queries if qid in test_qids]
        qrels = {qid: v for qid, v in all_qrels.items() if qid in test_qids}

        if args.n_queries > 0 and len(queries) > args.n_queries:
            import random

            queries = random.Random(42).sample(queries, args.n_queries)

        qids = [q[0] for q in queries]
        texts = [q[1] for q in queries]
        log.info("Loaded %d test queries", len(queries))

        article_ids, article_texts = load_article_ids_and_texts()
        if args.mode == "full":
            articles_for_ce = load_articles()
        else:
            articles_for_ce = None

        splade = SpladeRetriever(device="mps" if torch.backends.mps.is_available() else "cpu")
        splade.encode_articles(
            article_ids,
            article_texts,
            batch_size=args.splade_batch_size,
            force=args.force_encode,
        )

        splade_lists = splade.search_batch(texts, top_k=args.top_k_rerank,
                                             query_chunk=args.splade_query_chunk)
        if args.skip_known:
            log.info("SKIP Phase1_SPLADE_only (already logged: nDCG@10=0.0464)")
        else:
            splade_dict = {qid: lst for qid, lst in zip(qids, splade_lists)}
            all_results["Phase1_SPLADE_only"] = evaluate(splade_dict, qrels, label="Phase1_SPLADE_only")

        # Free SPLADE model before loading FashionCLIP (avoids MPS memory segfault)
        if not args.ner:
            splade.free_model()

        dense_lists = None
        if not args.skip_dense:
            log.info("Phase 1 — FashionCLIP dense (same as original Phase 1 dense baseline path)")
            dense_lists = dense_retrieval_fashionclip(
                queries,
                article_ids,
                top_k=args.top_k_rerank,
                faiss_query_chunk=args.faiss_query_chunk,
            )
            if args.skip_known:
                log.info("SKIP Phase1_FashionCLIP_dense (already logged: nDCG@10=0.0300)")
            else:
                dense_dict = {qid: lst for qid, lst in zip(qids, dense_lists)}
                all_results["Phase1_FashionCLIP_dense"] = evaluate(
                    dense_dict, qrels, label="Phase1_FashionCLIP_dense",
                )

        if args.mode == "part1":
            OVERNIGHT_CACHE.mkdir(parents=True, exist_ok=True)
            ckpt = {
                "queries": queries,
                "qrels": qrels,
                "qids": qids,
                "texts": texts,
                "splade_lists": splade_lists,
                "dense_lists": dense_lists,
                "all_results": all_results,
                "article_ids": article_ids,
                "article_texts": article_texts,
                "args": vars(args),
            }
            with open(PHASE1_2_PART1_PKL, "wb") as f:
                pickle.dump(ckpt, f)
            log.info("part1 checkpoint saved → %s", PHASE1_2_PART1_PKL)
            return all_results

    if articles_for_ce is None:
        articles_for_ce = load_articles()

    if not args.no_offshelf_ce:
        log.info("Phase 2 style — SPLADE + off-shelf CE (ms-marco L6)...")
        splade_offshelf = ce_rerank_batch(
            texts, splade_lists, articles_for_ce,
            model_name=OFF_SHELF_CE, top_k=10,
        )
        all_results["Phase2_SPLADE_OffshelfCE"] = evaluate(
            {qid: lst for qid, lst in zip(qids, splade_offshelf)},
            qrels,
            label="Phase2_SPLADE_OffshelfCE",
        )

    if args.skip_known:
        log.info("SKIP SPLADE_LLMCE (already logged: nDCG@10=0.0903)")
    elif not args.no_llm_ce and Path(LLM_TRAINED_CE).exists():
        log.info("Phase 2 style — SPLADE + LLM-trained CE...")
        splade_llm = ce_rerank_batch(
            texts, splade_lists, articles_for_ce,
            model_name=LLM_TRAINED_CE, top_k=10,
        )
        all_results["SPLADE_LLMCE"] = evaluate(
            {qid: lst for qid, lst in zip(qids, splade_llm)},
            qrels,
            label="SPLADE_LLMCE",
        )
    elif not args.no_llm_ce:
        log.warning("LLM CE not found at %s — skip", LLM_TRAINED_CE)

    best_hybrid_name = None
    best_hybrid_ndcg = -1.0
    best_hybrid_lists: list[list[str]] | None = None

    if not args.skip_hybrid and dense_lists is not None:
        log.info("Phase 2 — SPLADE + FashionCLIP RRF (same weight grid as BM25 hybrid Phase 2)")
        for w_sp, w_dn, label in PHASE2_RRF_WEIGHTS:
            hybrid_lists = rrf_fusion(
                splade_lists,
                dense_lists,
                k=RRF_K,
                bm25_weight=w_sp,
                dense_weight=w_dn,
                top_k=args.top_k_rerank,
            )
            hybrid_dict = {qid: lst for qid, lst in zip(qids, hybrid_lists)}
            res = evaluate(hybrid_dict, qrels, label=label)
            all_results[label] = res
            ndcg = res["metrics"]["ndcg@10"]
            if ndcg > best_hybrid_ndcg:
                best_hybrid_ndcg = ndcg
                best_hybrid_name = label
                best_hybrid_lists = hybrid_lists

        log.info("Best Phase-2 RRF fusion: %s (nDCG@10=%.4f)", best_hybrid_name, best_hybrid_ndcg)

        if not args.no_offshelf_ce and best_hybrid_lists is not None:
            h_off = ce_rerank_batch(
                texts, best_hybrid_lists, articles_for_ce,
                model_name=OFF_SHELF_CE, top_k=10,
            )
            all_results["Phase2_BestHybrid_OffshelfCE"] = evaluate(
                {qid: lst for qid, lst in zip(qids, h_off)},
                qrels,
                label="Phase2_BestHybrid_OffshelfCE",
            )

        if (
            not args.no_llm_ce
            and Path(LLM_TRAINED_CE).exists()
            and best_hybrid_lists is not None
        ):
            h_llm = ce_rerank_batch(
                texts, best_hybrid_lists, articles_for_ce,
                model_name=LLM_TRAINED_CE, top_k=10,
            )
            all_results["SPLADE_BestHybrid_LLMCE"] = evaluate(
                {qid: lst for qid, lst in zip(qids, h_llm)},
                qrels,
                label="SPLADE_BestHybrid_LLMCE",
            )

    if args.ner and dense_lists is not None:
        log.info("NER-expanded SPLADE queries (proxy for NER-boosted BM25) + RRF with dense")
        ner_cache = load_or_compute_ner(queries, force_recompute=False)
        ner_texts = [_ner_expand_text(qid, t, ner_cache) for qid, t in queries]
        splade_ner_lists = splade.search_batch(ner_texts, top_k=args.top_k_rerank,
                                                query_chunk=args.splade_query_chunk)
        all_results["Phase2_SPLADE_NERexpand_only"] = evaluate(
            {qid: lst for qid, lst in zip(qids, splade_ner_lists)},
            qrels,
            label="Phase2_SPLADE_NERexpand_only",
        )
        best_ner_ndcg = -1.0
        best_ner_label = None
        for w_sp, w_dn, label in PHASE2_RRF_WEIGHTS:
            fused = rrf_fusion(
                splade_ner_lists,
                dense_lists,
                k=RRF_K,
                bm25_weight=w_sp,
                dense_weight=w_dn,
                top_k=args.top_k_rerank,
            )
            res = evaluate(
                {qid: lst for qid, lst in zip(qids, fused)},
                qrels,
                label=f"NER_{label}",
            )
            all_results[f"NER_{label}"] = res
            n = res["metrics"]["ndcg@10"]
            if n > best_ner_ndcg:
                best_ner_ndcg = n
                best_ner_label = f"NER_{label}"
        log.info("Best NER hybrid: %s (nDCG@10=%.4f)", best_ner_label, best_ner_ndcg)

    out_path = RESULTS_DIR / "phase1_2_splade_eval.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    log.info("Wrote %s (%.1f min)", out_path, (time.time() - t0) / 60.0)
    return all_results


if __name__ == "__main__":
    main()
