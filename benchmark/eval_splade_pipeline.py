"""
MODA — SPLADE + FashionCLIP Evaluation Pipeline

Evaluates retrieval configurations:
  1. SPLADE-only retrieval
  2. SPLADE + LLM-trained CE rerank
  3. SPLADE + vanilla FashionCLIP (RRF fusion + weight sweep)
  4. SPLADE + vanilla FashionCLIP + CE rerank
  5. FT-FashionCLIP-only retrieval
  6. SPLADE + FT-FashionCLIP (RRF fusion + weight sweep)
  7. SPLADE + FT-FashionCLIP + CE rerank
  8. BM25 + SPLADE + Dense 3-way RRF fusion + CE rerank

Usage:
  python -m benchmark.eval_splade_pipeline
  python -m benchmark.eval_splade_pipeline --n_queries 500 --no_ce
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from benchmark.eval_full_pipeline import (
    load_benchmark,
    load_articles,
    load_or_compute_ner,
    bm25_ner_search,
    evaluate,
    ce_rerank_batch,
    rrf_fusion,
    RESULTS_DIR,
    TOP_K_RERANK,
)
from benchmark.eval_three_tower import load_article_ids_and_texts, load_test_query_ids
from benchmark.eval_finetuned_biencoder import encode_with_finetuned
from benchmark.splade_retriever import SpladeRetriever

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

MODEL_DIR = _REPO_ROOT / "models"
LLM_TRAINED_CE = str(MODEL_DIR / "moda-fashion-ce-llm-best")
EMBED_DIR = _REPO_ROOT / "data" / "processed" / "embeddings"
OVERNIGHT_CACHE = RESULTS_DIR / "overnight_cache"
SPLADE_FIRST_HALF_PKL = OVERNIGHT_CACHE / "splade_pipeline_first_half.pkl"

DENSE_MODEL = "fashion-clip"
FINETUNED_BIENC = _REPO_ROOT / "models" / "moda-fashionclip-finetuned" / "best"
FINETUNED_EMB_CACHE = EMBED_DIR / "ft_fashionclip_article_emb.npy"
FINETUNED_IDS_CACHE = EMBED_DIR / "ft_fashionclip_article_ids.json"
RRF_K = 60


# ─── FAISS subprocess search (avoids MPS + FAISS segfault on Apple Silicon) ──

def _faiss_search_subprocess(
    article_emb: np.ndarray,
    q_emb: np.ndarray,
    article_ids: list[str],
    top_k: int,
    query_chunk: int = 512,
) -> list[list[str]]:
    """Run FAISS search in a child process to isolate from MPS runtime."""
    import subprocess, tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        art_path = Path(tmpdir) / "art.npy"
        q_path = Path(tmpdir) / "q.npy"
        ids_path = Path(tmpdir) / "ids.json"
        out_path = Path(tmpdir) / "out.json"

        np.save(str(art_path), article_emb)
        np.save(str(q_path), q_emb)
        with open(ids_path, "w") as f:
            json.dump(article_ids, f)

        worker = Path(__file__).parent / "_faiss_flat_worker.py"
        cmd = [
            sys.executable, str(worker),
            str(q_path), str(art_path), str(ids_path), str(out_path),
            str(top_k), str(query_chunk),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            log.error("FAISS worker failed:\nstdout: %s\nstderr: %s",
                      result.stdout, result.stderr)
            raise RuntimeError("FAISS subprocess search failed")

        with open(out_path) as f:
            return json.load(f)


# ─── Dense retrieval (FashionCLIP via FAISS) ────────────────────────────────

def dense_retrieval_fashionclip(
    queries: list[tuple[str, str]],
    article_ids: list[str],
    top_k: int = TOP_K_RERANK,
    faiss_query_chunk: int = 512,
) -> list[list[str]]:
    """Encode queries & articles with FashionCLIP, search via FAISS subprocess."""
    import open_clip

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        "hf-hub:Marqo/marqo-fashionCLIP"
    )
    clip_tokenizer = open_clip.get_tokenizer("hf-hub:Marqo/marqo-fashionCLIP")
    clip_model = clip_model.to(device).eval()

    emb_cache = EMBED_DIR / "fashionclip_article_emb.npy"
    ids_cache = EMBED_DIR / "fashionclip_article_ids.json"

    if emb_cache.exists() and ids_cache.exists():
        log.info("Loading cached FashionCLIP article embeddings...")
        article_emb = np.load(str(emb_cache))
        with open(ids_cache) as f:
            cached_ids = json.load(f)
        if len(cached_ids) == len(article_ids):
            log.info("  Cache hit: %d articles", len(cached_ids))
        else:
            log.warning("  Cache size mismatch, re-encoding...")
            article_emb = None
    else:
        article_emb = None

    if article_emb is None:
        _, article_texts = load_article_ids_and_texts()
        texts_list = [article_texts.get(aid, "") for aid in article_ids]
        log.info("Encoding %d articles with FashionCLIP...", len(article_ids))
        all_emb = []
        bs = 128
        for start in tqdm(range(0, len(texts_list), bs), desc="FashionCLIP articles"):
            batch = texts_list[start : start + bs]
            tokens = clip_tokenizer(batch).to(device)
            with torch.no_grad():
                emb = clip_model.encode_text(tokens)
                emb = emb / emb.norm(dim=-1, keepdim=True)
            all_emb.append(emb.cpu().float().numpy())
        article_emb = np.vstack(all_emb).astype(np.float32)
        EMBED_DIR.mkdir(parents=True, exist_ok=True)
        np.save(str(emb_cache), article_emb)
        with open(ids_cache, "w") as f:
            json.dump(article_ids, f)
        log.info("  Cached to %s", emb_cache)

    log.info("Encoding %d queries with FashionCLIP...", len(queries))
    query_texts = [q[1] for q in queries]
    all_q_emb = []
    bs = 128
    for start in tqdm(range(0, len(query_texts), bs), desc="FashionCLIP queries"):
        batch = query_texts[start : start + bs]
        tokens = clip_tokenizer(batch).to(device)
        with torch.no_grad():
            emb = clip_model.encode_text(tokens)
            emb = emb / emb.norm(dim=-1, keepdim=True)
        all_q_emb.append(emb.cpu().float().numpy())
    q_emb = np.vstack(all_q_emb).astype(np.float32)

    log.info(
        "Running FAISS search in subprocess (top_k=%d, query_chunk=%d)...",
        top_k,
        faiss_query_chunk,
    )
    return _faiss_search_subprocess(article_emb, q_emb, article_ids, top_k, faiss_query_chunk)


# ─── Dense retrieval (Fine-tuned FashionCLIP via FAISS) ───────────────────────

def dense_retrieval_finetuned(
    queries: list[tuple[str, str]],
    article_ids: list[str],
    article_texts: dict[str, str],
    top_k: int = TOP_K_RERANK,
    faiss_query_chunk: int = 512,
) -> list[list[str]]:
    """Encode queries & articles with fine-tuned FashionCLIP, search via FAISS subprocess."""
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    if FINETUNED_EMB_CACHE.exists() and FINETUNED_IDS_CACHE.exists():
        log.info("Loading cached FT-FashionCLIP article embeddings...")
        article_emb = np.load(str(FINETUNED_EMB_CACHE))
        with open(FINETUNED_IDS_CACHE) as f:
            cached_ids = json.load(f)
        if len(cached_ids) != len(article_ids):
            log.warning("  FT cache size mismatch (%d vs %d), re-encoding...",
                        len(cached_ids), len(article_ids))
            article_emb = None
    else:
        article_emb = None

    if article_emb is None:
        texts_list = [article_texts.get(aid, "") for aid in article_ids]
        log.info("Encoding %d articles with FT-FashionCLIP...", len(article_ids))
        article_emb = encode_with_finetuned(texts_list, FINETUNED_BIENC, device)
        EMBED_DIR.mkdir(parents=True, exist_ok=True)
        np.save(str(FINETUNED_EMB_CACHE), article_emb)
        with open(FINETUNED_IDS_CACHE, "w") as f:
            json.dump(article_ids, f)
        log.info("  Cached FT embeddings → %s", FINETUNED_EMB_CACHE)

    query_texts = [q[1] for q in queries]
    log.info("Encoding %d queries with FT-FashionCLIP...", len(queries))
    q_emb = encode_with_finetuned(query_texts, FINETUNED_BIENC, device)

    log.info(
        "Running FAISS search in subprocess (top_k=%d, query_chunk=%d)...",
        top_k,
        faiss_query_chunk,
    )
    return _faiss_search_subprocess(article_emb, q_emb, article_ids, top_k, faiss_query_chunk)


# ─── 3-way RRF fusion (BM25 + SPLADE + Dense) ────────────────────────────────

def rrf_fusion_3way(
    bm25_lists: list[list[str]],
    splade_lists: list[list[str]],
    dense_lists: list[list[str]],
    k: int = RRF_K,
    w_bm25: float = 0.2,
    w_splade: float = 0.3,
    w_dense: float = 0.5,
    top_k: int = TOP_K_RERANK,
) -> list[list[str]]:
    fused = []
    for bm25, splade, dense in zip(bm25_lists, splade_lists, dense_lists):
        scores: dict[str, float] = {}
        for rank, doc_id in enumerate(bm25, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + w_bm25 / (k + rank)
        for rank, doc_id in enumerate(splade, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + w_splade / (k + rank)
        for rank, doc_id in enumerate(dense, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + w_dense / (k + rank)
        fused.append([d for d, _ in sorted(scores.items(), key=lambda x: -x[1])[:top_k]])
    return fused


def bm25_retrieval(
    queries: list[tuple[str, str]],
    top_k: int = TOP_K_RERANK,
) -> list[list[str]]:
    """Run BM25+NER retrieval via OpenSearch."""
    from opensearchpy import OpenSearch

    client = OpenSearch(
        hosts=[{"host": "localhost", "port": 9200}],
        http_compress=True, timeout=30,
    )
    ner_cache = load_or_compute_ner(queries)

    results = []
    for qid, text in tqdm(queries, desc="BM25+NER", ncols=80):
        ner_entities = ner_cache.get(qid, {})
        results.append(bm25_ner_search(client, text, ner_entities, top_k=top_k))
    return results


def _splade_ft_triway_and_finalize(
    args: argparse.Namespace,
    all_results: dict,
    qids: list,
    texts: list[str],
    qrels: dict,
    splade_lists: list[list[str]],
    dense_lists: list[list[str]],
    queries: list[tuple[str, str]],
    article_ids: list[str],
    article_texts: dict[str, str],
    articles_for_ce: dict,
    t_start: float,
) -> dict:
    """Fine-tuned dense, 3-way fusion, save JSON, print summary (shared: full & second_half)."""
    _save = getattr(args, "save_retrieved", False)
    ft_dense_lists = None

    # ── 6. Fine-tuned FashionCLIP dense retrieval ──────────────────────
    if FINETUNED_BIENC.exists():
        log.info("Running Fine-tuned FashionCLIP dense retrieval...")
        ft_dense_lists = dense_retrieval_finetuned(
            queries,
            article_ids,
            article_texts,
            top_k=args.top_k_rerank,
            faiss_query_chunk=args.faiss_query_chunk,
        )
        ft_dense_dict = {qid: lst for qid, lst in zip(qids, ft_dense_lists)}
        all_results["FT_FashionCLIP_only"] = evaluate(
            ft_dense_dict, qrels, label="FT_FashionCLIP_only",
        )

        FT_WEIGHT_CONFIGS = [
            (0.3, 0.7, "SPLADE03_FTCLIP07"),
            (0.4, 0.6, "SPLADE04_FTCLIP06"),
            (0.5, 0.5, "SPLADE05_FTCLIP05"),
            (0.6, 0.4, "SPLADE06_FTCLIP04"),
        ]

        best_ft_hybrid_name = None
        best_ft_hybrid_ndcg = -1.0
        best_ft_hybrid_lists = None

        for w_sp, w_dn, label in FT_WEIGHT_CONFIGS:
            hybrid_lists = rrf_fusion(
                splade_lists, ft_dense_lists,
                k=RRF_K, bm25_weight=w_sp, dense_weight=w_dn,
                top_k=args.top_k_rerank,
            )
            hybrid_dict = {qid: lst for qid, lst in zip(qids, hybrid_lists)}
            res = evaluate(hybrid_dict, qrels, label=label)
            all_results[label] = res

            ndcg = res["metrics"]["ndcg@10"]
            if ndcg > best_ft_hybrid_ndcg:
                best_ft_hybrid_ndcg = ndcg
                best_ft_hybrid_name = label
                best_ft_hybrid_lists = hybrid_lists

        log.info("Best FT hybrid fusion: %s (nDCG@10=%.4f)",
                 best_ft_hybrid_name, best_ft_hybrid_ndcg)

        if not args.no_ce and Path(LLM_TRAINED_CE).exists() and best_ft_hybrid_lists is not None:
            log.info("Best FT hybrid + LLM-trained CE reranking...")
            ft_hybrid_ce_lists = ce_rerank_batch(
                texts, best_ft_hybrid_lists, articles_for_ce,
                model_name=LLM_TRAINED_CE, top_k=10,
            )
            ft_hybrid_ce_dict = {qid: lst for qid, lst in zip(qids, ft_hybrid_ce_lists)}
            all_results["SPLADE_FTCLIP_CE"] = evaluate(
                ft_hybrid_ce_dict, qrels, label="SPLADE_FTCLIP_CE",
            )

            for w_sp, w_dn, label in FT_WEIGHT_CONFIGS:
                if label == best_ft_hybrid_name:
                    continue
                h_lists = rrf_fusion(
                    splade_lists, ft_dense_lists,
                    k=RRF_K, bm25_weight=w_sp, dense_weight=w_dn,
                    top_k=args.top_k_rerank,
                )
                ce_lists = ce_rerank_batch(
                    texts, h_lists, articles_for_ce,
                    model_name=LLM_TRAINED_CE, top_k=10,
                )
                ce_dict = {qid: lst for qid, lst in zip(qids, ce_lists)}
                ce_label = f"{label}_CE"
                all_results[ce_label] = evaluate(ce_dict, qrels, label=ce_label)
    else:
        log.warning("FT-FashionCLIP model not found at %s — skipping", FINETUNED_BIENC)

    log.info("Running BM25+NER retrieval for 3-way fusion...")
    bm25_lists = bm25_retrieval(queries, top_k=args.top_k_rerank)
    if args.skip_known:
        log.info("SKIP BM25_only (already logged: nDCG@10=0.0187)")
    else:
        bm25_dict = {qid: lst for qid, lst in zip(qids, bm25_lists)}
        all_results["BM25_only"] = evaluate(bm25_dict, qrels, label="BM25_only")

    TRIWAY_CONFIGS = [
        (0.1, 0.3, 0.6, "BM25_01_SPL03_DNS06"),
        (0.2, 0.3, 0.5, "BM25_02_SPL03_DNS05"),
        (0.2, 0.4, 0.4, "BM25_02_SPL04_DNS04"),
        (0.1, 0.4, 0.5, "BM25_01_SPL04_DNS05"),
        (0.15, 0.35, 0.5, "BM25_015_SPL035_DNS05"),
    ]

    if ft_dense_lists is not None:
        triway_dense_lists = ft_dense_lists
        triway_dense_tag = "FT"
    else:
        triway_dense_lists = dense_lists
        triway_dense_tag = "Vanilla"

    best_3way_name = None
    best_3way_ndcg = -1.0
    best_3way_lists = None

    for w_b, w_s, w_d, label in TRIWAY_CONFIGS:
        tri_lists = rrf_fusion_3way(
            bm25_lists, splade_lists, triway_dense_lists,
            k=RRF_K, w_bm25=w_b, w_splade=w_s, w_dense=w_d,
            top_k=args.top_k_rerank,
        )
        tri_dict = {qid: lst for qid, lst in zip(qids, tri_lists)}
        res = evaluate(tri_dict, qrels, label=f"3Way_{label}")
        all_results[f"3Way_{label}"] = res

        ndcg = res["metrics"]["ndcg@10"]
        if ndcg > best_3way_ndcg:
            best_3way_ndcg = ndcg
            best_3way_name = f"3Way_{label}"
            best_3way_lists = tri_lists

    log.info("Best 3-way fusion (%s dense): %s (nDCG@10=%.4f)",
             triway_dense_tag, best_3way_name, best_3way_ndcg)

    if not args.no_ce and Path(LLM_TRAINED_CE).exists() and best_3way_lists is not None:
        log.info("Best 3-way + LLM-trained CE reranking...")
        tri_ce_lists = ce_rerank_batch(
            texts, best_3way_lists, articles_for_ce,
            model_name=LLM_TRAINED_CE, top_k=10,
        )
        tri_ce_dict = {qid: lst for qid, lst in zip(qids, tri_ce_lists)}
        all_results["3Way_Best_CE"] = evaluate(
            tri_ce_dict, qrels, label="3Way_Best_CE",
        )

        for w_b, w_s, w_d, label in TRIWAY_CONFIGS:
            if f"3Way_{label}" == best_3way_name:
                continue
            tri_lists = rrf_fusion_3way(
                bm25_lists, splade_lists, triway_dense_lists,
                k=RRF_K, w_bm25=w_b, w_splade=w_s, w_dense=w_d,
                top_k=args.top_k_rerank,
            )
            ce_lists = ce_rerank_batch(
                texts, tri_lists, articles_for_ce,
                model_name=LLM_TRAINED_CE, top_k=10,
            )
            ce_dict = {qid: lst for qid, lst in zip(qids, ce_lists)}
            ce_label = f"3Way_{label}_CE"
            all_results[ce_label] = evaluate(ce_dict, qrels, label=ce_label)

    out_path = RESULTS_DIR / "splade_pipeline_eval.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    log.info("Results saved → %s", out_path)

    bm25_baseline_ndcg = 0.0815
    elapsed = time.time() - t_start
    print("\n" + "=" * 80)
    print("SPLADE PIPELINE RESULTS")
    print("=" * 80)
    print(f"{'Config':<30} {'nDCG@10':>9} {'MRR':>9} {'R@10':>9}  {'vs BM25 baseline'}")
    print("-" * 80)
    for name, res in all_results.items():
        m = res["metrics"]
        ndcg = m["ndcg@10"]
        delta = (ndcg / bm25_baseline_ndcg - 1) * 100
        sign = "+" if delta >= 0 else ""
        icon = ">>>" if delta > 5 else "++" if delta > 0 else "==" if abs(delta) < 1 else "--"
        print(
            f"{name:<30} {ndcg:>9.4f} {m['mrr']:>9.4f}"
            f" {m.get('recall@10', 0):>9.4f}  {sign}{delta:.1f}% {icon}"
        )
    print("=" * 80)
    print(f"BM25+Dense+CE baseline: nDCG@10 = {bm25_baseline_ndcg}")
    print(f"Total elapsed: {elapsed / 60:.1f} min")
    print()

    return all_results


def _splade_pipeline_second_half(args: argparse.Namespace, t_start: float) -> dict:
    if not SPLADE_FIRST_HALF_PKL.exists():
        log.error("Missing checkpoint %s — run --mode first_half first", SPLADE_FIRST_HALF_PKL)
        sys.exit(1)
    with open(SPLADE_FIRST_HALF_PKL, "rb") as f:
        ckpt = pickle.load(f)
    all_results = ckpt["all_results"]
    qids = ckpt["qids"]
    texts = ckpt["texts"]
    qrels = ckpt["qrels"]
    splade_lists = ckpt["splade_lists"]
    dense_lists = ckpt["dense_lists"]
    queries = ckpt["queries"]
    article_ids = ckpt["article_ids"]
    article_texts = ckpt["article_texts"]
    articles_for_ce = load_articles()
    log.info("Loaded first_half checkpoint — continuing with FT + 3-way fusion")
    return _splade_ft_triway_and_finalize(
        args,
        all_results,
        qids,
        texts,
        qrels,
        splade_lists,
        dense_lists,
        queries,
        article_ids,
        article_texts,
        articles_for_ce,
        t_start,
    )


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SPLADE + FashionCLIP eval")
    parser.add_argument("--n_queries", type=int, default=0,
                        help="0 = all queries")
    parser.add_argument("--top_k_rerank", type=int, default=TOP_K_RERANK)
    parser.add_argument("--no_ce", action="store_true",
                        help="Skip CE reranking (fast mode)")
    parser.add_argument("--splade_batch_size", type=int, default=32)
    parser.add_argument("--splade_query_chunk", type=int, default=1000,
                        help="Queries per chunk for sparse dot product (lower = less RAM)")
    parser.add_argument("--force_encode", action="store_true",
                        help="Force re-encode articles even if cache exists")
    parser.add_argument(
        "--skip_known",
        action="store_true",
        help="Skip experiments already logged (SPLADE-only, SPLADE+CE, FashionCLIP-only, BM25-only)",
    )
    parser.add_argument(
        "--faiss_query_chunk",
        type=int,
        default=512,
        help="Query batch size for FAISS search (lower uses less RAM; default 512)",
    )
    parser.add_argument(
        "--mode",
        choices=("full", "first_half", "second_half"),
        default="full",
        help="full=one process; first_half=through hybrid CE then checkpoint; second_half=FT+3way from checkpoint",
    )
    parser.add_argument(
        "--save_retrieved",
        action="store_true",
        help="Include per-query retrieved lists in output JSON (needed for bootstrap CIs)",
    )
    args = parser.parse_args()

    t_start = time.time()
    all_results: dict[str, dict] = {}

    if args.mode == "second_half":
        return _splade_pipeline_second_half(args, t_start)

    # ── 1. Load data (use same test split as eval_three_tower) ──────────
    test_qids = load_test_query_ids()
    all_queries, all_qrels = load_benchmark(n_queries=None)
    queries = [(qid, qt) for qid, qt in all_queries if qid in test_qids]
    qrels = {qid: v for qid, v in all_qrels.items() if qid in test_qids}

    if args.n_queries > 0 and len(queries) > args.n_queries:
        import random
        queries = random.Random(42).sample(queries, args.n_queries)

    qids = [q[0] for q in queries]
    texts = [q[1] for q in queries]
    log.info("Loaded %d test queries (from test split)", len(queries))

    article_ids, article_texts = load_article_ids_and_texts()
    log.info("Loaded %d articles", len(article_ids))

    articles_for_ce = load_articles()

    # ── 2. SPLADE retrieval ──────────────────────────────────────────────
    splade = SpladeRetriever(device="mps" if torch.backends.mps.is_available() else "cpu")
    splade.encode_articles(
        article_ids, article_texts,
        batch_size=args.splade_batch_size,
        force=args.force_encode,
    )

    splade_lists = splade.search_batch(texts, top_k=args.top_k_rerank,
                                       query_chunk=args.splade_query_chunk)
    splade_dict = {qid: lst for qid, lst in zip(qids, splade_lists)}

    # Config A: SPLADE-only
    if args.skip_known:
        log.info("SKIP SPLADE_only (already logged: nDCG@10=0.0464)")
    else:
        all_results["SPLADE_only"] = evaluate(splade_dict, qrels, label="SPLADE_only")

    # Config B: SPLADE + LLM-trained CE
    if args.skip_known:
        log.info("SKIP SPLADE_CE (already logged: nDCG@10=0.0903)")
    elif not args.no_ce and Path(LLM_TRAINED_CE).exists():
        log.info("SPLADE + LLM-trained CE reranking...")
        splade_ce_lists = ce_rerank_batch(
            texts, splade_lists, articles_for_ce,
            model_name=LLM_TRAINED_CE, top_k=10,
        )
        splade_ce_dict = {qid: lst for qid, lst in zip(qids, splade_ce_lists)}
        all_results["SPLADE_CE"] = evaluate(splade_ce_dict, qrels, label="SPLADE_CE")

    # Free SPLADE model before loading FashionCLIP (avoids MPS memory segfault)
    splade.free_model()

    # ── 3. FashionCLIP dense retrieval ───────────────────────────────────
    log.info("Running FashionCLIP dense retrieval...")
    dense_lists = dense_retrieval_fashionclip(
        queries,
        article_ids,
        top_k=args.top_k_rerank,
        faiss_query_chunk=args.faiss_query_chunk,
    )
    dense_dict = {qid: lst for qid, lst in zip(qids, dense_lists)}

    if args.skip_known:
        log.info("SKIP FashionCLIP_only (already logged: nDCG@10=0.0300)")
    else:
        all_results["FashionCLIP_only"] = evaluate(dense_dict, qrels, label="FashionCLIP_only")

    # ── 4. SPLADE + FashionCLIP RRF fusion ───────────────────────────────
    WEIGHT_CONFIGS = [
        (0.3, 0.7, "SPLADE03_CLIP07"),
        (0.4, 0.6, "SPLADE04_CLIP06"),
        (0.5, 0.5, "SPLADE05_CLIP05"),
        (0.6, 0.4, "SPLADE06_CLIP04"),
    ]

    best_hybrid_name = None
    best_hybrid_ndcg = -1.0
    best_hybrid_lists = None

    for w_sp, w_dn, label in WEIGHT_CONFIGS:
        hybrid_lists = rrf_fusion(
            splade_lists, dense_lists,
            k=RRF_K, bm25_weight=w_sp, dense_weight=w_dn,
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

    log.info("Best hybrid fusion: %s (nDCG@10=%.4f)", best_hybrid_name, best_hybrid_ndcg)

    # ── 5. Best fusion + CE reranking ────────────────────────────────────
    if not args.no_ce and Path(LLM_TRAINED_CE).exists() and best_hybrid_lists is not None:
        log.info("Best hybrid + LLM-trained CE reranking...")
        hybrid_ce_lists = ce_rerank_batch(
            texts, best_hybrid_lists, articles_for_ce,
            model_name=LLM_TRAINED_CE, top_k=10,
        )
        hybrid_ce_dict = {qid: lst for qid, lst in zip(qids, hybrid_ce_lists)}
        all_results["SPLADE_CLIP_CE"] = evaluate(
            hybrid_ce_dict, qrels, label="SPLADE_CLIP_CE",
        )

        # Also evaluate all weight configs with CE for completeness
        for w_sp, w_dn, label in WEIGHT_CONFIGS:
            if label == best_hybrid_name:
                continue
            h_lists = rrf_fusion(
                splade_lists, dense_lists,
                k=RRF_K, bm25_weight=w_sp, dense_weight=w_dn,
                top_k=args.top_k_rerank,
            )
            ce_lists = ce_rerank_batch(
                texts, h_lists, articles_for_ce,
                model_name=LLM_TRAINED_CE, top_k=10,
            )
            ce_dict = {qid: lst for qid, lst in zip(qids, ce_lists)}
            ce_label = f"{label}_CE"
            all_results[ce_label] = evaluate(ce_dict, qrels, label=ce_label)

    if args.mode == "first_half":
        OVERNIGHT_CACHE.mkdir(parents=True, exist_ok=True)
        ckpt = {
            "all_results": all_results,
            "qids": qids,
            "texts": texts,
            "qrels": qrels,
            "splade_lists": splade_lists,
            "dense_lists": dense_lists,
            "queries": queries,
            "article_ids": article_ids,
            "article_texts": article_texts,
            "args": vars(args),
        }
        with open(SPLADE_FIRST_HALF_PKL, "wb") as f:
            pickle.dump(ckpt, f)
        log.info("first_half checkpoint saved → %s", SPLADE_FIRST_HALF_PKL)
        return all_results

    return _splade_ft_triway_and_finalize(
        args,
        all_results,
        qids,
        texts,
        qrels,
        splade_lists,
        dense_lists,
        queries,
        article_ids,
        article_texts,
        articles_for_ce,
        t_start,
    )


if __name__ == "__main__":
    main()
