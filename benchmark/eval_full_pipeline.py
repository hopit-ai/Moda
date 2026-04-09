"""
MODA Phase 2 — Config 7 & 8: Full Pipeline Evaluation

Implements the complete SOTA pipeline:

  Config 7:  NER-boosted BM25 × 0.4  +  Dense × 0.6  →  CE rerank
  Config 8:  Config 7  (this IS the full pipeline for our ablation)

NER integration strategy:
  - Pre-compute GLiNER entities for all queries (saved to disk for reuse)
  - NER boosts: `bool.should` clauses — boost but don't hard-exclude
  - Only NER on BM25 component; dense embedding naturally handles semantics
  - CE reranker sees full (query, product-text) pairs → implicitly handles attributes

Research basis:
  - GLiNER (NAACL 2024): zero-shot NER, outperforms ChatGPT
  - EcomBERT-NER: 23 entity labels, field-specific boosting
  - RRF (Cormack 2009): robust fusion, no score normalization needed
  - CE reranking: cross-encoder ms-marco-MiniLM-L-6-v2

Finding from Config 2D ablation:
  - NER alone improves BM25 by +14%
  - Synonyms HURT BM25 on H&M (-35%) due to query pollution
    (H&M brand names + BM25 IDF = precision-sensitive to over-expansion)
  - Therefore: NER in BM25 + dense hybrid > plain hybrid

Usage:
  python benchmark/eval_full_pipeline.py
"""

from __future__ import annotations

import csv
import json
import logging
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from opensearchpy import OpenSearch
from sentence_transformers import CrossEncoder
from tqdm import tqdm

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from benchmark.metrics import compute_all_metrics, aggregate_metrics
from benchmark.query_expansion import (
    SynonymExpander, FashionNER, FashionNER2,
    LABEL_TO_FIELD, COLOR_MAP, GARMENT_TYPE_MAP, GENDER_MAP,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ─── Paths & constants ────────────────────────────────────────────────────────
HNM_DIR       = _REPO_ROOT / "data" / "raw" / "hnm_real"
EMBEDDINGS_DIR = _REPO_ROOT / "data" / "processed" / "embeddings"
RESULTS_DIR   = _REPO_ROOT / "results" / "real"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

NER_CACHE_PATH = RESULTS_DIR / "ner_cache_10k.json"

INDEX_NAME  = "moda_hnm"
DENSE_MODEL = "fashion-clip"      # best from Phase 1
N_QUERIES   = 10_000
BM25_WEIGHT = 0.4                 # Config C optimal weights
DENSE_WEIGHT = 0.6
RRF_K        = 60
TOP_K_RERANK = 100
TOP_K_FINAL  = 50
KEYWORD_FIELDS = {
    "garment_group_name", "graphical_appearance_name",
    "index_group_name", "index_name", "perceived_colour_master_name",
    "perceived_colour_value_name", "product_group_name",
}

# ─── Data loading ─────────────────────────────────────────────────────────────

def load_benchmark(n_queries: int = N_QUERIES):
    log.info("Loading real H&M benchmark...")
    qrels: dict[str, dict[str, int]] = {}
    with open(HNM_DIR / "qrels.csv", newline="") as f:
        for row in csv.DictReader(f):
            qid     = row["query_id"].strip()
            pos_ids = [x.strip() for x in row.get("positive_ids", "").split() if x.strip()]
            neg_ids = [x.strip() for x in row.get("negative_ids", "").split() if x.strip()]
            grades  = {aid: 2 for aid in pos_ids}
            grades.update({aid: 1 for aid in neg_ids if aid not in grades})
            qrels[qid] = grades

    df = pd.read_csv(HNM_DIR / "queries.csv")
    valid = set(qrels.keys())
    df = df[df["query_id"].astype(str).isin(valid)]
    if n_queries and len(df) > n_queries:
        df = df.sample(n=n_queries, random_state=42)

    queries = [(str(r["query_id"]), str(r["query_text"])) for _, r in df.iterrows()]
    log.info("  %d queries, %d qrels entries", len(queries), len(qrels))
    return queries, qrels


def load_articles() -> dict[str, dict]:
    """Load article text for CE reranking."""
    log.info("Loading articles for CE reranking...")
    df = pd.read_csv(HNM_DIR / "articles.csv", dtype=str).fillna("")
    articles = {}
    for _, row in df.iterrows():
        aid = str(row.get("article_id", "")).strip()
        if aid:
            articles[aid] = row.to_dict()
    log.info("  %d articles loaded", len(articles))
    return articles


# ─── NER pre-computation ──────────────────────────────────────────────────────

def load_or_compute_ner(
    queries: list[tuple[str, str]],
    force_recompute: bool = False,
    ner_model: str | None = None,
) -> dict[str, dict[str, list[str]]]:
    """Load NER cache from disk or compute fresh with GLiNER/GLiNER2.

    Args:
        ner_model: If provided, use this model (path or HF name) via
                   FashionNER2 and store results in a separate cache file.
                   If None, use the default GLiNER v1 + default cache.
    """
    if ner_model:
        safe = Path(ner_model).name
        cache_path = RESULTS_DIR / f"ner_cache_{safe}.json"
    else:
        cache_path = NER_CACHE_PATH

    if cache_path.exists() and not force_recompute:
        log.info("Loading NER cache from disk: %s", cache_path)
        with open(cache_path) as f:
            cache = json.load(f)
        qids = {qid for qid, _ in queries}
        if qids.issubset(set(cache.keys())):
            log.info("NER cache hit — %d entries", len(cache))
            return cache
        log.info("NER cache stale — recomputing...")

    if ner_model:
        log.info("Computing NER with GLiNER2 model: %s ...", ner_model)
        ner = FashionNER2(model_name=ner_model, threshold=0.4)
    else:
        log.info("Computing NER with GLiNER (urchade/gliner_medium-v2.1)...")
        ner = FashionNER(model_name="urchade/gliner_medium-v2.1", threshold=0.4)

    cache: dict[str, dict[str, list[str]]] = {}
    t0 = time.time()
    for i, (qid, text) in enumerate(queries):
        cache[qid] = ner.extract(text)
        if (i + 1) % 1000 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            log.info("  NER: %d/%d  (%.1f q/s, ~%.0fs remaining)",
                     i + 1, len(queries), rate,
                     (len(queries) - i - 1) / rate)
    del ner

    with open(cache_path, "w") as f:
        json.dump(cache, f)
    log.info("NER cache saved → %s  (%.1fs)", cache_path, time.time() - t0)
    return cache


# ─── BM25 retrieval (NER-boosted) ─────────────────────────────────────────────

def bm25_ner_search(
    client: OpenSearch,
    query_text: str,
    ner_entities: dict[str, list[str]],
    top_k: int = TOP_K_RERANK,
) -> list[str]:
    """
    NER-boosted BM25 query.
    base multi_match MUST + NER should boosts.
    """
    base = {
        "multi_match": {
            "query": query_text,
            "fields": [
                "prod_name^4", "product_type_name^3", "colour_group_name^2",
                "section_name^1.5", "garment_group_name^1.5",
                "detail_desc^1", "search_text^1",
            ],
            "type": "best_fields",
            "tie_breaker": 0.3,
            "operator": "or",
        }
    }

    should_clauses = []
    for label, values in ner_entities.items():
        if label not in LABEL_TO_FIELD:
            continue
        field, boost, value_map = LABEL_TO_FIELD[label]
        for raw_val in values:
            mapped = value_map.get(raw_val, raw_val.title())
            clause = (
                {"term":  {field: {"value": mapped, "boost": boost}}}
                if field in KEYWORD_FIELDS
                else {"match": {field: {"query": mapped, "boost": boost}}}
            )
            should_clauses.append(clause)

    if should_clauses:
        query_body = {
            "query": {
                "bool": {"must": [base], "should": should_clauses}
            },
            "size": top_k, "_source": False,
        }
    else:
        query_body = {"query": base, "size": top_k, "_source": False}

    resp = client.search(index=INDEX_NAME, body=query_body)
    return [h["_id"] for h in resp["hits"]["hits"]]


# ─── Dense retrieval (FAISS subprocess) ───────────────────────────────────────

def dense_search_batch(
    queries: list[tuple[str, str]],
    model_name: str = DENSE_MODEL,
    top_k: int = TOP_K_RERANK,
) -> dict[str, list[str]]:
    """
    Encode queries with CLIP model, then search FAISS in a subprocess.
    Subprocess isolation avoids PyTorch+FAISS BLAS library conflicts.
    Uses the same encode_queries pattern as run_hnm_eval.py.
    """
    import torch
    from benchmark.models import load_clip_model, encode_texts_clip

    safe_name  = model_name.replace("/", "_").replace(":", "_")
    faiss_path = EMBEDDINGS_DIR / f"{safe_name}_faiss.index"
    ids_path   = EMBEDDINGS_DIR / f"{safe_name}_article_ids.json"
    assert faiss_path.exists(), f"FAISS index not found: {faiss_path}"

    qids  = [q[0] for q in queries]
    texts = [q[1] for q in queries]

    # Step 1: Encode with PyTorch (in-process, before FAISS loads)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    log.info("Encoding %d queries with %s on %s...", len(queries), model_name, device)
    model, _, tokenizer = load_clip_model(model_name, device=device)
    q_emb = encode_texts_clip(texts, model, tokenizer, device, batch_size=128)
    del model
    if device == "mps":
        torch.mps.empty_cache()

    # Step 2: FAISS search in subprocess (avoids BLAS conflict)
    with tempfile.TemporaryDirectory() as tmp:
        q_path   = Path(tmp) / "queries.npy"
        out_path = Path(tmp) / "results.json"
        np.save(str(q_path), q_emb.astype("float32"))

        worker = Path(__file__).parent / "_faiss_search_worker.py"
        # Positional args: q_emb.npy faiss_index ids.json out.json top_k
        cmd = [
            sys.executable, str(worker),
            str(q_path), str(faiss_path), str(ids_path),
            str(out_path), str(top_k),
        ]
        log.info("Running FAISS search subprocess (top_k=%d)...", top_k)
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            log.error("FAISS worker failed:\nstdout: %s\nstderr: %s",
                      result.stdout[-1000:], result.stderr[-1000:])
            raise RuntimeError("FAISS search failed")

        with open(out_path) as f:
            raw_results = json.load(f)   # list of {ids: [...], scores: [...]}

    # Worker returns list of lists of article_ids (plain list, no dict wrapper)
    return {qid: lst for qid, lst in zip(qids, raw_results)}


# ─── RRF fusion ───────────────────────────────────────────────────────────────

def rrf_fusion(
    bm25_lists: list[list[str]],
    dense_lists: list[list[str]],
    k: int = RRF_K,
    bm25_weight: float = BM25_WEIGHT,
    dense_weight: float = DENSE_WEIGHT,
    top_k: int = TOP_K_RERANK,
) -> list[list[str]]:
    fused = []
    for bm25, dense in zip(bm25_lists, dense_lists):
        scores: dict[str, float] = {}
        for rank, doc_id in enumerate(bm25, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + bm25_weight / (k + rank)
        for rank, doc_id in enumerate(dense, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + dense_weight / (k + rank)
        fused.append([d for d, _ in sorted(scores.items(), key=lambda x: -x[1])[:top_k]])
    return fused


# ─── CE reranking ─────────────────────────────────────────────────────────────

def _build_article_text(row: dict) -> str:
    parts = []
    for field in ["prod_name", "product_type_name", "colour_group_name",
                  "section_name", "detail_desc"]:
        val = str(row.get(field, "")).strip()
        if val:
            parts.append(val[:150] if field == "detail_desc" else val)
    return " | ".join(parts)


def ce_rerank_batch(
    queries: list[str],
    candidate_lists: list[list[str]],
    articles: dict[str, dict],
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    batch_size: int = 64,
    top_k: int = TOP_K_FINAL,
) -> list[list[str]]:
    log.info("Loading cross-encoder: %s...", model_name)
    ce = CrossEncoder(model_name, max_length=512)

    article_texts: dict[str, str] = {}

    def get_text(aid: str) -> str:
        if aid not in article_texts:
            article_texts[aid] = _build_article_text(articles.get(aid, {}))
        return article_texts[aid]

    results = []
    log.info("CE reranking %d query-candidate sets...", len(queries))
    for query, candidates in tqdm(zip(queries, candidate_lists),
                                   total=len(queries), desc="CE rerank", ncols=80):
        if not candidates:
            results.append([])
            continue
        pairs  = [(query, get_text(cid)) for cid in candidates]
        scores = ce.predict(pairs, batch_size=batch_size, show_progress_bar=False)
        ranked = sorted(zip(candidates, scores), key=lambda x: -x[1])
        results.append([cid for cid, _ in ranked[:top_k]])
    return results


# ─── Evaluation ───────────────────────────────────────────────────────────────

def evaluate(
    retrieved_dict: dict[str, list[str]],
    qrels: dict[str, dict[str, int]],
    ks: list[int] = None,
    label: str = "config",
) -> dict:
    ks = ks or [5, 10, 20, 50]
    per_query = []
    for qid, retrieved in retrieved_dict.items():
        q_qrels = qrels.get(qid, {})
        if not q_qrels:
            continue
        per_query.append(compute_all_metrics(retrieved, q_qrels, ks=ks))
    agg = aggregate_metrics(per_query)
    log.info(
        "%s → nDCG@10=%.4f  MRR=%.4f  R@10=%.4f  (%d queries)",
        label, agg.get("ndcg@10", 0), agg.get("mrr", 0),
        agg.get("recall@10", 0), len(per_query),
    )
    return {"config": label, "n_queries": len(per_query), "metrics": agg}


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    t_start = time.time()

    # ── 1. Load data ──────────────────────────────────────────────────────────
    queries, qrels = load_benchmark()
    articles = load_articles()
    qids   = [q[0] for q in queries]
    texts  = [q[1] for q in queries]

    # ── 2. NER (load cache or compute) ───────────────────────────────────────
    ner_cache = load_or_compute_ner(queries)

    # ── 3. BM25 (NER-boosted) retrieval ──────────────────────────────────────
    client = OpenSearch(
        hosts=[{"host": "localhost", "port": 9200}],
        http_compress=True, timeout=30,
    )
    log.info("Running NER-boosted BM25 retrieval...")
    bm25_results: dict[str, list[str]] = {}
    for qid, text in tqdm(queries, desc="BM25+NER", ncols=80):
        ner_entities = ner_cache.get(qid, {})
        bm25_results[qid] = bm25_ner_search(client, text, ner_entities, top_k=TOP_K_RERANK)

    # ── 4. Dense (FAISS) retrieval ────────────────────────────────────────────
    log.info("Running dense retrieval (FAISS)...")
    dense_results = dense_search_batch(queries, model_name=DENSE_MODEL, top_k=TOP_K_RERANK)

    # ── 5. RRF Hybrid (Config C weights: BM25×0.4 + Dense×0.6) ───────────────
    log.info("Fusing with RRF (BM25=%.1f, Dense=%.1f)...", BM25_WEIGHT, DENSE_WEIGHT)
    bm25_lists  = [bm25_results.get(qid, []) for qid in qids]
    dense_lists = [dense_results.get(qid, []) for qid in qids]
    hybrid_lists = rrf_fusion(bm25_lists, dense_lists)
    hybrid_results = {qid: lst for qid, lst in zip(qids, hybrid_lists)}

    # ── 6. Evaluate hybrid BEFORE reranking ───────────────────────────────────
    res_hybrid_ner = evaluate(hybrid_results, qrels, label="Config7_Hybrid_NER")

    # ── 7. CE reranking ───────────────────────────────────────────────────────
    log.info("Cross-encoder reranking (this will take ~20-30 min)...")
    reranked_lists = ce_rerank_batch(texts, hybrid_lists, articles)
    reranked_results = {qid: lst for qid, lst in zip(qids, reranked_lists)}

    # ── 8. Evaluate full pipeline ─────────────────────────────────────────────
    res_full = evaluate(reranked_results, qrels, label="Config8_Full_Pipeline")

    # ── 9. Save results ───────────────────────────────────────────────────────
    all_results = {
        "Config7_Hybrid_NER":    res_hybrid_ner,
        "Config8_Full_Pipeline": res_full,
    }
    out_path = RESULTS_DIR / "hnm_full_pipeline.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    log.info("Results saved → %s", out_path)

    # ── 10. Print summary ─────────────────────────────────────────────────────
    prior_best = 0.0533   # Hybrid Config C + CE rerank

    print("\n" + "=" * 72)
    print("FULL PIPELINE RESULTS — Configs 7 & 8")
    print("=" * 72)
    print(f"{'Config':<35} {'nDCG@10':>9} {'MRR':>9} {'R@10':>9}  {'vs CE-rerank'}")
    print("-" * 72)
    for name, res in all_results.items():
        m = res["metrics"]
        delta = (m["ndcg@10"] / prior_best - 1) * 100
        sign  = "+" if delta >= 0 else ""
        icon  = "✅" if delta > 0 else "❌" if delta < -1 else "≈"
        print(
            f"{name:<35} {m['ndcg@10']:>9.4f} {m['mrr']:>9.4f}"
            f" {m['recall@10']:>9.4f}  {sign}{delta:.1f}% {icon}"
        )
    print("=" * 72)
    print(f"\nPrior best: Hybrid Config C + CE rerank  → nDCG@10 = {prior_best}")
    print(f"Total elapsed: {(time.time()-t_start)/60:.1f} min\n")

    return all_results


if __name__ == "__main__":
    main()
