"""
MODA — GLiNER v1 vs GLiNER2 NER Ablation

Side-by-side evaluation of two NER backends on the same 10K test queries:
  - GLiNER v1:  urchade/gliner_medium-v2.1 (NAACL 2024)
  - GLiNER2:    fastino/gliner2-base-v1    (EMNLP 2025)

Configs evaluated per NER backend:
  1. BM25 + NER boost only
  2. Hybrid (BM25+NER × 0.4  +  Dense × 0.6)
  3. Full Pipeline (Hybrid + CE rerank)

Shared baselines (NER-independent, computed once):
  - BM25 only (no NER)
  - Dense only (FashionCLIP)

Usage:
  python benchmark/eval_gliner2_ablation.py
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

from benchmark.article_text import build_article_text as _build_article_text
from benchmark.metrics import compute_all_metrics, aggregate_metrics
from benchmark.query_expansion import (
    FashionNER, FashionNER2,
    LABEL_TO_FIELD, COLOR_MAP, GARMENT_TYPE_MAP, GENDER_MAP,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ─── Paths & constants ────────────────────────────────────────────────────────
HNM_DIR        = _REPO_ROOT / "data" / "raw" / "hnm_real"
EMBEDDINGS_DIR = _REPO_ROOT / "data" / "processed" / "embeddings"
RESULTS_DIR    = _REPO_ROOT / "results" / "real"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

INDEX_NAME   = "moda_hnm"
DENSE_MODEL  = "fashion-clip"
N_QUERIES    = 10_000
BM25_WEIGHT  = 0.4
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

def compute_ner(
    queries: list[tuple[str, str]],
    ner_backend,
    label: str,
) -> dict[str, dict[str, list[str]]]:
    """Run NER over all queries, return {qid: {label: [values]}}."""
    log.info("Computing NER with %s on %d queries...", label, len(queries))
    cache: dict[str, dict[str, list[str]]] = {}
    t0 = time.time()
    for i, (qid, text) in enumerate(queries):
        cache[qid] = ner_backend.extract(text)
        if (i + 1) % 1000 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            log.info("  %s NER: %d/%d  (%.1f q/s, ~%.0fs remaining)",
                     label, i + 1, len(queries), rate,
                     (len(queries) - i - 1) / rate)
    log.info("%s NER done in %.1fs", label, time.time() - t0)
    return cache


# ─── BM25 retrieval ───────────────────────────────────────────────────────────

def bm25_plain_search(client: OpenSearch, query_text: str,
                      top_k: int = TOP_K_RERANK) -> list[str]:
    body = {
        "query": {
            "multi_match": {
                "query": query_text,
                "fields": [
                    "prod_name^4", "product_type_name^3", "colour_group_name^2",
                    "section_name^1.5", "garment_group_name^1.5",
                    "detail_desc^1", "search_text^1",
                ],
                "type": "best_fields", "tie_breaker": 0.3, "operator": "or",
            }
        },
        "size": top_k, "_source": False,
    }
    resp = client.search(index=INDEX_NAME, body=body)
    return [h["_id"] for h in resp["hits"]["hits"]]


def bm25_ner_search(client: OpenSearch, query_text: str,
                    ner_entities: dict[str, list[str]],
                    top_k: int = TOP_K_RERANK) -> list[str]:
    base = {
        "multi_match": {
            "query": query_text,
            "fields": [
                "prod_name^4", "product_type_name^3", "colour_group_name^2",
                "section_name^1.5", "garment_group_name^1.5",
                "detail_desc^1", "search_text^1",
            ],
            "type": "best_fields", "tie_breaker": 0.3, "operator": "or",
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
            "query": {"bool": {"must": [base], "should": should_clauses}},
            "size": top_k, "_source": False,
        }
    else:
        query_body = {"query": base, "size": top_k, "_source": False}

    resp = client.search(index=INDEX_NAME, body=query_body)
    return [h["_id"] for h in resp["hits"]["hits"]]


# ─── Dense retrieval ──────────────────────────────────────────────────────────

def dense_search_batch(
    queries: list[tuple[str, str]],
    model_name: str = DENSE_MODEL,
    top_k: int = TOP_K_RERANK,
) -> dict[str, list[str]]:
    import torch
    from benchmark.models import load_clip_model, encode_texts_clip

    safe_name  = model_name.replace("/", "_").replace(":", "_")
    faiss_path = EMBEDDINGS_DIR / f"{safe_name}_faiss.index"
    ids_path   = EMBEDDINGS_DIR / f"{safe_name}_article_ids.json"
    assert faiss_path.exists(), f"FAISS index not found: {faiss_path}"

    qids  = [q[0] for q in queries]
    texts = [q[1] for q in queries]

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    log.info("Encoding %d queries with %s on %s...", len(queries), model_name, device)
    model, _, tokenizer = load_clip_model(model_name, device=device)
    q_emb = encode_texts_clip(texts, model, tokenizer, device, batch_size=128)
    del model
    if device == "mps":
        torch.mps.empty_cache()

    with tempfile.TemporaryDirectory() as tmp:
        q_path   = Path(tmp) / "queries.npy"
        out_path = Path(tmp) / "results.json"
        np.save(str(q_path), q_emb.astype("float32"))

        worker = Path(__file__).parent / "_faiss_search_worker.py"
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
            raw_results = json.load(f)

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
        "%s -> nDCG@10=%.4f  MRR=%.4f  R@10=%.4f  (%d queries)",
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
    qids  = [q[0] for q in queries]
    texts = [q[1] for q in queries]

    # ── 2. Compute NER with both backends ─────────────────────────────────────
    ner_v1 = FashionNER(model_name="urchade/gliner_medium-v2.1", threshold=0.4)
    ner_v1_cache = compute_ner(queries, ner_v1, label="GLiNER_v1")
    del ner_v1

    ner_v2 = FashionNER2(model_name="fastino/gliner2-base-v1", threshold=0.4)
    ner_v2_cache = compute_ner(queries, ner_v2, label="GLiNER2")
    del ner_v2

    # Save NER caches for inspection
    for name, cache in [("gliner_v1", ner_v1_cache), ("gliner2", ner_v2_cache)]:
        path = RESULTS_DIR / f"ner_cache_{name}_10k.json"
        with open(path, "w") as f:
            json.dump(cache, f)
        log.info("NER cache saved -> %s", path)

    # ── 3. Shared baselines (NER-independent) ─────────────────────────────────
    client = OpenSearch(
        hosts=[{"host": "localhost", "port": 9200}],
        http_compress=True, timeout=30,
    )
    info = client.info()
    log.info("OpenSearch %s connected", info["version"]["number"])

    # BM25 plain (no NER)
    log.info("Running plain BM25 retrieval (shared baseline)...")
    bm25_plain: dict[str, list[str]] = {}
    for qid, text in tqdm(queries, desc="BM25 plain", ncols=80):
        bm25_plain[qid] = bm25_plain_search(client, text)

    # Dense retrieval (shared)
    log.info("Running dense retrieval (shared baseline)...")
    dense_results = dense_search_batch(queries, model_name=DENSE_MODEL, top_k=TOP_K_RERANK)

    all_results = {}

    # ── 4. Shared baselines evaluation ────────────────────────────────────────
    all_results["BM25_only"] = evaluate(bm25_plain, qrels, label="BM25_only")

    dense_dict = {qid: dense_results.get(qid, []) for qid in qids}
    all_results["Dense_only"] = evaluate(dense_dict, qrels, label="Dense_only")

    # Plain hybrid (no NER in BM25)
    plain_bm25_lists = [bm25_plain.get(qid, []) for qid in qids]
    dense_lists      = [dense_results.get(qid, []) for qid in qids]
    hybrid_plain     = rrf_fusion(plain_bm25_lists, dense_lists)
    hybrid_plain_d   = {qid: lst for qid, lst in zip(qids, hybrid_plain)}
    all_results["Hybrid_plain"] = evaluate(hybrid_plain_d, qrels, label="Hybrid_plain")

    # ── 5. Per-NER-backend evaluation ─────────────────────────────────────────
    for ner_label, ner_cache in [("GLiNER_v1", ner_v1_cache), ("GLiNER2", ner_v2_cache)]:
        log.info("=" * 60)
        log.info("Evaluating NER backend: %s", ner_label)
        log.info("=" * 60)

        # BM25 + NER
        bm25_ner: dict[str, list[str]] = {}
        for qid, text in tqdm(queries, desc=f"BM25+{ner_label}", ncols=80):
            ner_entities = ner_cache.get(qid, {})
            bm25_ner[qid] = bm25_ner_search(client, text, ner_entities)
        all_results[f"BM25_NER_{ner_label}"] = evaluate(
            bm25_ner, qrels, label=f"BM25_NER_{ner_label}")

        # Hybrid (BM25+NER fused with Dense)
        bm25_ner_lists = [bm25_ner.get(qid, []) for qid in qids]
        hybrid_lists   = rrf_fusion(bm25_ner_lists, dense_lists)
        hybrid_dict    = {qid: lst for qid, lst in zip(qids, hybrid_lists)}
        all_results[f"Hybrid_NER_{ner_label}"] = evaluate(
            hybrid_dict, qrels, label=f"Hybrid_NER_{ner_label}")

        # Full pipeline (Hybrid+NER + CE rerank)
        log.info("CE reranking for %s pipeline...", ner_label)
        reranked = ce_rerank_batch(texts, hybrid_lists, articles)
        reranked_dict = {qid: lst for qid, lst in zip(qids, reranked)}
        all_results[f"Full_{ner_label}"] = evaluate(
            reranked_dict, qrels, label=f"Full_{ner_label}")

    # ── 6. Also rerank the plain hybrid for comparison ────────────────────────
    log.info("CE reranking for plain hybrid (no NER)...")
    reranked_plain = ce_rerank_batch(texts, hybrid_plain, articles)
    reranked_plain_d = {qid: lst for qid, lst in zip(qids, reranked_plain)}
    all_results["Full_plain"] = evaluate(reranked_plain_d, qrels, label="Full_plain")

    # ── 7. Save results ───────────────────────────────────────────────────────
    out_path = RESULTS_DIR / "gliner2_ablation.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    log.info("Results saved -> %s", out_path)

    # ── 8. Print summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 90)
    print("GLiNER v1 vs GLiNER2 — NER Ablation (10K queries)")
    print("=" * 90)
    print(f"{'Config':<35} {'nDCG@10':>9} {'MRR':>9} {'R@10':>9} {'AP':>9}")
    print("-" * 90)

    display_order = [
        "BM25_only", "Dense_only", "Hybrid_plain", "Full_plain",
        "",
        "BM25_NER_GLiNER_v1", "Hybrid_NER_GLiNER_v1", "Full_GLiNER_v1",
        "",
        "BM25_NER_GLiNER2", "Hybrid_NER_GLiNER2", "Full_GLiNER2",
    ]
    for key in display_order:
        if key == "":
            print("-" * 90)
            continue
        res = all_results.get(key)
        if not res:
            continue
        m = res["metrics"]
        print(f"{key:<35} {m.get('ndcg@10',0):>9.4f} {m.get('mrr',0):>9.4f}"
              f" {m.get('recall@10',0):>9.4f} {m.get('ap',0):>9.4f}")

    print("=" * 90)

    # Head-to-head deltas
    for stage in ["BM25_NER", "Hybrid_NER", "Full"]:
        v1_key = f"{stage}_GLiNER_v1"
        v2_key = f"{stage}_GLiNER2"
        if v1_key in all_results and v2_key in all_results:
            v1_ndcg = all_results[v1_key]["metrics"]["ndcg@10"]
            v2_ndcg = all_results[v2_key]["metrics"]["ndcg@10"]
            delta_pct = (v2_ndcg / v1_ndcg - 1) * 100 if v1_ndcg > 0 else 0
            sign = "+" if delta_pct >= 0 else ""
            print(f"  {stage}: GLiNER2 vs v1  {sign}{delta_pct:.1f}%  "
                  f"(v1={v1_ndcg:.4f}, v2={v2_ndcg:.4f})")

    print(f"\nTotal elapsed: {(time.time()-t_start)/60:.1f} min\n")
    return all_results


if __name__ == "__main__":
    main()
