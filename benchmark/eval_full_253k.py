"""
MODA — Full 253K Query Evaluation Pipeline

Evaluates all 8 ablation configs on the complete H&M benchmark (253,685 queries).
Designed for overnight execution with checkpointing — each stage saves to disk and
can be resumed without re-running previous stages.

Execution plan (dominated by CE rerank ×2):
  Stage 1 — BM25 + FAISS            ~30 min   → bm25_top100.json, dense_top100.json
  Stage 2 — NER + BM25+NER          ~3 hrs    → ner_cache_253k.json, bm25_ner_top100.json
  Stage 3 — CE on plain hybrid      ~8.5 hrs  → ce_reranked.json (Config 6)
          — CE on NER hybrid        ~8.5 hrs  → ce_ner_reranked.json (Config 8)
  Stage 4 — Metrics (all q; empty BM25 = 0)     → full_ablation.json

Run with:
  python benchmark/eval_full_253k.py --stages all          # full overnight run
  python benchmark/eval_full_253k.py --stages 1            # only BM25 + FAISS
  python benchmark/eval_full_253k.py --stages 2            # only NER
  python benchmark/eval_full_253k.py --stages 3            # only CE rerank + metrics
  python benchmark/eval_full_253k.py --stages 4            # only metrics (reuse cached)
  python benchmark/eval_full_253k.py --stages 1,2,3,4      # explicit order
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
from opensearchpy import OpenSearch
from tqdm import tqdm

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from benchmark.metrics import compute_all_metrics, aggregate_metrics
from benchmark.query_expansion import (
    FashionNER, SynonymExpander,
    LABEL_TO_FIELD, COLOR_MAP, GARMENT_TYPE_MAP, GENDER_MAP,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ─── Paths ────────────────────────────────────────────────────────────────────
HNM_DIR        = _REPO_ROOT / "data" / "raw" / "hnm_real"
EMBEDDINGS_DIR = _REPO_ROOT / "data" / "processed" / "embeddings"
OUT_DIR        = _REPO_ROOT / "results" / "full"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Stage output files (cache)
BM25_CACHE      = OUT_DIR / "bm25_top100.json"
BM25_NER_CACHE  = OUT_DIR / "bm25_ner_top100.json"   # NER-boosted BM25
DENSE_EMB_CACHE = OUT_DIR / "dense_query_embeddings.npy"
DENSE_CACHE     = OUT_DIR / "dense_top100.json"
DENSE_QIDS      = OUT_DIR / "dense_query_ids.json"
NER_CACHE       = OUT_DIR / "ner_cache_253k.json"
CE_CACHE        = OUT_DIR / "ce_reranked.json"        # Hybrid C (plain BM25+dense) → CE → Config 6
CE_NER_CACHE    = OUT_DIR / "ce_ner_reranked.json"    # Hybrid C (NER-BM25+dense) → CE → Config 8

INDEX_NAME  = "moda_hnm"
DENSE_MODEL = "fashion-clip"
TOP_K       = 100   # retrieval depth (reranked to 50)
KS          = [5, 10, 20, 50]

KEYWORD_FIELDS = {
    "garment_group_name", "graphical_appearance_name",
    "index_group_name", "index_name", "perceived_colour_master_name",
    "perceived_colour_value_name", "product_group_name",
}

RANDOM_SEED = 42

# ─── Data loading ─────────────────────────────────────────────────────────────

def load_benchmark() -> tuple[list[tuple[str, str]], dict[str, dict[str, int]]]:
    log.info("Loading full H&M benchmark (253K queries)...")
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

    queries = [(str(r["query_id"]), str(r["query_text"])) for _, r in df.iterrows()]
    log.info("Loaded %d queries, %d qrels entries", len(queries), len(qrels))
    return queries, qrels


def load_articles() -> dict[str, dict]:
    log.info("Loading articles for CE reranking...")
    df = pd.read_csv(HNM_DIR / "articles.csv", dtype=str).fillna("")
    return {str(r.get("article_id", "")).strip(): r.to_dict() for _, r in df.iterrows()}


# ─── Stage 1A: BM25 retrieval ─────────────────────────────────────────────────

def bm25_query(client: OpenSearch, query_text: str, top_k: int = TOP_K) -> list[str]:
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


def bm25_ner_query(
    client: OpenSearch, query_text: str,
    ner_entities: dict[str, list[str]], top_k: int = TOP_K,
) -> list[str]:
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
    should = []
    for label, values in ner_entities.items():
        if label not in LABEL_TO_FIELD:
            continue
        field, boost, value_map = LABEL_TO_FIELD[label]
        for raw in values:
            mapped = value_map.get(raw, raw.title())
            clause = (
                {"term":  {field: {"value": mapped, "boost": boost}}}
                if field in KEYWORD_FIELDS
                else {"match": {field: {"query": mapped, "boost": boost}}}
            )
            should.append(clause)
    if should:
        body = {"query": {"bool": {"must": [base], "should": should}}, "size": top_k, "_source": False}
    else:
        body = {"query": base, "size": top_k, "_source": False}
    resp = client.search(index=INDEX_NAME, body=body)
    return [h["_id"] for h in resp["hits"]["hits"]]


def stage1_bm25(queries: list[tuple[str, str]], force: bool = False,
                n_workers: int = 12):
    if BM25_CACHE.exists() and not force:
        log.info("BM25 cache hit: %s (%d entries)", BM25_CACHE,
                 len(json.loads(BM25_CACHE.read_text())))
        return

    from concurrent.futures import ThreadPoolExecutor, as_completed
    import threading

    # Thread-local OpenSearch clients (one per thread to avoid connection sharing)
    _tl = threading.local()
    def get_client():
        if not hasattr(_tl, "client"):
            _tl.client = OpenSearch(hosts=[{"host": "localhost", "port": 9200}],
                                    http_compress=True, timeout=30,
                                    maxsize=n_workers)
        return _tl.client

    def fetch_one(args):
        qid, text = args
        try:
            return qid, bm25_query(get_client(), text)
        except Exception as e:
            log.warning("BM25 error qid=%s: %s", qid, e)
            return qid, []

    log.info("Stage 1A: BM25 retrieval for %d queries (%d threads)...",
             len(queries), n_workers)
    results: dict[str, list[str]] = {}
    t0 = time.time()
    CHECKPOINT_EVERY = 10_000

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        pbar = tqdm(total=len(queries), desc="BM25", ncols=80)
        futures = {pool.submit(fetch_one, q): q[0] for q in queries}
        for i, fut in enumerate(as_completed(futures), 1):
            qid, res = fut.result()
            results[qid] = res
            pbar.update(1)
            if i % CHECKPOINT_EVERY == 0:
                elapsed = time.time() - t0
                rate = i / elapsed
                log.info("  BM25 %d/%d  (%.1f q/s, %.0f min remaining)",
                         i, len(queries), rate, (len(queries)-i)/rate/60)
                BM25_CACHE.write_text(json.dumps(results))
        pbar.close()

    elapsed = time.time() - t0
    BM25_CACHE.write_text(json.dumps(results))
    log.info("BM25 done: %d queries in %.1f min (%.1f q/s) → %s",
             len(results), elapsed/60, len(results)/elapsed, BM25_CACHE)


# ─── Stage 1B: FAISS dense retrieval ─────────────────────────────────────────

def stage1_dense(queries: list[tuple[str, str]], force: bool = False):
    if DENSE_CACHE.exists() and not force:
        log.info("Dense cache hit: %s (%d entries)",
                 DENSE_CACHE, len(json.loads(DENSE_CACHE.read_text())))
        return

    import torch
    from benchmark.models import load_clip_model, encode_texts_clip

    qids  = [q[0] for q in queries]
    texts = [q[1] for q in queries]

    safe = DENSE_MODEL.replace("/", "_").replace(":", "_")
    faiss_path = EMBEDDINGS_DIR / f"{safe}_faiss.index"
    ids_path   = EMBEDDINGS_DIR / f"{safe}_article_ids.json"

    # Encode
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    log.info("Stage 1B: Encoding %d queries with %s on %s...", len(texts), DENSE_MODEL, device)
    model, _, tokenizer = load_clip_model(DENSE_MODEL, device=device)
    q_emb = encode_texts_clip(texts, model, tokenizer, device, batch_size=256)
    del model
    if device == "mps":
        torch.mps.empty_cache()

    # Save embeddings
    np.save(str(DENSE_EMB_CACHE), q_emb.astype("float32"))
    DENSE_QIDS.write_text(json.dumps(qids))
    log.info("Query embeddings saved: %s  shape=%s", DENSE_EMB_CACHE, q_emb.shape)

    # FAISS search in subprocess
    with tempfile.TemporaryDirectory() as tmp:
        out_path = Path(tmp) / "results.json"
        worker = Path(__file__).parent / "_faiss_search_worker.py"
        cmd = [sys.executable, str(worker),
               str(DENSE_EMB_CACHE), str(faiss_path), str(ids_path),
               str(out_path), str(TOP_K)]
        log.info("Running FAISS search subprocess (top_k=%d)...", TOP_K)
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            log.error("FAISS failed:\n%s", result.stderr[-2000:])
            raise RuntimeError("FAISS search failed")
        raw = json.loads(out_path.read_text())

    # Worker returns a list of lists; convert to {qid: [aids]} dict
    if isinstance(raw, list):
        dense_results = {qid: lst for qid, lst in zip(qids, raw)}
    else:
        dense_results = raw  # already a dict (future-proofing)

    DENSE_CACHE.write_text(json.dumps(dense_results))
    log.info("Dense done: %d queries → %s", len(dense_results), DENSE_CACHE)


# ─── Stage 2: NER pre-computation ────────────────────────────────────────────

def stage2_ner(queries: list[tuple[str, str]], force: bool = False):
    if NER_CACHE.exists() and not force:
        cache = json.loads(NER_CACHE.read_text())
        qids = {q[0] for q in queries}
        if qids.issubset(set(cache.keys())):
            log.info("NER cache hit: %s (%d entries)", NER_CACHE, len(cache))
            return
        log.info("NER cache stale — recomputing...")

    log.info("Stage 2: GLiNER NER on %d queries (~%.1f hrs)...",
             len(queries), len(queries) / 26 / 3600)

    ner = FashionNER(model_name="urchade/gliner_medium-v2.1", threshold=0.4)
    cache: dict[str, dict] = {}
    t0 = time.time()
    for i, (qid, text) in enumerate(queries):
        cache[qid] = ner.extract(text)
        if (i + 1) % 5000 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            remaining = (len(queries) - i - 1) / rate
            log.info("  NER %d/%d  (%.1f q/s, %.0f min remaining)",
                     i+1, len(queries), rate, remaining/60)
            # Checkpoint every 5K
            NER_CACHE.write_text(json.dumps(cache))

    del ner
    NER_CACHE.write_text(json.dumps(cache))
    log.info("NER done: %d queries in %.1f min → %s",
             len(cache), (time.time()-t0)/60, NER_CACHE)


def stage2b_bm25_ner(queries: list[tuple[str, str]],
                     ner_cache: dict[str, dict],
                     force: bool = False, n_workers: int = 12):
    """Pre-compute NER-boosted BM25 results for all queries (cached)."""
    if BM25_NER_CACHE.exists() and not force:
        log.info("BM25+NER cache hit: %s", BM25_NER_CACHE)
        return

    from concurrent.futures import ThreadPoolExecutor, as_completed
    import threading

    _tl = threading.local()
    def get_client():
        if not hasattr(_tl, "client"):
            _tl.client = OpenSearch(hosts=[{"host": "localhost", "port": 9200}],
                                    http_compress=True, timeout=30, maxsize=n_workers)
        return _tl.client

    def fetch_one(args):
        qid, text = args
        entities = ner_cache.get(qid, {})
        try:
            return qid, bm25_ner_query(get_client(), text, entities)
        except Exception as e:
            log.warning("BM25+NER error qid=%s: %s", qid, e)
            return qid, []

    log.info("Stage 2B: BM25+NER retrieval for %d queries (%d threads)...",
             len(queries), n_workers)
    results: dict[str, list[str]] = {}
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        pbar = tqdm(total=len(queries), desc="BM25+NER", ncols=80)
        futures = {pool.submit(fetch_one, q): q[0] for q in queries}
        for i, fut in enumerate(as_completed(futures), 1):
            qid, res = fut.result()
            results[qid] = res
            pbar.update(1)
            if i % 10_000 == 0:
                BM25_NER_CACHE.write_text(json.dumps(results))
        pbar.close()
    BM25_NER_CACHE.write_text(json.dumps(results))
    log.info("BM25+NER done: %.1f min → %s", (time.time()-t0)/60, BM25_NER_CACHE)


# ─── RRF fusion ───────────────────────────────────────────────────────────────

def rrf_fusion(
    bm25_lists: list[list[str]], dense_lists: list[list[str]],
    k: int = 60, bm25_w: float = 0.4, dense_w: float = 0.6,
    top_k: int = TOP_K,
) -> list[list[str]]:
    fused = []
    for bm25, dense in zip(bm25_lists, dense_lists):
        scores: dict[str, float] = {}
        for rank, doc in enumerate(bm25, 1):
            scores[doc] = scores.get(doc, 0.0) + bm25_w / (k + rank)
        for rank, doc in enumerate(dense, 1):
            scores[doc] = scores.get(doc, 0.0) + dense_w / (k + rank)
        fused.append([d for d, _ in sorted(scores.items(), key=lambda x: -x[1])[:top_k]])
    return fused


# ─── Stage 3: CE reranking ────────────────────────────────────────────────────

def build_article_text(row: dict) -> str:
    parts = []
    for field, limit in [("prod_name", None), ("product_type_name", None),
                          ("colour_group_name", None), ("section_name", None),
                          ("detail_desc", 200)]:
        val = str(row.get(field, "")).strip()
        if val and val.lower() not in ("nan", "none", ""):
            parts.append(val[:limit] if limit else val)
    return " | ".join(parts)


DEFAULT_CE_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def _ce_rerank_pass(
    label: str,
    cache_path: Path,
    qids: list[str],
    texts: list[str],
    hybrid_lists: list[list[str]],
    articles: dict[str, dict],
    force: bool,
    ce_model_path: str | None = None,
    CHECKPOINT_EVERY: int = 5000,
) -> dict[str, list[str]]:
    """Run cross-encoder over pre-computed hybrid candidate lists; resume from cache_path."""
    from sentence_transformers.cross_encoder import CrossEncoder

    if cache_path.exists() and not force:
        results: dict[str, list[str]] = json.loads(cache_path.read_text())
        n_done = len(results)
        if n_done >= len(qids):
            log.info("%s: cache complete (%d entries) → %s", label, n_done, cache_path)
            return results
        log.info("%s: resuming from %d / %d queries...", label, n_done, len(qids))
    else:
        results = {}

    done_qids = set(results.keys())
    import torch
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model_name = ce_model_path or DEFAULT_CE_MODEL
    log.info("%s: loading CE model → %s", label, model_name)
    ce = CrossEncoder(model_name, max_length=512, device=device)

    art_text: dict[str, str] = {}

    def get_text(aid: str) -> str:
        if aid not in art_text:
            art_text[aid] = build_article_text(articles.get(aid, {}))
        return art_text[aid]

    t0 = time.time()
    processed = 0
    already_done = len(done_qids)
    log.info("%s: CE reranking → top-50...", label)
    for qid, query_text, candidates in zip(qids, texts, hybrid_lists):
        if qid in done_qids:
            continue
        if not candidates:
            results[qid] = []
        else:
            pairs = [(query_text, get_text(cid)) for cid in candidates]
            scores = ce.predict(pairs, batch_size=64, show_progress_bar=False)
            ranked = sorted(zip(candidates, scores), key=lambda x: -x[1])
            results[qid] = [cid for cid, _ in ranked[:50]]
        processed += 1
        total_done = already_done + processed
        if processed % CHECKPOINT_EVERY == 0:
            elapsed = time.time() - t0
            rate = processed / elapsed
            remaining_q = len(qids) - total_done
            log.info("  %s  %d/%d  (%.1f q/s, %.0f min remaining)",
                     label, total_done, len(qids), rate,
                     remaining_q / rate / 60 if rate > 0 else 0)
            cache_path.write_text(json.dumps(results))

    cache_path.write_text(json.dumps(results))
    log.info("%s done: %.1f min → %s  (entries: %d)",
             label, (time.time() - t0) / 60, cache_path, len(results))
    return results


def stage3_ce_rerank(
    queries: list[tuple[str, str]],
    bm25_results: dict[str, list[str]],
    dense_results: dict[str, list[str]],
    articles: dict[str, dict],
    force: bool = False,
    ce_model_path: str | None = None,
):
    """CE rerank: (1) plain Hybrid C → Config 6; (2) NER-Hybrid → Config 8."""
    qids = [q[0] for q in queries]
    texts = [q[1] for q in queries]
    bm25_lists = [bm25_results.get(qid, []) for qid in qids]
    dense_lists = [dense_results.get(qid, []) for qid in qids]

    hybrid_plain = rrf_fusion(bm25_lists, dense_lists, bm25_w=0.4, dense_w=0.6)
    _ce_rerank_pass(
        "CE [plain Hybrid C → Config 6]", CE_CACHE,
        qids, texts, hybrid_plain, articles, force,
        ce_model_path=ce_model_path,
    )

    if not BM25_NER_CACHE.exists():
        log.warning(
            "No %s — skip NER-hybrid CE (run Stage 2). Config 8 will be missing in metrics.",
            BM25_NER_CACHE,
        )
        return

    bm25_ner = json.loads(BM25_NER_CACHE.read_text())
    ner_bm25_lists = [bm25_ner.get(qid, []) for qid in qids]
    hybrid_ner = rrf_fusion(ner_bm25_lists, dense_lists, bm25_w=0.4, dense_w=0.6)
    _ce_rerank_pass(
        "CE [NER Hybrid → Config 8]", CE_NER_CACHE,
        qids, texts, hybrid_ner, articles, force,
        ce_model_path=ce_model_path,
    )


# ─── Stage 4: Evaluate all configs ───────────────────────────────────────────

def measure_latency(
    queries: list[tuple[str, str]],
    bm25_results: dict,
    dense_results: dict,
    ner_cache: dict | None,
    n_sample: int = 500,
) -> dict[str, float]:
    """
    Measure mean query latency (ms) for each retrieval stage on a small sample.
    CE latency measured separately (it's batch-optimised, not per-query online).
    """
    rng = random.Random(RANDOM_SEED)
    sample = rng.sample(queries, min(n_sample, len(queries)))
    client = OpenSearch(hosts=[{"host": "localhost", "port": 9200}],
                        http_compress=True, timeout=30)
    latencies: dict[str, list[float]] = {"bm25": [], "dense_lookup": [], "rrf": []}

    log.info("Measuring latency on %d sample queries...", len(sample))
    for qid, text in tqdm(sample, desc="latency", ncols=80):
        # BM25 latency
        t0 = time.perf_counter()
        bm25_query(client, text, top_k=TOP_K)
        latencies["bm25"].append((time.perf_counter() - t0) * 1000)

        # Dense lookup (pre-computed, just dict access)
        t0 = time.perf_counter()
        dense_results.get(qid, [])
        latencies["dense_lookup"].append((time.perf_counter() - t0) * 1000)

        # RRF (in-memory)
        t0 = time.perf_counter()
        rrf_fusion([bm25_results.get(qid, [])], [dense_results.get(qid, [])],
                   bm25_w=0.4, dense_w=0.6)
        latencies["rrf"].append((time.perf_counter() - t0) * 1000)

    return {k: float(np.mean(v)) for k, v in latencies.items()}


def bootstrap_ci(values: list[float], n_boot: int = 1000, ci: float = 0.95) -> tuple[float, float]:
    rng = np.random.default_rng(RANDOM_SEED)
    boot_means = [rng.choice(values, size=len(values), replace=True).mean()
                  for _ in range(n_boot)]
    alpha = (1 - ci) / 2
    return float(np.quantile(boot_means, alpha)), float(np.quantile(boot_means, 1 - alpha))


def stage4_evaluate(
    queries: list[tuple[str, str]],
    qrels: dict[str, dict[str, int]],
    bm25_results: dict[str, list[str]],
    dense_results: dict[str, list[str]],
    bm25_ner_results: dict[str, list[str]] | None,
    ce_results: dict[str, list[str]] | None,
) -> dict:
    log.info("Stage 4: Evaluating all configs on %d queries...", len(queries))
    qids = [q[0] for q in queries]

    bm25_lists  = [bm25_results.get(qid, []) for qid in qids]
    dense_lists = [dense_results.get(qid, []) for qid in qids]
    ner_bm25_lists = (
        [bm25_ner_results.get(qid, []) for qid in qids]
        if bm25_ner_results else None
    )

    # All configs — all pre-computed, no live network calls
    configs = {
        "1_BM25":          bm25_lists,
        "3_Dense":         dense_lists,
        "4c_Hybrid_C":     rrf_fusion(bm25_lists, dense_lists, bm25_w=0.4, dense_w=0.6),
        "2b_BM25_NER":     ner_bm25_lists,
        "7_Hybrid_NER":    rrf_fusion(ner_bm25_lists, dense_lists, bm25_w=0.4, dense_w=0.6)
                           if ner_bm25_lists else None,
        "6_Hybrid_CE":     [ce_results.get(qid, []) for qid in qids] if ce_results else None,
        "8_Full_Pipeline": [ce_results.get(qid, []) for qid in qids] if ce_results else None,
    }

    all_results = {}
    for name, retrieved_lists in configs.items():
        if retrieved_lists is None:
            log.info("  Skipping %s (no data)", name)
            continue
        per_q = []
        ndcg10_vals = []
        for qid, retrieved in zip(qids, retrieved_lists):
            q_qrels = qrels.get(qid, {})
            if not q_qrels:
                continue
            # Count queries with empty retrieval (e.g. BM25 miss) as zero-score, not skipped.
            m = compute_all_metrics(retrieved, q_qrels, ks=KS)
            per_q.append(m)
            ndcg10_vals.append(m.get("ndcg@10", 0))

        agg = aggregate_metrics(per_q)
        lo, hi = bootstrap_ci(ndcg10_vals)
        all_results[name] = {
            "n_queries": len(per_q),
            "metrics": agg,
            "ndcg10_ci95": [round(lo, 4), round(hi, 4)],
        }
        log.info(
            "  %-25s nDCG@10=%.4f [%.4f–%.4f]  MRR=%.4f  R@10=%.4f",
            name, agg.get("ndcg@10", 0), lo, hi, agg.get("mrr", 0),
            agg.get("recall@10", 0),
        )

    # Save — use distinct filenames so test-only results don't overwrite full results
    suffix = "_test" if len(queries) < 100_000 else ""
    out_path = OUT_DIR / f"full_ablation{suffix}.json"
    out_path.write_text(json.dumps(all_results, indent=2))
    log.info("Ablation results (%d queries) → %s", len(queries), out_path)
    return all_results


def print_final_table(results: dict):
    p1_best = 0.0300
    print("\n" + "=" * 82)
    print("MODA — Full 253K Query Benchmark — Final Results")
    print("=" * 82)
    print(f"{'Config':<28} {'nDCG@10':>9} {'95% CI':>14} {'MRR':>9} {'R@10':>9} {'vs P1 Dense'}")
    print("-" * 82)
    for name, res in results.items():
        m   = res["metrics"]
        ci  = res.get("ndcg10_ci95", [0, 0])
        val = m.get("ndcg@10", 0)
        delta = (val / p1_best - 1) * 100
        sign  = "+" if delta >= 0 else ""
        print(
            f"{name:<28} {val:>9.4f} [{ci[0]:.4f}–{ci[1]:.4f}] "
            f"{m.get('mrr',0):>9.4f} {m.get('recall@10',0):>9.4f}"
            f"  {sign}{delta:.1f}%"
        )
    print("=" * 82)
    print(f"\nPhase 1 dense baseline (FashionCLIP, 10K sample): nDCG@10 = {p1_best}")
    print(f"Full benchmark: 253,685 queries × 105,542 articles")
    print(f"Ground truth:   purchase-based relevance (1 positive + ~9 hard negatives)\n")


# ─── Main ─────────────────────────────────────────────────────────────────────

QUERY_SPLITS_PATH = _REPO_ROOT / "data" / "processed" / "query_splits.json"


def load_test_qids() -> set[str]:
    """Load the held-out test query IDs from the split file."""
    if not QUERY_SPLITS_PATH.exists():
        raise FileNotFoundError(
            f"No query splits found at {QUERY_SPLITS_PATH}. "
            "Run train_cross_encoder.py first to generate train/val/test splits."
        )
    obj = json.loads(QUERY_SPLITS_PATH.read_text())
    return set(obj["test"])


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--stages", default="all",
                   help="Comma-separated stages: 1,2,3,4 or 'all'")
    p.add_argument("--force", action="store_true",
                   help="Recompute even if cache exists")
    p.add_argument("--test-only", action="store_true",
                   help="Evaluate only on held-out test queries (prevents data leakage "
                        "when comparing fine-tuned vs zero-shot models)")
    p.add_argument("--ce-model", type=str, default=None,
                   help="Path to a fine-tuned CrossEncoder model for Stage 3. "
                        "If not set, uses the default ms-marco zero-shot model.")
    return p.parse_args()


def main():
    args = parse_args()
    stages_str = args.stages.lower()
    if stages_str == "all":
        run_stages = {1, 2, 3, 4}
    else:
        run_stages = set(int(s.strip()) for s in stages_str.split(","))

    log.info("=" * 60)
    log.info("MODA Full 253K Evaluation Pipeline")
    log.info("Stages to run: %s", sorted(run_stages))
    log.info("=" * 60)

    queries, qrels = load_benchmark()

    if args.test_only:
        test_qids = load_test_qids()
        all_count = len(queries)
        queries = [(qid, qt) for qid, qt in queries if qid in test_qids]
        qrels   = {qid: v for qid, v in qrels.items() if qid in test_qids}
        log.info("--test-only: filtered %d → %d queries (held-out test set)",
                 all_count, len(queries))

    # ── Stage 1: BM25 + FAISS retrieval (~24 min BM25, ~8 min FAISS) ─────────
    if 1 in run_stages:
        stage1_bm25(queries, force=args.force)
        stage1_dense(queries, force=args.force)

    # ── Stage 2: NER + NER-boosted BM25 (~2.7 hrs NER, then ~24 min BM25+NER)
    if 2 in run_stages:
        stage2_ner(queries, force=args.force)
        if NER_CACHE.exists():
            ner_cache = json.loads(NER_CACHE.read_text())
            stage2b_bm25_ner(queries, ner_cache, force=args.force)

    # ── Stage 3: CE rerank hybrid C for all 253K (~8.5 hrs) ──────────────────
    if 3 in run_stages:
        if not BM25_CACHE.exists() or not DENSE_CACHE.exists():
            log.error("Stage 1 caches not found — run --stages 1 first")
            sys.exit(1)
        bm25_results  = json.loads(BM25_CACHE.read_text())
        dense_results = json.loads(DENSE_CACHE.read_text())
        articles = load_articles()
        stage3_ce_rerank(queries, bm25_results, dense_results, articles,
                         force=args.force, ce_model_path=args.ce_model)

    # ── Stage 4: Compute metrics for all configs (fast, all in-memory) ────────
    if 4 in run_stages:
        if not BM25_CACHE.exists() or not DENSE_CACHE.exists():
            log.error("Stage 1 caches not found — run --stages 1 first")
            sys.exit(1)
        bm25_results     = json.loads(BM25_CACHE.read_text())
        dense_results    = json.loads(DENSE_CACHE.read_text())
        bm25_ner_results = json.loads(BM25_NER_CACHE.read_text()) if BM25_NER_CACHE.exists() else None
        ce_results       = json.loads(CE_CACHE.read_text()) if CE_CACHE.exists() else None

        results = stage4_evaluate(
            queries, qrels, bm25_results, dense_results, bm25_ner_results, ce_results,
        )
        print_final_table(results)

    log.info("Pipeline complete.")


if __name__ == "__main__":
    main()
