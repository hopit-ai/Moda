"""
MODA Phase 2 — ColBERT v2 Late Interaction Reranking

Evaluates ColBERT v2 (zero-shot, off-the-shelf) as a reranker in the Phase 2
pipeline, as a direct comparison against the cross-encoder (ms-marco-MiniLM-L-6-v2).

Three-way comparison of query–document interaction depth (all zero-shot):
  Bi-encoder (FashionCLIP):    single-vector dot product (no interaction)
  ColBERT v2 (late interaction): per-token MaxSim (precomputable doc tokens)
  Cross-encoder (MiniLM-L6):    full cross-attention (no precomputation)

Configs evaluated:
  Hybrid_NER_baseline:       Hybrid (NER-BM25×0.4 + Dense×0.6), no reranking
  Hybrid_NER → ColBERT@50:   ColBERT reranks hybrid top-100 → top-50
  Hybrid_NER → CE@50:        CE reranks hybrid top-100 → top-50 (reference)
  ColBERT → CE cascade:      ColBERT@100→50 then CE@50→50 (cascade)

ColBERT implementation uses transformers + torch directly (no ragatouille/colbert-ai).
Model weights loaded from HuggingFace: colbert-ir/colbertv2.0 (BERT-base + 768→128 projection).
Scoring: per-token MaxSim — for each query token, find max cosine sim with any doc token, sum.

Same data split as all Phase 2 evals:
  10K queries, random_state=42, data/raw/hnm_real/

Usage:
  python benchmark/eval_colbert.py
  python benchmark/eval_colbert.py --n_queries 1000         # quick test
  python benchmark/eval_colbert.py --skip_ce                # ColBERT only
  python benchmark/eval_colbert.py --measure_latency        # include latency
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from opensearchpy import OpenSearch
from tqdm import tqdm
from transformers import BertModel, BertTokenizerFast

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT))

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
    N_QUERIES,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

DEFAULT_COLBERT_MODEL = "colbert-ir/colbertv2.0"
COLBERT_RESULTS_PATH = RESULTS_DIR / "hnm_colbert_rerank.json"


# ─── ColBERT v2 reranker (direct implementation) ─────────────────────────────

class ColBERTReranker:
    """
    ColBERT v2 late-interaction reranker.

    Loads colbert-ir/colbertv2.0 weights directly via transformers:
      - BERT-base encoder (768-dim hidden states)
      - Linear projection (768 → 128, no bias)
      - Per-token L2 normalization

    Query encoding:  [CLS] [Q] tokens... [SEP] [MASK]... (padded to query_maxlen)
    Doc encoding:    [CLS] [D] tokens... [SEP]

    Scoring: MaxSim — sum of per-query-token max cosine similarities.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_COLBERT_MODEL,
        device: str = "cpu",
        query_maxlen: int = 32,
        doc_maxlen: int = 180,
        doc_batch_size: int = 32,
    ):
        self.device = device
        self.query_maxlen = query_maxlen
        self.doc_maxlen = doc_maxlen
        self.doc_batch_size = doc_batch_size

        log.info("Loading ColBERT tokenizer: %s", model_name)
        self.tokenizer = BertTokenizerFast.from_pretrained(model_name)

        log.info("Loading ColBERT BERT encoder on %s...", device)
        self.bert = BertModel.from_pretrained(
            model_name, add_pooling_layer=False,
        ).to(device)
        self.bert.eval()

        log.info("Loading ColBERT projection layer (768→128)...")
        from huggingface_hub import hf_hub_download
        from safetensors import safe_open

        ckpt_path = hf_hub_download(model_name, "model.safetensors")
        with safe_open(ckpt_path, framework="pt") as f:
            proj_weight = f.get_tensor("linear.weight")  # [128, 768]

        self.linear = torch.nn.Linear(768, 128, bias=False)
        self.linear.weight.data = proj_weight
        self.linear = self.linear.to(device)
        self.linear.eval()

        self.Q_id = self.tokenizer.convert_tokens_to_ids("[unused0]")
        self.D_id = self.tokenizer.convert_tokens_to_ids("[unused1]")
        self.CLS = self.tokenizer.cls_token_id
        self.SEP = self.tokenizer.sep_token_id
        self.MASK = self.tokenizer.mask_token_id
        self.PAD = self.tokenizer.pad_token_id

        log.info("ColBERT loaded  (query_maxlen=%d, doc_maxlen=%d)",
                 query_maxlen, doc_maxlen)

    @torch.no_grad()
    def _encode_query(self, query: str) -> torch.Tensor:
        """Encode a single query → (query_maxlen, 128) L2-normalized."""
        tokens = self.tokenizer.encode(
            query, add_special_tokens=False,
            max_length=self.query_maxlen - 3, truncation=True,
        )
        # [CLS] [Q] tokens... [SEP] [MASK]... → pad to query_maxlen
        ids = [self.CLS, self.Q_id] + tokens + [self.SEP]
        ids += [self.MASK] * (self.query_maxlen - len(ids))
        mask = [1] * len(ids)

        inputs = {
            "input_ids": torch.tensor([ids], device=self.device),
            "attention_mask": torch.tensor([mask], device=self.device),
        }
        hidden = self.bert(**inputs).last_hidden_state      # (1, qlen, 768)
        projected = self.linear(hidden)                      # (1, qlen, 128)
        return F.normalize(projected.squeeze(0), dim=-1)     # (qlen, 128)

    @torch.no_grad()
    def _encode_docs(self, doc_texts: list[str]) -> list[torch.Tensor]:
        """Encode documents → list of (n_tokens, 128) L2-normalized tensors."""
        all_token_ids = []
        for text in doc_texts:
            tokens = self.tokenizer.encode(
                text, add_special_tokens=False,
                max_length=self.doc_maxlen - 3, truncation=True,
            )
            all_token_ids.append([self.CLS, self.D_id] + tokens + [self.SEP])

        results: list[torch.Tensor] = []
        bs = self.doc_batch_size
        for start in range(0, len(all_token_ids), bs):
            batch_ids = all_token_ids[start:start + bs]
            max_len = max(len(t) for t in batch_ids)
            lengths = [len(t) for t in batch_ids]

            padded_ids = [t + [self.PAD] * (max_len - len(t)) for t in batch_ids]
            attn_mask = [[1] * l + [0] * (max_len - l) for l in lengths]

            inputs = {
                "input_ids": torch.tensor(padded_ids, device=self.device),
                "attention_mask": torch.tensor(attn_mask, device=self.device),
            }
            hidden = self.bert(**inputs).last_hidden_state   # (B, max_len, 768)
            projected = self.linear(hidden)                   # (B, max_len, 128)

            for i, real_len in enumerate(lengths):
                doc_emb = F.normalize(projected[i, :real_len], dim=-1)
                results.append(doc_emb)

        return results

    def rerank(
        self,
        query: str,
        doc_texts: list[str],
        candidate_ids: list[str],
        top_k: int = 50,
    ) -> list[str]:
        """Rerank candidates by ColBERT MaxSim score, return top_k IDs."""
        if not doc_texts:
            return []

        Q = self._encode_query(query)               # (qlen, 128)
        D_list = self._encode_docs(doc_texts)        # list of (dlen_i, 128)

        scores = []
        for D in D_list:
            sim = Q @ D.T                            # (qlen, dlen)
            scores.append(sim.max(dim=1).values.sum().item())

        ranked = sorted(zip(candidate_ids, scores), key=lambda x: -x[1])
        return [cid for cid, _ in ranked[:top_k]]


# ─── Batch reranking ──────────────────────────────────────────────────────────

def colbert_rerank_batch(
    queries: list[str],
    candidate_lists: list[list[str]],
    articles: dict[str, dict],
    model_name: str = DEFAULT_COLBERT_MODEL,
    top_k: int = TOP_K_FINAL,
    device: str | None = None,
) -> list[list[str]]:
    """Rerank all candidate lists using ColBERT v2 late interaction."""
    if device is None:
        device = "mps" if torch.backends.mps.is_available() else "cpu"

    colbert = ColBERTReranker(model_name=model_name, device=device)

    article_texts: dict[str, str] = {}

    def get_text(aid: str) -> str:
        if aid not in article_texts:
            article_texts[aid] = _build_article_text(articles.get(aid, {}))
        return article_texts[aid]

    max_pool = max(len(c) for c in candidate_lists) if candidate_lists else 0
    log.info("ColBERT reranking %d queries (pool=%d → top-%d, device=%s)...",
             len(queries), max_pool, top_k, device)

    results: list[list[str]] = []
    t0 = time.time()
    for i, (query, candidates) in enumerate(zip(queries, candidate_lists)):
        if not candidates:
            results.append([])
            continue

        doc_texts = [get_text(cid) for cid in candidates]
        valid = [(cid, txt) for cid, txt in zip(candidates, doc_texts)
                 if txt.strip()]
        if not valid:
            results.append(candidates[:top_k])
            continue

        valid_cids, valid_texts = zip(*valid)
        reranked = colbert.rerank(
            query, list(valid_texts), list(valid_cids), top_k=top_k)
        results.append(reranked)

        if (i + 1) % 200 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            remaining = (len(queries) - i - 1) / rate / 60
            log.info("  ColBERT: %d/%d  (%.1f q/s, ~%.0f min remaining)",
                     i + 1, len(queries), rate, remaining)

    elapsed = time.time() - t0
    log.info("ColBERT reranking done: %d queries in %.1f min (%.1f q/s)",
             len(queries), elapsed / 60, len(queries) / elapsed)
    return results


# ─── Latency comparison ──────────────────────────────────────────────────────

def measure_rerank_latency(
    queries: list[str],
    candidate_lists: list[list[str]],
    articles: dict[str, dict],
    colbert_model: str = DEFAULT_COLBERT_MODEL,
    n_sample: int = 100,
) -> dict[str, float]:
    """Measure per-query reranking latency for ColBERT vs CE."""
    from sentence_transformers import CrossEncoder

    device = "mps" if torch.backends.mps.is_available() else "cpu"

    rng = np.random.default_rng(42)
    idx = rng.choice(len(queries), size=min(n_sample, len(queries)), replace=False)
    sample_q = [queries[i] for i in idx]
    sample_c = [candidate_lists[i] for i in idx]

    article_texts: dict[str, str] = {}

    def get_text(aid: str) -> str:
        if aid not in article_texts:
            article_texts[aid] = _build_article_text(articles.get(aid, {}))
        return article_texts[aid]

    # ── ColBERT ──────────────────────────────────────────────────────────────
    log.info("Latency: measuring ColBERT on %d queries...", len(sample_q))
    colbert = ColBERTReranker(model_name=colbert_model, device=device)
    cb_times: list[float] = []
    for q, cands in zip(sample_q, sample_c):
        valid = [(c, get_text(c)) for c in cands if get_text(c).strip()]
        if not valid:
            continue
        cids, texts = zip(*valid)
        t0 = time.perf_counter()
        colbert.rerank(q, list(texts), list(cids), top_k=50)
        cb_times.append((time.perf_counter() - t0) * 1000)

    # ── CE ────────────────────────────────────────────────────────────────────
    log.info("Latency: measuring CE on %d queries...", len(sample_q))
    ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=512)
    ce_times: list[float] = []
    for q, cands in zip(sample_q, sample_c):
        pairs = [(q, get_text(c)) for c in cands if get_text(c).strip()]
        if not pairs:
            continue
        t0 = time.perf_counter()
        ce.predict(pairs, batch_size=64, show_progress_bar=False)
        ce_times.append((time.perf_counter() - t0) * 1000)

    return {
        "colbert_mean_ms": float(np.mean(cb_times)) if cb_times else 0,
        "colbert_p50_ms":  float(np.median(cb_times)) if cb_times else 0,
        "colbert_p95_ms":  float(np.percentile(cb_times, 95)) if cb_times else 0,
        "ce_mean_ms":      float(np.mean(ce_times)) if ce_times else 0,
        "ce_p50_ms":       float(np.median(ce_times)) if ce_times else 0,
        "ce_p95_ms":       float(np.percentile(ce_times, 95)) if ce_times else 0,
        "n_samples":       len(idx),
        "pool_size":       TOP_K_RERANK,
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="MODA Phase 2 — ColBERT v2 Late Interaction Reranking Eval")
    p.add_argument("--n_queries", type=int, default=N_QUERIES,
                   help="Number of queries (default: 10,000; 0 = all)")
    p.add_argument("--colbert_model", default=DEFAULT_COLBERT_MODEL,
                   help="ColBERT model name or path")
    p.add_argument("--skip_ce", action="store_true",
                   help="Skip CE reranking (no CE reference / cascade)")
    p.add_argument("--no_cascade", action="store_true",
                   help="Skip ColBERT → CE cascade evaluation")
    p.add_argument("--measure_latency", action="store_true",
                   help="Measure per-query reranking latency (ColBERT vs CE)")
    p.add_argument("--latency_samples", type=int, default=100,
                   help="Queries to sample for latency measurement")
    p.add_argument("--device", default=None,
                   help="Device for ColBERT (default: auto-detect mps/cpu)")
    args = p.parse_args()

    t_start = time.time()

    # ── 1. Load data (same split as all Phase 2 evals) ──────────────────────
    queries, qrels = load_benchmark(args.n_queries)
    articles = load_articles()
    qids  = [q[0] for q in queries]
    texts = [q[1] for q in queries]

    # ── 2. NER (load from 10K cache or compute) ────────────────────────────
    ner_cache = load_or_compute_ner(queries)

    # ── 3. BM25 (NER-boosted) retrieval ─────────────────────────────────────
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

    # ── 4. Dense (FAISS) retrieval ──────────────────────────────────────────
    log.info("Running dense retrieval (FAISS)...")
    dense_results = dense_search_batch(
        queries, model_name=DENSE_MODEL, top_k=TOP_K_RERANK)

    # ── 5. RRF Hybrid fusion ────────────────────────────────────────────────
    log.info("Fusing with RRF (BM25=%.1f, Dense=%.1f)...",
             BM25_WEIGHT, DENSE_WEIGHT)
    bm25_lists  = [bm25_results.get(qid, []) for qid in qids]
    dense_lists = [dense_results.get(qid, []) for qid in qids]
    hybrid_lists = rrf_fusion(bm25_lists, dense_lists)
    hybrid_results = {qid: lst for qid, lst in zip(qids, hybrid_lists)}

    # ── 6. Evaluate hybrid baseline (sanity check, matches Config 7) ────────
    res_hybrid = evaluate(hybrid_results, qrels, label="Hybrid_NER_baseline")

    # ── 7. ColBERT reranking ────────────────────────────────────────────────
    log.info("ColBERT reranking (model: %s)...", args.colbert_model)
    colbert_lists = colbert_rerank_batch(
        texts, hybrid_lists, articles,
        model_name=args.colbert_model, top_k=TOP_K_FINAL,
        device=args.device,
    )
    colbert_results = {qid: lst for qid, lst in zip(qids, colbert_lists)}
    res_colbert = evaluate(
        colbert_results, qrels, label="Hybrid_NER_ColBERT@50")

    all_results = {
        "Hybrid_NER_baseline":   res_hybrid,
        "Hybrid_NER_ColBERT@50": res_colbert,
    }

    # ── 8. CE reranking (reference comparison) ──────────────────────────────
    if not args.skip_ce:
        log.info("CE reranking (reference comparison)...")
        ce_lists = ce_rerank_batch(texts, hybrid_lists, articles)
        ce_results_dict = {qid: lst for qid, lst in zip(qids, ce_lists)}
        res_ce = evaluate(
            ce_results_dict, qrels, label="Hybrid_NER_CE@50")
        all_results["Hybrid_NER_CE@50"] = res_ce

    # ── 9. ColBERT → CE cascade ─────────────────────────────────────────────
    if not args.skip_ce and not args.no_cascade:
        log.info("ColBERT → CE cascade (ColBERT@100→50, then CE@50→50)...")
        cascade_lists = ce_rerank_batch(
            texts, colbert_lists, articles, top_k=TOP_K_FINAL)
        cascade_results = {qid: lst for qid, lst in zip(qids, cascade_lists)}
        res_cascade = evaluate(
            cascade_results, qrels, label="ColBERT_CE_cascade")
        all_results["ColBERT_CE_cascade"] = res_cascade

    # ── 10. Latency measurement ─────────────────────────────────────────────
    latency = None
    if args.measure_latency:
        latency = measure_rerank_latency(
            texts, hybrid_lists, articles,
            colbert_model=args.colbert_model,
            n_sample=args.latency_samples,
        )
        log.info(
            "Latency — ColBERT: %.1fms  CE: %.1fms  (mean, pool=%d)",
            latency["colbert_mean_ms"], latency["ce_mean_ms"], TOP_K_RERANK)

    # ── 11. Save results ────────────────────────────────────────────────────
    output = {
        "configs": all_results,
        "latency": latency,
        "settings": {
            "n_queries": len(queries),
            "random_state": 42,
            "colbert_model": args.colbert_model,
            "ce_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "pool_size": TOP_K_RERANK,
            "rerank_top_k": TOP_K_FINAL,
            "bm25_weight": BM25_WEIGHT,
            "dense_weight": DENSE_WEIGHT,
        },
    }
    with open(COLBERT_RESULTS_PATH, "w") as f:
        json.dump(output, f, indent=2)
    log.info("Results saved → %s", COLBERT_RESULTS_PATH)

    # ── 12. Print comparison table ──────────────────────────────────────────
    ref_key = "Hybrid_NER_CE@50"
    if ref_key in all_results:
        ref_ndcg = all_results[ref_key]["metrics"]["ndcg@10"]
        ref_label = "CE"
    else:
        ref_ndcg = res_hybrid["metrics"]["ndcg@10"]
        ref_label = "Baseline"

    print("\n" + "=" * 85)
    print("COLBERT v2 vs CROSS-ENCODER — Phase 2 Reranking Comparison")
    print(f"  {len(queries):,} queries  |  Pool: {TOP_K_RERANK} → Top-{TOP_K_FINAL}"
          f"  |  ColBERT: {args.colbert_model}")
    print("=" * 85)
    print(f"{'Config':<30} {'nDCG@10':>9} {'MRR':>9} {'R@10':>9}"
          f"  vs {ref_label}")
    print("-" * 85)
    for name, res in all_results.items():
        m = res["metrics"]
        ndcg = m["ndcg@10"]
        delta = (ndcg / ref_ndcg - 1) * 100 if ref_ndcg > 0 else 0
        sign  = "+" if delta >= 0 else ""
        icon  = "✅" if delta > 0 else ("❌" if delta < -1 else "≈")
        print(
            f"{name:<30} {ndcg:>9.4f} {m['mrr']:>9.4f}"
            f" {m['recall@10']:>9.4f}  {sign}{delta:.1f}% {icon}"
        )
    print("=" * 85)

    if latency:
        print(f"\n  Latency (reranking {TOP_K_RERANK} candidates per query):")
        print(f"    ColBERT:  {latency['colbert_mean_ms']:>7.1f}ms mean  "
              f"{latency['colbert_p50_ms']:>7.1f}ms p50  "
              f"{latency['colbert_p95_ms']:>7.1f}ms p95")
        print(f"    CE:       {latency['ce_mean_ms']:>7.1f}ms mean  "
              f"{latency['ce_p50_ms']:>7.1f}ms p50  "
              f"{latency['ce_p95_ms']:>7.1f}ms p95")

    print(f"\nTotal elapsed: {(time.time() - t_start) / 60:.1f} min\n")

    return all_results


if __name__ == "__main__":
    main()
