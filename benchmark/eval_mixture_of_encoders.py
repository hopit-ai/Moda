"""
MODA Phase 2 — Mixture of Encoders (Superlinked-style) Retrieval

STATUS: EXPLORATORY — NOT PART OF THE CORE PHASE 1–2 BENCHMARK
========================================================================
This experiment is included for transparency but is excluded from the
official Phase 1–2 results. Superlinked's MoE architecture assumes each
field is encoded by a *type-specific trained encoder* (e.g. a learned
color embedding that maps "navy" near "dark blue" but away from "dark
brown", a categorical product-type encoder, etc.). Our implementation
reuses the same general-purpose FashionCLIP text encoder for all four
fields. FashionCLIP was trained on full product descriptions, not isolated
attribute values — encoding "Dark Blue" in isolation produces a poor
colour representation. The -12% result vs single-vector dense is expected:
we're diluting one well-functioning 512-dim text representation with
three poorly-functioning 512-dim attribute representations.

This is NOT a verdict on the MoE concept. It is a verdict on applying the
same text encoder four times, which is not what Superlinked proposes.
Proper MoE requires trained field-specific encoders and is deferred to
future work.
========================================================================

Encodes each product attribute with a separate FashionCLIP encoder and
concatenates into a structured product vector.  At query time, NER-extracted
entities determine per-field weights — no retraining, no re-indexing.

Architecture (per product):
  text_block  = FashionCLIP(prod_name + detail_desc)          512-dim
  color_block = FashionCLIP(colour_group_name)                512-dim
  type_block  = FashionCLIP(product_type_name)                512-dim
  group_block = FashionCLIP(product_group_name)               512-dim

  score(q, p) = w_text  * cos(q_text,  p_text)
              + w_color * cos(q_color, p_color)
              + w_type  * cos(q_type,  p_type)
              + w_group * cos(q_group, p_group)

Query-side:
  q_text  = FashionCLIP(query_text)
  q_color = FashionCLIP(NER color entity)     — zero if no color detected
  q_type  = FashionCLIP(NER garment type)     — zero if no type detected
  q_group = FashionCLIP(NER product group)    — zero if no group detected

Weights are *adaptive*: when NER detects an entity for a field, that field's
weight is boosted.  This gives structured queries (e.g. "navy slim fit jeans")
a big advantage over single-vector dense retrieval.

Configs evaluated:
  Dense_only (FashionCLIP):       single-vector baseline (existing Config 3)
  MoE_retrieval:                  multi-field structured retrieval
  Hybrid_NER_MoE:                 BM25 (NER-boosted) + MoE via RRF
  Hybrid_NER_MoE → CE@50:        structured retrieval + CE reranking

Same data split as all Phase 2 evals: 10K queries, random_state=42.

Usage:
  python benchmark/eval_mixture_of_encoders.py
  python benchmark/eval_mixture_of_encoders.py --n_queries 1000
  python benchmark/eval_mixture_of_encoders.py --skip_ce
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

from benchmark.eval_full_pipeline import (
    load_benchmark,
    load_articles,
    load_or_compute_ner,
    bm25_ner_search,
    rrf_fusion,
    evaluate,
    ce_rerank_batch,
    RESULTS_DIR,
    DENSE_MODEL,
    BM25_WEIGHT,
    DENSE_WEIGHT,
    TOP_K_RERANK,
    TOP_K_FINAL,
    N_QUERIES,
    HNM_DIR,
    EMBEDDINGS_DIR,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

MOE_RESULTS_PATH = RESULTS_DIR / "hnm_mixture_of_encoders.json"

ADAPTIVE_BOOSTS = {
    "color": 0.25,
    "type": 0.30,
    "group": 0.15,
}


# ─── Mixture of Encoders Retriever ────────────────────────────────────────────

class MixtureOfEncodersRetriever:
    """
    Superlinked-style multi-field retriever.

    Pre-computes per-field embeddings for all 105K products using FashionCLIP.
    At query time, scores each field independently via vectorized numpy ops
    and fuses with adaptive weights based on NER-detected entities.
    """

    def __init__(
        self,
        articles: dict[str, dict],
        model_name: str = DENSE_MODEL,
        device: str | None = None,
    ):
        import torch
        from benchmark.models import load_clip_model, encode_texts_clip

        if device is None:
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = device
        self.embed_dim = 512

        safe_name = model_name.replace("/", "_").replace(":", "_")
        emb_path = EMBEDDINGS_DIR / f"{safe_name}_embeddings.npy"
        ids_path = EMBEDDINGS_DIR / f"{safe_name}_article_ids.json"

        log.info("Loading pre-computed text embeddings: %s", emb_path)
        self.text_embeddings = np.load(str(emb_path)).astype(np.float32)
        with open(ids_path) as f:
            self.article_ids: list[str] = json.load(f)
        self.n_articles = len(self.article_ids)
        self.aid_to_idx = {aid: i for i, aid in enumerate(self.article_ids)}

        norms = np.linalg.norm(self.text_embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.text_embeddings /= norms

        log.info("Building category vocabularies from %d articles...", len(articles))
        color_set: set[str] = set()
        type_set: set[str] = set()
        group_set: set[str] = set()
        for aid in self.article_ids:
            art = articles.get(aid, {})
            c = str(art.get("colour_group_name", "")).strip()
            t = str(art.get("product_type_name", "")).strip()
            g = str(art.get("product_group_name", "")).strip()
            if c: color_set.add(c)
            if t: type_set.add(t)
            if g: group_set.add(g)

        self.color_vocab = sorted(color_set)
        self.type_vocab = sorted(type_set)
        self.group_vocab = sorted(group_set)
        log.info("  Vocabularies: %d colors, %d types, %d groups",
                 len(self.color_vocab), len(self.type_vocab), len(self.group_vocab))

        color_to_idx = {v: i for i, v in enumerate(self.color_vocab)}
        type_to_idx = {v: i for i, v in enumerate(self.type_vocab)}
        group_to_idx = {v: i for i, v in enumerate(self.group_vocab)}

        self.product_color_idx = np.zeros(self.n_articles, dtype=np.int32)
        self.product_type_idx = np.zeros(self.n_articles, dtype=np.int32)
        self.product_group_idx = np.zeros(self.n_articles, dtype=np.int32)
        for i, aid in enumerate(self.article_ids):
            art = articles.get(aid, {})
            c = str(art.get("colour_group_name", "")).strip()
            t = str(art.get("product_type_name", "")).strip()
            g = str(art.get("product_group_name", "")).strip()
            self.product_color_idx[i] = color_to_idx.get(c, 0)
            self.product_type_idx[i] = type_to_idx.get(t, 0)
            self.product_group_idx[i] = group_to_idx.get(g, 0)

        log.info("Encoding category names with FashionCLIP (%d total)...",
                 len(self.color_vocab) + len(self.type_vocab) + len(self.group_vocab))
        model, _, tokenizer = load_clip_model(model_name, device=device)
        self._model = model
        self._tokenizer = tokenizer

        self.color_name_emb = self._encode_texts(self.color_vocab)
        self.type_name_emb = self._encode_texts(self.type_vocab)
        self.group_name_emb = self._encode_texts(self.group_vocab)

        self.product_color_emb = self.color_name_emb[self.product_color_idx]
        self.product_type_emb = self.type_name_emb[self.product_type_idx]
        self.product_group_emb = self.group_name_emb[self.product_group_idx]

        log.info("MoE retriever ready: %d products × 4 fields", self.n_articles)

    def _encode_texts(self, texts: list[str]) -> np.ndarray:
        from benchmark.models import encode_texts_clip
        if not texts:
            return np.zeros((0, self.embed_dim), dtype=np.float32)
        return encode_texts_clip(
            texts, self._model, self._tokenizer, self.device, batch_size=128
        )

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        return self._encode_texts(texts)


# ─── Batch search (vectorized) ────────────────────────────────────────────────

def moe_search_batch(
    retriever: MixtureOfEncodersRetriever,
    queries: list[tuple[str, str]],
    ner_cache: dict[str, dict[str, list[str]]],
    top_k: int = TOP_K_RERANK,
) -> dict[str, list[str]]:
    """Vectorized MoE retrieval: batch-encode queries, then score per-field."""
    qids = [q[0] for q in queries]
    texts = [q[1] for q in queries]
    n_q = len(queries)

    log.info("MoE: batch-encoding %d query texts...", n_q)
    q_text_emb = retriever.encode_queries(texts)

    unique_ner_strings: dict[str, int] = {}
    ner_string_list: list[str] = []

    def get_ner_emb_idx(entity_text: str) -> int:
        if entity_text not in unique_ner_strings:
            unique_ner_strings[entity_text] = len(ner_string_list)
            ner_string_list.append(entity_text)
        return unique_ner_strings[entity_text]

    q_color_info: list[int | None] = []
    q_type_info: list[int | None] = []
    q_group_info: list[int | None] = []

    for qid, _ in queries:
        ner = ner_cache.get(qid, {})
        if "color" in ner and ner["color"]:
            q_color_info.append(get_ner_emb_idx(" ".join(ner["color"])))
        else:
            q_color_info.append(None)
        if "garment type" in ner and ner["garment type"]:
            q_type_info.append(get_ner_emb_idx(" ".join(ner["garment type"])))
        else:
            q_type_info.append(None)
        group_text = None
        for label in ("garment type", "occasion"):
            if label in ner and ner[label]:
                group_text = " ".join(ner[label])
                break
        q_group_info.append(get_ner_emb_idx(group_text) if group_text else None)

    log.info("MoE: batch-encoding %d unique NER entity strings...", len(ner_string_list))
    if ner_string_list:
        ner_embeddings = retriever.encode_queries(ner_string_list)
    else:
        ner_embeddings = np.zeros((0, retriever.embed_dim), dtype=np.float32)

    log.info("MoE: scoring %d queries × %d products...", n_q, retriever.n_articles)
    t0 = time.time()

    text_scores = q_text_emb @ retriever.text_embeddings.T

    color_sims = ner_embeddings @ retriever.product_color_emb.T if len(ner_embeddings) > 0 else None
    type_sims = ner_embeddings @ retriever.product_type_emb.T if len(ner_embeddings) > 0 else None
    group_sims = ner_embeddings @ retriever.product_group_emb.T if len(ner_embeddings) > 0 else None

    results: dict[str, list[str]] = {}
    for i in range(n_q):
        w_color = ADAPTIVE_BOOSTS["color"] if q_color_info[i] is not None else 0.0
        w_type = ADAPTIVE_BOOSTS["type"] if q_type_info[i] is not None else 0.0
        w_group = ADAPTIVE_BOOSTS["group"] if q_group_info[i] is not None else 0.0

        scores = text_scores[i].copy()
        if w_color > 0:
            scores += w_color * color_sims[q_color_info[i]]
        if w_type > 0:
            scores += w_type * type_sims[q_type_info[i]]
        if w_group > 0:
            scores += w_group * group_sims[q_group_info[i]]

        top_idx = np.argpartition(-scores, top_k)[:top_k]
        top_idx = top_idx[np.argsort(-scores[top_idx])]
        results[qids[i]] = [retriever.article_ids[j] for j in top_idx]

    elapsed = time.time() - t0
    log.info("MoE scoring done: %d queries in %.1f s (%.0f q/s)",
             n_q, elapsed, n_q / elapsed)
    return results


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="MODA Phase 2 — Mixture of Encoders (Superlinked-style) Eval")
    p.add_argument("--n_queries", type=int, default=N_QUERIES)
    p.add_argument("--skip_ce", action="store_true",
                   help="Skip CE reranking comparison")
    p.add_argument("--device", default=None)
    args = p.parse_args()

    t_start = time.time()

    # ── 1. Load data ────────────────────────────────────────────────────────
    queries, qrels = load_benchmark(args.n_queries)
    articles = load_articles()
    qids = [q[0] for q in queries]
    texts = [q[1] for q in queries]

    # ── 2. NER ──────────────────────────────────────────────────────────────
    ner_cache = load_or_compute_ner(queries)

    # ── 3. Build MoE retriever ──────────────────────────────────────────────
    retriever = MixtureOfEncodersRetriever(
        articles, model_name=DENSE_MODEL, device=args.device)

    # ── 4. MoE retrieval (standalone) ───────────────────────────────────────
    moe_results = moe_search_batch(retriever, queries, ner_cache, top_k=TOP_K_RERANK)
    res_moe = evaluate(moe_results, qrels, label="MoE_retrieval")

    # ── 5. BM25 (NER-boosted) for hybrid ────────────────────────────────────
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

    # ── 6. RRF Hybrid: BM25 + MoE ──────────────────────────────────────────
    log.info("Fusing BM25 + MoE with RRF (BM25=%.1f, MoE=%.1f)...",
             BM25_WEIGHT, DENSE_WEIGHT)
    bm25_lists = [bm25_results.get(qid, []) for qid in qids]
    moe_lists = [moe_results.get(qid, []) for qid in qids]
    hybrid_moe_lists = rrf_fusion(bm25_lists, moe_lists)
    hybrid_moe_results = {qid: lst for qid, lst in zip(qids, hybrid_moe_lists)}
    res_hybrid_moe = evaluate(
        hybrid_moe_results, qrels, label="Hybrid_NER_MoE")

    # ── 7. Dense-only baseline for reference ────────────────────────────────
    from benchmark.eval_full_pipeline import dense_search_batch
    dense_results = dense_search_batch(queries, model_name=DENSE_MODEL, top_k=TOP_K_RERANK)
    res_dense = evaluate(dense_results, qrels, label="Dense_only_baseline")

    # ── 7b. Hybrid (BM25 + Dense) baseline for reference ───────────────────
    dense_lists = [dense_results.get(qid, []) for qid in qids]
    hybrid_dense_lists = rrf_fusion(bm25_lists, dense_lists)
    hybrid_dense_results = {qid: lst for qid, lst in zip(qids, hybrid_dense_lists)}
    res_hybrid_dense = evaluate(
        hybrid_dense_results, qrels, label="Hybrid_NER_Dense_baseline")

    all_results = {
        "Dense_only_baseline": res_dense,
        "MoE_retrieval": res_moe,
        "Hybrid_NER_Dense_baseline": res_hybrid_dense,
        "Hybrid_NER_MoE": res_hybrid_moe,
    }

    # ── 8. CE reranking on both hybrid configs ──────────────────────────────
    if not args.skip_ce:
        log.info("CE reranking Hybrid+MoE candidates...")
        ce_moe_lists = ce_rerank_batch(texts, hybrid_moe_lists, articles)
        ce_moe_results = {qid: lst for qid, lst in zip(qids, ce_moe_lists)}
        res_ce_moe = evaluate(
            ce_moe_results, qrels, label="Hybrid_NER_MoE_CE@50")
        all_results["Hybrid_NER_MoE_CE@50"] = res_ce_moe

        log.info("CE reranking Hybrid+Dense candidates (reference)...")
        ce_dense_lists = ce_rerank_batch(texts, hybrid_dense_lists, articles)
        ce_dense_results = {qid: lst for qid, lst in zip(qids, ce_dense_lists)}
        res_ce_dense = evaluate(
            ce_dense_results, qrels, label="Hybrid_NER_Dense_CE@50")
        all_results["Hybrid_NER_Dense_CE@50"] = res_ce_dense

    # ── 9. Save results ────────────────────────────────────────────────────
    output = {
        "configs": all_results,
        "settings": {
            "n_queries": len(queries),
            "random_state": 42,
            "dense_model": DENSE_MODEL,
            "pool_size": TOP_K_RERANK,
            "rerank_top_k": TOP_K_FINAL,
            "bm25_weight": BM25_WEIGHT,
            "dense_weight": DENSE_WEIGHT,
            "adaptive_boosts": ADAPTIVE_BOOSTS,
        },
    }
    with open(MOE_RESULTS_PATH, "w") as f:
        json.dump(output, f, indent=2)
    log.info("Results saved → %s", MOE_RESULTS_PATH)

    # ── 10. Print comparison ────────────────────────────────────────────────
    ref_key = "Hybrid_NER_Dense_CE@50" if "Hybrid_NER_Dense_CE@50" in all_results else "Hybrid_NER_Dense_baseline"
    ref_ndcg = all_results[ref_key]["metrics"]["ndcg@10"]
    ref_label = "CE" if "CE" in ref_key else "HybDense"

    print("\n" + "=" * 90)
    print("MIXTURE OF ENCODERS — Phase 2 Structured Retrieval Comparison")
    print(f"  {len(queries):,} queries  |  Pool: {TOP_K_RERANK} → Top-{TOP_K_FINAL}"
          f"  |  Model: {DENSE_MODEL}")
    print("=" * 90)
    print(f"{'Config':<35} {'nDCG@10':>9} {'MRR':>9} {'R@10':>9}"
          f"  vs {ref_label}")
    print("-" * 90)
    for name, res in all_results.items():
        m = res["metrics"]
        ndcg = m["ndcg@10"]
        delta = (ndcg / ref_ndcg - 1) * 100 if ref_ndcg > 0 else 0
        sign = "+" if delta >= 0 else ""
        icon = "✅" if delta > 0 else ("❌" if delta < -1 else "≈")
        print(
            f"{name:<35} {ndcg:>9.4f} {m['mrr']:>9.4f}"
            f" {m['recall@10']:>9.4f}  {sign}{delta:.1f}% {icon}"
        )
    print("=" * 90)
    print(f"\nTotal elapsed: {(time.time() - t_start) / 60:.1f} min\n")

    return all_results


if __name__ == "__main__":
    main()
