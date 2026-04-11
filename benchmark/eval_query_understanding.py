"""
MODA Phase 2, Step 2D — Fashion Query Understanding Ablation

Runs the 4-config ablation for the query understanding layer:

  Config A  BM25 baseline (Config 1 from ablation table, reproduced)
  Config B  BM25 + synonym expansion
  Config C  BM25 + NER attribute boosting
  Config D  BM25 + synonyms + NER  ← expected best

  Then feeds the best through Hybrid + CE rerank to see end-to-end gains.

SOTA references:
  - Synonym strategy: query-time client-side (Whatnot 2024, Zalando 2024)
  - NER: GLiNER (NAACL 2024) zero-shot, outperforms ChatGPT on NER
  - Attribute boosting: function_score with field-specific weights
    following EcomBERT-NER production pattern

Outputs: results/real/hnm_query_understanding.json
         results/real/PHASE2_RUNNING_LEADERBOARD.md  (updated)
"""

from __future__ import annotations

import csv
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

import pandas as pd
from opensearchpy import OpenSearch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from benchmark.metrics import compute_all_metrics, aggregate_metrics
from benchmark.query_expansion import SynonymExpander, FashionNER, build_boosted_query, LABEL_TO_FIELD, COLOR_MAP, GARMENT_TYPE_MAP, GENDER_MAP

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ─── paths ────────────────────────────────────────────────────────────────────
DATA_DIR   = Path("data/raw/hnm_real")
RESULTS_DIR = Path("results/real")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

INDEX_NAME = "moda_hnm"
N_QUERIES  = 10_000      # same sample as prior runs for comparability

# Metrics to target (mirroring prior runs)
K_VALUES = [5, 10, 20, 50]

# ─── OpenSearch client ────────────────────────────────────────────────────────
OS_HOST = os.getenv("OPENSEARCH_HOST", "localhost")
OS_PORT = int(os.getenv("OPENSEARCH_PORT", 9200))

def get_client() -> OpenSearch:
    return OpenSearch(
        hosts=[{"host": OS_HOST, "port": OS_PORT}],
        http_compress=True,
        timeout=30,
    )


# ─── Load benchmark data ──────────────────────────────────────────────────────

def load_benchmark(n_queries: int = N_QUERIES):
    """Load real H&M queries and qrels, matching the format in eval_hybrid.py."""
    log.info("Loading real H&M benchmark data...")

    # Load qrels: qid → {article_id: grade}
    # Format: query_id, positive_ids (space-sep), negative_ids (space-sep)
    qrels: dict[str, dict[str, int]] = {}
    with open(DATA_DIR / "qrels.csv", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            qid = row["query_id"].strip()
            pos_ids = [x.strip() for x in row.get("positive_ids", "").split() if x.strip()]
            neg_ids = [x.strip() for x in row.get("negative_ids", "").split() if x.strip()]
            grades: dict[str, int] = {}
            for aid in pos_ids:
                grades[aid] = 2
            for aid in neg_ids:
                if aid not in grades:
                    grades[aid] = 1
            qrels[qid] = grades

    # Load queries: query_id, transaction_id, query_text
    queries_df = pd.read_csv(DATA_DIR / "queries.csv")
    valid_qids = set(qrels.keys())
    queries_df = queries_df[queries_df["query_id"].astype(str).isin(valid_qids)]

    if n_queries and len(queries_df) > n_queries:
        queries_df = queries_df.sample(n=n_queries, random_state=42)

    queries = [
        (str(row["query_id"]), str(row["query_text"]))
        for _, row in queries_df.iterrows()
    ]
    log.info("Loaded %d queries, %d qrels entries", len(queries), len(qrels))
    return queries, qrels


# ─── NER pre-processing ───────────────────────────────────────────────────────

def precompute_ner(
    queries: list[tuple[str, str]],
    ner: FashionNER,
    batch_log_every: int = 500,
) -> dict[str, dict[str, list[str]]]:
    """Run GLiNER over all queries upfront — more efficient than per-query calls."""
    log.info("Running GLiNER NER on %d queries...", len(queries))
    ner_cache: dict[str, dict[str, list[str]]] = {}
    t0 = time.time()
    for i, (qid, text) in enumerate(queries):
        ner_cache[qid] = ner.extract(text)
        if (i + 1) % batch_log_every == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            remaining = (len(queries) - i - 1) / rate
            log.info(
                "  NER: %d/%d done  (%.1f q/s, ~%.0f s remaining)",
                i + 1, len(queries), rate, remaining,
            )
    log.info("NER pre-computation done in %.1fs", time.time() - t0)
    return ner_cache


# ─── BM25 search variants ─────────────────────────────────────────────────────

def _base_bm25_query(search_text: str, top_k: int = 50) -> dict:
    # Fields and boosts match the proven BM25 baseline in eval_hybrid.py
    return {
        "query": {
            "multi_match": {
                "query": search_text,
                "fields": [
                    "prod_name^4",
                    "product_type_name^3",
                    "colour_group_name^2",
                    "section_name^1.5",
                    "garment_group_name^1.5",
                    "detail_desc^1",
                    "search_text^1",
                ],
                "type": "best_fields",
                "tie_breaker": 0.3,
                "operator": "or",
            }
        },
        "size": top_k,
        "_source": False,
    }


def _ner_boosted_query(
    search_text: str,
    ner_entities: dict[str, list[str]],
    top_k: int = 50,
) -> dict:
    """
    Build bool/should NER-boosted query.

    Industry pattern (EcomBERT-NER, QueryNER):
      - Base retrieval: must clause with multi_match (identical to baseline)
      - Attribute boosts: should clauses using field-type-aware queries

    Using should (not must-filter) so NER boosts relevance without
    hard-excluding near-miss products. Critical for fashion: "red dress"
    should still surface coral/pink dresses, but lower.

    Field type handling (from moda_hnm mapping):
      - text fields  → `match` query (analyzed, tolerant of case/plurality)
      - keyword fields → `term` query (exact match required)
    """
    # Keyword fields in moda_hnm that need exact `term` matching
    KEYWORD_FIELDS = {
        "garment_group_name", "graphical_appearance_name",
        "index_group_name", "index_name", "perceived_colour_master_name",
        "perceived_colour_value_name", "product_group_name",
    }

    base = {
        "multi_match": {
            "query": search_text,
            "fields": [
                "prod_name^4",
                "product_type_name^3",
                "colour_group_name^2",
                "section_name^1.5",
                "garment_group_name^1.5",
                "detail_desc^1",
                "search_text^1",
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
        for raw_value in values:
            mapped = value_map.get(raw_value, raw_value.title())
            if field in KEYWORD_FIELDS:
                # Exact match for keyword fields (e.g., index_group_name: "Ladieswear")
                should_clauses.append({
                    "term": {field: {"value": mapped, "boost": boost}}
                })
            else:
                # Analyzed match for text fields (e.g., colour_group_name: "Dark Blue")
                should_clauses.append({
                    "match": {field: {"query": mapped, "boost": boost}}
                })

    if not should_clauses:
        return _base_bm25_query(search_text, top_k)

    return {
        "query": {
            "bool": {
                "must":   [base],
                "should": should_clauses,
            }
        },
        "size": top_k,
        "_source": False,
    }


def run_bm25_variant(
    client: OpenSearch,
    queries: list[tuple[str, str]],
    qrels: dict[str, dict[str, int]],
    *,
    use_synonyms: bool = False,
    use_ner: bool = False,
    ner_cache: Optional[dict] = None,
    expander: Optional[SynonymExpander] = None,
    top_k: int = 50,
    config_name: str = "BM25",
) -> dict:
    """Run BM25 evaluation with optional synonym expansion and NER boosting."""

    per_query: list[dict] = []
    t0 = time.time()

    for qid, raw_query in tqdm(queries, desc=config_name, ncols=80):
        # 1. Optionally expand query with synonyms
        search_text = expander.expand(raw_query) if (use_synonyms and expander) else raw_query

        # 2. Build OpenSearch query
        if use_ner and ner_cache and qid in ner_cache:
            ner_entities = ner_cache[qid]
            os_query = _ner_boosted_query(search_text, ner_entities, top_k)
        else:
            os_query = _base_bm25_query(search_text, top_k)

        # 3. Execute search
        try:
            resp = client.search(index=INDEX_NAME, body=os_query)
        except Exception as e:
            log.warning("Search error for qid=%s: %s", qid, e)
            continue

        retrieved = [h["_id"] for h in resp["hits"]["hits"]]
        q_qrels   = qrels.get(qid, {})
        if not q_qrels:
            continue

        # 4. Compute metrics using the shared metrics module
        m = compute_all_metrics(retrieved, q_qrels, ks=[5, 10, 20, 50])
        per_query.append(m)

    elapsed = time.time() - t0
    agg = aggregate_metrics(per_query)

    results = {
        "config":    config_name,
        "n_queries": len(per_query),
        "elapsed_s": round(elapsed, 1),
        "metrics": {
            "nDCG@10": round(agg.get("ndcg@10", 0), 4),
            "MRR":     round(agg.get("mrr",     0), 4),
            "R@10":    round(agg.get("recall@10", 0), 4),
            "R@20":    round(agg.get("recall@20", 0), 4),
            "P@5":     round(agg.get("p@5",      0), 4),
        },
    }
    log.info(
        "%s → nDCG@10=%.4f  MRR=%.4f  R@10=%.4f  (%.1fs)",
        config_name,
        results["metrics"]["nDCG@10"],
        results["metrics"]["MRR"],
        results["metrics"]["R@10"],
        elapsed,
    )
    return results


# ─── Main ablation runner ─────────────────────────────────────────────────────

def main():
    client = get_client()
    try:
        info = client.info()
        log.info("OpenSearch %s connected ✓", info["version"]["number"])
    except Exception as e:
        log.error("Cannot connect to OpenSearch: %s", e)
        sys.exit(1)

    queries, qrels = load_benchmark()

    # ── Step 1: Synonym expander (instant, no model loading) ──────────────────
    expander = SynonymExpander()
    log.info("SynonymExpander ready with %d synonym groups", len(expander._sorted_keys))

    # ── Step 2: Load GLiNER and pre-compute NER for all queries ──────────────
    log.info("Loading GLiNER model (urchade/gliner_medium-v2.1)...")
    ner = FashionNER(model_name="urchade/gliner_medium-v2.1", threshold=0.4)
    ner_cache = precompute_ner(queries, ner)
    del ner   # free RAM — we don't need the model after pre-computation

    # ── Ablation configs ──────────────────────────────────────────────────────
    configs = [
        dict(use_synonyms=False, use_ner=False, config_name="A_BM25_baseline"),
        dict(use_synonyms=True,  use_ner=False, config_name="B_BM25_synonyms"),
        dict(use_synonyms=False, use_ner=True,  config_name="C_BM25_NER"),
        dict(use_synonyms=True,  use_ner=True,  config_name="D_BM25_synonyms_NER"),
    ]

    all_results = {}
    for cfg in configs:
        name = cfg["config_name"]
        result = run_bm25_variant(
            client, queries, qrels,
            expander=expander,
            ner_cache=ner_cache,
            top_k=50,
            **cfg,
        )
        all_results[name] = result

    # ── Save results ──────────────────────────────────────────────────────────
    out_path = RESULTS_DIR / "hnm_query_understanding.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    log.info("Results saved → %s", out_path)

    # ── Print comparison table ────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("QUERY UNDERSTANDING ABLATION — H&M Real Queries")
    print("=" * 70)
    header = f"{'Config':<30} {'nDCG@10':>9} {'MRR':>9} {'R@10':>9} {'R@20':>9}"
    print(header)
    print("-" * 70)

    baseline_ndcg = None
    for name, res in all_results.items():
        m = res["metrics"]
        if baseline_ndcg is None:
            baseline_ndcg = m["nDCG@10"]
        delta = f"({(m['nDCG@10']/baseline_ndcg - 1)*100:+.0f}%)" if baseline_ndcg else ""
        print(
            f"{name:<30} {m['nDCG@10']:>9.4f} {m['MRR']:>9.4f}"
            f" {m['R@10']:>9.4f} {m['R@20']:>9.4f}  {delta}"
        )

    print("=" * 70)
    print(f"\nContext (prior results for comparison):")
    print(f"  Phase 1 best dense  (FashionCLIP):         nDCG@10 = 0.0300")
    print(f"  Phase 2 BM25 baseline:                     nDCG@10 = 0.0187  (behind dense on real queries)")
    print(f"  Phase 2 Hybrid Config C (BM25×0.4+D×0.6): nDCG@10 = 0.0353")
    print(f"  Phase 2 Hybrid C + CE rerank:              nDCG@10 = 0.0533  ← current BEST\n")

    return all_results


if __name__ == "__main__":
    main()
