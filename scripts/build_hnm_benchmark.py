"""
MODA Phase 1 — Build H&M Benchmark from Articles Dataset

Constructs queries.csv and qrels.csv from the H&M articles catalog
using category-based relevance — a clean, reproducible approach for
Phase 1 (no transaction data required).

Methodology:
  - Query text: prod_name of a sampled article (simulates user searching
    for a product by name)
  - Positive IDs (grade=2): all articles with the same product_type_name
    (exact category match = "purchased equivalent")
  - Negative IDs (grade=1): articles with the same garment_group_name but
    different product_type_name (viewed similar category)

This creates a realistic retrieval task where:
  - Models must find the right product type within a broader category
  - Naive category-level retrieval would score ~P@1=0.3 (random baseline)
  - Good text encoders should score much higher

Phase 2 will extend this with actual purchase data from transactions.

Usage:
  python scripts/build_hnm_benchmark.py
  python scripts/build_hnm_benchmark.py --n_queries 5000 --seed 42
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw" / "hnm"
LOG_FILE = BASE_DIR / "logs" / "build_hnm_benchmark.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE),
    ],
)
log = logging.getLogger(__name__)


def load_articles_from_hf(cache_dir: Path) -> pd.DataFrame:
    """Load H&M articles from local HF cache."""
    from datasets import load_dataset
    log.info("Loading articles from HuggingFace cache …")
    ds = load_dataset(
        "microsoft/hnm-search-data",
        "articles",
        cache_dir=str(cache_dir),
    )
    df = ds["train"].to_pandas()
    df["article_id"] = df["article_id"].astype(str)
    log.info("Loaded %d articles.", len(df))
    return df


def build_benchmark(df: pd.DataFrame, n_queries: int, seed: int, max_pos: int = 20, max_neg: int = 20) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build queries.csv and qrels.csv from articles DataFrame.

    Uses vectorized pandas operations (no iterrows) for speed.

    Strategy:
    - Sample n_queries articles as query "seeds" (stratified by product_type_name)
    - Query text: prod_name
    - Positives (grade=2): up to max_pos articles with same product_type_name
    - Negatives (grade=1): up to max_neg articles with same garment_group_name
      but different product_type_name

    Returns:
        (queries_df, qrels_df)
    """
    rng = np.random.default_rng(seed)
    df = df.copy()
    df["article_id"] = df["article_id"].astype(str)

    # Filter to product types with enough peers
    type_counts = df["product_type_name"].value_counts()
    valid_types = type_counts[type_counts >= 5].index
    df_valid = df[df["product_type_name"].isin(valid_types)].reset_index(drop=True)
    log.info("Articles with ≥5 peers in product_type: %d / %d", len(df_valid), len(df))

    # Build lookup tables using groupby (vectorized, fast)
    log.info("Building type/group lookup tables …")
    type_to_ids: dict[str, list[str]] = df_valid.groupby("product_type_name")["article_id"].apply(list).to_dict()
    group_to_type_ids: dict[tuple, list[str]] = (
        df_valid.groupby(["garment_group_name", "product_type_name"])["article_id"]
        .apply(list).to_dict()
    )
    group_to_ids: dict[str, list[str]] = df_valid.groupby("garment_group_name")["article_id"].apply(list).to_dict()

    # Sample query seeds — stratified by product_type_name
    log.info("Sampling query seeds …")
    n_per_type = max(1, n_queries // len(valid_types))
    sampled_parts = []
    for ptype in valid_types:
        group = df_valid[df_valid["product_type_name"] == ptype]
        n_sample = min(n_per_type, len(group))
        sampled_parts.append(group.sample(n=n_sample, random_state=int(rng.integers(1 << 31))))

    query_seeds = pd.concat(sampled_parts, ignore_index=True).sample(frac=1, random_state=seed)
    if len(query_seeds) > n_queries:
        query_seeds = query_seeds.iloc[:n_queries]
    log.info("Query seeds sampled: %d", len(query_seeds))

    # Build queries and qrels as lists (fast, no iterrows)
    seeds_records = query_seeds[["article_id", "prod_name", "product_type_name", "garment_group_name"]].to_dict("records")

    queries_rows = []
    qrels_rows = []

    for qid_idx, seed_row in enumerate(seeds_records):
        aid = seed_row["article_id"]
        ptype = seed_row["product_type_name"]
        gname = seed_row["garment_group_name"]

        # Positives: same product_type, capped at max_pos, excluding seed
        all_pos = [a for a in type_to_ids.get(ptype, []) if a != aid]
        if not all_pos:
            continue
        # Sample max_pos from positives
        if len(all_pos) > max_pos:
            all_pos = list(rng.choice(all_pos, size=max_pos, replace=False))

        # Negatives: same garment_group but different product_type, capped at max_neg
        pos_set = set(all_pos) | {aid}
        neg_candidates = [a for a in group_to_ids.get(gname, []) if a not in pos_set]
        if len(neg_candidates) > max_neg:
            neg_candidates = list(rng.choice(neg_candidates, size=max_neg, replace=False))

        queries_rows.append({
            "query_id": f"q{qid_idx:06d}",
            "query_text": str(seed_row["prod_name"]).strip(),
            "seed_article_id": aid,
            "product_type_name": ptype,
        })
        qrels_rows.append({
            "query_id": f"q{qid_idx:06d}",
            "positive_ids": " ".join(all_pos),
            "negative_ids": " ".join(neg_candidates),
        })

    return pd.DataFrame(queries_rows), pd.DataFrame(qrels_rows)


def save_articles_csv(df: pd.DataFrame, output_dir: Path) -> Path:
    """Save articles.csv to the HNM directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "articles.csv"
    df.to_csv(out_path, index=False)
    log.info("Saved articles.csv: %d rows → %s", len(df), out_path)
    return out_path


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build H&M benchmark from articles dataset.")
    p.add_argument("--n_queries", type=int, default=5000,
                   help="Number of queries to generate (default: 5000)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", type=Path, default=RAW_DIR,
                   help="Where to save articles.csv, queries.csv, qrels.csv")
    p.add_argument("--cache_dir", type=Path,
                   default=BASE_DIR / "data" / "raw" / "hnm",
                   help="HuggingFace cache directory for H&M dataset")
    return p.parse_args()


def main():
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    log.info("=" * 60)
    log.info("Building H&M Benchmark (Phase 1 — category-based relevance)")
    log.info("n_queries=%d, seed=%d", args.n_queries, args.seed)
    log.info("=" * 60)

    # Load articles
    df = load_articles_from_hf(args.cache_dir)

    # Save articles.csv
    save_articles_csv(df, args.output_dir)

    # Build queries and qrels
    queries_df, qrels_df = build_benchmark(df, n_queries=args.n_queries, seed=args.seed)

    # Save
    q_path = args.output_dir / "queries.csv"
    qr_path = args.output_dir / "qrels.csv"
    queries_df.to_csv(q_path, index=False)
    qrels_df.to_csv(qr_path, index=False)

    log.info("Saved queries.csv: %d rows → %s", len(queries_df), q_path)
    log.info("Saved qrels.csv:   %d rows → %s", len(qrels_df), qr_path)

    # Stats
    avg_pos = qrels_df["positive_ids"].apply(lambda x: len(x.split())).mean()
    avg_neg = qrels_df["negative_ids"].apply(lambda x: len(x.split()) if x else 0).mean()
    n_types = queries_df["product_type_name"].nunique()

    log.info("")
    log.info("=== Benchmark Stats ===")
    log.info("  Queries              : %d", len(queries_df))
    log.info("  Unique product types : %d", n_types)
    log.info("  Avg positive IDs     : %.1f", avg_pos)
    log.info("  Avg negative IDs     : %.1f", avg_neg)
    log.info("  Total articles       : %d", len(df))

    # Save metadata
    meta = {
        "n_queries": len(queries_df),
        "n_articles": len(df),
        "n_product_types": int(n_types),
        "avg_positives": float(avg_pos),
        "avg_negatives": float(avg_neg),
        "methodology": "category-based: positives=same product_type_name, negatives=same garment_group_name",
        "seed": args.seed,
        "phase": 1,
        "note": "Phase 2 will extend with purchase-based relevance from transaction data",
    }
    meta_path = args.output_dir / "benchmark_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    log.info("Metadata saved to %s", meta_path)
    log.info("H&M benchmark build complete.")


if __name__ == "__main__":
    main()
