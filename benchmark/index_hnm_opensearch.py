"""
MODA Phase 2 — Step 1: Index H&M Articles into OpenSearch for BM25

Creates index 'moda_hnm' with:
- Proper text analysis (English analyzer + synonym support)
- All article metadata fields for BM25 retrieval
- article_id as document _id (matches our FAISS benchmark)

Usage:
  python benchmark/index_hnm_opensearch.py
  python benchmark/index_hnm_opensearch.py --host localhost --port 9200 --recreate
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import pandas as pd
from opensearchpy import OpenSearch, helpers

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).parent.parent
ARTICLES_CSV = REPO_ROOT / "data" / "raw" / "hnm" / "articles.csv"
INDEX_NAME = "moda_hnm"

INDEX_SETTINGS = {
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0,
        "analysis": {
            "analyzer": {
                "hnm_text": {
                    "type": "custom",
                    "tokenizer": "standard",
                    "filter": ["lowercase", "asciifolding", "english_stop", "english_stemmer"]
                }
            },
            "filter": {
                "english_stop": {"type": "stop", "stopwords": "_english_"},
                "english_stemmer": {"type": "stemmer", "language": "english"}
            }
        }
    },
    "mappings": {
        "properties": {
            "article_id":          {"type": "keyword"},
            "prod_name":           {"type": "text", "analyzer": "hnm_text", "boost": 3.0},
            "product_type_name":   {"type": "text", "analyzer": "hnm_text", "boost": 2.0,
                                    "fields": {"keyword": {"type": "keyword"}}},
            "product_group_name":  {"type": "keyword"},
            "colour_group_name":   {"type": "text", "analyzer": "hnm_text",
                                    "fields": {"keyword": {"type": "keyword"}}},
            "perceived_colour_value_name": {"type": "keyword"},
            "perceived_colour_master_name": {"type": "keyword"},
            "department_name":     {"type": "text", "analyzer": "hnm_text",
                                    "fields": {"keyword": {"type": "keyword"}}},
            "index_name":          {"type": "keyword"},
            "index_group_name":    {"type": "keyword"},
            "section_name":        {"type": "text", "analyzer": "hnm_text",
                                    "fields": {"keyword": {"type": "keyword"}}},
            "garment_group_name":  {"type": "keyword"},
            "graphical_appearance_name": {"type": "keyword"},
            "detail_desc":         {"type": "text", "analyzer": "hnm_text"},
            # Composite search field — main BM25 target
            "search_text":         {"type": "text", "analyzer": "hnm_text", "boost": 1.5},
        }
    }
}


def build_search_text(row: pd.Series) -> str:
    """Combine key fields into a single searchable text blob."""
    parts = [
        str(row.get("prod_name", "") or ""),
        str(row.get("product_type_name", "") or ""),
        str(row.get("colour_group_name", "") or ""),
        str(row.get("perceived_colour_master_name", "") or ""),
        str(row.get("department_name", "") or ""),
        str(row.get("section_name", "") or ""),
        str(row.get("garment_group_name", "") or ""),
        str(row.get("detail_desc", "") or ""),
    ]
    return " ".join(p for p in parts if p and p != "nan")


def generate_docs(df: pd.DataFrame):
    for _, row in df.iterrows():
        doc = {k: (None if pd.isna(v) else v) for k, v in row.items()}
        doc["search_text"] = build_search_text(row)
        yield {
            "_index": INDEX_NAME,
            "_id": str(row["article_id"]),
            "_source": doc,
        }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=9200)
    p.add_argument("--recreate", action="store_true", help="Delete and recreate index")
    p.add_argument("--batch_size", type=int, default=500)
    args = p.parse_args()

    client = OpenSearch(
        hosts=[{"host": args.host, "port": args.port}],
        http_compress=True,
        use_ssl=False,
        verify_certs=False,
    )

    # Check connection
    info = client.info()
    log.info("Connected to OpenSearch %s", info["version"]["number"])

    # Handle index
    if client.indices.exists(index=INDEX_NAME):
        if args.recreate:
            log.info("Deleting existing index '%s'...", INDEX_NAME)
            client.indices.delete(index=INDEX_NAME)
        else:
            count = client.count(index=INDEX_NAME)["count"]
            log.info("Index '%s' already exists with %d docs.", INDEX_NAME, count)
            if count > 100000:
                log.info("Index looks complete. Use --recreate to rebuild.")
                return
            log.info("Index incomplete — will re-index.")
            client.indices.delete(index=INDEX_NAME)

    log.info("Creating index '%s'...", INDEX_NAME)
    client.indices.create(index=INDEX_NAME, body=INDEX_SETTINGS)

    # Load articles
    log.info("Loading articles from %s...", ARTICLES_CSV)
    df = pd.read_csv(ARTICLES_CSV, dtype=str)
    log.info("Loaded %d articles with columns: %s", len(df), list(df.columns))

    # Bulk index
    log.info("Bulk indexing %d articles...", len(df))
    t0 = time.time()
    success, failed = 0, 0

    for ok, info in helpers.parallel_bulk(
        client,
        generate_docs(df),
        chunk_size=args.batch_size,
        thread_count=4,
        raise_on_error=False,
    ):
        if ok:
            success += 1
        else:
            failed += 1
            if failed <= 3:
                log.warning("Failed doc: %s", info)

        if (success + failed) % 10000 == 0:
            elapsed = time.time() - t0
            rate = (success + failed) / elapsed
            log.info("  Progress: %d/%d docs (%.0f docs/sec)", success + failed, len(df), rate)

    elapsed = time.time() - t0
    log.info("Indexing complete: %d success, %d failed in %.1fs", success, failed, elapsed)

    # Refresh and verify
    client.indices.refresh(index=INDEX_NAME)
    count = client.count(index=INDEX_NAME)["count"]
    log.info("Index '%s' now has %d documents.", INDEX_NAME, count)

    # Quick BM25 test
    log.info("Testing BM25 search...")
    result = client.search(index=INDEX_NAME, body={
        "query": {"multi_match": {
            "query": "black dress",
            "fields": ["prod_name^3", "product_type_name^2", "colour_group_name", "search_text"]
        }},
        "size": 3
    })
    log.info("Test query 'black dress' → top results:")
    for hit in result["hits"]["hits"]:
        src = hit["_source"]
        log.info("  [%.3f] %s — %s %s",
                 hit["_score"],
                 src.get("article_id"),
                 src.get("prod_name"),
                 src.get("colour_group_name", ""))

    log.info("Step 1 complete. OpenSearch BM25 index 'moda_hnm' is ready.")


if __name__ == "__main__":
    main()
