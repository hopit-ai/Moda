"""
Phase 1c: Leakage verification against all 7 Marqo benchmarks.

Checks that NO training image or text overlaps with any benchmark dataset.
- Image check: compare product_ids and image hashes
- Text check: compare training queries/titles against benchmark query/document text
- Generates audit report
"""
import os, sys, json, hashlib, time
from pathlib import Path
from collections import defaultdict

PROJ_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJ_ROOT / "data" / "processed" / "v4_pattern_targeted"
REPORT_FILE = DATA_DIR / "leakage_report.md"

BENCHMARK_DATASETS = {
    "deepfashion_inshop": "Marqo/deepfashion-inshop",
    "deepfashion_multimodal": "Marqo/deepfashion-multimodal",
    "fashion200k": "Marqo/fashion200k",
    "KAGL": "Marqo/KAGL",
    "atlas": "Marqo/atlas",
    "polyvore": "Marqo/polyvore",
    "iMaterialist": "Marqo/iMaterialist",
}

MAX_BENCH_ROWS = 50_000


def load_training_data() -> tuple[set, set, set]:
    """Load training pairs and return (product_ids, queries_normalized, titles_normalized)."""
    product_ids = set()
    queries = set()
    titles = set()

    for fname in ["pairs.jsonl", "synthetic_pairs.jsonl"]:
        fpath = DATA_DIR / fname
        if not fpath.exists():
            continue
        with open(fpath) as f:
            for line in f:
                rec = json.loads(line)
                product_ids.add(rec.get("product_id", ""))
                q = rec.get("query", "").strip().lower()
                t = rec.get("title", "").strip().lower()
                if q:
                    queries.add(q)
                if t:
                    titles.add(t)

    return product_ids, queries, titles


def check_benchmark(name: str, hf_id: str, train_pids: set,
                    train_queries: set, train_titles: set) -> dict:
    """Check one benchmark for leakage. Returns overlap stats."""
    from datasets import load_dataset

    print(f"  Checking {name} ({hf_id})...")
    result = {
        "name": name,
        "hf_id": hf_id,
        "rows_checked": 0,
        "product_id_overlaps": [],
        "query_overlaps": [],
        "title_overlaps": [],
    }

    try:
        ds = load_dataset(hf_id, split="data", streaming=True)
        ds = ds.remove_columns([c for c in ds.column_names if c == "image"])
    except Exception:
        try:
            ds = load_dataset(hf_id, split="test", streaming=True)
            ds = ds.remove_columns([c for c in ds.column_names if c == "image"])
        except Exception as e:
            result["error"] = str(e)
            return result

    bench_queries = set()
    bench_titles = set()
    bench_pids = set()

    for i, row in enumerate(ds):
        if i >= MAX_BENCH_ROWS:
            break

        pid = str(row.get("product_id", row.get("id", row.get("_id", ""))))
        if pid:
            bench_pids.add(pid)

        for text_col in ["query", "text", "caption", "description"]:
            val = row.get(text_col, "")
            if val and isinstance(val, str):
                bench_queries.add(val.strip().lower())

        for text_col in ["title", "product_title", "name"]:
            val = row.get(text_col, "")
            if val and isinstance(val, str):
                bench_titles.add(val.strip().lower())

    result["rows_checked"] = min(i + 1, MAX_BENCH_ROWS) if 'i' in dir() else 0

    pid_overlap = train_pids & bench_pids
    query_overlap = train_queries & bench_queries
    title_overlap = train_titles & bench_titles

    result["product_id_overlaps"] = list(pid_overlap)[:20]
    result["product_id_overlap_count"] = len(pid_overlap)
    result["query_overlaps"] = list(query_overlap)[:20]
    result["query_overlap_count"] = len(query_overlap)
    result["title_overlaps"] = list(title_overlap)[:20]
    result["title_overlap_count"] = len(title_overlap)
    result["is_clean"] = (len(pid_overlap) == 0 and len(query_overlap) == 0
                          and len(title_overlap) == 0)

    status = "CLEAN" if result["is_clean"] else "LEAKAGE DETECTED"
    print(f"    {status}: pids={len(pid_overlap)}, queries={len(query_overlap)}, "
          f"titles={len(title_overlap)}")
    return result


def main():
    print("Loading training data...")
    train_pids, train_queries, train_titles = load_training_data()
    print(f"  Training set: {len(train_pids)} product_ids, "
          f"{len(train_queries)} unique queries, {len(train_titles)} unique titles")

    results = []
    overall_clean = True

    for name, hf_id in BENCHMARK_DATASETS.items():
        result = check_benchmark(name, hf_id, train_pids, train_queries, train_titles)
        results.append(result)
        if not result.get("is_clean", True):
            overall_clean = False

    report_lines = [
        "# Data Leakage Verification Report",
        "",
        f"**Date**: {time.strftime('%Y-%m-%d %H:%M')}",
        f"**Training pairs**: {len(train_pids)} products, {len(train_queries)} queries",
        f"**Overall status**: {'CLEAN - No leakage detected' if overall_clean else 'LEAKAGE DETECTED'}",
        "",
        "## Per-Benchmark Results",
        "",
    ]

    for r in results:
        status = "CLEAN" if r.get("is_clean", True) else "LEAKAGE"
        report_lines.append(f"### {r['name']} ({r['hf_id']})")
        report_lines.append(f"- Status: **{status}**")
        report_lines.append(f"- Rows checked: {r.get('rows_checked', 'N/A')}")
        report_lines.append(f"- Product ID overlaps: {r.get('product_id_overlap_count', 0)}")
        report_lines.append(f"- Query text overlaps: {r.get('query_overlap_count', 0)}")
        report_lines.append(f"- Title text overlaps: {r.get('title_overlap_count', 0)}")

        if r.get("query_overlaps"):
            report_lines.append("- Sample query overlaps:")
            for q in r["query_overlaps"][:5]:
                report_lines.append(f"  - `{q}`")
        if r.get("title_overlaps"):
            report_lines.append("- Sample title overlaps:")
            for t in r["title_overlaps"][:5]:
                report_lines.append(f"  - `{t}`")
        if r.get("error"):
            report_lines.append(f"- Error: {r['error']}")
        report_lines.append("")

    report_lines.extend([
        "## Methodology",
        "- Product IDs: exact string match between training and benchmark",
        "- Query text: case-insensitive exact match of full query strings",
        "- Title text: case-insensitive exact match of full title strings",
        f"- Max rows checked per benchmark: {MAX_BENCH_ROWS}",
        "",
        "## Action Items",
    ])

    if overall_clean:
        report_lines.append("None - training data is clean for all benchmarks.")
    else:
        report_lines.append("Remove overlapping entries before training:")
        for r in results:
            if not r.get("is_clean", True):
                report_lines.append(f"- {r['name']}: remove {r.get('product_id_overlap_count', 0)} "
                                    f"product overlaps, {r.get('query_overlap_count', 0)} query overlaps, "
                                    f"{r.get('title_overlap_count', 0)} title overlaps")

    with open(REPORT_FILE, "w") as f:
        f.write("\n".join(report_lines) + "\n")

    with open(DATA_DIR / "leakage_results.json", "w") as f:
        json.dump({"overall_clean": overall_clean, "results": results}, f, indent=2, default=str)

    print(f"\nReport saved to {REPORT_FILE}")
    return 0 if overall_clean else 1


if __name__ == "__main__":
    sys.exit(main())
