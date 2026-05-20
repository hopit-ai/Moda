"""
Error analysis on Marqo-FashionSigLIP's fashion200k 10K-corpus retrieval results.

Goal: stop training blind. Look at WHICH queries the teacher fails on, and
WHY, to inform whether/how to fine-tune.

Inputs (already on disk from prior eval runs):
  - retrieved_text-image.json: {query: {doc_id: similarity_score, ...}}
  - ground_truth_text-image.json: {query: {doc_id: relevance, ...}}

Outputs:
  - results/error_analysis/fashion200k_per_query.csv
      one row per query with: rank_of_first_positive, hit@1/10/100, query_length,
      n_words, attribute_count, has_color, has_brand_word, has_material_word
  - results/error_analysis/fashion200k_summary.md
      human-readable summary: bucketed metrics, top-50 worst, top-50 best,
      vocabulary patterns
  - results/error_analysis/fashion200k_worst_queries.html
      gallery of worst queries with gold image vs top-5 retrieved (inline base64)

Comparison: optionally pass --compare <run_prefix> to see the SAME analysis
for another model and identify queries where the comparison model wins/loses.

Usage:
  .venv/bin/python scripts/analyze_fashion200k_failures.py \
      --teacher Marqo-FashionSigLIP_subsample10000 \
      --compare Google-SigLIP2-B16-384_subsample10000

Why the HTML report uses inline base64: fashion200k images come from a HF
dataset; we materialise the images we need (worst-50 + their top-5 retrieved
docs = ~300 images) so the report is self-contained and shareable.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Optional

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("err-analysis")

RESULTS_ROOT = REPO / "repos" / "marqo-FashionCLIP" / "results" / "fashion200k"
GT_PATH = REPO / "repos" / "marqo-FashionCLIP" / "data" / "Fashion200k" / "gt_query_doc" / "ground_truth_text-image.json"
OUT_DIR = REPO / "results" / "error_analysis"

# Fashion vocabulary buckets — used to characterise queries.
# Not exhaustive; designed to surface patterns, not be a perfect classifier.
COLORS = {
    "black", "white", "blue", "red", "green", "yellow", "pink", "purple",
    "orange", "brown", "grey", "gray", "beige", "navy", "khaki", "cream",
    "maroon", "burgundy", "gold", "silver", "tan", "ivory", "olive", "mint",
    "coral", "turquoise", "lavender", "fuchsia", "magenta", "teal",
}
MATERIALS = {
    "cotton", "wool", "leather", "silk", "denim", "polyester", "linen",
    "suede", "velvet", "cashmere", "knit", "lace", "satin", "chiffon",
    "nylon", "spandex", "rayon", "fleece", "tweed", "corduroy", "fur",
}
GARMENT_TYPES = {
    "dress", "shirt", "blouse", "top", "pants", "trousers", "jeans",
    "skirt", "jacket", "coat", "sweater", "hoodie", "t-shirt", "tee",
    "shorts", "blazer", "vest", "cardigan", "jumpsuit", "romper",
    "swimsuit", "bikini", "lingerie", "bra", "underwear", "socks",
    "shoes", "boots", "heels", "sneakers", "sandals", "bag", "purse",
    "scarf", "hat", "cap", "gloves", "belt",
}
PATTERNS = {
    "striped", "stripe", "floral", "polka", "checkered", "plaid",
    "geometric", "paisley", "abstract", "graphic", "solid", "printed",
    "embroidered", "sequined", "ruffled", "lace", "mesh", "sheer",
    "metallic", "ribbed", "quilted",
}
WORD_RE = re.compile(r"[a-zA-Z]+")


def tokenize(s: str) -> list[str]:
    return [w.lower() for w in WORD_RE.findall(s)]


def query_features(q: str) -> dict:
    toks = tokenize(q)
    tok_set = set(toks)
    return {
        "n_chars": len(q),
        "n_words": len(toks),
        "n_unique_words": len(tok_set),
        "n_colors": len(tok_set & COLORS),
        "n_materials": len(tok_set & MATERIALS),
        "n_garments": len(tok_set & GARMENT_TYPES),
        "n_patterns": len(tok_set & PATTERNS),
        "has_color": int(bool(tok_set & COLORS)),
        "has_material": int(bool(tok_set & MATERIALS)),
        "has_garment": int(bool(tok_set & GARMENT_TYPES)),
        "has_pattern": int(bool(tok_set & PATTERNS)),
        "n_sentences": q.count(".") + q.count("!") + q.count("?"),
    }


def per_query_metrics(retrieved: dict, gt: dict, ks=(1, 5, 10, 100)) -> list[dict]:
    """For each query, compute rank of first positive + hit@K."""
    rows: list[dict] = []
    for q, retrieval_scores in retrieved.items():
        if q not in gt:
            continue
        positives = set(gt[q].keys())
        # retrieved is dict {doc_id: score} ALREADY in score-descending order
        # (verified by sniffing the file).
        ranked = list(retrieval_scores.keys())
        first_pos_rank = None
        for i, doc in enumerate(ranked):
            if doc in positives:
                first_pos_rank = i + 1
                break
        row = {
            "query": q,
            "n_positives": len(positives),
            "rank_first_positive": first_pos_rank,  # None if missed
            "missed": int(first_pos_rank is None),
        }
        for k in ks:
            row[f"hit@{k}"] = int(first_pos_rank is not None and first_pos_rank <= k)
        # Reciprocal rank for the first positive (caps at 1/(corpus size))
        row["rr"] = (1.0 / first_pos_rank) if first_pos_rank else 0.0
        # Add query features
        row.update(query_features(q))
        rows.append(row)
    return rows


def buckets_summary(rows: list[dict]) -> str:
    """Markdown summary of metrics bucketed by query features."""
    import statistics as st

    def m(rows, key="hit@10"):
        vals = [r[key] for r in rows]
        return sum(vals) / len(vals) if vals else 0.0

    out = []
    out += [f"\n### Overall ({len(rows)} queries)\n"]
    out += [
        f"- hit@1   = {m(rows,'hit@1'):.4f}",
        f"- hit@10  = {m(rows,'hit@10'):.4f}",
        f"- hit@100 = {m(rows,'hit@100'):.4f}",
        f"- MRR     = {m(rows,'rr'):.4f}",
        f"- median rank of first positive (excluding misses) = "
        f"{st.median([r['rank_first_positive'] for r in rows if r['rank_first_positive']]):.0f}",
        f"- queries that miss top-100 entirely = {sum(r['missed'] for r in rows)} ({100*sum(r['missed'] for r in rows)/len(rows):.1f}%)",
    ]

    # Bucket by n_words
    out += ["\n### By query length (#words)\n",
            "| #words | n queries | hit@1 | hit@10 | hit@100 |",
            "| --- | ---: | ---: | ---: | ---: |"]
    buckets = [(0, 5), (5, 10), (10, 20), (20, 30), (30, 50), (50, 100), (100, 9999)]
    for lo, hi in buckets:
        sub = [r for r in rows if lo <= r["n_words"] < hi]
        if not sub: continue
        out.append(f"| {lo}-{hi if hi<9999 else '∞'} | {len(sub)} | {m(sub,'hit@1'):.4f} | {m(sub,'hit@10'):.4f} | {m(sub,'hit@100'):.4f} |")

    # Bucket by attribute density
    out += ["\n### By attribute richness\n",
            "| has_color | has_material | has_pattern | n queries | hit@10 |",
            "| --- | --- | --- | ---: | ---: |"]
    for c in (0, 1):
        for mat in (0, 1):
            for pat in (0, 1):
                sub = [r for r in rows if r["has_color"]==c and r["has_material"]==mat and r["has_pattern"]==pat]
                if len(sub) < 10: continue
                out.append(f"| {c} | {mat} | {pat} | {len(sub)} | {m(sub,'hit@10'):.4f} |")

    # Bucket by #colors mentioned
    out += ["\n### By #colors mentioned\n",
            "| #colors | n queries | hit@10 |",
            "| --- | ---: | ---: |"]
    for n in range(0, 5):
        sub = [r for r in rows if r["n_colors"] == n]
        if not sub: continue
        out.append(f"| {n} | {len(sub)} | {m(sub,'hit@10'):.4f} |")

    # Bucket by #garment-type words mentioned
    out += ["\n### By #garment-type words mentioned\n",
            "| #garments | n queries | hit@10 |",
            "| --- | ---: | ---: |"]
    for n in range(0, 5):
        sub = [r for r in rows if r["n_garments"] == n]
        if not sub: continue
        out.append(f"| {n} | {len(sub)} | {m(sub,'hit@10'):.4f} |")

    return "\n".join(out)


def vocab_failure_analysis(rows: list[dict], topk_words: int = 30) -> str:
    """Compare word frequencies in failed (rank>10) vs successful (rank<=10) queries."""
    fail_words = Counter()
    succ_words = Counter()
    n_fail = 0; n_succ = 0
    for r in rows:
        toks = set(tokenize(r["query"]))
        if r["hit@10"]:
            n_succ += 1
            for t in toks: succ_words[t] += 1
        else:
            n_fail += 1
            for t in toks: fail_words[t] += 1
    if n_fail == 0 or n_succ == 0:
        return "\n(no failures or no successes — skipping vocab analysis)\n"
    # Word "failure rate": P(query fails | word in query)
    rates = []
    all_words = set(fail_words) | set(succ_words)
    for w in all_words:
        f = fail_words.get(w, 0)
        s = succ_words.get(w, 0)
        total = f + s
        if total < 20: continue   # skip rare words
        rate = f / total
        rates.append((w, rate, total))
    rates.sort(key=lambda x: -x[1])
    out = ["\n### Words most associated with FAILURE (sorted by failure rate, min 20 occurrences)\n",
           "| Word | Failure rate | n queries containing it |",
           "| --- | ---: | ---: |"]
    for w, rate, total in rates[:topk_words]:
        out.append(f"| {w} | {rate:.3f} | {total} |")
    out += ["\n### Words most associated with SUCCESS (lowest failure rate)\n",
            "| Word | Failure rate | n queries containing it |",
            "| --- | ---: | ---: |"]
    for w, rate, total in rates[-topk_words:][::-1]:
        out.append(f"| {w} | {rate:.3f} | {total} |")
    return "\n".join(out)


def comparison_summary(teacher_rows: list[dict], compare_rows: list[dict], compare_label: str) -> str:
    """Compute pairs of (teacher, compare) per query to find divergence."""
    t_by_q = {r["query"]: r for r in teacher_rows}
    c_by_q = {r["query"]: r for r in compare_rows}
    common = set(t_by_q) & set(c_by_q)
    if not common:
        return "\n(no common queries with comparison model)\n"

    # 4-way: teacher_hit@10 × compare_hit@10
    matrix = {(0,0):0,(0,1):0,(1,0):0,(1,1):0}
    for q in common:
        matrix[(t_by_q[q]["hit@10"], c_by_q[q]["hit@10"])] += 1
    n = len(common)
    out = [f"\n### {compare_label} vs Marqo-FashionSigLIP (n={n} common queries)\n",
           "| | compare hit@10 = 0 | compare hit@10 = 1 |",
           "| --- | ---: | ---: |",
           f"| **teacher hit@10 = 0** | both miss: {matrix[(0,0)]} ({100*matrix[(0,0)]/n:.1f}%) | only compare wins: **{matrix[(0,1)]}** ({100*matrix[(0,1)]/n:.1f}%) |",
           f"| **teacher hit@10 = 1** | only teacher wins: **{matrix[(1,0)]}** ({100*matrix[(1,0)]/n:.1f}%) | both hit: {matrix[(1,1)]} ({100*matrix[(1,1)]/n:.1f}%) |"]

    teacher_hits = sum(1 for q in common if t_by_q[q]["hit@10"])
    compare_hits = sum(1 for q in common if c_by_q[q]["hit@10"])
    union_hits = sum(1 for q in common if t_by_q[q]["hit@10"] or c_by_q[q]["hit@10"])
    out += [
        f"\n- teacher hit@10 alone: {teacher_hits/n:.4f}",
        f"- compare hit@10 alone: {compare_hits/n:.4f}",
        f"- **ENSEMBLE upper bound (any model hits)**: {union_hits/n:.4f}",
        f"- ⇒ ensembling could lift hit@10 by **{(union_hits-teacher_hits)/n:+.4f}** absolute over teacher alone "
        f"(if we had a perfect router that picked the right model per query).",
    ]
    return "\n".join(out)


def render_html_gallery(
    teacher_rows: list[dict],
    retrieved: dict,
    gt: dict,
    n_worst: int = 30,
    n_best: int = 10,
) -> str:
    """Build a self-contained HTML report with worst+best queries.

    For each query: gold image(s) + top-5 retrieved images with their rank +
    whether the retrieval matched a positive.
    """
    # Decide which queries (and therefore which doc_ids) we need BEFORE touching
    # the dataset. Indexing all 200k items costs ~5 GB of RAM and is what OOM'd
    # the first attempt. We only need ~300 specific doc_ids.
    sorted_by_rank = sorted(
        teacher_rows,
        key=lambda r: (r["missed"], -1 / max(r["rank_first_positive"] or 1e9, 1)),
        reverse=True,
    )
    worst = sorted_by_rank[:n_worst]
    best = [r for r in teacher_rows if r["rank_first_positive"] == 1][:n_best]
    selected_queries = [r["query"] for r in worst] + [r["query"] for r in best]

    needed_doc_ids: set[str] = set()
    for q in selected_queries:
        for d in list(retrieved.get(q, {}).keys())[:5]:
            needed_doc_ids.add(d)
        for d in gt.get(q, {}).keys():
            needed_doc_ids.add(d)
    log.info("gallery needs %d unique images (worst %d + best %d queries)",
             len(needed_doc_ids), len(worst), len(best))

    log.info("loading fashion200k HF dataset to materialise needed images")
    try:
        from datasets import load_dataset
        ds = load_dataset("Marqo/fashion200k", split="data", cache_dir=str(REPO / "data" / "hf_cache"))
    except Exception as e:
        log.warning("could not load fashion200k dataset (%s) — gallery will skip images", e)
        ds = None

    img_lookup: dict[str, "PIL.Image.Image"] = {}
    if ds is not None:
        log.info("scanning fashion200k for needed item_IDs (streaming, low-memory) ...")
        # Iterate once but only keep images for items we need. Memory cost = ~300 thumbs.
        for i, row in enumerate(ds):
            iid = row.get("item_ID")
            if iid in needed_doc_ids:
                # Thumbnail immediately to keep memory tiny
                try:
                    img = row.get("image")
                    if img is not None:
                        img = img.copy()
                        img.thumbnail((200, 200))
                        img_lookup[iid] = img
                except Exception:
                    pass
                if len(img_lookup) >= len(needed_doc_ids):
                    break
            if i and i % 40000 == 0:
                log.info("  scanned %d, found %d/%d", i, len(img_lookup), len(needed_doc_ids))
        log.info("found %d/%d needed images", len(img_lookup), len(needed_doc_ids))

    def img_to_b64(img, size: int = 160) -> str:
        if img is None:
            return ""
        try:
            img2 = img.copy()
            img2.thumbnail((size, size))
            buf = io.BytesIO()
            img2.save(buf, format="JPEG", quality=70)
            return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
        except Exception:
            return ""

    def render_one(q: str, label: str) -> str:
        gold_ids = list(gt.get(q, {}).keys())
        ranked = list(retrieved.get(q, {}).keys())[:5]
        rank = next((i + 1 for i, d in enumerate(retrieved.get(q, {}).keys()) if d in set(gold_ids)), None)

        gold_html = ""
        for gid in gold_ids[:3]:
            b64 = img_to_b64(img_lookup.get(gid))
            gold_html += f'<div class="thumb"><img src="{b64}" /><div class="cap">gold {gid}</div></div>'
        ret_html = ""
        gold_set = set(gold_ids)
        for i, did in enumerate(ranked):
            cls = "hit" if did in gold_set else "miss"
            b64 = img_to_b64(img_lookup.get(did))
            ret_html += f'<div class="thumb {cls}"><img src="{b64}" /><div class="cap">#{i+1} {did}</div></div>'

        rank_str = f"rank {rank}" if rank else "<b>missed top-100</b>"
        return f"""
<div class="case">
  <div class="meta"><span class="lbl">{label}</span> · {rank_str}</div>
  <div class="query">{q}</div>
  <div class="row"><div class="row-lbl">Gold:</div>{gold_html}</div>
  <div class="row"><div class="row-lbl">Top-5 retrieved:</div>{ret_html}</div>
</div>"""

    body = ""
    body += "<h2>Worst queries (rank of first positive worst-first)</h2>"
    for r in worst:
        body += render_one(r["query"], "WORST")
    body += "<h2>Best queries (perfect rank-1 hits)</h2>"
    for r in best:
        body += render_one(r["query"], "BEST")

    css = """
body{font-family:-apple-system,system-ui,sans-serif;max-width:1100px;margin:20px auto;padding:0 16px;color:#222}
h1,h2{border-bottom:1px solid #ddd;padding-bottom:6px}
.case{border:1px solid #e0e0e0;border-radius:8px;padding:12px;margin:14px 0;background:#fafafa}
.meta{font-size:12px;color:#666}
.lbl{display:inline-block;padding:2px 6px;border-radius:3px;background:#222;color:#fff;font-weight:600;margin-right:6px}
.query{margin:8px 0;font-size:14px;font-style:italic;color:#333}
.row{display:flex;align-items:center;gap:10px;margin-top:8px;flex-wrap:wrap}
.row-lbl{font-size:12px;font-weight:600;color:#444;min-width:120px}
.thumb{display:inline-block;text-align:center}
.thumb img{max-width:140px;max-height:140px;border:2px solid #ccc;border-radius:4px}
.thumb.hit img{border-color:#2c7a2c}
.thumb.miss img{border-color:#999}
.cap{font-size:10px;color:#666;margin-top:2px;max-width:140px;overflow:hidden;white-space:nowrap;text-overflow:ellipsis}
"""
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>fashion200k error analysis</title><style>{css}</style></head><body>
<h1>fashion200k 10K — error analysis</h1>
<p>Green border = retrieved doc is a gold positive. Grey border = miss.</p>
{body}
</body></html>"""


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--teacher", default="Marqo-FashionSigLIP_subsample10000",
                   help="Eval run dir prefix for the model whose failures we analyse")
    p.add_argument("--compare", default=None,
                   help="Optional second model prefix for query-level comparison")
    p.add_argument("--n-worst-html", type=int, default=30)
    p.add_argument("--skip-html", action="store_true",
                   help="Skip HTML gallery (saves ~5min — needed when fashion200k images aren't already cached)")
    args = p.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    def find_run(prefix):
        cands = sorted([d for d in RESULTS_ROOT.iterdir() if d.is_dir() and d.name.startswith(prefix)],
                       key=lambda p: p.stat().st_mtime, reverse=True)
        return cands[0] if cands else None

    teacher_dir = find_run(args.teacher)
    if not teacher_dir:
        log.error("could not find run dir for %s under %s", args.teacher, RESULTS_ROOT)
        sys.exit(2)
    log.info("teacher run dir: %s", teacher_dir)

    with open(teacher_dir / "text-to-image" / "retrieved_text-image.json") as f:
        teacher_retrieved = json.load(f)
    with open(GT_PATH) as f:
        gt = json.load(f)
    log.info("teacher retrievals: %d queries; gt: %d queries", len(teacher_retrieved), len(gt))

    teacher_rows = per_query_metrics(teacher_retrieved, gt)
    log.info("computed per-query metrics for %d teacher queries", len(teacher_rows))

    # CSV out
    import csv
    csv_path = OUT_DIR / "fashion200k_per_query.csv"
    if teacher_rows:
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(teacher_rows[0].keys()))
            w.writeheader()
            w.writerows(teacher_rows)
        log.info("wrote %s (%d rows)", csv_path, len(teacher_rows))

    # Markdown summary
    md = ["# fashion200k error analysis — Marqo-FashionSigLIP",
          "",
          f"Teacher run: `{teacher_dir.name}`",
          f"Queries analysed: {len(teacher_rows)}",
          ""]
    md.append(buckets_summary(teacher_rows))
    md.append(vocab_failure_analysis(teacher_rows))

    if args.compare:
        compare_dir = find_run(args.compare)
        if compare_dir:
            log.info("compare run dir: %s", compare_dir)
            with open(compare_dir / "text-to-image" / "retrieved_text-image.json") as f:
                compare_retrieved = json.load(f)
            compare_rows = per_query_metrics(compare_retrieved, gt)
            md.append(comparison_summary(teacher_rows, compare_rows, args.compare))
        else:
            md.append(f"\n(comparison run not found: {args.compare})")

    md_path = OUT_DIR / "fashion200k_summary.md"
    md_path.write_text("\n".join(md))
    log.info("wrote %s", md_path)

    # HTML gallery
    if not args.skip_html:
        try:
            html = render_html_gallery(teacher_rows, teacher_retrieved, gt, n_worst=args.n_worst_html)
            html_path = OUT_DIR / "fashion200k_worst_queries.html"
            html_path.write_text(html)
            log.info("wrote %s (%.1f MB)", html_path, html_path.stat().st_size / 1e6)
        except Exception as e:
            log.warning("HTML gallery failed: %s — skipping", e)

    log.info("DONE")


if __name__ == "__main__":
    main()
