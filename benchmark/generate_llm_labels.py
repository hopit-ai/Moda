"""
MODA Phase 3B — LLM-Judged Relevance Labels

Generates graded relevance labels (0-3) for query-product pairs using
GPT-4o-mini via PaleblueDot API. Replaces noisy binary purchase labels
with clean semantic relevance judgments.

Output: data/processed/llm_relevance_labels.jsonl
Each line: {"query_id": str, "article_id": str, "query_text": str,
            "product_text": str, "score": int, "reason": str}

Usage:
  export PALEBLUEDOT_API_KEY=your_key
  python benchmark/generate_llm_labels.py                    # full run (~200K pairs)
  python benchmark/generate_llm_labels.py --max_queries 500  # smoke test
  python benchmark/generate_llm_labels.py --audit            # GPT-4o quality audit
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
import os
import random
import time
from pathlib import Path

import pandas as pd
from openai import AsyncOpenAI

from benchmark.article_text import build_article_texts_from_df as build_article_texts

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).parent.parent
HNM_DIR = _REPO_ROOT / "data" / "raw" / "hnm_real"
PROCESSED_DIR = _REPO_ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

LABELS_PATH = PROCESSED_DIR / "llm_relevance_labels.jsonl"
AUDIT_PATH = PROCESSED_DIR / "llm_audit_labels.jsonl"
SPLIT_PATH = PROCESSED_DIR / "query_splits.json"

BASE_URL = "https://open.palebluedot.ai/v1"
BULK_MODEL = "anthropic/claude-sonnet-4.6"
AUDIT_MODEL = "anthropic/claude-sonnet-4.6"

RANDOM_SEED = 42
HARD_NEGS_PER_QUERY = 15
RANDOM_NEGS_PER_QUERY = 5
MAX_CONCURRENT = 15
MAX_RETRIES = 3

RELEVANCE_PROMPT = """\
You are a fashion search relevance judge. Given a user's search query and a product description, rate how relevant the product is to the query.

Rating scale:
3 = Exact match — product clearly matches ALL aspects of the query (garment type, color, style, gender, fit)
2 = Good match — correct garment category with minor attribute differences (e.g. "dark blue" when query says "navy", or "slim" vs "regular" fit)
1 = Partial match — same broad category but significant mismatches (e.g. query asks for "dress" but product is a "skirt", or correct type but wrong gender)
0 = Not relevant — completely different product category or purpose

Important color mappings in fashion: navy = dark blue, coral = light orange, burgundy = dark red, cream = off-white, charcoal = dark grey. Treat these as equivalent.

Examples:
- Query "navy slim fit jeans mens", Product "Slim Fit Stretch Jeans | Trousers | Dark Blue | Menswear | detail: five-pocket slim jeans in stretch denim" → score 3 (exact: slim jeans, dark blue=navy, menswear)
- Query "red summer dress", Product "Jersey Dress | Dress | Red | Ladieswear | detail: short sleeveless dress in cotton jersey" → score 3 (exact: red dress, summer-appropriate)
- Query "black leather jacket women", Product "Biker Jacket | Jacket | Black | Ladieswear | detail: biker jacket in imitation leather with zip" → score 2 (good: right type/color/gender but imitation leather, not real)
- Query "blue denim shorts", Product "Regular Fit Jeans | Trousers | Denim Blue | Menswear | detail: five-pocket jeans in washed denim" → score 1 (partial: denim but trousers not shorts)
- Query "floral maxi dress", Product "Knitted Jumper | Jumper | Light Pink | Ladieswear | detail: fine-knit jumper in cotton" → score 0 (not relevant: jumper vs dress)

Think step by step: identify what the query wants (garment type, color, style, gender), then compare against the product attributes.

Query: "{query}"
Product: "{product}"

Respond ONLY with JSON: {{"reasoning": "<your step-by-step analysis in 20 words>", "score": <0-3>}}"""


# ─── Data Loading ────────────────────────────────────────────────────────────


def load_train_pairs(
    article_texts: dict[str, str],
    max_queries: int | None = None,
    seed: int = RANDOM_SEED,
) -> list[dict]:
    """Load query-product pairs from training split for LLM labeling."""
    rng = random.Random(seed)

    splits = json.loads(SPLIT_PATH.read_text())
    train_qids = set(splits["train"])
    log.info("Train split: %d query IDs", len(train_qids))

    queries: dict[str, str] = {}
    with open(HNM_DIR / "queries.csv", newline="") as f:
        for row in csv.DictReader(f):
            queries[row["query_id"].strip()] = row["query_text"].strip()

    all_article_ids = list(article_texts.keys())

    qrels_by_qid: list[dict] = []
    with open(HNM_DIR / "qrels.csv", newline="") as f:
        for row in csv.DictReader(f):
            qid = row["query_id"].strip()
            if qid in train_qids and qid in queries:
                qrels_by_qid.append(row)

    rng.shuffle(qrels_by_qid)
    if max_queries:
        qrels_by_qid = qrels_by_qid[:max_queries]

    pairs = []
    for row in qrels_by_qid:
        qid = row["query_id"].strip()
        query_text = queries[qid]
        pos_ids = [x.strip() for x in row.get("positive_ids", "").split() if x.strip()]
        neg_ids = [x.strip() for x in row.get("negative_ids", "").split() if x.strip()]

        for pid in pos_ids:
            if pid in article_texts:
                pairs.append({
                    "query_id": qid,
                    "article_id": pid,
                    "query_text": query_text,
                    "product_text": article_texts[pid],
                    "source": "positive",
                })

        hard_sample = rng.sample(neg_ids, min(HARD_NEGS_PER_QUERY, len(neg_ids)))
        for nid in hard_sample:
            if nid in article_texts:
                pairs.append({
                    "query_id": qid,
                    "article_id": nid,
                    "query_text": query_text,
                    "product_text": article_texts[nid],
                    "source": "hard_negative",
                })

        for _ in range(RANDOM_NEGS_PER_QUERY):
            rand_id = rng.choice(all_article_ids)
            if rand_id not in pos_ids:
                pairs.append({
                    "query_id": qid,
                    "article_id": rand_id,
                    "query_text": query_text,
                    "product_text": article_texts[rand_id],
                    "source": "random_negative",
                })

    log.info("Total pairs to label: %d", len(pairs))
    return pairs


def load_checkpoint(output_path: Path) -> set[str]:
    """Load already-labeled pair keys for checkpoint resumption."""
    done = set()
    if output_path.exists():
        with open(output_path) as f:
            for line in f:
                if line.strip():
                    obj = json.loads(line)
                    done.add(f"{obj['query_id']}_{obj['article_id']}")
    log.info("Checkpoint: %d pairs already labeled", len(done))
    return done


# ─── LLM Labeling ───────────────────────────────────────────────────────────

def _extract_json(text: str) -> dict:
    """Extract JSON from LLM response, handling markdown fences and surrounding text."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    start = text.find("{")
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        text = text[start:end]
    return json.loads(text)


async def label_one(
    client: AsyncOpenAI,
    pair: dict,
    model: str,
    semaphore: asyncio.Semaphore,
) -> dict | None:
    """Call LLM for a single query-product relevance judgment."""
    prompt = RELEVANCE_PROMPT.format(
        query=pair["query_text"],
        product=pair["product_text"],
    )

    use_json_mode = any(k in model for k in ("gpt-4o", "gpt-5"))
    is_reasoning = any(k in model for k in ("gpt-5-mini", "gpt-5.4", "o1", "o3", "o4"))

    for attempt in range(MAX_RETRIES):
        try:
            max_tok = 2048 if is_reasoning else 300
            kwargs: dict = dict(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tok,
            )
            if not is_reasoning:
                kwargs["temperature"] = 0.0
            if use_json_mode:
                kwargs["response_format"] = {"type": "json_object"}

            async with semaphore:
                resp = await client.chat.completions.create(**kwargs)

            content = resp.choices[0].message.content
            if content is None:
                raise ValueError(f"Empty response (finish={resp.choices[0].finish_reason})")
            obj = _extract_json(content)
            score = int(obj.get("score", -1))
            if score < 0 or score > 3:
                raise ValueError(f"Invalid score: {score}")

            return {
                "query_id": pair["query_id"],
                "article_id": pair["article_id"],
                "query_text": pair["query_text"],
                "product_text": pair["product_text"],
                "score": score,
                "reason": str(obj.get("reasoning", obj.get("reason", "")))[:150],
                "source": pair["source"],
                "model": model,
            }
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(2 ** attempt)
            else:
                log.warning("Failed after %d retries: %s — %s",
                            MAX_RETRIES, pair["article_id"], e)
                return None


async def label_batch(
    pairs: list[dict],
    output_path: Path,
    model: str = BULK_MODEL,
    concurrency: int = MAX_CONCURRENT,
):
    """Label all pairs with concurrent LLM calls, writing results incrementally."""
    api_key = os.environ.get("PALEBLUEDOT_API_KEY", "")
    if not api_key:
        raise ValueError(
            "Set PALEBLUEDOT_API_KEY environment variable. "
            "Get it from https://palebluedot.ai"
        )

    client = AsyncOpenAI(base_url=BASE_URL, api_key=api_key)
    semaphore = asyncio.Semaphore(concurrency)

    done_keys = load_checkpoint(output_path)
    todo = [p for p in pairs if f"{p['query_id']}_{p['article_id']}" not in done_keys]
    log.info("To label: %d pairs (skipping %d already done)", len(todo), len(done_keys))

    if not todo:
        log.info("All pairs already labeled!")
        return

    t0 = time.time()
    completed = 0
    failed = 0

    CHUNK_SIZE = 200
    for chunk_start in range(0, len(todo), CHUNK_SIZE):
        chunk = todo[chunk_start:chunk_start + CHUNK_SIZE]
        tasks = [label_one(client, p, model, semaphore) for p in chunk]
        results = await asyncio.gather(*tasks)

        with open(output_path, "a") as f:
            for r in results:
                if r is not None:
                    f.write(json.dumps(r) + "\n")
                    completed += 1
                else:
                    failed += 1

        total_done = len(done_keys) + completed
        elapsed = time.time() - t0
        rate = completed / elapsed if elapsed > 0 else 0
        remaining = (len(todo) - completed - failed) / rate if rate > 0 else 0
        log.info(
            "  Progress: %d/%d done (%.1f/s, ~%.0fs remaining, %d failed)",
            total_done, len(pairs), rate, remaining, failed,
        )

    elapsed = time.time() - t0
    log.info(
        "Labeling complete: %d labeled, %d failed in %.1f min (%.1f pairs/s)",
        completed, failed, elapsed / 60, completed / elapsed,
    )


# ─── Quality Audit ──────────────────────────────────────────────────────────

def run_audit(n_sample: int = 2000):
    """Re-label a sample with GPT-4o and compute agreement metrics."""
    if not LABELS_PATH.exists():
        raise FileNotFoundError(f"Run labeling first: {LABELS_PATH}")

    labels = []
    with open(LABELS_PATH) as f:
        for line in f:
            if line.strip():
                labels.append(json.loads(line))

    rng = random.Random(RANDOM_SEED)
    sample = rng.sample(labels, min(n_sample, len(labels)))
    log.info("Audit: re-labeling %d pairs with %s", len(sample), AUDIT_MODEL)

    audit_pairs = []
    for item in sample:
        audit_pairs.append({
            "query_id": item["query_id"],
            "article_id": item["article_id"],
            "query_text": item["query_text"],
            "product_text": item["product_text"],
            "source": item["source"],
            "mini_score": item["score"],
        })

    asyncio.run(label_batch(audit_pairs, AUDIT_PATH, model=AUDIT_MODEL, concurrency=10))

    mini_scores = {}
    for item in sample:
        mini_scores[f"{item['query_id']}_{item['article_id']}"] = item["score"]

    audit_scores = {}
    with open(AUDIT_PATH) as f:
        for line in f:
            if line.strip():
                obj = json.loads(line)
                audit_scores[f"{obj['query_id']}_{obj['article_id']}"] = obj["score"]

    exact = 0
    within_1 = 0
    total = 0
    diffs = []
    for key, mini_s in mini_scores.items():
        if key in audit_scores:
            audit_s = audit_scores[key]
            total += 1
            diff = abs(mini_s - audit_s)
            diffs.append(diff)
            if diff == 0:
                exact += 1
            if diff <= 1:
                within_1 += 1

    if total > 0:
        log.info("=" * 50)
        log.info("AUDIT RESULTS (%d pairs compared)", total)
        log.info("  Exact agreement:    %d/%d (%.1f%%)", exact, total, 100 * exact / total)
        log.info("  Within-1 agreement: %d/%d (%.1f%%)", within_1, total, 100 * within_1 / total)
        log.info("  Mean absolute diff: %.2f", sum(diffs) / len(diffs))
        log.info("=" * 50)
    else:
        log.warning("No matching pairs found for audit comparison")


# ─── Score Distribution Report ───────────────────────────────────────────────

def report_distribution():
    """Print score distribution from the labels file."""
    if not LABELS_PATH.exists():
        print(f"No labels file: {LABELS_PATH}")
        return

    counts = {0: 0, 1: 0, 2: 0, 3: 0}
    source_counts = {}
    total = 0
    with open(LABELS_PATH) as f:
        for line in f:
            if line.strip():
                obj = json.loads(line)
                s = obj["score"]
                counts[s] = counts.get(s, 0) + 1
                src = obj.get("source", "unknown")
                if src not in source_counts:
                    source_counts[src] = {0: 0, 1: 0, 2: 0, 3: 0}
                source_counts[src][s] += 1
                total += 1

    print(f"\nTotal labels: {total:,}")
    print(f"\nScore distribution:")
    for s in range(4):
        pct = 100 * counts[s] / total if total else 0
        bar = "#" * int(pct / 2)
        print(f"  {s}: {counts[s]:>6,}  ({pct:5.1f}%)  {bar}")

    print(f"\nBy source:")
    for src, sc in sorted(source_counts.items()):
        src_total = sum(sc.values())
        avg = sum(s * c for s, c in sc.items()) / src_total if src_total else 0
        print(f"  {src:20s}: {src_total:>6,} pairs, avg score={avg:.2f}")


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate LLM relevance labels")
    parser.add_argument("--max_queries", type=int, default=None,
                        help="Cap number of queries to label (default: all train)")
    parser.add_argument("--concurrency", type=int, default=MAX_CONCURRENT,
                        help="Max concurrent API calls (default: 20)")
    parser.add_argument("--audit", action="store_true",
                        help="Run GPT-4o quality audit on existing labels")
    parser.add_argument("--report", action="store_true",
                        help="Print score distribution report")
    parser.add_argument("--model", type=str, default=BULK_MODEL,
                        help=f"Model to use (default: {BULK_MODEL})")
    args = parser.parse_args()

    if args.report:
        report_distribution()
        return

    if args.audit:
        run_audit()
        return

    log.info("=" * 60)
    log.info("MODA Phase 3B — LLM Relevance Label Generation")
    log.info("Model: %s via PaleblueDot API", args.model)
    log.info("=" * 60)

    log.info("Loading H&M articles...")
    articles_df = pd.read_csv(HNM_DIR / "articles.csv", dtype=str).fillna("")
    article_texts = build_article_texts(articles_df)
    log.info("Articles loaded: %d", len(article_texts))

    pairs = load_train_pairs(article_texts, max_queries=args.max_queries)

    asyncio.run(label_batch(
        pairs, LABELS_PATH,
        model=args.model,
        concurrency=args.concurrency,
    ))

    report_distribution()


if __name__ == "__main__":
    main()
