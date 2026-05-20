"""
Phase A.2 — Extract structured multi-field labels from existing pairs.

Input:  pairs.jsonl with at least {query, title, pair_id} per record.
Output: pairs_labeled.jsonl with the same fields plus 8 new structured fields:
        category_l1, category_l2, color, material, pattern, fit_style, gender, occasion

The LLM is given the (query, title) text plus any pre-existing partial labels
(category1, colors, materials, brand from v4 mining) as context — its job is to
ENRICH and FILL GAPS, not redo what regex already did.

Resumes on failure: skips pair_ids already present in the output file.

Usage:
    PALEBLUEDOT_API_KEY=... python scripts/v5/phase_a_extract_multifield.py \\
        --input data/processed/v4_pattern_targeted/pairs.jsonl \\
        --output data/processed/v5_multifield/pairs_labeled.jsonl \\
        --limit 200            # for validation run; omit to do all

    # Stratified sampling for the 200-pair validation:
    python scripts/v5/phase_a_extract_multifield.py \\
        --input data/processed/v4_pattern_targeted/pairs.jsonl \\
        --output data/processed/v5_multifield/validation_200.jsonl \\
        --limit 200 --stratify
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from tqdm import tqdm

API_URL = "https://open.palebluedot.ai/v1/chat/completions"
MODEL_NAME = "qwen/qwen3.5-flash"
BATCH_SIZE = 25            # pairs per API call — tuned for qwen3.5-flash context
MAX_TOKENS = 8192
TEMPERATURE = 0.0          # deterministic
MAX_RETRIES = 5
DEFAULT_WORKERS = 4        # concurrent API calls
DEFAULT_TIMEOUT = 240      # per-request timeout (s); bigger batches need more

# 9-class L1 taxonomy — must match what we'll use for stratified eval
ALLOWED_L1 = {
    "dress", "top", "bottom", "outerwear", "footwear",
    "bag", "accessory", "beauty", "home_lifestyle",
}
ALLOWED_GENDER = {"women", "men", "unisex", "kids", "unknown"}


# ---------------------------------------------------------------------------
# API client
# ---------------------------------------------------------------------------

def load_env():
    """Load .env if present so the API key is available."""
    env = Path(__file__).resolve().parents[2] / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def call_llm(messages: list[dict], api_key: str, timeout: int = DEFAULT_TIMEOUT) -> str:
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
    }
    last_err = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.post(API_URL, headers=headers, json=payload, timeout=timeout)
            resp.raise_for_status()
            msg = resp.json()["choices"][0]["message"]
            text = msg.get("content") or msg.get("reasoning_content") or ""
            return text.strip()
        except Exception as e:
            last_err = e
            wait = min(2 ** attempt, 30)
            print(f"  [retry {attempt + 1}/{MAX_RETRIES}] {e} — waiting {wait}s",
                  file=sys.stderr)
            time.sleep(wait)
    raise RuntimeError(f"LLM call failed after {MAX_RETRIES} retries: {last_err}")


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You label fashion and lifestyle products with structured fields. You receive product data and return a JSON array of label objects, one per product, in the SAME order as input.

Output format — EXACT JSON, no prose, no markdown fences:
[
  {
    "id": "<pair_id verbatim from input>",
    "category_l1": "<one of: dress, top, bottom, outerwear, footwear, bag, accessory, beauty, home_lifestyle>",
    "category_l2": "<specific type: e.g. mini-dress, crew-neck-tee, slim-fit-jeans, parka, ankle-boot, crossbody-bag, earrings, lipstick, throw-pillow>",
    "color": "<primary color in plain English: 'navy blue', 'cream', 'burgundy', 'multicolor'. 'unknown' if absent>",
    "material": "<primary material: cotton, denim, leather, wool, silk, polyester, etc. 'unknown' if absent>",
    "pattern": "<one of: solid, striped, floral, plaid, polka-dot, geometric, animal-print, abstract, color-block, graphic, unknown>",
    "fit_style": "<one of: slim, regular, relaxed, oversized, fitted, cropped, flowy, tailored, unknown>",
    "gender": "<one of: women, men, unisex, kids, unknown>",
    "occasion": "<one of: casual, formal, athletic, evening, beach, business, lounge, outdoor, unknown>"
  }
]

Rules:
- Use ONLY the values listed above for the constrained fields (category_l1, pattern, fit_style, gender, occasion). If unsure, use "unknown".
- color and material are free-form lowercase strings.
- For non-fashion items (home/beauty/tech), use category_l1="home_lifestyle" or "beauty"; gender="unisex"; occasion="unknown" is fine.
- Do not invent fields. Do not include any text outside the JSON array.
- The "id" must be copied EXACTLY from the input; do not modify it.
- The QUERY represents user search intent. When the query and title seem to disagree, prefer the query. Example: query="vest women", title="Brown Pillow Down Vest" → this is a vest (outerwear), not a pillow. The word "pillow" describes the down filling, not the product type.
- The regex_hints are noisy and incomplete — use them as a hint only, do not blindly copy. If your reading of (query, title) disagrees with a hint, trust your reading.
"""


def build_user_prompt(batch: list[dict]) -> str:
    """Build the user message for one batch of pairs."""
    lines = ["Label each of the following products. Return JSON array only.\n"]
    for i, rec in enumerate(batch, 1):
        # Pass existing partial labels as context — LLM should enrich/correct, not ignore
        ctx_parts = []
        if rec.get("category1"):
            ctx_parts.append(f"hint_category={rec['category1']}")
        if rec.get("category2") and rec["category2"] != "general":
            ctx_parts.append(f"hint_subcategory={rec['category2']}")
        if rec.get("colors"):
            ctx_parts.append(f"hint_colors={','.join(rec['colors'])}")
        if rec.get("materials"):
            ctx_parts.append(f"hint_materials={','.join(rec['materials'])}")
        if rec.get("brand"):
            ctx_parts.append(f"hint_brand={rec['brand']}")
        ctx = " | ".join(ctx_parts) if ctx_parts else "—"

        lines.append(f"{i}. id={rec['pair_id']}")
        lines.append(f"   query: {rec.get('query', '')}")
        lines.append(f"   title: {rec.get('title', '')}")
        lines.append(f"   regex_hints: {ctx}")
        lines.append("")
    return "\n".join(lines)


def parse_response(text: str, expected_ids: list[str]) -> list[dict]:
    """Parse LLM response, returning list of label dicts. Best-effort."""
    # Strip markdown fences if model added them
    t = text.strip()
    if t.startswith("```"):
        t = t.split("```", 2)[1]
        if t.startswith("json"):
            t = t[4:]
        t = t.strip()
        if t.endswith("```"):
            t = t[:-3].strip()
    # Locate the JSON array
    start = t.find("[")
    end = t.rfind("]")
    if start < 0 or end < 0:
        raise ValueError(f"No JSON array found in response: {text[:200]!r}")
    arr = json.loads(t[start:end + 1])
    if not isinstance(arr, list):
        raise ValueError(f"Response is not a JSON array: {type(arr)}")

    # Validate and normalize each entry
    cleaned = []
    by_id = {item.get("id"): item for item in arr if isinstance(item, dict)}
    for pid in expected_ids:
        item = by_id.get(pid)
        if not item:
            # Model dropped one — emit a stub flagged unknown
            cleaned.append({
                "id": pid, "category_l1": "unknown", "category_l2": "unknown",
                "color": "unknown", "material": "unknown", "pattern": "unknown",
                "fit_style": "unknown", "gender": "unknown", "occasion": "unknown",
                "_llm_dropped": True,
            })
            continue
        # Coerce constrained fields
        c1 = str(item.get("category_l1", "unknown")).lower().strip()
        if c1 not in ALLOWED_L1 and c1 != "unknown":
            c1 = "unknown"
        g = str(item.get("gender", "unknown")).lower().strip()
        if g not in ALLOWED_GENDER:
            g = "unknown"
        cleaned.append({
            "id": pid,
            "category_l1": c1,
            "category_l2": str(item.get("category_l2", "unknown")).lower().strip(),
            "color": str(item.get("color", "unknown")).lower().strip(),
            "material": str(item.get("material", "unknown")).lower().strip(),
            "pattern": str(item.get("pattern", "unknown")).lower().strip(),
            "fit_style": str(item.get("fit_style", "unknown")).lower().strip(),
            "gender": g,
            "occasion": str(item.get("occasion", "unknown")).lower().strip(),
        })
    return cleaned


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def stratified_sample(records: list[dict], n: int, seed: int = 1337) -> list[dict]:
    """Sample n records balanced across category1 buckets."""
    by_bucket = defaultdict(list)
    for r in records:
        by_bucket[r.get("category1", "unknown")].append(r)
    rng = random.Random(seed)
    out = []
    buckets = list(by_bucket.keys())
    per_bucket = max(1, n // max(1, len(buckets)))
    for b in buckets:
        rng.shuffle(by_bucket[b])
        out.extend(by_bucket[b][:per_bucket])
    rng.shuffle(out)
    return out[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path,
                    help="Path to pairs.jsonl (with at minimum query, title, pair_id)")
    ap.add_argument("--output", required=True, type=Path,
                    help="Path to write pairs_labeled.jsonl")
    ap.add_argument("--limit", type=int, default=0,
                    help="Process only this many records (0 = all)")
    ap.add_argument("--stratify", action="store_true",
                    help="Stratified sample by category1 (use for validation runs)")
    ap.add_argument("--batch_size", type=int, default=BATCH_SIZE,
                    help="Pairs per API call (default 40)")
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                    help="Concurrent API calls (default 4)")
    ap.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT,
                    help="Per-request timeout in seconds (default 240)")
    args = ap.parse_args()

    load_env()
    api_key = os.environ.get("PALEBLUEDOT_API_KEY")
    if not api_key:
        sys.exit("ERROR: PALEBLUEDOT_API_KEY not set (check .env)")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Resume support: load already-processed ids
    done_ids: set[str] = set()
    if args.output.exists():
        with args.output.open() as f:
            for line in f:
                try:
                    done_ids.add(json.loads(line)["pair_id"])
                except Exception:
                    pass
        print(f"Resume: {len(done_ids):,} ids already in {args.output}")

    # Load input
    print(f"Loading input from {args.input} ...")
    records = []
    with args.input.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec["pair_id"] in done_ids:
                continue
            records.append(rec)
    print(f"  {len(records):,} records remaining to label")

    if args.stratify:
        records = stratified_sample(records, args.limit or len(records))
        print(f"  Stratified sample: {len(records)} records")
    elif args.limit:
        records = records[: args.limit]

    if not records:
        print("Nothing to do.")
        return

    # Build batches
    n_batches = (len(records) + args.batch_size - 1) // args.batch_size
    batches = [
        records[bi * args.batch_size : (bi + 1) * args.batch_size]
        for bi in range(n_batches)
    ]
    print(f"Processing {len(records):,} records in {n_batches} batches "
          f"of size {args.batch_size}, with {args.workers} concurrent workers")

    LABEL_KEYS = ("category_l1", "category_l2", "color", "material",
                  "pattern", "fit_style", "gender", "occasion")

    write_lock = threading.Lock()
    counters = {"ok": 0, "drop": 0, "fail": 0}
    fout = args.output.open("a")
    t_start = time.time()

    def process_one(batch_idx_pair):
        bi, batch = batch_idx_pair
        user_msg = build_user_prompt(batch)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]
        try:
            response = call_llm(messages, api_key, timeout=args.timeout)
            labels = parse_response(response, [r["pair_id"] for r in batch])
        except Exception as e:
            return bi, batch, None, str(e)
        return bi, batch, labels, None

    try:
        with tqdm(total=len(records), desc="labeling") as pbar:
            with ThreadPoolExecutor(max_workers=args.workers) as ex:
                futures = [ex.submit(process_one, (bi, b)) for bi, b in enumerate(batches)]
                for fut in as_completed(futures):
                    bi, batch, labels, err = fut.result()
                    if err is not None:
                        counters["fail"] += len(batch)
                        print(f"  [batch {bi}] FAILED: {err}", file=sys.stderr)
                        pbar.update(len(batch))
                        continue
                    lbl_by_id = {l["id"]: l for l in labels}
                    out_lines = []
                    local_ok = local_drop = 0
                    for rec in batch:
                        lbl = lbl_by_id.get(rec["pair_id"])
                        if lbl is None:
                            local_drop += 1
                            continue
                        if lbl.get("_llm_dropped"):
                            local_drop += 1
                        out = dict(rec)
                        for k in LABEL_KEYS:
                            out[k] = lbl[k]
                        out_lines.append(json.dumps(out, ensure_ascii=False) + "\n")
                        local_ok += 1
                    with write_lock:
                        fout.writelines(out_lines)
                        fout.flush()
                        counters["ok"] += local_ok
                        counters["drop"] += local_drop
                    pbar.update(len(batch))
                    pbar.set_postfix(
                        ok=counters["ok"],
                        drop=counters["drop"],
                        fail=counters["fail"],
                    )
    finally:
        fout.close()

    elapsed = time.time() - t_start
    rate = counters["ok"] / elapsed if elapsed > 0 else 0
    print(f"\nDone in {elapsed:.1f}s ({rate:.2f} pairs/sec)")
    print(f"  ok={counters['ok']}, dropped={counters['drop']}, failed={counters['fail']}")
    print(f"  output: {args.output}")


if __name__ == "__main__":
    main()
