"""
Iteration 2 — Generate synthetic fashion200k-style prose queries.

For each fashion-relevant pair in pairs_50k.jsonl, ask the LLM to write a
descriptive prose search query (5-15 words) that someone might type to find
that product. Each pair generates ONE synthetic query, paired with the
ORIGINAL image. Output is added as additional training pairs.

This directly addresses the v1 failure mode: the model regressed on
fashion200k because only 52 of 50K training records had prose-style queries.
Adding ~10K synthetic prose queries gives the text tower exposure to that
distribution.

Usage:
    PALEBLUEDOT_API_KEY=... python scripts/v5/v2_generate_long_desc.py \\
        --n 10000 --workers 8
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from tqdm import tqdm

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data" / "processed" / "v5_multifield"
DEFAULT_INPUT = DATA / "pairs_50k.jsonl"
DEFAULT_OUTPUT = DATA / "pairs_v2_long_desc.jsonl"

API_URL = "https://open.palebluedot.ai/v1/chat/completions"
MODEL_NAME = "qwen/qwen3.5-flash"
BATCH_SIZE = 25
MAX_TOKENS = 4096
TEMPERATURE = 0.7  # higher than labeling — we want diversity in synthetic queries
MAX_RETRIES = 5
DEFAULT_WORKERS = 8
DEFAULT_TIMEOUT = 240

FASHION_L1 = {"dress", "top", "bottom", "outerwear", "footwear", "bag", "accessory"}


SYSTEM_PROMPT = """You generate fashion product search queries in the style of fashion200k benchmark queries.

You receive product data and return a JSON array of query objects, one per product.

Style:
- 5 to 15 words per query
- DESCRIPTIVE PROSE — what a stylist or fashion blogger would write
- Include attributes: color, material, length, fit, style, occasion, pattern
- Sound like natural human descriptions, not catalog product names
- NO brand names
- NO "the" at the start

Output format — EXACT JSON, no prose, no markdown fences:
[
  {"id": "<pair_id verbatim>", "query": "<5-15 word descriptive query>"}
]

Examples:
Title: "Calvin Klein Women's Sleeveless Wrap Midi Dress in Rich Black"
→ "elegant black sleeveless wrap midi dress for evening occasions"

Title: "ASOS Design Slim Fit Indigo Wash Stretch Jeans"
→ "slim fit indigo wash stretch denim jeans for casual wear"

Title: "Coach Signature Canvas Soho Hobo Bag in Tan"
→ "tan signature canvas hobo handbag with shoulder strap"

Title: "Adidas Originals Ozweego White Mesh Sneakers"
→ "white mesh chunky sole athletic sneakers retro design"

Title: "Free People Floral Print Maxi Dress with Bell Sleeves"
→ "bohemian floral print maxi dress with flowing bell sleeves"

Now generate queries for the products provided. ONE query per product. Maintain the order. Use the EXACT pair_id from input.
"""


# ---------------------------------------------------------------------------
# API client (shared with extractor; reused inline)
# ---------------------------------------------------------------------------

def load_env():
    env = REPO / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def call_llm(messages, api_key, timeout):
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
            return (msg.get("content") or msg.get("reasoning_content") or "").strip()
        except Exception as e:
            last_err = e
            time.sleep(min(2 ** attempt, 30))
    raise RuntimeError(f"LLM failed after {MAX_RETRIES} retries: {last_err}")


def build_prompt(batch):
    lines = ["Generate ONE descriptive prose query per product. Return JSON array.\n"]
    for i, p in enumerate(batch, 1):
        lines.append(f"{i}. id={p['pair_id']}")
        lines.append(f"   title: {p.get('title', '')}")
        lines.append("")
    return "\n".join(lines)


def parse_response(text, expected_ids):
    t = text.strip()
    if t.startswith("```"):
        t = t.split("```", 2)[1]
        if t.startswith("json"):
            t = t[4:]
        t = t.strip()
        if t.endswith("```"):
            t = t[:-3].strip()
    start, end = t.find("["), t.rfind("]")
    if start < 0 or end < 0:
        raise ValueError(f"no JSON array in response: {text[:200]!r}")
    arr = json.loads(t[start:end + 1])
    by_id = {item.get("id"): item for item in arr if isinstance(item, dict)}
    out = []
    for pid in expected_ids:
        item = by_id.get(pid)
        q = (item or {}).get("query", "").strip() if item else ""
        out.append({"id": pid, "query": q, "_dropped": not q})
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--n", type=int, default=10000, help="Number of synthetic queries to generate")
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    ap.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    ap.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    load_env()
    api_key = os.environ.get("PALEBLUEDOT_API_KEY")
    if not api_key:
        sys.exit("PALEBLUEDOT_API_KEY not set")

    # Resume support
    done_ids: set[str] = set()
    if args.output.exists():
        with args.output.open() as f:
            for line in f:
                try:
                    done_ids.add(json.loads(line)["pair_id"])
                except Exception:
                    pass
        print(f"Resume: {len(done_ids):,} already in output")

    # Load + filter
    print(f"Loading {args.input}")
    pairs = []
    with args.input.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            # Skip non-fashion buckets — fashion200k only contains fashion items
            if r.get("category1") in {"non_fashion", "other"}:
                continue
            if not r.get("title"):
                continue
            if r["pair_id"] in done_ids:
                continue
            pairs.append(r)

    rng = random.Random(args.seed)
    rng.shuffle(pairs)
    pairs = pairs[: args.n]
    print(f"Will generate prose queries for {len(pairs):,} pairs")

    n_batches = (len(pairs) + args.batch_size - 1) // args.batch_size
    batches = [pairs[i * args.batch_size : (i + 1) * args.batch_size] for i in range(n_batches)]
    print(f"  {n_batches} batches of size {args.batch_size}, workers={args.workers}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fout = args.output.open("a")
    write_lock = threading.Lock()
    counters = {"ok": 0, "drop": 0, "fail": 0}

    def process_one(idx_batch):
        bi, batch = idx_batch
        prompt = build_prompt(batch)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        try:
            response = call_llm(messages, api_key, args.timeout)
            results = parse_response(response, [p["pair_id"] for p in batch])
        except Exception as e:
            return bi, batch, None, str(e)
        return bi, batch, results, None

    t0 = time.time()
    with tqdm(total=len(pairs), desc="prose-gen") as pbar:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = [ex.submit(process_one, (bi, b)) for bi, b in enumerate(batches)]
            for fut in as_completed(futures):
                bi, batch, results, err = fut.result()
                if err is not None:
                    counters["fail"] += len(batch)
                    pbar.update(len(batch))
                    print(f"  [batch {bi}] FAILED: {err}", file=sys.stderr)
                    continue
                lines = []
                local_ok = local_drop = 0
                by_id = {r["id"]: r for r in results}
                for p in batch:
                    r = by_id.get(p["pair_id"])
                    if r is None or r.get("_dropped"):
                        local_drop += 1
                        continue
                    out = dict(p)
                    out["query_original"] = out.get("query", "")
                    out["query"] = r["query"]
                    out["pair_id"] = p["pair_id"] + "__synth_long"
                    out["score_linear"] = max(70, int(p.get("score_linear", 80)) - 10)
                    out["_synth_source"] = "v2_long_desc"
                    lines.append(json.dumps(out, ensure_ascii=False) + "\n")
                    local_ok += 1
                with write_lock:
                    fout.writelines(lines)
                    fout.flush()
                    counters["ok"] += local_ok
                    counters["drop"] += local_drop
                pbar.update(len(batch))
                pbar.set_postfix(ok=counters["ok"], drop=counters["drop"], fail=counters["fail"])

    fout.close()
    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s")
    print(f"  ok={counters['ok']:,} drop={counters['drop']} fail={counters['fail']}")
    print(f"  output: {args.output}")


if __name__ == "__main__":
    main()
