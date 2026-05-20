"""
Fast data prep using only already-downloaded data.
Sources (no network required):
  1. data/processed/fashion_stratified_gs_df/  — 91K short queries
  2. data/processed/v4_pattern_targeted/        — 118K + 37K (short + medium)

Produces data/train_v1/pairs_combined.jsonl + data/manifest.json.
No strict stratification — we use all valid pairs, which gives ~247K with
~84% short / ~16% medium distribution. This is acceptable for bootstrapping.

Usage:
    python train/prep_data_v1.py
    python train/prep_data_v1.py --no_leakage   # skip pHash (faster, less safe)
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from tqdm import tqdm

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "data" / "train_v1"


def _nwords(s: str) -> int:
    return len(s.split())


def _bucket(s: str) -> str:
    n = _nwords(s)
    if n <= 8:   return "short"
    if n <= 25:  return "medium"
    return "long"


# ──────────────────────────────────────────────────────────────────────────────
# Source 1: fashion_stratified_gs_df
# ──────────────────────────────────────────────────────────────────────────────
def load_gs_df() -> list[dict]:
    src = REPO / "data" / "processed" / "fashion_stratified_gs_df" / "pairs.jsonl"
    if not src.exists():
        print("  [SKIP] fashion_stratified_gs_df not found")
        return []

    records = []
    for line in tqdm(src.open(), desc="gs_df", unit=" pairs"):
        r = json.loads(line)
        # stored as relative path from repo root
        img_path = REPO / r["image_path"]
        if not img_path.exists():
            continue
        records.append({
            "pair_id":          r.get("pair_id", f"gsdf_{len(records)}"),
            "query":            r["query"].strip(),
            "title":            r.get("title", r["query"])[:200],
            "long_description": "",
            "score":            float(r.get("score_linear", r.get("weight", 50)) * (1 if r.get("score_linear") else 100)),
            "source":           "gs10m_gsdf",
            "image_path":       str(img_path),
            "n_words":          _nwords(r["query"]),
        })
    print(f"  gs_df: {len(records):,} valid pairs")
    return records


# ──────────────────────────────────────────────────────────────────────────────
# Source 2: v4 pattern-targeted pairs (GS-10M bucketed)
# ──────────────────────────────────────────────────────────────────────────────
def load_v4_pairs() -> list[dict]:
    v4_dir  = REPO / "data" / "processed" / "v4_pattern_targeted"
    img_dir = v4_dir / "images"
    src = v4_dir / "pairs.jsonl"
    if not src.exists():
        print("  [SKIP] v4_pattern_targeted/pairs.jsonl not found")
        return []

    records = []
    for line in tqdm(src.open(), desc="v4_pairs", unit=" pairs"):
        r = json.loads(line)
        img_path = img_dir / r["image_file"]
        if not img_path.exists():
            continue
        score = float(r.get("score_linear", 70))
        records.append({
            "pair_id":          r["pair_id"],
            "query":            r["query"].strip(),
            "title":            r.get("title", r["query"])[:200],
            "long_description": "",
            "score":            score,
            "source":           "v4_gs10m",
            "image_path":       str(img_path),
            "n_words":          _nwords(r["query"]),
        })
    print(f"  v4_pairs: {len(records):,} valid pairs")
    return records


# ──────────────────────────────────────────────────────────────────────────────
# Source 3: v4 synthetic (template + LLM captions, medium queries)
# ──────────────────────────────────────────────────────────────────────────────
def load_v4_synthetic() -> list[dict]:
    v4_dir  = REPO / "data" / "processed" / "v4_pattern_targeted"
    img_dir = v4_dir / "images"
    src = v4_dir / "synthetic_pairs.jsonl"
    if not src.exists():
        print("  [SKIP] v4_pattern_targeted/synthetic_pairs.jsonl not found")
        return []

    records = []
    for line in tqdm(src.open(), desc="v4_synth", unit=" pairs"):
        r = json.loads(line)
        img_path = img_dir / r["image_file"]
        if not img_path.exists():
            continue
        query = r["query"].strip()
        nw = _nwords(query)
        records.append({
            "pair_id":          r["pair_id"],
            "query":            query,
            "title":            r.get("title", query)[:200],
            "long_description": query if nw > 15 else "",
            "score":            float(r.get("score_linear", 80)),
            "source":           "v4_synthetic",
            "image_path":       str(img_path),
            "n_words":          nw,
        })
    print(f"  v4_synthetic: {len(records):,} valid pairs")
    return records


# ──────────────────────────────────────────────────────────────────────────────
# Leakage check (pHash)
# ──────────────────────────────────────────────────────────────────────────────
def run_leakage_check(records: list[dict]) -> list[dict]:
    try:
        from imagededup.methods import PHash
    except ImportError:
        print("  [WARN] imagededup not installed — skipping pHash leakage check")
        return records

    phasher = PHash()
    eval_hashes: set[str] = set()

    # Load cached eval hashes if available
    cache_dir = REPO / "data" / "processed" / "leakage_hashes"
    if cache_dir.exists() and any(cache_dir.glob("*.json")):
        for f in cache_dir.glob("*.json"):
            for h in json.loads(f.read_text()):
                if h:
                    eval_hashes.add(h)
        print(f"  Loaded {len(eval_hashes):,} eval hashes from cache")
    else:
        # Hash images from eval sets (HuggingFace cache)
        hf_cache_dir = REPO / ".cache"
        eval_ds_names = ["atlas", "deepfashion_inshop", "deepfashion_multimodal",
                         "fashion200k", "KAGL", "polyvore"]
        raw_dirs = [REPO / "data" / "raw" / d for d in eval_ds_names]
        for d in raw_dirs:
            if not d.exists():
                continue
            for img in d.rglob("*.jpg"):
                try:
                    h = phasher.encode_image(str(img))
                    if h:
                        eval_hashes.add(h)
                except Exception:
                    pass
        print(f"  Hashed {len(eval_hashes):,} raw eval images")

    if not eval_hashes:
        print("  [WARN] No eval images found for leakage check — proceeding without check")
        return records

    kept, removed = [], 0
    for r in tqdm(records, desc="leakage-check"):
        try:
            h = phasher.encode_image(r["image_path"])
            if h and any(phasher.hamming_distance(h, eh) <= 6 for eh in eval_hashes):
                removed += 1
                continue
        except Exception:
            pass
        kept.append(r)

    pct = removed / max(len(records), 1)
    print(f"  Leakage: removed={removed:,}, kept={len(kept):,} ({pct:.2%})")
    if pct > 0.05:
        print(f"  [WARN] Leak rate {pct:.1%} > 5% — review data sources before training.")

    (REPO / "data" / "leakage_report.json").write_text(json.dumps({
        "total_checked": len(records), "removed": removed,
        "kept": len(kept), "pct": pct,
    }, indent=2))
    return kept


# ──────────────────────────────────────────────────────────────────────────────
# Dedup by pair_id, then write
# ──────────────────────────────────────────────────────────────────────────────
def dedup(records: list[dict]) -> list[dict]:
    seen: set[str] = set()
    out = []
    for r in records:
        pid = r["pair_id"]
        if pid in seen:
            continue
        seen.add(pid)
        out.append(r)
    print(f"  Dedup: {len(records):,} → {len(out):,} unique pairs")
    return out


def print_stats(records: list[dict]):
    buckets = Counter(_bucket(r["query"]) for r in records)
    sources = Counter(r["source"] for r in records)
    total = len(records)
    print(f"\n  Total pairs: {total:,}")
    for label, tgt in [("short", 0.25), ("medium", 0.35), ("long", 0.40)]:
        cnt = buckets.get(label, 0)
        pct = cnt / total
        warn = " ← OFF >10pp" if abs(pct - tgt) > 0.10 else ""
        print(f"    {label:8s}: {cnt:7,} ({pct:.1%}) target {tgt:.0%}{warn}")
    print("  Sources:", dict(sources.most_common()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no_leakage", action="store_true", help="Skip pHash leakage check")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_jsonl = OUT_DIR / "pairs_combined.jsonl"

    print("\n[1/3] Loading existing GS-10M (fashion_stratified_gs_df) ...")
    records = load_gs_df()

    print("\n[2/3] Loading v4 pattern-targeted pairs ...")
    records += load_v4_pairs()

    print("\n[2b] Loading v4 synthetic pairs ...")
    records += load_v4_synthetic()

    print(f"\n[3/3] Dedup ...")
    records = dedup(records)

    random.shuffle(records)

    if not args.no_leakage:
        print("\n[Leakage check] pHash ...")
        records = run_leakage_check(records)
    else:
        print("\n[Leakage check] SKIPPED (--no_leakage)")

    print_stats(records)

    print(f"\nWriting {len(records):,} pairs → {out_jsonl}")
    with out_jsonl.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    # manifest
    buckets = Counter(_bucket(r["query"]) for r in records)
    sources = Counter(r["source"] for r in records)
    total = len(records)
    manifest = {
        "total_pairs": total,
        "query_length": {k: round(v / total, 3) for k, v in buckets.items()},
        "sources": dict(sources.most_common()),
    }
    (REPO / "data" / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Manifest → data/manifest.json")
    print("\nData prep complete.")


if __name__ == "__main__":
    main()
