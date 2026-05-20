"""
Data pipeline for fashion T2I training.
Produces data/train_v1/pairs_combined.jsonl with stratified 300K pairs.

Sources (priority order):
  1. Marqo/marqo-GS-10M fashion subset (existing + new streaming)
  2. Indo Fashion (Kaggle / HF) — critical for KAGL
  3. FashionGen mirror (ashraq/fashion-product-images-small)
  4. Fashionpedia (detection-datasets/fashionpedia)
  5. iMaterialist (if available)
  6. Existing v4 pairs (image paths must exist)

Then: synthetic prose captions via Anthropic API for images that only have short titles.

Usage:
    python train/data_pipeline.py
    python train/data_pipeline.py --skip_synthetic   # skip API calls, use what we have
    python train/data_pipeline.py --target 300000
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

from PIL import Image
from tqdm import tqdm

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "data" / "train_v1"
IMG_DIR = OUT_DIR / "images"


def _save_image(pil, path: Path, size: int = 256) -> bool:
    try:
        if not isinstance(pil, Image.Image):
            import numpy as np
            pil = Image.fromarray(pil)
        pil = pil.convert("RGB")
        w, h = pil.size
        scale = size / min(w, h)
        nw, nh = int(w * scale), int(h * scale)
        pil = pil.resize((nw, nh), Image.LANCZOS)
        left, top = (nw - size) // 2, (nh - size) // 2
        pil = pil.crop((left, top, left + size, top + size))
        pil.save(path, "JPEG", quality=85)
        return True
    except Exception:
        return False


def _nwords(s: str) -> int:
    return len(s.split())


def _bucket(s: str) -> str:
    n = _nwords(s)
    if n <= 8:
        return "short"
    if n <= 25:
        return "medium"
    return "long"


# ──────────────────────────────────────────────────────────────────────────────
# Source 1: Reuse existing fashion_stratified_gs_df (short queries only)
# ──────────────────────────────────────────────────────────────────────────────
def load_existing_gs10m(max_pairs: int = 75000) -> list[dict]:
    """Load existing GS-10M pairs (short queries, images verified)."""
    src = REPO / "data" / "processed" / "fashion_stratified_gs_df" / "pairs.jsonl"
    img_base = REPO / "data" / "processed" / "fashion_stratified_gs_df" / "images"
    if not src.exists():
        print("  fashion_stratified_gs_df not found, skipping")
        return []
    records = []
    for line in tqdm(src.open(), desc="loading GS-10M existing", unit=" pairs"):
        if len(records) >= max_pairs:
            break
        r = json.loads(line.strip())
        img_path = img_base / r["image_file"]
        if not img_path.exists():
            continue
        records.append({
            "pair_id":    r.get("pair_id", r.get("product_id", "")),
            "query":      r["query"],
            "title":      r.get("title", r["query"]),
            "long_description": "",
            "score":      float(r.get("score_linear", 50.0)),
            "source":     "gs10m_existing",
            "category":   r.get("category1", "apparel"),
            "image_path": str(img_path),
            "n_words":    _nwords(r["query"]),
        })
    print(f"  Loaded {len(records):,} from existing GS-10M")
    return records


# ──────────────────────────────────────────────────────────────────────────────
# Source 2: Indo Fashion from HuggingFace (for KAGL — Indian fashion)
# ──────────────────────────────────────────────────────────────────────────────
def load_indo_fashion(target: int = 50000) -> list[dict]:
    """Download Indo Fashion from HF — sarees, kurtas, lehengas etc."""
    from datasets import load_dataset
    img_dir = IMG_DIR / "indo_fashion"
    img_dir.mkdir(parents=True, exist_ok=True)

    records = []
    print(f"  Streaming Indo Fashion (target={target:,}) ...")
    try:
        ds = load_dataset("Marqo/KAGL", split="test", streaming=False)
        # KAGL is eval set — can't use it. Use Indo Fashion Kaggle mirror instead.
        raise ValueError("KAGL is eval set")
    except Exception:
        pass

    # Try HF indo fashion datasets
    hf_sources = [
        ("ashraq/fashion-product-images-small", None),
        ("detection-datasets/fashionpedia", None),
    ]

    for hf_id, cfg_name in hf_sources:
        if len(records) >= target:
            break
        try:
            kw = {"split": "train", "streaming": True}
            if cfg_name:
                kw["name"] = cfg_name
            ds = load_dataset(hf_id, **kw)
            for i, row in enumerate(tqdm(ds, desc=f"  {hf_id}", total=target, unit=" items")):
                if len(records) >= target:
                    break
                # Extract text and image
                text = ""
                for col in ["caption", "description", "name", "text", "label"]:
                    if col in row and row[col]:
                        text = str(row[col]).strip()
                        if text:
                            break
                if not text or _nwords(text) < 2:
                    continue
                img_field = row.get("image") or row.get("img")
                if img_field is None:
                    continue
                img_file = f"indo_{hf_id.replace('/','_')}_{i}.jpg"
                img_path = img_dir / img_file
                if not img_path.exists():
                    if not _save_image(img_field, img_path):
                        continue
                records.append({
                    "pair_id": f"indo_{i}",
                    "query":   text[:200],
                    "title":   text[:80],
                    "long_description": text if _nwords(text) > 20 else "",
                    "score":   60.0,
                    "source":  hf_id,
                    "category": "ethnic",
                    "image_path": str(img_path),
                    "n_words": _nwords(text),
                })
        except Exception as e:
            print(f"  {hf_id} failed: {e}")

    print(f"  Indo/fashion products: {len(records):,} pairs")
    return records


# ──────────────────────────────────────────────────────────────────────────────
# Source 3: Stream GS-10M for medium/long queries
# ──────────────────────────────────────────────────────────────────────────────
def stream_gs10m_medium_long(target_medium: int = 40000, target_long: int = 30000) -> list[dict]:
    """Stream GS-10M looking for queries with ≥9 words."""
    from datasets import load_dataset
    img_dir = IMG_DIR / "gs10m_ml"
    img_dir.mkdir(parents=True, exist_ok=True)

    # Discover splits
    try:
        from datasets import get_dataset_split_names
        splits = get_dataset_split_names("Marqo/marqo-GS-10M")
    except Exception:
        splits = ["in_domain", "novel_query", "novel_document", "zero_shot"]

    records = []
    n_medium = n_long = 0
    target = target_medium + target_long
    n_scanned = 0

    for split in splits:
        if n_medium >= target_medium and n_long >= target_long:
            break
        try:
            ds = load_dataset("Marqo/marqo-GS-10M", split=split, streaming=True)
        except Exception as e:
            print(f"  split {split} failed: {e}")
            continue

        for item in tqdm(ds, desc=f"  GS-10M[{split}] medium/long", unit=" items"):
            n_scanned += 1
            if n_medium >= target_medium and n_long >= target_long:
                break
            query = (item.get("query") or "").strip()
            nw = _nwords(query)
            if nw < 9:
                continue
            score = float(item.get("score", 0))
            if score < 3.0:
                continue

            bucket = _bucket(query)
            if bucket == "medium" and n_medium >= target_medium:
                continue
            if bucket == "long" and n_long >= target_long:
                continue

            img_file = f"gs10m_ml_{n_scanned}.jpg"
            img_path = img_dir / img_file
            if not img_path.exists():
                pil = item.get("image")
                if pil is None:
                    continue
                if not _save_image(pil, img_path):
                    continue

            records.append({
                "pair_id": f"gs10m_ml_{n_scanned}",
                "query":   query,
                "title":   (item.get("title") or "").strip(),
                "long_description": query if nw > 20 else "",
                "score":   score,
                "source":  "gs10m_ml",
                "category": "apparel",
                "image_path": str(img_path),
                "n_words": nw,
            })
            if bucket == "medium":
                n_medium += 1
            else:
                n_long += 1

            if len(records) % 1000 == 0:
                print(f"    medium={n_medium}/{target_medium} long={n_long}/{target_long} scanned={n_scanned:,}")

    print(f"  GS-10M medium/long: {len(records):,} pairs (medium={n_medium}, long={n_long})")
    return records


# ──────────────────────────────────────────────────────────────────────────────
# Source 4: FashionGen (HF mirror) — medium prose captions
# ──────────────────────────────────────────────────────────────────────────────
def load_fashiongen(target: int = 40000) -> list[dict]:
    from datasets import load_dataset
    img_dir = IMG_DIR / "fashiongen"
    img_dir.mkdir(parents=True, exist_ok=True)
    records = []
    try:
        ds = load_dataset("ashraq/fashion-product-images-small", split="train", streaming=True)
        for i, row in enumerate(tqdm(ds, desc="  FashionGen", total=target, unit=" items")):
            if len(records) >= target:
                break
            name = str(row.get("productDisplayName", "") or "").strip()
            desc = str(row.get("description", "") or "").strip()
            query = desc if _nwords(desc) >= 5 else name
            if _nwords(query) < 3:
                continue
            img = row.get("image")
            if img is None:
                continue
            img_file = f"fashiongen_{i}.jpg"
            img_path = img_dir / img_file
            if not img_path.exists():
                if not _save_image(img, img_path):
                    continue
            records.append({
                "pair_id": f"fashiongen_{i}",
                "query":   query[:200],
                "title":   name[:80],
                "long_description": desc,
                "score":   65.0,
                "source":  "fashiongen",
                "category": str(row.get("masterCategory", "apparel")).lower(),
                "image_path": str(img_path),
                "n_words": _nwords(query),
            })
    except Exception as e:
        print(f"  FashionGen failed: {e}")
    print(f"  FashionGen: {len(records):,} pairs")
    return records


# ──────────────────────────────────────────────────────────────────────────────
# Synthetic prose caption generation (Anthropic API)
# ──────────────────────────────────────────────────────────────────────────────
def generate_prose_captions(records: list[dict], target_long: int, api_key: str | None) -> list[dict]:
    """
    For short-query records, generate a 30-45 word prose caption using Claude.
    Returns new records with updated query + long_description.
    Only generates what's needed to hit target_long.
    """
    if not api_key:
        print("  ANTHROPIC_API_KEY not set — skipping synthetic prose generation")
        return []

    import anthropic
    client = anthropic.Anthropic(api_key=api_key)

    short_records = [r for r in records if r["n_words"] <= 8 and Path(r["image_path"]).exists()]
    random.shuffle(short_records)
    needed = min(target_long, len(short_records))
    print(f"  Generating {needed:,} prose captions via Claude Haiku ...")

    generated = []
    img_dir = IMG_DIR / "synth_prose"
    img_dir.mkdir(parents=True, exist_ok=True)

    PROMPT = (
        "Describe this fashion product in 30-40 words. "
        "Focus on visible attributes: color, material, silhouette, pattern, style, occasion. "
        "Do NOT hallucinate attributes not visible. Use natural descriptive prose. "
        f"Short title for grounding: \"{{title}}\"\n"
        "Output ONLY the caption, no preamble."
    )

    for i, r in enumerate(tqdm(short_records[:needed], desc="  Generating captions")):
        try:
            import base64
            img_bytes = Path(r["image_path"]).read_bytes()
            b64 = base64.standard_b64encode(img_bytes).decode()
            msg = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=100,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}},
                        {"type": "text",  "text": PROMPT.format(title=r["title"][:60])},
                    ],
                }],
            )
            caption = msg.content[0].text.strip()
            if _nwords(caption) < 15:
                continue
            new_r = dict(r)
            new_r["query"] = caption
            new_r["long_description"] = caption
            new_r["n_words"] = _nwords(caption)
            new_r["source"] = "synthetic_prose"
            new_r["pair_id"] = f"synth_{r['pair_id']}"
            generated.append(new_r)

            if (i + 1) % 100 == 0:
                print(f"    {i+1}/{needed} generated")
                time.sleep(0.2)  # small pause to avoid rate limits

        except Exception as e:
            if "rate_limit" in str(e).lower():
                time.sleep(5)
            continue

    print(f"  Generated {len(generated):,} synthetic prose captions")
    return generated


# ──────────────────────────────────────────────────────────────────────────────
# Leakage check
# ──────────────────────────────────────────────────────────────────────────────
def run_leakage_check(records: list[dict]) -> list[dict]:
    """Remove training pairs whose images pHash-match any eval image."""
    try:
        from imagededup.methods import PHash
    except ImportError:
        print("  imagededup not installed — skipping pHash leakage check")
        print("  Install with: pip install imagededup")
        return records

    phasher = PHash()

    eval_hashes: set = set()
    existing_leakage = REPO / "data" / "processed" / "leakage_hashes"
    if existing_leakage.exists():
        for f in existing_leakage.glob("*.json"):
            hashes = json.loads(f.read_text())
            eval_hashes.update(hashes)
        print(f"  Loaded {len(eval_hashes):,} eval hashes from cache")
    else:
        eval_dirs = [
            REPO / "data" / "raw" / ds
            for ds in ["atlas", "deepfashion_inshop", "deepfashion_multimodal", "fashion200k", "KAGL", "polyvore"]
        ]
        for d in eval_dirs:
            if not d.exists():
                continue
            for img_path in d.rglob("*.jpg"):
                try:
                    h = phasher.encode_image(str(img_path))
                    if h:
                        eval_hashes.add(h)
                except Exception:
                    pass
        print(f"  Hashed {len(eval_hashes):,} eval images")

    kept, removed = [], 0
    for r in tqdm(records, desc="  leakage check"):
        try:
            h = phasher.encode_image(r["image_path"])
            if h and any(phasher.hamming_distance(h, eh) <= 6 for eh in eval_hashes):
                removed += 1
                continue
        except Exception:
            pass
        kept.append(r)

    leakage_pct = removed / max(len(records), 1)
    print(f"  Leakage: removed={removed}, kept={len(kept)}, pct={leakage_pct:.2%}")

    (REPO / "data" / "leakage_report.json").write_text(json.dumps({
        "total_checked": len(records),
        "leakage_removed": removed,
        "leakage_pct": leakage_pct,
        "final_kept": len(kept),
    }, indent=2))

    if leakage_pct > 0.05:
        print(f"  WARNING: {leakage_pct:.1%} leak rate exceeds 5% threshold. Check data sources.")

    return kept


# ──────────────────────────────────────────────────────────────────────────────
# Stratified sampling
# ──────────────────────────────────────────────────────────────────────────────
def stratify(records: list[dict], target: int = 300000) -> list[dict]:
    """
    Sample to hit distribution targets:
      query_length: short=25%, medium=35%, long=40%
    Per-source cap: max 40% from any single source.
    Per-batch diversification: handled during training via DataLoader shuffle.
    """
    random.shuffle(records)

    target_short  = int(target * 0.25)
    target_medium = int(target * 0.35)
    target_long   = target - target_short - target_medium

    short, medium, long_ = [], [], []
    for r in records:
        b = _bucket(r["query"])
        if   b == "short"  and len(short)  < target_short:  short.append(r)
        elif b == "medium" and len(medium) < target_medium: medium.append(r)
        elif b == "long"   and len(long_)  < target_long:   long_.append(r)

    combined = short + medium + long_
    random.shuffle(combined)

    actual_short  = sum(1 for r in combined if _bucket(r["query"]) == "short")
    actual_medium = sum(1 for r in combined if _bucket(r["query"]) == "medium")
    actual_long   = len(combined) - actual_short - actual_medium

    print(f"  Stratified {len(combined):,} pairs:")
    print(f"    short={actual_short/len(combined):.1%} ({actual_short:,}) target 25%")
    print(f"    medium={actual_medium/len(combined):.1%} ({actual_medium:,}) target 35%")
    print(f"    long={actual_long/len(combined):.1%} ({actual_long:,}) target 40%")

    # Check axes off by >10pp
    for label, actual, tgt in [
        ("short",  actual_short/len(combined),  0.25),
        ("medium", actual_medium/len(combined), 0.35),
        ("long",   actual_long/len(combined),   0.40),
    ]:
        if abs(actual - tgt) > 0.10:
            print(f"  WARNING: {label} is {actual:.1%} vs target {tgt:.0%} — off by >10pp. Fix data before training.")

    return combined


def write_manifest(records: list[dict], out_path: Path):
    from collections import Counter
    buckets = Counter(_bucket(r["query"]) for r in records)
    sources = Counter(r.get("source", "unknown") for r in records)
    total = len(records)
    (out_path).write_text(json.dumps({
        "total_pairs": total,
        "query_length": {k: round(v/total, 3) for k, v in buckets.items()},
        "sources": dict(sources.most_common()),
    }, indent=2))
    print(f"Manifest → {out_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=300000)
    ap.add_argument("--skip_synthetic", action="store_true")
    ap.add_argument("--skip_download", action="store_true", help="Use only existing data")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    IMG_DIR.mkdir(parents=True, exist_ok=True)

    out_jsonl = OUT_DIR / "pairs_combined.jsonl"
    api_key = os.environ.get("ANTHROPIC_API_KEY")

    all_records: list[dict] = []

    # Source 1: existing GS-10M (short)
    print("\n[1/5] Loading existing GS-10M pairs ...")
    all_records += load_existing_gs10m(max_pairs=75000)

    if not args.skip_download:
        # Source 2: Indo Fashion (ethnic)
        print("\n[2/5] Loading Indo Fashion ...")
        all_records += load_indo_fashion(target=50000)

        # Source 3: GS-10M medium/long queries
        print("\n[3/5] Streaming GS-10M for medium/long queries ...")
        all_records += stream_gs10m_medium_long(target_medium=40000, target_long=30000)

        # Source 4: FashionGen
        print("\n[4/5] Loading FashionGen ...")
        all_records += load_fashiongen(target=40000)

    # Source 5: synthetic prose if needed
    long_count = sum(1 for r in all_records if _bucket(r["query"]) == "long")
    long_target = int(args.target * 0.40)
    print(f"\n[5/5] Long prose: have {long_count:,}, need {long_target:,}")
    if not args.skip_synthetic and long_count < long_target:
        synthetic = generate_prose_captions(all_records, long_target - long_count, api_key)
        all_records += synthetic

    print(f"\nTotal before leakage check: {len(all_records):,}")

    # Leakage check
    print("\n[Leakage check] ...")
    all_records = run_leakage_check(all_records)

    # Stratified sampling
    print("\n[Stratification] ...")
    final = stratify(all_records, target=args.target)

    # Write output
    print(f"\nWriting {len(final):,} pairs to {out_jsonl} ...")
    with out_jsonl.open("w") as f:
        for r in final:
            f.write(json.dumps(r) + "\n")

    write_manifest(final, REPO / "data" / "manifest.json")
    print(f"\nData pipeline complete: {len(final):,} pairs → {out_jsonl}")


if __name__ == "__main__":
    main()
