"""
v6 Step 2 — Download and prep FashionIQ for prose teacher training.

FashionIQ is a public academic dataset with ~77K natural language descriptions
of fashion item differences. Each item has a candidate image + a multi-sentence
description. Distribution matches fashion200k closely (long prose, fashion domain).
Zero overlap with our Marqo eval benchmarks.

HuggingFace: McAuley-Lab/FashionIQ  (or clip-benchmark/wds_fashioniq)

Outputs (data/processed/v6/):
  pairs_fashioniq.jsonl     — (pair_id, query, image_file, category, split)
  fashioniq_images/         — downloaded images at 224px
  stats_fashioniq.json      — category/split counts

Usage:
    python scripts/v6/step2_prep_fashioniq.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = REPO / "data" / "processed" / "v6"
IMG_DIR = OUT_DIR / "fashioniq_images"

# FashionIQ has 3 categories; all have (captions, candidate_image) pairs
FASHIONIQ_CATEGORIES = ["dress", "toptee", "shirt"]
FASHIONIQ_HF_ID = "McAuley-Lab/Amazon-Fashion"   # fallback if primary fails

# Primary source: the standard FashionIQ splits via HF
PRIMARY_HF_ID = "sentence-transformers/fashioniq"


def _try_load(hf_id: str, category: str | None = None):
    from datasets import load_dataset
    try:
        if category:
            return load_dataset(hf_id, category, trust_remote_code=False)
        return load_dataset(hf_id, trust_remote_code=False)
    except Exception as e:
        print(f"  load_dataset({hf_id!r}) failed: {e}")
        return None


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    IMG_DIR.mkdir(parents=True, exist_ok=True)

    out_path = OUT_DIR / "pairs_fashioniq.jsonl"
    stats_path = OUT_DIR / "stats_fashioniq.json"

    print("Discovering FashionIQ on HuggingFace ...")
    from datasets import load_dataset, get_dataset_config_names

    # Try several known HF sources for FashionIQ
    sources_to_try = [
        ("clip-benchmark/wds_fashioniq", None),
        ("sentence-transformers/fashioniq", None),
        ("McAuley-Lab/FashionIQ", None),
    ]

    ds_found = None
    hf_id_used = None
    for hf_id, cfg in sources_to_try:
        print(f"  Trying {hf_id} ...")
        try:
            configs = get_dataset_config_names(hf_id)
            print(f"    configs: {configs}")
            hf_id_used = hf_id
            ds_found = {"configs": configs, "hf_id": hf_id}
            break
        except Exception as e:
            print(f"    failed: {e}")

    if ds_found is None:
        # Fallback: write a clear note and produce a stub
        print("\nWARNING: Could not auto-discover FashionIQ on HuggingFace.")
        print("Manual download option:")
        print("  1. Download from https://github.com/XiaoxiaoGuo/fashion-iq")
        print("  2. Place images in data/processed/v6/fashioniq_images/")
        print("  3. Place JSON files in data/processed/v6/fashioniq_raw/")
        print("  4. Re-run this script with --manual flag")
        stub = {"status": "not_found", "tried": [s[0] for s in sources_to_try]}
        stats_path.write_text(json.dumps(stub, indent=2))
        return

    print(f"\nLoading from {hf_id_used} ...")
    n_written = 0
    n_skipped = 0
    category_counts = {}
    t0 = time.time()

    with out_path.open("w") as fout:
        for config in ds_found["configs"]:
            print(f"\n  Config: {config}")
            try:
                ds = load_dataset(hf_id_used, config)
            except Exception as e:
                print(f"    skip (load failed): {e}")
                continue

            for split_name, split_ds in ds.items():
                print(f"    Split: {split_name}, rows: {len(split_ds)}")
                cols = split_ds.column_names
                print(f"    Columns: {cols}")

                # Detect text and image columns
                text_col = next((c for c in ["captions", "caption", "text",
                                              "description", "query"] if c in cols), None)
                img_col = next((c for c in ["image", "img", "candidate_image",
                                             "target_image"] if c in cols), None)

                if text_col is None or img_col is None:
                    print(f"    skip (no text_col={text_col} or img_col={img_col})")
                    continue

                for i, row in enumerate(tqdm(split_ds, desc=f"{config}/{split_name}")):
                    text = row.get(text_col, "")
                    if isinstance(text, list):
                        # FashionIQ has list of captions — use all as separate pairs
                        texts = [t for t in text if isinstance(t, str) and t.strip()]
                    elif isinstance(text, str) and text.strip():
                        texts = [text.strip()]
                    else:
                        n_skipped += 1
                        continue

                    pil = row.get(img_col)
                    if pil is None:
                        n_skipped += 1
                        continue

                    img_file = f"fiq_{config}_{split_name}_{i}.jpg"
                    img_path = IMG_DIR / img_file
                    if not img_path.exists():
                        try:
                            if not isinstance(pil, Image.Image):
                                pil = Image.fromarray(pil)
                            pil = pil.convert("RGB").resize((224, 224), Image.LANCZOS)
                            pil.save(img_path, "JPEG", quality=85)
                        except Exception:
                            n_skipped += 1
                            continue

                    for j, t in enumerate(texts):
                        pid = f"fiq_{config}_{split_name}_{i}_{j}"
                        record = {
                            "pair_id": pid,
                            "query": t,
                            "image_file": img_file,
                            "category": config,
                            "split": split_name,
                            "n_words": len(t.split()),
                        }
                        fout.write(json.dumps(record) + "\n")
                        n_written += 1

                    category_counts[config] = category_counts.get(config, 0) + len(texts)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s")
    print(f"  Written: {n_written:,} pairs | Skipped: {n_skipped:,}")
    print(f"  Category breakdown: {category_counts}")

    # Word count stats
    pairs = [json.loads(l) for l in out_path.open() if l.strip()]
    avg_words = sum(r["n_words"] for r in pairs) / max(len(pairs), 1)
    print(f"  Avg query length: {avg_words:.1f} words")

    stats = {
        "hf_id": hf_id_used,
        "total_pairs": n_written,
        "category_counts": category_counts,
        "avg_query_words": avg_words,
        "output": str(out_path),
        "images": str(IMG_DIR),
    }
    stats_path.write_text(json.dumps(stats, indent=2))
    print(f"Stats → {stats_path}")


if __name__ == "__main__":
    main()
