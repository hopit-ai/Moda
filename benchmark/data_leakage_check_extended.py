"""
Extended data-leakage audit for Phase 5.X (Recipe X attribute extraction).

Goal: confirm that DeepFashion-InShop and DeepFashion-MultiModal (which we
plan to use for attribute linear-probing) do not overlap with any LookBench
image (query, subset gallery, or noise gallery).

Method: hash every image at a normalized representation (resized to
224x224 RGB, JPEG-encoded at quality 95) using MD5. This makes the hash
invariant to source format / original resolution differences while
remaining a strict "exact image" check.

Outputs results/lookbench/data_leakage_check_v2.json with:
    - per-dataset image counts and unique-hash counts
    - intersection sizes between (lookbench, df-inshop) and (lookbench, df-mm)
    - sample overlapping item_IDs if any (else empty list)
    - PASS / FAIL verdict

Run:
    python benchmark/data_leakage_check_extended.py
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from datasets import load_dataset, load_from_disk
from PIL import Image
from tqdm import tqdm

BASE = Path(__file__).resolve().parent.parent
LB_ROOT = BASE / "data/raw/lookbench/datasets"
LB_NOISE = BASE / "data/raw/lookbench/datasets/noise"
RESULTS = BASE / "results/lookbench"
RESULTS.mkdir(parents=True, exist_ok=True)
OUT = RESULTS / "data_leakage_check_v2.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

LB_SUBSETS = ["real_studio_flat", "aigen_studio", "real_streetlook", "aigen_streetlook"]
NORM_SIZE = (224, 224)
JPEG_QUALITY = 95


def normalized_md5(img: Image.Image) -> str:
    """Resize to 224x224 RGB, re-encode as JPEG q=95, MD5 of bytes."""
    if img.mode != "RGB":
        img = img.convert("RGB")
    img = img.resize(NORM_SIZE, Image.BICUBIC)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=JPEG_QUALITY)
    return hashlib.md5(buf.getvalue()).hexdigest()


def hash_dataset(rows: Iterable[dict], image_key: str, id_key: str | None,
                 desc: str, limit: int | None = None) -> dict[str, list[str]]:
    """Hash every image in `rows`. Returns {hash: [item_ids]}."""
    h2ids: dict[str, list[str]] = defaultdict(list)
    n = 0
    for row in tqdm(rows, desc=desc, total=limit):
        if limit and n >= limit:
            break
        try:
            img = row[image_key]
            if not isinstance(img, Image.Image):
                continue
            h = normalized_md5(img)
            iid = row.get(id_key, f"{desc}_{n}") if id_key else f"{desc}_{n}"
            h2ids[h].append(str(iid))
            n += 1
        except Exception as e:
            log.debug("skip row %d: %s", n, e)
    log.info("  %s: %d images, %d unique hashes (%d duplicates)",
             desc, n, len(h2ids), n - len(h2ids))
    return dict(h2ids)


def load_lookbench_hashes() -> dict[str, list[str]]:
    """Hash all LookBench images: each subset's query + gallery + the shared noise."""
    log.info("Hashing LookBench images...")
    lb_hashes: dict[str, list[str]] = defaultdict(list)
    for subset in LB_SUBSETS:
        path = LB_ROOT / subset
        if not path.exists():
            log.warning("missing LB subset %s", subset)
            continue
        dsd = load_from_disk(str(path))
        for split_name in ["query", "gallery"]:
            if split_name not in dsd:
                continue
            split = dsd[split_name]
            tag = f"lb_{subset}_{split_name}"
            id_key = ("item_ID" if "item_ID" in split.column_names
                      else ("article_id" if "article_id" in split.column_names else None))
            partial = hash_dataset(split, image_key="image", id_key=id_key, desc=tag)
            for h, ids in partial.items():
                lb_hashes[h].extend([f"{tag}:{i}" for i in ids])
    if LB_NOISE.exists():
        noise_ds = load_from_disk(str(LB_NOISE))
        if hasattr(noise_ds, "items"):
            for split_name, split in noise_ds.items():
                tag = f"lb_noise_{split_name}"
                id_key = ("item_ID" if "item_ID" in split.column_names
                          else ("article_id" if "article_id" in split.column_names else None))
                partial = hash_dataset(split, image_key="image", id_key=id_key, desc=tag)
                for h, ids in partial.items():
                    lb_hashes[h].extend([f"{tag}:{i}" for i in ids])
        else:
            tag = "lb_noise"
            id_key = ("item_ID" if "item_ID" in noise_ds.column_names
                      else ("article_id" if "article_id" in noise_ds.column_names else None))
            partial = hash_dataset(noise_ds, image_key="image", id_key=id_key, desc=tag)
            for h, ids in partial.items():
                lb_hashes[h].extend([f"{tag}:{i}" for i in ids])
    return dict(lb_hashes)


def load_marqo_dataset_hashes(hf_id: str, cache_dir: Path,
                              tag: str) -> dict[str, list[str]]:
    log.info("Loading + hashing %s ...", hf_id)
    ds = load_dataset(hf_id, cache_dir=str(cache_dir))
    split_name = next(iter(ds.keys()))
    split = ds[split_name]
    return hash_dataset(split, image_key="image", id_key="item_ID", desc=tag)


def intersect(lb: dict[str, list[str]],
              other: dict[str, list[str]],
              other_name: str) -> dict:
    overlap_hashes = set(lb.keys()) & set(other.keys())
    samples = []
    for h in list(overlap_hashes)[:25]:
        samples.append({
            "hash": h,
            "lookbench_ids": lb[h][:5],
            f"{other_name}_ids": other[h][:5],
        })
    return {
        "n_lookbench_hashes": len(lb),
        f"n_{other_name}_hashes": len(other),
        "n_overlapping_hashes": len(overlap_hashes),
        "overlap_pct_of_other": (
            round(100.0 * len(overlap_hashes) / max(1, len(other)), 4)
        ),
        "sample_overlaps": samples,
    }


def main() -> int:
    t0 = time.time()
    log.info("=" * 70)
    log.info("EXTENDED DATA-LEAKAGE AUDIT (Recipe X attribute extraction)")
    log.info("=" * 70)

    lb_hashes = load_lookbench_hashes()
    log.info("LookBench: %d unique image hashes total", len(lb_hashes))

    inshop_hashes = load_marqo_dataset_hashes(
        "Marqo/deepfashion-inshop",
        BASE / "data/raw/deepfashion_inshop",
        "df_inshop",
    )
    mm_hashes = load_marqo_dataset_hashes(
        "Marqo/deepfashion-multimodal",
        BASE / "data/raw/deepfashion_multimodal",
        "df_multimodal",
    )

    inshop_check = intersect(lb_hashes, inshop_hashes, "df_inshop")
    mm_check = intersect(lb_hashes, mm_hashes, "df_multimodal")

    elapsed = time.time() - t0
    verdict_inshop = "PASS" if inshop_check["n_overlapping_hashes"] == 0 else "FAIL"
    verdict_mm = "PASS" if mm_check["n_overlapping_hashes"] == 0 else "FAIL"
    overall = "PASS" if (verdict_inshop == "PASS" and verdict_mm == "PASS") else "FAIL"

    out = {
        "verification_date": time.strftime("%Y-%m-%d"),
        "purpose": ("Confirm that DeepFashion-InShop and DeepFashion-MultiModal "
                    "(planned for attribute linear-probing in Recipe X) do not "
                    "leak any image into LookBench (queries, subset galleries, "
                    "or noise gallery)."),
        "method": {
            "hash": "MD5 of JPEG-q95-encoded 224x224 RGB normalized image",
            "scope": "ALL images in both datasets compared against ALL LookBench images",
        },
        "evaluation_data": {
            "name": "LookBench",
            "subsets": LB_SUBSETS + ["noise"],
        },
        "training_data_audited": [
            {
                "name": "DeepFashion-InShop (Marqo)",
                "hf_id": "Marqo/deepfashion-inshop",
                "n_images_audited": sum(len(v) for v in inshop_hashes.values()),
                "n_unique_hashes": len(inshop_hashes),
            },
            {
                "name": "DeepFashion-MultiModal (Marqo)",
                "hf_id": "Marqo/deepfashion-multimodal",
                "n_images_audited": sum(len(v) for v in mm_hashes.values()),
                "n_unique_hashes": len(mm_hashes),
            },
        ],
        "leakage_checks": {
            "df_inshop_vs_lookbench": {**inshop_check, "verdict": verdict_inshop},
            "df_multimodal_vs_lookbench": {**mm_check, "verdict": verdict_mm},
        },
        "overall_verdict": overall,
        "elapsed_seconds": round(elapsed, 1),
    }

    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)

    log.info("=" * 70)
    log.info("DF-InShop      vs LookBench: %d overlapping hashes -> %s",
             inshop_check["n_overlapping_hashes"], verdict_inshop)
    log.info("DF-MultiModal  vs LookBench: %d overlapping hashes -> %s",
             mm_check["n_overlapping_hashes"], verdict_mm)
    log.info("OVERALL VERDICT: %s", overall)
    log.info("Saved to: %s (elapsed %.1fs)", OUT, elapsed)
    return 0 if overall == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
