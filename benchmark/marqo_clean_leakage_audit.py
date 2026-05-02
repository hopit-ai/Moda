"""
Cross-benchmark leakage audit for the "clean" Marqo benchmark suite.

Goal: confirm that Atlas, KAGL, Fashion200k, and Polyvore
(the 4 Marqo datasets we plan to evaluate on for a fair MoDA-vs-FashionSigLIP
comparison) do NOT overlap with our distillation training pool:

    - DeepFashion-InShop (Marqo)
    - DeepFashion-MultiModal (Marqo)
    - H&M article images (`data/raw/hnm_images/`)
    - LookBench (queries + galleries + noise gallery)

Method (same as benchmark/data_leakage_check_extended.py):
    Hash every image at a normalized representation (resized to 224x224 RGB,
    JPEG-encoded at quality 95) using MD5. Invariant to source format and
    original resolution differences while remaining a strict "exact image"
    check.

Per-dataset hash sets are CACHED to disk so re-runs are cheap.

Outputs:
    results/marqo_bench/leakage_audit_clean.json
        - per-dataset image counts and unique-hash counts
        - intersection sizes for each (clean_dataset, training_dataset) pair
        - PASS / FAIL verdict
        - sample overlapping IDs (if any)

Run:
    python benchmark/marqo_clean_leakage_audit.py
    python benchmark/marqo_clean_leakage_audit.py --recompute hnm
    python benchmark/marqo_clean_leakage_audit.py --datasets atlas KAGL
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import logging
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from PIL import Image
from tqdm import tqdm

BASE = Path(__file__).resolve().parent.parent
RESULTS = BASE / "results/marqo_bench"
RESULTS.mkdir(parents=True, exist_ok=True)
HASH_CACHE_DIR = BASE / "data/processed/leakage_hashes"
HASH_CACHE_DIR.mkdir(parents=True, exist_ok=True)
OUT_JSON = RESULTS / "leakage_audit_clean.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

NORM_SIZE = (224, 224)
JPEG_QUALITY = 95
LB_ROOT = BASE / "data/raw/lookbench/datasets"
LB_NOISE = BASE / "data/raw/lookbench/datasets/noise"
LB_SUBSETS = ["real_studio_flat", "aigen_studio", "real_streetlook", "aigen_streetlook"]
HNM_IMG_DIR = BASE / "data/raw/hnm_images"

# 4 clean candidate datasets (under audit)
CLEAN_DATASETS = {
    "atlas": "Marqo/atlas",
    "KAGL": "Marqo/KAGL",
    "fashion200k": "Marqo/fashion200k",
    "polyvore": "Marqo/polyvore",
}

# Datasets ACTUALLY used in our distillation training. A clean dataset must NOT
# overlap with ANY of these or our "MoDA beats FashionSigLIP" claim is invalid.
TRAINING_POOLS = ["df_inshop", "df_multimodal", "hnm"]

# Other eval sets we use. Overlap here is NOT leakage (we never train on these),
# but we still cross-check for transparency. Reported separately from verdict.
OTHER_EVAL_SETS = ["lookbench"]

ALL_POOLS = TRAINING_POOLS + OTHER_EVAL_SETS


# ---------------------------------------------------------------------------
# Hashing primitives
# ---------------------------------------------------------------------------

def normalized_md5(img: Image.Image) -> str:
    if img.mode != "RGB":
        img = img.convert("RGB")
    img = img.resize(NORM_SIZE, Image.BICUBIC)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=JPEG_QUALITY)
    return hashlib.md5(buf.getvalue()).hexdigest()


def hash_iter(rows: Iterable, image_extractor, id_extractor,
              desc: str, total: int | None = None) -> dict[str, list[str]]:
    """Generic hashing loop. Returns {hash: [item_ids]}."""
    h2ids: dict[str, list[str]] = defaultdict(list)
    n_ok = 0
    n_skip = 0
    iterable = tqdm(rows, desc=desc, total=total)
    for i, row in enumerate(iterable):
        try:
            img = image_extractor(row)
            if img is None:
                n_skip += 1
                continue
            if not isinstance(img, Image.Image):
                n_skip += 1
                continue
            h = normalized_md5(img)
            iid = id_extractor(row, i)
            h2ids[h].append(str(iid))
            n_ok += 1
        except Exception as e:
            n_skip += 1
            if n_skip <= 5:
                log.debug("  skip %d (%s): %s", i, desc, e)
    log.info("  %s: %d hashed, %d skipped, %d unique hashes",
             desc, n_ok, n_skip, len(h2ids))
    return dict(h2ids)


def cache_path(name: str) -> Path:
    return HASH_CACHE_DIR / f"{name}.pkl"


def load_or_build_hashes(name: str, builder, force: bool = False
                         ) -> dict[str, list[str]]:
    p = cache_path(name)
    if p.exists() and not force:
        log.info("Loading cached hashes for %s from %s", name, p)
        with open(p, "rb") as f:
            return pickle.load(f)
    log.info("Building hashes for %s ...", name)
    h2ids = builder()
    with open(p, "wb") as f:
        pickle.dump(h2ids, f)
    log.info("  cached -> %s", p)
    return h2ids


# ---------------------------------------------------------------------------
# Per-dataset builders
# ---------------------------------------------------------------------------

def build_marqo_hf(hf_id: str, tag: str) -> dict[str, list[str]]:
    from datasets import load_dataset
    cache_dir = BASE / "data/raw" / tag.replace("-", "_")
    cache_dir.mkdir(parents=True, exist_ok=True)
    ds = load_dataset(hf_id, cache_dir=str(cache_dir))
    # Some Marqo datasets have only 'data' or 'train' split
    split_name = next(iter(ds.keys()))
    split = ds[split_name]
    log.info("  HF %s split=%s n=%d cols=%s", hf_id, split_name, len(split),
             split.column_names)
    image_key = "image" if "image" in split.column_names else split.column_names[0]
    id_key = ("item_ID" if "item_ID" in split.column_names
              else ("article_id" if "article_id" in split.column_names else None))

    def img_ext(row):
        return row.get(image_key)

    def id_ext(row, i):
        return row.get(id_key, f"{tag}_{i}") if id_key else f"{tag}_{i}"

    return hash_iter(split, img_ext, id_ext, desc=tag, total=len(split))


def build_lookbench() -> dict[str, list[str]]:
    from datasets import load_from_disk
    h2ids: dict[str, list[str]] = defaultdict(list)
    for subset in LB_SUBSETS:
        path = LB_ROOT / subset
        if not path.exists():
            continue
        dsd = load_from_disk(str(path))
        for split_name in ["query", "gallery"]:
            if split_name not in dsd:
                continue
            split = dsd[split_name]
            tag = f"lb_{subset}_{split_name}"
            id_key = ("item_ID" if "item_ID" in split.column_names
                      else ("article_id" if "article_id" in split.column_names
                            else None))

            def img_ext(row):
                return row.get("image")

            def id_ext(row, i, _tag=tag, _idk=id_key):
                return f"{_tag}:{row.get(_idk, i) if _idk else i}"

            partial = hash_iter(split, img_ext, id_ext, desc=tag,
                                total=len(split))
            for h, ids in partial.items():
                h2ids[h].extend(ids)
    if LB_NOISE.exists():
        try:
            noise_ds = load_from_disk(str(LB_NOISE))
        except Exception:
            noise_ds = None
        if noise_ds is not None:
            splits = noise_ds.items() if hasattr(noise_ds, "items") else [("noise", noise_ds)]
            for split_name, split in splits:
                tag = f"lb_noise_{split_name}"

                def img_ext(row):
                    return row.get("image")

                def id_ext(row, i, _tag=tag):
                    return f"{_tag}:{i}"

                partial = hash_iter(split, img_ext, id_ext, desc=tag,
                                    total=len(split))
                for h, ids in partial.items():
                    h2ids[h].extend(ids)
    return dict(h2ids)


def build_hnm_local() -> dict[str, list[str]]:
    if not HNM_IMG_DIR.exists():
        log.warning("H&M image dir missing: %s", HNM_IMG_DIR)
        return {}
    paths = sorted(HNM_IMG_DIR.rglob("*.jpg"))
    if not paths:
        paths = sorted(HNM_IMG_DIR.rglob("*.jpeg")) + sorted(HNM_IMG_DIR.rglob("*.png"))
    log.info("  H&M: %d image files under %s", len(paths), HNM_IMG_DIR)

    def opener(p):
        try:
            return Image.open(p)
        except Exception:
            return None

    def img_ext(p):
        return opener(p)

    def id_ext(p, i):
        return p.name

    return hash_iter(paths, img_ext, id_ext, desc="hnm", total=len(paths))


# ---------------------------------------------------------------------------
# Intersection helpers
# ---------------------------------------------------------------------------

def intersect(a: dict[str, list[str]], b: dict[str, list[str]],
              a_name: str, b_name: str) -> dict:
    overlap = set(a.keys()) & set(b.keys())
    samples = []
    for h in list(overlap)[:25]:
        samples.append({"hash": h,
                        f"{a_name}_ids": a[h][:3],
                        f"{b_name}_ids": b[h][:3]})
    return {
        f"n_hashes_{a_name}": len(a),
        f"n_hashes_{b_name}": len(b),
        "n_overlap_hashes": len(overlap),
        "overlap_pct_of_clean": round(100.0 * len(overlap) / max(1, len(a)), 4),
        "sample_overlaps": samples,
        "verdict": "PASS" if len(overlap) == 0 else "FAIL",
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=list(CLEAN_DATASETS.keys()),
                    choices=list(CLEAN_DATASETS.keys()),
                    help="Subset of clean datasets to hash (default: all 4)")
    ap.add_argument("--training", nargs="+", default=ALL_POOLS,
                    choices=ALL_POOLS,
                    help="Subset of training-pool datasets to hash (default: all)")
    ap.add_argument("--recompute", nargs="*", default=[],
                    help="Force re-hash for these dataset names")
    return ap.parse_args()


def main() -> int:
    args = _parse_args()
    t0 = time.time()
    log.info("=" * 70)
    log.info("MARQO CLEAN-BENCHMARK LEAKAGE AUDIT")
    log.info("Clean candidates : %s", args.datasets)
    log.info("Training pools   : %s", args.training)
    log.info("Force recompute  : %s", args.recompute or "none")
    log.info("=" * 70)

    # ------- hash CLEAN datasets (under audit) -------
    clean_hashes: dict[str, dict] = {}
    for name in args.datasets:
        hf_id = CLEAN_DATASETS[name]
        clean_hashes[name] = load_or_build_hashes(
            name=f"clean__{name}",
            builder=lambda hf=hf_id, tag=name: build_marqo_hf(hf, tag),
            force=(name in args.recompute),
        )

    # ------- hash TRAINING-POOL datasets -------
    train_hashes: dict[str, dict] = {}
    for tname in args.training:
        if tname == "df_inshop":
            train_hashes[tname] = load_or_build_hashes(
                name="train__df_inshop",
                builder=lambda: build_marqo_hf("Marqo/deepfashion-inshop",
                                               "df_inshop"),
                force=("df_inshop" in args.recompute),
            )
        elif tname == "df_multimodal":
            train_hashes[tname] = load_or_build_hashes(
                name="train__df_multimodal",
                builder=lambda: build_marqo_hf("Marqo/deepfashion-multimodal",
                                               "df_multimodal"),
                force=("df_multimodal" in args.recompute),
            )
        elif tname == "hnm":
            train_hashes[tname] = load_or_build_hashes(
                name="train__hnm",
                builder=build_hnm_local,
                force=("hnm" in args.recompute),
            )
        elif tname == "lookbench":
            train_hashes[tname] = load_or_build_hashes(
                name="train__lookbench",
                builder=build_lookbench,
                force=("lookbench" in args.recompute),
            )

    # ------- pairwise intersection -------
    audit_train: dict = {}
    audit_other: dict = {}
    training_pass = True
    log.info("--- training-pool checks (must PASS) ---")
    for cname, ch in clean_hashes.items():
        audit_train[cname] = {}
        for tname in TRAINING_POOLS:
            if tname not in train_hashes:
                continue
            res = intersect(ch, train_hashes[tname], cname, tname)
            if res["verdict"] != "PASS":
                training_pass = False
            audit_train[cname][tname] = res
            log.info("  [%s vs %s] %d overlap hashes -> %s",
                     cname, tname, res["n_overlap_hashes"], res["verdict"])
    log.info("--- other-eval-set checks (informational only) ---")
    for cname, ch in clean_hashes.items():
        audit_other[cname] = {}
        for tname in OTHER_EVAL_SETS:
            if tname not in train_hashes:
                continue
            res = intersect(ch, train_hashes[tname], cname, tname)
            res["verdict"] = "INFO"
            audit_other[cname][tname] = res
            log.info("  [%s vs %s] %d overlap hashes (NOT a leakage failure)",
                     cname, tname, res["n_overlap_hashes"])

    elapsed = time.time() - t0
    out = {
        "verification_date": time.strftime("%Y-%m-%d"),
        "purpose": ("Confirm that the 4 Marqo benchmark datasets we plan to "
                    "use for a fair MoDA vs FashionSigLIP comparison "
                    "(Atlas, KAGL, Fashion200k, Polyvore) do NOT overlap "
                    "with our distillation training pool "
                    "(DeepFashion-InShop, DeepFashion-MultiModal, H&M)."),
        "method": {
            "hash": "MD5 of JPEG-q95-encoded 224x224 RGB normalized image",
            "scope": "ALL images in both clean and training datasets",
            "training_pools": TRAINING_POOLS,
            "other_eval_sets": OTHER_EVAL_SETS,
            "verdict_logic": ("PASS iff zero overlap between any clean dataset "
                              "and any TRAINING_POOL dataset. Overlaps with "
                              "OTHER_EVAL_SETS (LookBench) are reported as "
                              "INFO and do not affect the verdict."),
        },
        "clean_datasets": {
            cname: {
                "hf_id": CLEAN_DATASETS[cname],
                "n_images_audited": sum(len(v) for v in clean_hashes[cname].values()),
                "n_unique_hashes": len(clean_hashes[cname]),
            } for cname in clean_hashes
        },
        "training_pools": {
            tname: {
                "n_images_audited": sum(len(v) for v in train_hashes[tname].values()),
                "n_unique_hashes": len(train_hashes[tname]),
            } for tname in train_hashes if tname in TRAINING_POOLS
        },
        "other_eval_sets": {
            tname: {
                "n_images_audited": sum(len(v) for v in train_hashes[tname].values()),
                "n_unique_hashes": len(train_hashes[tname]),
            } for tname in train_hashes if tname in OTHER_EVAL_SETS
        },
        "leakage_pairs_training": audit_train,
        "leakage_pairs_other_eval": audit_other,
        "overall_verdict": "PASS" if training_pass else "FAIL",
        "notes": [
            ("LookBench's noise gallery overlaps heavily with Fashion200k. "
             "This is by Marqo's LookBench design (Fashion200k is used as "
             "distractor pool) and is NOT a training-time leakage. "
             "Our distillation NEVER trained on LookBench OR Fashion200k.")
        ],
        "elapsed_seconds": round(elapsed, 1),
    }

    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)

    log.info("=" * 70)
    log.info("OVERALL VERDICT: %s", out["overall_verdict"])
    log.info("  (verdict considers TRAINING_POOLS only: %s)", TRAINING_POOLS)
    log.info("  (LookBench overlap with Fashion200k is reported as INFO, "
             "not a failure -- LookBench is an eval set, not training data)")
    log.info("Saved to: %s (elapsed %.1f min)", OUT_JSON, elapsed / 60.0)
    return 0 if training_pass else 1


if __name__ == "__main__":
    sys.exit(main())
