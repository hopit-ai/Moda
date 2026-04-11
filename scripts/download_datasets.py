"""
Phase 0: Download Batch 1 datasets from HuggingFace.
Skips iMaterialist (71.5GB) and marqo-GS-10M (100GB+) — deferred to Phase 3.
"""

import json
import logging
import sys
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
LOG_FILE = BASE_DIR / "logs" / "phase0.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

DATASETS = [
    {"hf_id": "Marqo/deepfashion-inshop",    "local_name": "deepfashion_inshop"},
    {"hf_id": "Marqo/deepfashion-multimodal", "local_name": "deepfashion_multimodal"},
    {"hf_id": "Marqo/fashion200k",            "local_name": "fashion200k"},
    {"hf_id": "Marqo/atlas",                  "local_name": "atlas"},
    {"hf_id": "Marqo/polyvore",               "local_name": "polyvore"},
    {"hf_id": "microsoft/hnm-search-data",    "local_name": "hnm"},
]


def download_dataset(hf_id: str, local_name: str) -> dict:
    out_dir = RAW_DIR / local_name
    result = {"hf_id": hf_id, "local_name": local_name, "status": "unknown", "splits": {}}

    try:
        log.info(f"Downloading {hf_id} → {out_dir}")
        ds = load_dataset(hf_id, cache_dir=str(out_dir))

        for split_name, split_data in ds.items():
            rows = len(split_data)
            cols = split_data.column_names
            result["splits"][split_name] = {"rows": rows, "columns": cols}
            log.info(f"  [{split_name}] {rows:,} rows | columns: {cols}")

        sample_path = RAW_DIR / f"{local_name}_sample.json"
        first_split = next(iter(ds.values()))
        sample = [first_split[i] for i in range(min(5, len(first_split)))]

        def make_serializable(obj):
            if isinstance(obj, bytes):
                return "<bytes>"
            if hasattr(obj, "tolist"):
                return obj.tolist()
            return str(obj)

        with open(sample_path, "w") as f:
            json.dump(sample, f, indent=2, default=make_serializable)

        result["status"] = "success"
        log.info(f"  Sample saved to {sample_path}")

    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
        log.error(f"  FAILED {hf_id}: {e}")

    return result


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

    log.info("=" * 60)
    log.info("Phase 0: Dataset Download (Batch 1)")
    log.info("Skipping: iMaterialist (71.5GB), marqo-GS-10M (100GB+)")
    log.info("=" * 60)

    results = []
    for ds_config in tqdm(DATASETS, desc="Datasets"):
        result = download_dataset(ds_config["hf_id"], ds_config["local_name"])
        results.append(result)

    summary_path = BASE_DIR / "logs" / "download_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)

    log.info("\n" + "=" * 60)
    log.info("DOWNLOAD SUMMARY")
    log.info("=" * 60)
    success = [r for r in results if r["status"] == "success"]
    failed = [r for r in results if r["status"] == "error"]
    log.info(f"  Succeeded: {len(success)}/{len(DATASETS)}")
    for r in success:
        total_rows = sum(s["rows"] for s in r["splits"].values())
        log.info(f"    ✓ {r['hf_id']} ({total_rows:,} rows total)")
    if failed:
        log.info(f"  Failed: {len(failed)}")
        for r in failed:
            log.info(f"    ✗ {r['hf_id']}: {r.get('error', 'unknown error')}")

    log.info(f"\nFull summary saved to {summary_path}")
    return len(failed) == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
