"""
Phase 0: Download pre-trained models from HuggingFace.
Downloads: CLIP ViT-B/32, FashionCLIP, FashionSigLIP
"""

import logging
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
LOG_FILE = BASE_DIR / "logs" / "phase0.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="a"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

MODELS = [
    {"hf_id": "Marqo/marqo-fashionSigLIP",     "local_name": "marqo-fashionSigLIP"},
    {"hf_id": "Marqo/marqo-fashionCLIP",        "local_name": "marqo-fashionCLIP"},
    {"hf_id": "openai/clip-vit-base-patch32",   "local_name": "clip-vit-base-patch32"},
]


def download_model(hf_id: str, local_name: str) -> dict:
    out_dir = MODELS_DIR / local_name
    result = {"hf_id": hf_id, "local_name": local_name, "status": "unknown"}

    try:
        log.info(f"Downloading model {hf_id} → {out_dir}")
        path = snapshot_download(repo_id=hf_id, local_dir=str(out_dir))
        files = list(Path(path).glob("*"))
        result["status"] = "success"
        result["path"] = str(path)
        result["files"] = [f.name for f in files]
        log.info(f"  ✓ {hf_id} → {path}")
        log.info(f"    Files: {[f.name for f in files]}")
    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
        log.error(f"  ✗ FAILED {hf_id}: {e}")

    return result


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

    log.info("=" * 60)
    log.info("Phase 0: Model Download")
    log.info("=" * 60)

    results = []
    for m in MODELS:
        result = download_model(m["hf_id"], m["local_name"])
        results.append(result)

    log.info("\n" + "=" * 60)
    log.info("MODEL DOWNLOAD SUMMARY")
    success = [r for r in results if r["status"] == "success"]
    failed = [r for r in results if r["status"] == "error"]
    log.info(f"  Succeeded: {len(success)}/{len(MODELS)}")
    for r in success:
        log.info(f"    ✓ {r['hf_id']}")
    if failed:
        for r in failed:
            log.info(f"    ✗ {r['hf_id']}: {r.get('error')}")

    return len(failed) == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
