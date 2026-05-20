"""
Phase 4.3 — Download and prepare LookBench (Tier 3).

1. Pulls all benchmark subsets from Hugging Face (srpone/look-bench) and saves
   them with datasets.save_to_disk() under data/raw/lookbench/datasets/<config>/.
2. Optionally shallow-clones the official evaluation repo into
   data/raw/lookbench/look-bench/.

Prereq: pip install datasets huggingface_hub tqdm

Usage:
  python scripts/download_lookbench.py
  python scripts/download_lookbench.py --dry-run
  python scripts/download_lookbench.py --subsets real_studio_flat aigen_studio
  python scripts/download_lookbench.py --no-clone
  python scripts/download_lookbench.py --no-hf
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

_REPO = Path(__file__).resolve().parent.parent
OUT_ROOT = _REPO / "data" / "raw" / "lookbench"
DATASETS_DIR = OUT_ROOT / "datasets"
CODE_DIR = OUT_ROOT / "look-bench"
MANIFEST_PATH = _REPO / "results" / "real" / "lookbench_download_manifest.json"

LOOKBENCH_GIT = "https://github.com/SerendipityOneInc/look-bench.git"

# HF configs (from datasets.get_dataset_config_names("srpone/look-bench")).
# "noise" = distractor pool used in gallery construction; include for faithful repro.
DEFAULT_SUBSETS = (
    "real_studio_flat",
    "aigen_studio",
    "real_streetlook",
    "aigen_streetlook",
    "noise",
)


def _summarize_saved(path: Path) -> dict:
    from datasets import load_from_disk

    root = load_from_disk(str(path))
    out: dict = {"path": str(path.relative_to(_REPO))}
    if hasattr(root, "keys"):
        out["splits"] = {}
        for k in root.keys():
            sp = root[k]
            out["splits"][k] = {"rows": len(sp), "columns": sp.column_names}
    else:
        out["rows"] = len(root)
        out["columns"] = root.column_names
    return out


def download_hf_subsets(subsets: tuple[str, ...], dry_run: bool) -> list[dict]:
    from datasets import load_dataset

    results = []
    for cfg in subsets:
        dest = DATASETS_DIR / cfg
        if dry_run:
            log.info("[dry-run] Would load HF srpone/look-bench[%s] → %s", cfg, dest)
            results.append({"config": cfg, "status": "dry_run", "path": str(dest)})
            continue
        if dest.exists():
            log.info("Skip %s (already exists: %s)", cfg, dest)
            try:
                results.append({"config": cfg, "status": "cached", **_summarize_saved(dest)})
            except Exception as e:
                results.append({"config": cfg, "status": "cached", "note": str(e)})
            continue
        log.info("Downloading srpone/look-bench config=%s ...", cfg)
        ds = load_dataset("srpone/look-bench", cfg)
        dest.parent.mkdir(parents=True, exist_ok=True)
        ds.save_to_disk(str(dest))
        log.info("Saved → %s", dest)
        results.append({"config": cfg, "status": "ok", **_summarize_saved(dest)})
    return results


def clone_official_repo(dry_run: bool) -> dict:
    if dry_run:
        log.info("[dry-run] Would git clone %s → %s", LOOKBENCH_GIT, CODE_DIR)
        return {"status": "dry_run", "path": str(CODE_DIR)}

    if CODE_DIR.exists() and any(CODE_DIR.iterdir()):
        log.info("Skip clone (non-empty %s)", CODE_DIR)
        return {"status": "skipped", "path": str(CODE_DIR.relative_to(_REPO))}

    CODE_DIR.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "git",
        "clone",
        "--depth",
        "1",
        LOOKBENCH_GIT,
        str(CODE_DIR),
    ]
    log.info("Running: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)
    return {"status": "ok", "path": str(CODE_DIR.relative_to(_REPO))}


def main() -> int:
    p = argparse.ArgumentParser(description="Download LookBench data + optional eval repo")
    p.add_argument(
        "--subsets",
        nargs="*",
        default=list(DEFAULT_SUBSETS),
        help=f"HF configs to save (default: all). Choices include {DEFAULT_SUBSETS}",
    )
    p.add_argument("--no-hf", action="store_true", help="Skip Hugging Face dataset download")
    p.add_argument("--no-clone", action="store_true", help="Skip git clone of look-bench repo")
    p.add_argument("--dry-run", action="store_true", help="Print plan only")
    args = p.parse_args()
    subsets = tuple(args.subsets)

    manifest: dict = {
        "out_root": str(OUT_ROOT.relative_to(_REPO)),
        "hf_dataset_id": "srpone/look-bench",
        "official_repo": LOOKBENCH_GIT,
        "subsets_requested": list(subsets),
    }

    if not args.no_hf:
        manifest["hf_downloads"] = download_hf_subsets(subsets, args.dry_run)
    else:
        manifest["hf_downloads"] = []

    if not args.no_clone:
        manifest["git_clone"] = clone_official_repo(args.dry_run)
    else:
        manifest["git_clone"] = {"status": "skipped"}

    if not args.dry_run:
        MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(MANIFEST_PATH, "w") as f:
            json.dump(manifest, f, indent=2)
        log.info("Wrote manifest → %s", MANIFEST_PATH)

    log.info("Done. Next: cd %s && pip install -e .  (then follow their README / main.py)", CODE_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
