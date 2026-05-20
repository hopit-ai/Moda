"""
Local LookBench layouts under MODA (Phase 4.3 / 4.4).

Reads Hugging Face ``save_to_disk`` trees produced by ``scripts/download_lookbench.py``.
Optional: merge the ``noise`` distractor split into the gallery (paper-style corpus).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

_REPO = Path(__file__).resolve().parent.parent
DEFAULT_DATASETS_ROOT = _REPO / "data" / "raw" / "lookbench" / "datasets"

SUBSET_ALIASES: dict[str, str] = {
    "RealStudioFlat": "real_studio_flat",
    "real_studio_flat": "real_studio_flat",
    "AIGen-Studio": "aigen_studio",
    "aigen_studio": "aigen_studio",
    "RealStreetLook": "real_streetlook",
    "real_streetlook": "real_streetlook",
    "AIGen-StreetLook": "aigen_streetlook",
    "aigen_streetlook": "aigen_streetlook",
}


def resolve_subset(name: str) -> str:
    key = name.strip()
    if key in SUBSET_ALIASES:
        return SUBSET_ALIASES[key]
    raise ValueError(f"Unknown subset {name!r}. Known: {sorted(set(SUBSET_ALIASES.values()))}")


def load_subset_dict(
    subset: str,
    datasets_root: Path | None = None,
    include_noise_in_gallery: bool = True,
) -> dict[str, Any]:
    """Load query + gallery (+ optional noise) as HuggingFace ``Dataset`` objects.

    Returns:
        dict with keys ``query``, ``gallery``, ``subset``, ``noise_rows`` (int).
    """
    from datasets import Dataset, concatenate_datasets, load_from_disk

    root = Path(datasets_root or DEFAULT_DATASETS_ROOT)
    cfg = resolve_subset(subset)
    path = root / cfg
    if not path.is_dir():
        raise FileNotFoundError(
            f"LookBench split not found: {path}\n"
            "Run: python scripts/download_lookbench.py"
        )

    ddict = load_from_disk(str(path))
    query = ddict["query"]
    gallery = ddict["gallery"]

    noise_n = 0
    if include_noise_in_gallery:
        noise_path = root / "noise"
        if noise_path.is_dir():
            noise_ds = load_from_disk(str(noise_path))
            if "gallery" in noise_ds:
                ng = noise_ds["gallery"]
                noise_n = len(ng)
                gallery = concatenate_datasets([gallery, ng])
                log.info("Merged %d noise distractors into gallery", noise_n)
        else:
            log.warning("Noise split not found at %s — gallery has no distractors", noise_path)

    return {
        "query": query,
        "gallery": gallery,
        "subset": cfg,
        "noise_rows": noise_n,
    }


def subset_rows_summary(bundle: dict[str, Any]) -> dict[str, int]:
    return {
        "query": len(bundle["query"]),
        "gallery": len(bundle["gallery"]),
        "noise_merged": bundle["noise_rows"],
    }
