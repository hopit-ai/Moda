"""
Phase 4.4 — LookBench baselines inside MODA (vision-only, image→image).

Uses local ``save_to_disk`` trees (``benchmark/lookbench_local.py``) and open_clip
models already used in Tier 1 (Marqo FashionCLIP / SigLIP, generic CLIP-B/32).

Metrics (per query, then macro-average, percentages 0–100):
  * **instance_recall@K** — any top-K gallery row with same ``item_ID`` as the query
  * **coarse@1** — top-1 gallery row has same ``category`` as the query
  * **fine@1** — top-1 row matches ``category``, ``main_attribute``, and ``other_attributes``
  * **ndcg@5** — binary relevance for gallery rows that share the query ``item_ID``

These align with the benchmark’s *retrieval* difficulty but are **not** a byte-for-byte
copy of the official LookBench leaderboard code (which lives in the cloned repo).
For leaderboard-identical numbers, run their ``main.py`` after parquet export.

Usage:
  python benchmark/eval_lookbench_baseline.py
  python benchmark/eval_lookbench_baseline.py --subsets real_studio_flat --models fashion-clip
  python benchmark/eval_lookbench_baseline.py --no-noise --batch_size 32
  python benchmark/eval_lookbench_baseline.py --skip-existing  # resume without re-running finished (subset, model) pairs
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from benchmark.lookbench_local import DEFAULT_DATASETS_ROOT, load_subset_dict, subset_rows_summary
from benchmark.models import encode_pil_images_clip, load_clip_model, load_clip_model_from_checkpoint

_4F_CHECKPOINT = _REPO / "models" / "moda-fashionclip-multimodal" / "best" / "model_state_dict.pt"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

RESULTS_DIR = _REPO / "results" / "real"

MODEL_CHOICES = ("fashion-clip", "fashion-siglip", "clip", "fashion-clip-4f")


def _encode_split(ds, model, preprocess, device: str, batch_size: int) -> np.ndarray:
    images = [ds[i]["image"] for i in range(len(ds))]
    return encode_pil_images_clip(
        images, model, preprocess, device, batch_size=batch_size, normalize=True
    )


def _row_attrs(ds, idx: int) -> tuple[str, str, str, str]:
    r = ds[idx]
    return (
        str(r["item_ID"]),
        str(r.get("category", "")),
        str(r.get("main_attribute", "")),
        str(r.get("other_attributes", "")),
    )


def _gallery_positive_mask(
    q_item: str, g_items: list[str]
) -> np.ndarray:
    return np.array([g == q_item for g in g_items], dtype=np.float64)


def _ndcg_at_k_from_ranking(relevant_mask: np.ndarray, sorted_idx: np.ndarray, k: int) -> float:
    rel = relevant_mask[sorted_idx[:k]]
    dcg = np.sum(rel / np.log2(np.arange(2, len(rel) + 2)))
    ideal = np.sort(relevant_mask)[::-1][:k]
    idcg = np.sum(ideal / np.log2(np.arange(2, len(ideal) + 2)))
    if idcg <= 0:
        return 0.0
    return float(dcg / idcg)


def evaluate_subset(
    subset: str,
    model_key: str,
    device: str,
    batch_size: int,
    include_noise: bool,
    datasets_root: Path | None,
    ks: tuple[int, ...] = (1, 5, 10, 20),
) -> dict:
    bundle = load_subset_dict(
        subset, datasets_root=datasets_root, include_noise_in_gallery=include_noise
    )
    q_ds = bundle["query"]
    g_ds = bundle["gallery"]

    g_items = [str(g_ds[i]["item_ID"]) for i in range(len(g_ds))]
    g_cat = [str(g_ds[i].get("category", "")) for i in range(len(g_ds))]
    g_main = [str(g_ds[i].get("main_attribute", "")) for i in range(len(g_ds))]
    g_other = [str(g_ds[i].get("other_attributes", "")) for i in range(len(g_ds))]

    log.info("Loading open_clip model %s ...", model_key)
    if model_key == "fashion-clip-4f":
        if not _4F_CHECKPOINT.is_file():
            raise FileNotFoundError(f"Phase 4F checkpoint not found: {_4F_CHECKPOINT}")
        model, preprocess, _ = load_clip_model_from_checkpoint(
            str(_4F_CHECKPOINT), base_model="fashion-clip", device=device,
        )
    else:
        model, preprocess, _ = load_clip_model(model_key, device=device)

    log.info("Encoding gallery (%d images)...", len(g_ds))
    g_emb = _encode_split(g_ds, model, preprocess, device, batch_size)

    log.info("Encoding queries (%d images)...", len(q_ds))
    q_emb = _encode_split(q_ds, model, preprocess, device, batch_size)

    sim = q_emb @ g_emb.T
    n_q, n_g = sim.shape

    instance_hits = {k: [] for k in ks}
    coarse_hits = []
    fine_hits = []
    ndcgs = []

    for i in range(n_q):
        q_item, q_cat, q_main, q_other = _row_attrs(q_ds, i)
        pos_mask = _gallery_positive_mask(q_item, g_items)
        order = np.argsort(-sim[i])

        for k in ks:
            topk = set(order[:k].tolist())
            hit = any(pos_mask[j] > 0 for j in topk)
            instance_hits[k].append(1.0 if hit else 0.0)

        j1 = int(order[0])
        coarse_hits.append(1.0 if g_cat[j1] == q_cat else 0.0)
        fine_hits.append(
            1.0
            if (
                g_cat[j1] == q_cat
                and g_main[j1] == q_main
                and g_other[j1] == q_other
            )
            else 0.0
        )
        ndcgs.append(_ndcg_at_k_from_ranking(pos_mask, order, 5))

    metrics: dict[str, float] = {}
    for k in ks:
        metrics[f"instance_recall@{k}"] = round(100.0 * float(np.mean(instance_hits[k])), 2)
    metrics["coarse@1"] = round(100.0 * float(np.mean(coarse_hits)), 2)
    metrics["fine@1"] = round(100.0 * float(np.mean(fine_hits)), 2)
    metrics["ndcg@5"] = round(100.0 * float(np.mean(ndcgs)), 2)

    return {
        "subset": bundle["subset"],
        "model": model_key,
        "rows": subset_rows_summary(bundle),
        "metrics": metrics,
    }


def main() -> int:
    p = argparse.ArgumentParser(description="LookBench vision baselines (MODA)")
    p.add_argument(
        "--datasets-root",
        type=Path,
        default=None,
        help=f"Default: {DEFAULT_DATASETS_ROOT}",
    )
    p.add_argument(
        "--subsets",
        nargs="+",
        default=[
            "real_studio_flat",
            "aigen_studio",
            "real_streetlook",
            "aigen_streetlook",
        ],
        help="LookBench configs (local folder names)",
    )
    p.add_argument(
        "--models",
        nargs="+",
        default=list(MODEL_CHOICES),
        choices=MODEL_CHOICES,
    )
    p.add_argument("--no-noise", action="store_true", help="Do not merge noise distractors")
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="If lookbench_moda_baseline.json exists, keep its runs and only evaluate missing (subset, model) pairs",
    )
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--device", type=str, default=None)
    args = p.parse_args()

    device = args.device or (
        "cuda"
        if __import__("torch").cuda.is_available()
        else ("mps" if __import__("torch").backends.mps.is_available() else "cpu")
    )
    log.info("Device: %s", device)

    include_noise = not args.no_noise
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "lookbench_moda_baseline.json"

    all_results: dict = {
        "settings": {
            "datasets_root": str(args.datasets_root or DEFAULT_DATASETS_ROOT),
            "include_noise": include_noise,
            "device": device,
            "batch_size": args.batch_size,
        },
        "runs": [],
    }

    done: set[tuple[str, str]] = set()
    if args.skip_existing and out.is_file():
        try:
            prev = json.loads(out.read_text())
            prev_settings = prev.get("settings") or {}
            if prev_settings.get("include_noise") != include_noise:
                log.warning(
                    "--skip-existing: existing file has include_noise=%s but this run has %s — "
                    "mixing may be misleading",
                    prev_settings.get("include_noise"),
                    include_noise,
                )
            for r in prev.get("runs", []):
                if "metrics" in r and "subset" in r and "model" in r:
                    done.add((str(r["subset"]), str(r["model"])))
            all_results["runs"] = list(prev.get("runs", []))
            log.info("--skip-existing: loaded %d completed runs from %s", len(done), out)
        except (json.JSONDecodeError, OSError) as e:
            log.warning("Could not read %s for skip-existing: %s", out, e)

    for subset in args.subsets:
        for mk in args.models:
            if (subset, mk) in done:
                log.info("=== %s | %s === (skip-existing)", subset, mk)
                continue
            log.info("=== %s | %s ===", subset, mk)
            try:
                block = evaluate_subset(
                    subset,
                    mk,
                    device,
                    args.batch_size,
                    include_noise,
                    args.datasets_root,
                )
                all_results["runs"].append(block)
                log.info("Metrics: %s", block["metrics"])
            except Exception as e:
                log.exception("Failed %s %s: %s", subset, mk, e)
                all_results["runs"].append(
                    {"subset": subset, "model": mk, "error": str(e)}
                )
    out.write_text(json.dumps(all_results, indent=2))
    log.info("Wrote %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
