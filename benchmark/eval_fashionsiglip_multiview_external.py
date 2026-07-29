"""Select a zero-extra-parameter FashionSigLIP descriptor on external data.

The model weights remain the official 203M FashionSigLIP weights.  Candidate
recipes aggregate deterministic image views and text prompts into one
normalized 768-dimensional vector.  Selection uses only:

* 4,989 source-disjoint OpenVTON validation products; and
* 2,000 leakage-audited GLAMI development products with three query views.

No Marqo target example, image, qrel, retrieval, or metric is read.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image, ImageOps
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from benchmark.cache_glami_300m_teacher import (
    ParquetImageStore,
    VIEW_NAMES as GLAMI_QUERY_VIEWS,
    load_manifest,
    records_digest,
    sha256_file,
)
from benchmark.distill_mobileclip2_reinforced import (
    DEFAULT_OPENVTON_CACHE,
    OpenVTONValidationStore,
)
from benchmark.eval_legal_fusion import atomic_json, atomic_torch_save
from benchmark.gold_residual_head import retrieval_metrics
from benchmark.train_fsl_s0_glami_additive_300m import (
    FSL_PARAMETERS,
    load_fsl,
)


log = logging.getLogger("fashionsiglip-multiview-external")
DEFAULT_OPENVTON_BASE = (
    REPO / "results/fashionsiglip_vocab_adapter/fsl_validation_base.pt"
)
DEFAULT_GLAMI_MANIFEST = (
    REPO / "results/fsl_s0_glami_300m_pilot/data/DEV_MANIFEST.jsonl"
)
DEFAULT_OUTPUT = REPO / "results/fashionsiglip_multiview_external_v1"
EMBED_DIM = 768
OPENVTON_COUNT = 4_989
GLAMI_COUNT = 2_000
IMAGE_VIEWS = ("official", "pad", "center_crop", "foreground_pad")
PROMPT_VIEWS = ("raw", "photo", "fashion_product")

IMAGE_MIXES: dict[str, dict[str, float]] = {
    "official": {"official": 1.0},
    "pad025": {"official": 1.0, "pad": 0.25},
    "pad050": {"official": 1.0, "pad": 0.50},
    "pad100": {"official": 1.0, "pad": 1.0},
    "crop025": {"official": 1.0, "center_crop": 0.25},
    "crop050": {"official": 1.0, "center_crop": 0.50},
    "foreground025": {"official": 1.0, "foreground_pad": 0.25},
    "foreground050": {"official": 1.0, "foreground_pad": 0.50},
    "foreground100": {"official": 1.0, "foreground_pad": 1.0},
    "pad_crop025": {
        "official": 1.0,
        "pad": 0.25,
        "center_crop": 0.25,
    },
    "pad_foreground025": {
        "official": 1.0,
        "pad": 0.25,
        "foreground_pad": 0.25,
    },
}
PROMPT_MIXES: dict[str, dict[str, float]] = {
    "raw": {"raw": 1.0},
    "photo025": {"raw": 1.0, "photo": 0.25},
    "product025": {"raw": 1.0, "fashion_product": 0.25},
    "both025": {
        "raw": 1.0,
        "photo": 0.25,
        "fashion_product": 0.25,
    },
}
CENTERING_MODES = (False, True)
MINIMUM_MEAN_RELATIVE_MRR_GAIN = 0.005


def square_pad(image: Image.Image, fill: int = 128) -> Image.Image:
    image = image.convert("RGB")
    side = max(image.size)
    canvas = Image.new("RGB", (side, side), (fill, fill, fill))
    canvas.paste(
        image,
        ((side - image.width) // 2, (side - image.height) // 2),
    )
    return canvas


def center_square(image: Image.Image) -> Image.Image:
    side = min(image.size)
    left = (image.width - side) // 2
    top = (image.height - side) // 2
    return image.convert("RGB").crop((left, top, left + side, top + side))


def foreground_square(image: Image.Image) -> Image.Image:
    """Crop a near-white catalog border, then pad without aspect distortion."""
    image = image.convert("RGB")
    preview = image.copy()
    preview.thumbnail((256, 256), Image.Resampling.BILINEAR)
    array = np.asarray(preview, dtype=np.uint8)
    mask = np.any(array < 242, axis=-1)
    if float(mask.mean()) < 0.01:
        return square_pad(image)
    ys, xs = np.nonzero(mask)
    scale_x = image.width / preview.width
    scale_y = image.height / preview.height
    left = max(0, math.floor(float(xs.min()) * scale_x))
    right = min(image.width, math.ceil(float(xs.max() + 1) * scale_x))
    top = max(0, math.floor(float(ys.min()) * scale_y))
    bottom = min(image.height, math.ceil(float(ys.max() + 1) * scale_y))
    margin_x = max(1, round((right - left) * 0.05))
    margin_y = max(1, round((bottom - top) * 0.05))
    cropped = image.crop(
        (
            max(0, left - margin_x),
            max(0, top - margin_y),
            min(image.width, right + margin_x),
            min(image.height, bottom + margin_y),
        )
    )
    return square_pad(cropped)


def image_view(
    image: Image.Image,
    view: str,
    preprocess: Callable[[Image.Image], torch.Tensor],
) -> torch.Tensor:
    if view == "official":
        transformed = image
    elif view == "pad":
        transformed = square_pad(image)
    elif view == "center_crop":
        transformed = center_square(image)
    elif view == "foreground_pad":
        transformed = foreground_square(image)
    else:
        raise ValueError(f"unknown image view: {view}")
    return preprocess(transformed)


def prompted(text: str, view: str) -> str:
    if view == "raw":
        return text
    if view == "photo":
        return f"a photo of {text}"
    if view == "fashion_product":
        return f"a fashion product photo of {text}"
    raise ValueError(f"unknown prompt view: {view}")


def compose(
    features: dict[str, torch.Tensor],
    weights: dict[str, float],
) -> torch.Tensor:
    if not weights or any(value <= 0.0 for value in weights.values()):
        raise ValueError("composition weights must be positive")
    missing = set(weights).difference(features)
    if missing:
        raise KeyError(f"missing feature blocks: {sorted(missing)}")
    value = sum(float(weight) * features[name].float() for name, weight in weights.items())
    return F.normalize(value, dim=-1)


def center_pair(
    query: torch.Tensor,
    document: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    mean = document.float().mean(dim=0, keepdim=True)
    return (
        F.normalize(query.float() - mean, dim=-1),
        F.normalize(document.float() - mean, dim=-1),
    )


def predeclared_recipe(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "artifact_kind": "fashionsiglip_multiview_external_predeclared_recipe",
        "architecture": {
            "model": "Marqo/marqo-fashionSigLIP",
            "neural_parameters": FSL_PARAMETERS,
            "output_dim": EMBED_DIM,
            "stored_vectors_per_item": 1,
            "additional_neural_parameters": 0,
        },
        "development": {
            "openvton_records": OPENVTON_COUNT,
            "glami_records": GLAMI_COUNT,
            "glami_query_views": list(GLAMI_QUERY_VIEWS),
            "target_benchmarks_accessed": False,
        },
        "grid": {
            "image_views": list(IMAGE_VIEWS),
            "prompt_views": list(PROMPT_VIEWS),
            "image_mixes": IMAGE_MIXES,
            "prompt_mixes": PROMPT_MIXES,
            "centering_modes": list(CENTERING_MODES),
            "candidate_count": (
                len(IMAGE_MIXES) * len(PROMPT_MIXES) * len(CENTERING_MODES)
            ),
        },
        "gate": {
            "minimum_mrr_delta_each_external_slice": 0.0,
            "minimum_recall1_delta_each_external_slice": 0.0,
            "minimum_recall10_delta_each_external_slice": 0.0,
            "minimum_mean_relative_mrr_gain": MINIMUM_MEAN_RELATIVE_MRR_GAIN,
            "selection_order": [
                "higher_worst_relative_mrr_gain",
                "higher_mean_relative_mrr_gain",
                "candidate_id",
            ],
        },
        "inputs": {
            "openvton_base": str(Path(args.openvton_base).resolve()),
            "openvton_base_sha256": sha256_file(Path(args.openvton_base)),
            "glami_manifest": str(Path(args.glami_manifest).resolve()),
            "glami_manifest_sha256": sha256_file(Path(args.glami_manifest)),
        },
        "research_only": True,
    }


def encode_text_views(
    model,
    tokenizer,
    values: list[str],
    *,
    device: str,
    batch_size: int,
) -> dict[str, torch.Tensor]:
    output: dict[str, torch.Tensor] = {}
    with torch.inference_mode():
        for view in PROMPT_VIEWS:
            parts = []
            prompted_values = [prompted(value, view) for value in values]
            for start in range(0, len(values), batch_size):
                tokens = tokenizer(
                    prompted_values[start : start + batch_size]
                ).to(device)
                parts.append(
                    F.normalize(model.encode_text(tokens).float(), dim=-1)
                    .half()
                    .cpu()
                )
            output[view] = torch.cat(parts)
    return output


def encode_image_views(
    model,
    preprocess,
    store,
    *,
    source: str,
    output_dir: Path,
    device: str,
    batch_size: int,
    chunk_size: int,
    decode_workers: int,
) -> dict[str, torch.Tensor]:
    source_dir = output_dir / "image_features" / source
    source_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "artifact_kind": "fashionsiglip_multiview_external_image_features",
        "source": source,
        "record_count": len(store),
        "views": list(IMAGE_VIEWS),
        "embedding_dim": EMBED_DIM,
        "model_parameters": FSL_PARAMETERS,
    }
    final_path = source_dir / "features.pt"
    if final_path.exists():
        payload = torch.load(final_path, map_location="cpu", weights_only=False)
        if payload.get("metadata") != metadata:
            raise RuntimeError(f"incompatible completed feature cache: {source}")
        return payload["features"]

    def prepare(index: int) -> dict[str, torch.Tensor]:
        image = store.image(index)
        return {
            view: image_view(image, view, preprocess)
            for view in IMAGE_VIEWS
        }

    pool = ThreadPoolExecutor(max_workers=max(1, decode_workers))
    chunks: dict[str, list[torch.Tensor]] = {view: [] for view in IMAGE_VIEWS}
    try:
        with torch.inference_mode():
            for chunk_start in range(0, len(store), chunk_size):
                chunk_end = min(chunk_start + chunk_size, len(store))
                path = source_dir / f"{chunk_start:07d}_{chunk_end:07d}.pt"
                chunk_meta = {
                    **metadata,
                    "chunk_start": chunk_start,
                    "chunk_end": chunk_end,
                }
                if path.exists():
                    payload = torch.load(
                        path, map_location="cpu", weights_only=False
                    )
                    if payload.get("metadata") != chunk_meta:
                        raise RuntimeError(f"incompatible chunk: {path}")
                    for view in IMAGE_VIEWS:
                        chunks[view].append(payload["features"][view])
                    log.info("%s resumed %d:%d", source, chunk_start, chunk_end)
                    continue
                chunk_parts: dict[str, list[torch.Tensor]] = {
                    view: [] for view in IMAGE_VIEWS
                }
                for start in range(chunk_start, chunk_end, batch_size):
                    end = min(start + batch_size, chunk_end)
                    prepared = list(pool.map(prepare, range(start, end)))
                    for view in IMAGE_VIEWS:
                        images = torch.stack(
                            [row[view] for row in prepared]
                        ).to(device)
                        encoded = F.normalize(
                            model.encode_image(images).float(), dim=-1
                        )
                        chunk_parts[view].append(encoded.half().cpu())
                        del images, encoded
                payload_features = {
                    view: torch.cat(chunk_parts[view])
                    for view in IMAGE_VIEWS
                }
                atomic_torch_save(
                    path,
                    {"metadata": chunk_meta, "features": payload_features},
                )
                for view in IMAGE_VIEWS:
                    chunks[view].append(payload_features[view])
                log.info("%s cached %d:%d", source, chunk_start, chunk_end)
                if device == "mps":
                    torch.mps.empty_cache()
    finally:
        pool.shutdown(wait=True)
    features = {view: torch.cat(chunks[view]) for view in IMAGE_VIEWS}
    atomic_torch_save(
        final_path, {"metadata": metadata, "features": features}
    )
    return features


def metrics(
    query: torch.Tensor,
    document: torch.Tensor,
    labels: torch.Tensor,
) -> dict[str, float]:
    return retrieval_metrics(query, document, labels)


def candidate_id(
    image_mix: str,
    prompt_mix: str,
    centered: bool,
) -> str:
    return (
        f"img-{image_mix}__txt-{prompt_mix}__"
        f"{'centered' if centered else 'raw'}"
    )


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    recipe = predeclared_recipe(args)
    recipe_path = output_dir / "PREDECLARED_RECIPE.json"
    if recipe_path.exists():
        if json.loads(recipe_path.read_text()) != recipe:
            raise RuntimeError("predeclared recipe differs; use a fresh output")
    else:
        atomic_json(recipe_path, recipe)

    openvton = torch.load(
        args.openvton_base, map_location="cpu", weights_only=False
    )
    if (
        tuple(openvton["query"].shape) != (OPENVTON_COUNT, EMBED_DIM)
        or tuple(openvton["document"].shape) != (OPENVTON_COUNT, EMBED_DIM)
        or openvton["metadata"].get("benchmark_metrics_accessed") is not False
    ):
        raise RuntimeError("OpenVTON baseline cache differs")
    rows = load_manifest(Path(args.glami_manifest), max_records=GLAMI_COUNT)
    if len(rows) != GLAMI_COUNT:
        raise RuntimeError("GLAMI development row count differs")

    model, preprocess, tokenizer = load_fsl(args.device)
    openvton_store = OpenVTONValidationStore(
        list(openvton["queries"]),
        list(openvton["product_ids"]),
        Path(args.openvton_cache),
    )
    glami_store = ParquetImageStore(rows)
    openvton_images = encode_image_views(
        model,
        preprocess,
        openvton_store,
        source="openvton",
        output_dir=output_dir,
        device=args.device,
        batch_size=args.batch_size,
        chunk_size=args.chunk_size,
        decode_workers=args.decode_workers,
    )
    glami_images = encode_image_views(
        model,
        preprocess,
        glami_store,
        source="glami",
        output_dir=output_dir,
        device=args.device,
        batch_size=args.batch_size,
        chunk_size=args.chunk_size,
        decode_workers=args.decode_workers,
    )
    openvton_text = encode_text_views(
        model,
        tokenizer,
        [str(value) for value in openvton["queries"]],
        device=args.device,
        batch_size=args.batch_size,
    )
    glami_text = {
        view: encode_text_views(
            model,
            tokenizer,
            [str(row["text_views"][view]) for row in rows],
            device=args.device,
            batch_size=args.batch_size,
        )
        for view in GLAMI_QUERY_VIEWS
    }
    atomic_torch_save(
        output_dir / "text_features.pt",
        {
            "metadata": {
                "artifact_kind": "fashionsiglip_multiview_external_text_features",
                "prompt_views": list(PROMPT_VIEWS),
                "openvton_count": OPENVTON_COUNT,
                "glami_count": GLAMI_COUNT,
                "glami_records_digest": records_digest(rows),
                "target_benchmarks_accessed": False,
            },
            "openvton": openvton_text,
            "glami": glami_text,
        },
    )

    labels_openvton = openvton["labels"].long()
    labels_glami = torch.arange(GLAMI_COUNT)
    base_openvton = metrics(
        openvton_text["raw"],
        openvton_images["official"],
        labels_openvton,
    )
    base_glami = {
        view: metrics(
            glami_text[view]["raw"],
            glami_images["official"],
            labels_glami,
        )
        for view in GLAMI_QUERY_VIEWS
    }
    baselines = {"openvton": base_openvton, **base_glami}

    scores = []
    for image_name, image_weights in IMAGE_MIXES.items():
        openvton_document = compose(openvton_images, image_weights)
        glami_document = compose(glami_images, image_weights)
        for prompt_name, prompt_weights in PROMPT_MIXES.items():
            openvton_query = compose(openvton_text, prompt_weights)
            glami_query = {
                view: compose(glami_text[view], prompt_weights)
                for view in GLAMI_QUERY_VIEWS
            }
            for centered in CENTERING_MODES:
                ov_q, ov_d = openvton_query, openvton_document
                if centered:
                    ov_q, ov_d = center_pair(ov_q, ov_d)
                observed = {
                    "openvton": metrics(ov_q, ov_d, labels_openvton)
                }
                for view in GLAMI_QUERY_VIEWS:
                    gq, gd = glami_query[view], glami_document
                    if centered:
                        gq, gd = center_pair(gq, gd)
                    observed[view] = metrics(gq, gd, labels_glami)
                deltas = {
                    source: {
                        key: observed[source][key] - baselines[source][key]
                        for key in ("mrr@10", "recall@1", "recall@10")
                    }
                    for source in observed
                }
                relative_mrr = {
                    source: (
                        observed[source]["mrr@10"] / baselines[source]["mrr@10"]
                        - 1.0
                    )
                    for source in observed
                }
                eligible = (
                    min(value["mrr@10"] for value in deltas.values()) >= 0.0
                    and min(value["recall@1"] for value in deltas.values()) >= 0.0
                    and min(value["recall@10"] for value in deltas.values()) >= 0.0
                    and float(np.mean(list(relative_mrr.values())))
                    >= MINIMUM_MEAN_RELATIVE_MRR_GAIN
                )
                scores.append(
                    {
                        "candidate_id": candidate_id(
                            image_name, prompt_name, centered
                        ),
                        "image_mix": image_name,
                        "prompt_mix": prompt_name,
                        "centered": centered,
                        "metrics": observed,
                        "deltas": deltas,
                        "relative_mrr": relative_mrr,
                        "worst_relative_mrr_gain": min(
                            relative_mrr.values()
                        ),
                        "mean_relative_mrr_gain": float(
                            np.mean(list(relative_mrr.values()))
                        ),
                        "eligible": eligible,
                    }
                )
    eligible = [row for row in scores if row["eligible"]]
    eligible.sort(
        key=lambda row: (
            -row["worst_relative_mrr_gain"],
            -row["mean_relative_mrr_gain"],
            row["candidate_id"],
        )
    )
    selected = eligible[0] if eligible else None
    summary = {
        "schema_version": 1,
        "artifact_kind": "fashionsiglip_multiview_external_selection",
        "status": (
            "EXTERNAL_GATE_PASSED_CANDIDATE_LOCK_REQUIRED"
            if selected
            else "STOP_NO_EXTERNALLY_SAFE_MULTIVIEW_CANDIDATE"
        ),
        "deployment": recipe["architecture"],
        "baselines": baselines,
        "candidate_count": len(scores),
        "eligible_candidate_count": len(eligible),
        "selected_candidate": selected,
        "scores": scores,
        "target_benchmarks_accessed": False,
        "research_only": True,
    }
    atomic_json(output_dir / "SELECTION.json", summary)
    log.info(
        "%s (%d/%d eligible)",
        summary["status"],
        len(eligible),
        len(scores),
    )
    if selected:
        log.info(
            "selected %s worst MRR %+.3f%% mean %+.3f%%",
            selected["candidate_id"],
            100 * selected["worst_relative_mrr_gain"],
            100 * selected["mean_relative_mrr_gain"],
        )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--openvton-base", default=str(DEFAULT_OPENVTON_BASE))
    parser.add_argument("--openvton-cache", default=str(DEFAULT_OPENVTON_CACHE))
    parser.add_argument("--glami-manifest", default=str(DEFAULT_GLAMI_MANIFEST))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--chunk-size", type=int, default=1_024)
    parser.add_argument("--decode-workers", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    evaluate(parse_args())


if __name__ == "__main__":
    main()
