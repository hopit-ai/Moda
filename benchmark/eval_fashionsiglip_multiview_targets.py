"""Locked target evaluation of the externally selected 203M multi-view recipe.

The selected descriptor is:

    text  = normalize(raw + 0.25 * "a fashion product photo of {raw}")
    image = normalize(official + 0.25 * aspect_pad + 0.25 * center_crop)

It uses the official FashionSigLIP weights, zero additional neural parameters,
and one stored 768-dimensional vector.  Existing official image/text features
are reused only for the official view; pad/crop and prompted text are freshly
encoded from raw inputs.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "repos/marqo-FashionCLIP"))

from benchmark.cache_glami_300m_teacher import sha256_file
from benchmark.eval_fashionsiglip_multiview_external import (
    EMBED_DIM,
    FSL_PARAMETERS,
    IMAGE_MIXES,
    PROMPT_MIXES,
    image_view,
    prompted,
)
from benchmark.eval_fashionsiglip_r36_nearmiss_targets import streaming_topk
from benchmark.eval_fsl_s0_glami_300m import (
    BOOTSTRAP_SAMPLES,
    BOOTSTRAP_SEED,
    DATASETS,
    load_raw_image_dataset,
    retrieval_from_topk,
    stable_list_digest,
)
from benchmark.eval_legal_fusion import (
    _stable_seed,
    atomic_json,
    atomic_torch_save,
    load_official_fsl_baseline,
    load_or_encode_query_embeddings,
    paired_bootstrap,
    per_query_ap10,
)
from benchmark.fuse_and_eval import (
    evaluate_with_beir,
    load_image_embeddings,
    reconstruct_corpus_full,
)
from benchmark.train_fashionsiglip_dual_encoder_v4 import (
    load_pinned_tokenizer,
    verify_base_model_artifacts,
)


log = logging.getLogger("fashionsiglip-multiview-targets")
DEFAULT_EXTERNAL_DIR = REPO / "results/fashionsiglip_multiview_external_v1"
DEFAULT_SELECTION = DEFAULT_EXTERNAL_DIR / "SELECTION.json"
DEFAULT_RECIPE = DEFAULT_EXTERNAL_DIR / "PREDECLARED_RECIPE.json"
DEFAULT_QUERY_CACHE = REPO / "results/gold_three_tower_refine/cache"
DEFAULT_OUTPUT = REPO / "results/fashionsiglip_multiview_target_iteration5_sizeordered"
SELECTED_ID = "img-pad_crop025__txt-product025__raw"
IMAGE_WEIGHTS = {"official": 1.0, "pad": 0.25, "center_crop": 0.25}
TEXT_WEIGHTS = {"raw": 1.0, "fashion_product": 0.25}
EVALUATION_ORDER = (
    "deepfashion_multimodal",
    "KAGL",
    "deepfashion_inshop",
    "atlas",
    "polyvore",
    "fashion200k",
)
MINIMUM_SIGNIFICANT_WINS = 4
LOCK_NAME = "CANDIDATE_LOCK.json"
SUMMARY_NAME = "FINAL_SUMMARY.json"
REPORT_NAME = "FINAL_REPORT.md"


def model_weights_path() -> Path:
    root = REPO / "data/hf_cache/models--Marqo--marqo-fashionSigLIP/snapshots"
    paths = list(root.glob("*/open_clip_model.safetensors"))
    if len(paths) != 1:
        raise RuntimeError("FashionSigLIP local weight binding is ambiguous")
    return paths[0].resolve()


def load_locked_fsl(device: str):
    """Load model and tokenizer entirely from hash-pinned local snapshots."""
    import open_clip

    cache_dir = REPO / "data/hf_cache"
    artifacts = verify_base_model_artifacts(cache_dir)
    model, _, preprocess = open_clip.create_model_and_transforms(
        artifacts["load_name"],
        cache_dir=str(cache_dir),
    )
    tokenizer = load_pinned_tokenizer(cache_dir)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != FSL_PARAMETERS:
        raise RuntimeError("local FashionSigLIP parameter count differs")
    model.eval().to(device)
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model, preprocess, tokenizer


def create_lock(
    output_dir: Path,
    selection_path: Path,
    recipe_path: Path,
) -> dict[str, Any]:
    selection = json.loads(selection_path.read_text())
    recipe = json.loads(recipe_path.read_text())
    selected = selection.get("selected_candidate")
    if (
        selection.get("status")
        != "EXTERNAL_GATE_PASSED_CANDIDATE_LOCK_REQUIRED"
        or selection.get("target_benchmarks_accessed") is not False
        or not isinstance(selected, dict)
        or selected.get("candidate_id") != SELECTED_ID
        or selected.get("eligible") is not True
        or selected.get("image_mix") != "pad_crop025"
        or selected.get("prompt_mix") != "product025"
        or selected.get("centered") is not False
        or recipe.get("development", {}).get("target_benchmarks_accessed")
        is not False
    ):
        raise RuntimeError("external multi-view selection boundary differs")
    if IMAGE_MIXES["pad_crop025"] != IMAGE_WEIGHTS:
        raise RuntimeError("selected image recipe differs from evaluation code")
    if PROMPT_MIXES["product025"] != TEXT_WEIGHTS:
        raise RuntimeError("selected text recipe differs from evaluation code")
    for values in selected["deltas"].values():
        if min(
            values[key] for key in ("mrr@10", "recall@1", "recall@10")
        ) < 0.0:
            raise RuntimeError("selected recipe regresses an external metric")
    weights = model_weights_path()
    payload = {
        "schema_version": 1,
        "artifact_kind": "fashionsiglip_multiview_iteration5_candidate_lock",
        "status": "LOCKED_BEFORE_ITERATION5_TARGET_ACCESS",
        "candidate": {
            "candidate_id": SELECTED_ID,
            "model": "Marqo/marqo-fashionSigLIP",
            "model_weights": str(weights),
            "model_weights_sha256": sha256_file(weights),
            "neural_parameters": FSL_PARAMETERS,
            "additional_neural_parameters": 0,
            "output_dim": EMBED_DIM,
            "stored_vectors_per_item": 1,
            "image_weights": IMAGE_WEIGHTS,
            "text_weights": TEXT_WEIGHTS,
            "corpus_centering": False,
        },
        "external_selection": {
            "selection": str(selection_path.resolve()),
            "selection_sha256": sha256_file(selection_path),
            "recipe": str(recipe_path.resolve()),
            "recipe_sha256": sha256_file(recipe_path),
            "eligible_candidate_count": selection["eligible_candidate_count"],
            "selected_metrics": selected,
        },
        "protocol": {
            "datasets": list(DATASETS),
            "evaluation_order": list(EVALUATION_ORDER),
            "task": "text-to-image",
            "full_corpus": True,
            "primary_metric": "MAP@10",
            "paired_bootstrap_samples": BOOTSTRAP_SAMPLES,
            "minimum_significant_wins": MINIMUM_SIGNIFICANT_WINS,
            "same_global_recipe_all_datasets": True,
            "early_stop_when_four_wins_impossible": True,
        },
        "disclosure": {
            "fresh_blind_sota_evaluation": False,
            "benchmark_iteration": 5,
            "prior_aggregate_target_results_informed_research_direction": True,
            "this_recipe_selected_by_target_metrics": False,
            "target_examples_images_qrels_used_for_recipe_selection": False,
        },
        "research_only": True,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / LOCK_NAME
    if path.exists():
        if json.loads(path.read_text()) != payload:
            raise RuntimeError("multi-view target candidate lock differs")
    else:
        atomic_json(path, payload)
    return payload


def validate_lock(path: Path) -> dict[str, Any]:
    lock = json.loads(path.read_text())
    if lock.get("status") != "LOCKED_BEFORE_ITERATION5_TARGET_ACCESS":
        raise RuntimeError("multi-view target candidate is not locked")
    candidate = lock["candidate"]
    weights = Path(candidate["model_weights"])
    if sha256_file(weights) != candidate["model_weights_sha256"]:
        raise RuntimeError("locked FashionSigLIP weights changed")
    external = lock["external_selection"]
    for key in ("selection", "recipe"):
        artifact = Path(external[key])
        if sha256_file(artifact) != external[f"{key}_sha256"]:
            raise RuntimeError(f"locked external artifact changed: {key}")
    return lock


def encode_prompt_queries(
    model,
    tokenizer,
    queries: list[str],
    *,
    dataset: str,
    lock_sha: str,
    output_dir: Path,
    device: str,
    batch_size: int,
) -> torch.Tensor:
    path = output_dir / "prompt_queries" / f"{dataset}.pt"
    metadata = {
        "artifact_kind": "fashionsiglip_multiview_product_prompt_queries",
        "dataset": dataset,
        "candidate_lock_sha256": lock_sha,
        "query_count": len(queries),
        "queries_sha256": stable_list_digest(queries),
        "embedding_dim": EMBED_DIM,
    }
    if path.exists():
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("metadata") != metadata:
            raise RuntimeError(f"incompatible prompted query cache: {dataset}")
        return payload["embeddings"].float()
    parts = []
    values = [prompted(value, "fashion_product") for value in queries]
    with torch.inference_mode():
        for start in range(0, len(values), batch_size):
            tokens = tokenizer(values[start : start + batch_size]).to(device)
            parts.append(
                F.normalize(model.encode_text(tokens).float(), dim=-1)
                .half()
                .cpu()
            )
    embeddings = torch.cat(parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_torch_save(
        path, {"metadata": metadata, "embeddings": embeddings}
    )
    return embeddings.float()


def cache_extra_image_views(
    *,
    dataset: str,
    expected_item_ids: list[str],
    model,
    preprocess,
    lock_sha: str,
    output_dir: Path,
    device: str,
    batch_size: int,
    chunk_size: int,
    decode_workers: int,
) -> dict[str, torch.Tensor]:
    dataset_dir = output_dir / "extra_image_views" / dataset
    final_path = dataset_dir / "features.pt"
    views = ("pad", "center_crop")
    metadata = {
        "artifact_kind": "fashionsiglip_multiview_iteration5_extra_images",
        "dataset": dataset,
        "candidate_lock_sha256": lock_sha,
        "item_count": len(expected_item_ids),
        "item_ids_sha256": stable_list_digest(expected_item_ids),
        "views": list(views),
        "embedding_dim": EMBED_DIM,
    }
    if final_path.exists():
        payload = torch.load(final_path, map_location="cpu", weights_only=False)
        if payload.get("metadata") != metadata:
            raise RuntimeError(f"incompatible extra image cache: {dataset}")
        return payload["features"]
    raw, observed_item_ids = load_raw_image_dataset(dataset)
    if observed_item_ids != expected_item_ids:
        raise RuntimeError(f"raw corpus order differs: {dataset}")
    dataset_dir.mkdir(parents=True, exist_ok=True)

    def prepare(index: int) -> dict[str, torch.Tensor]:
        image = raw[int(index)]["image"]
        return {
            view: image_view(image, view, preprocess)
            for view in views
        }

    pool = ThreadPoolExecutor(max_workers=max(1, decode_workers))
    chunks: dict[str, list[torch.Tensor]] = {view: [] for view in views}
    try:
        with torch.inference_mode():
            for chunk_start in range(0, len(raw), chunk_size):
                chunk_end = min(chunk_start + chunk_size, len(raw))
                path = dataset_dir / f"{chunk_start:07d}_{chunk_end:07d}.pt"
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
                        raise RuntimeError(f"incompatible image chunk: {path}")
                    for view in views:
                        chunks[view].append(payload["features"][view])
                    log.info("%s resumed %d:%d", dataset, chunk_start, chunk_end)
                    continue
                parts: dict[str, list[torch.Tensor]] = {
                    view: [] for view in views
                }
                for start in range(chunk_start, chunk_end, batch_size):
                    end = min(start + batch_size, chunk_end)
                    prepared = list(pool.map(prepare, range(start, end)))
                    for view in views:
                        images = torch.stack(
                            [row[view] for row in prepared]
                        ).to(device)
                        encoded = F.normalize(
                            model.encode_image(images).float(), dim=-1
                        )
                        parts[view].append(encoded.half().cpu())
                        del images, encoded
                features = {
                    view: torch.cat(parts[view]) for view in views
                }
                atomic_torch_save(
                    path, {"metadata": chunk_meta, "features": features}
                )
                for view in views:
                    chunks[view].append(features[view])
                log.info("%s cached %d:%d", dataset, chunk_start, chunk_end)
                if device == "mps":
                    torch.mps.empty_cache()
    finally:
        pool.shutdown(wait=True)
    features = {view: torch.cat(chunks[view]) for view in views}
    atomic_torch_save(
        final_path, {"metadata": metadata, "features": features}
    )
    return features


def evaluate_dataset(
    *,
    dataset: str,
    model,
    preprocess,
    tokenizer,
    lock_sha: str,
    query_cache: Path,
    output_dir: Path,
    device: str,
    batch_size: int,
    chunk_size: int,
    decode_workers: int,
    score_document_batch_size: int,
) -> dict[str, Any]:
    output_path = output_dir / "datasets" / f"{dataset}.json"
    metadata = {
        "dataset": dataset,
        "candidate_id": SELECTED_ID,
        "candidate_lock_sha256": lock_sha,
        "full_corpus": True,
        "task": "text-to-image",
        "primary_metric": "MAP@10",
    }
    if output_path.exists():
        payload = json.loads(output_path.read_text())
        if payload.get("metadata") != metadata:
            raise RuntimeError(f"completed result differs: {dataset}")
        return payload
    item_ids, queries, gt = reconstruct_corpus_full(dataset)
    baseline_metrics, baseline_ap10, baseline_sources = load_official_fsl_baseline(
        dataset, gt, queries
    )
    raw_query = load_or_encode_query_embeddings(
        "fashion-siglip",
        dataset,
        queries,
        query_cache,
        batch_size,
        device,
        overwrite=False,
    ).float()
    prompted_query = encode_prompt_queries(
        model,
        tokenizer,
        queries,
        dataset=dataset,
        lock_sha=lock_sha,
        output_dir=output_dir,
        device=device,
        batch_size=batch_size,
    )
    query = F.normalize(raw_query + 0.25 * prompted_query, dim=-1)

    official_document = load_image_embeddings(
        "fashion-siglip", dataset, 0, 42, full_corpus=True
    ).float()
    if official_document.shape[0] != len(item_ids):
        raise RuntimeError(f"official image/corpus count differs: {dataset}")
    extra = cache_extra_image_views(
        dataset=dataset,
        expected_item_ids=item_ids,
        model=model,
        preprocess=preprocess,
        lock_sha=lock_sha,
        output_dir=output_dir,
        device=device,
        batch_size=batch_size,
        chunk_size=chunk_size,
        decode_workers=decode_workers,
    )
    document = F.normalize(
        official_document
        + 0.25 * extra["pad"].float()
        + 0.25 * extra["center_crop"].float(),
        dim=-1,
    )
    scores, indices = streaming_topk(
        query, document, document_batch_size=score_document_batch_size
    )
    retrieved = retrieval_from_topk(scores, indices, item_ids, queries)
    candidate_metrics = evaluate_with_beir(retrieved, gt)
    candidate_ap10 = per_query_ap10(retrieved, gt, queries)
    if not math.isclose(
        float(np.mean(candidate_ap10)),
        candidate_metrics["MAP@10"],
        abs_tol=5.1e-6,
    ):
        raise RuntimeError(f"candidate per-query MAP differs: {dataset}")
    significance = paired_bootstrap(
        candidate_ap10,
        baseline_ap10,
        BOOTSTRAP_SAMPLES,
        _stable_seed(BOOTSTRAP_SEED, dataset, lock_sha),
    )
    classification = (
        "significant_win"
        if significance["ci95_low"] > 0.0
        else "significant_loss"
        if significance["ci95_high"] < 0.0
        else "inconclusive"
    )
    payload = {
        "metadata": metadata,
        "candidate": {
            "metrics": candidate_metrics,
            "ap10": candidate_ap10,
        },
        "baseline": {
            "metrics": baseline_metrics,
            "ap10": baseline_ap10,
            "sources": baseline_sources,
        },
        "comparison": {
            **significance,
            "relative_MAP@10": (
                candidate_metrics["MAP@10"] / baseline_metrics["MAP@10"] - 1.0
            ),
            "classification": classification,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(output_path, payload)
    log.info(
        "%s candidate %.5f baseline %.5f delta %+.5f CI [%+.5f,%+.5f] %s",
        dataset,
        candidate_metrics["MAP@10"],
        baseline_metrics["MAP@10"],
        significance["delta"],
        significance["ci95_low"],
        significance["ci95_high"],
        classification,
    )
    return payload


def build_summary(
    results: dict[str, dict[str, Any]],
    lock: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    rows = {
        dataset: {
            "candidate_MAP@10": result["candidate"]["metrics"]["MAP@10"],
            "baseline_MAP@10": result["baseline"]["metrics"]["MAP@10"],
            "absolute_delta": result["comparison"]["delta"],
            "relative_delta": result["comparison"]["relative_MAP@10"],
            "ci95_low": result["comparison"]["ci95_low"],
            "ci95_high": result["comparison"]["ci95_high"],
            "classification": result["comparison"]["classification"],
        }
        for dataset, result in results.items()
    }
    wins = sum(row["classification"] == "significant_win" for row in rows.values())
    losses = sum(
        row["classification"] == "significant_loss" for row in rows.values()
    )
    impossible = wins + len(DATASETS) - len(rows) < MINIMUM_SIGNIFICANT_WINS
    complete = len(rows) == len(DATASETS)
    achieved = wins >= MINIMUM_SIGNIFICANT_WINS
    status = (
        "AT_LEAST_4_OF_6_ACHIEVED"
        if achieved
        else "EARLY_STOPPED_4_OF_6_IMPOSSIBLE"
        if impossible
        else "COMPLETE_4_OF_6_NOT_ACHIEVED"
        if complete
        else "IN_PROGRESS"
    )
    summary = {
        "schema_version": 1,
        "artifact_kind": "fashionsiglip_multiview_iteration5_summary",
        "status": status,
        "candidate": lock["candidate"],
        "external_selection": lock["external_selection"],
        "protocol": lock["protocol"],
        "disclosure": lock["disclosure"],
        "datasets": rows,
        "evaluated_datasets": len(rows),
        "significant_wins": wins,
        "significant_losses": losses,
        "goal_achieved": achieved,
        "research_only": True,
    }
    atomic_json(output_dir / SUMMARY_NAME, summary)
    report = [
        "# 203M Multi-View FashionSigLIP — Target Iteration 5",
        "",
        f"**Status:** {status}",
        "",
        f"**Result:** {wins} significant wins and {losses} significant losses "
        f"across {len(rows)} evaluated datasets.",
        "",
        "| Dataset | Candidate MAP@10 | FSL MAP@10 | Delta | 95% CI | Result |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for dataset, row in rows.items():
        report.append(
            f"| {dataset} | {row['candidate_MAP@10']:.5f} | "
            f"{row['baseline_MAP@10']:.5f} | {row['absolute_delta']:+.5f} | "
            f"[{row['ci95_low']:+.5f}, {row['ci95_high']:+.5f}] | "
            f"{row['classification']} |"
        )
    report.extend(
        [
            "",
            "Recipe: one official, aspect-padded, and center-cropped image "
            "embedding mixed at 1/.25/.25; raw and deterministic fashion-product "
            "text embeddings mixed at 1/.25. The result is one normalized 768-D "
            "vector with 203,155,970 neural parameters.",
            "",
            "This is benchmark iteration 5, not a fresh blind SOTA evaluation. "
            "The exact recipe was selected only on external OpenVTON and GLAMI "
            "development data; prior aggregate target results informed the broader "
            "research direction.",
            "",
        ]
    )
    (output_dir / REPORT_NAME).write_text("\n".join(report))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("lock", "evaluate", "all"), default="all")
    parser.add_argument("--selection", default=str(DEFAULT_SELECTION))
    parser.add_argument("--recipe", default=str(DEFAULT_RECIPE))
    parser.add_argument("--query-cache", default=str(DEFAULT_QUERY_CACHE))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--chunk-size", type=int, default=1_024)
    parser.add_argument("--decode-workers", type=int, default=4)
    parser.add_argument("--score-document-batch-size", type=int, default=8_192)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    output_dir = Path(args.output_dir)
    lock_path = output_dir / LOCK_NAME
    if args.stage in ("lock", "all"):
        lock = create_lock(
            output_dir, Path(args.selection), Path(args.recipe)
        )
        log.info("locked multi-view candidate %s", lock["candidate"]["candidate_id"])
    if args.stage == "lock":
        return
    lock = validate_lock(lock_path)
    lock_sha = sha256_file(lock_path)
    model, preprocess, tokenizer = load_locked_fsl(args.device)
    results: dict[str, dict[str, Any]] = {}
    for dataset in EVALUATION_ORDER:
        results[dataset] = evaluate_dataset(
            dataset=dataset,
            model=model,
            preprocess=preprocess,
            tokenizer=tokenizer,
            lock_sha=lock_sha,
            query_cache=Path(args.query_cache),
            output_dir=output_dir,
            device=args.device,
            batch_size=args.batch_size,
            chunk_size=args.chunk_size,
            decode_workers=args.decode_workers,
            score_document_batch_size=args.score_document_batch_size,
        )
        wins = sum(
            result["comparison"]["classification"] == "significant_win"
            for result in results.values()
        )
        if wins + len(DATASETS) - len(results) < MINIMUM_SIGNIFICANT_WINS:
            log.info("early stop: four significant wins are no longer possible")
            break
    summary = build_summary(results, lock, output_dir)
    log.info("%s", summary["status"])


if __name__ == "__main__":
    main()
