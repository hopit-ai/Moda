"""
MODA — Vision-Only Model Compression + LookBench Evaluation

1. Exports the vision-only encoder (strips text tower → 93M params, ~372MB)
2. Creates FP16 half-precision variant (~186MB, faster on GPU/MPS)
3. Evaluates both on full LookBench protocol (4 subsets + 58K noise)

Usage:
    python benchmark/eval_lookbench_compressed.py
    python benchmark/eval_lookbench_compressed.py --skip-export
    python benchmark/eval_lookbench_compressed.py --variants fp32
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import open_clip
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from benchmark.eval_lookbench_baseline import (
    SUBSETS,
    NOISE_LABEL,
    NOISE_CAT,
    extract_labels,
    compute_all_metrics,
    compute_mrr,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

RESULTS_DIR = REPO / "results" / "lookbench"
MODELS_DIR = REPO / "models"

SRC_CKPT = MODELS_DIR / "moda-siglip-distilled" / "best" / "model_state_dict.pt"
VISION_FP32_DIR = MODELS_DIR / "moda-vision-only-fp32"
VISION_FP16_DIR = MODELS_DIR / "moda-vision-only-fp16"


class VisionOnlyEncoder(nn.Module):
    """Standalone vision encoder extracted from an OpenCLIP model."""

    def __init__(self, visual_module):
        super().__init__()
        self.visual = visual_module

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.visual(images)

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        feats = self.forward(images)
        return F.normalize(feats, p=2, dim=-1)


def _build_full_model_with_vision_weights(vision_sd_path: Path):
    """Create base model and load our fine-tuned vision weights into it."""
    base_model, _, preprocess = open_clip.create_model_and_transforms(
        "hf-hub:Marqo/marqo-fashionSigLIP"
    )
    vision_sd = torch.load(vision_sd_path, map_location="cpu", weights_only=True)
    full_sd = base_model.state_dict()
    for k, v in vision_sd.items():
        full_sd[k] = v
    base_model.load_state_dict(full_sd, strict=True)
    return base_model, preprocess


def export_vision_only(src_ckpt: Path, dest_dir: Path) -> Path:
    """Extract vision encoder from full CLIP model, save standalone."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    out_path = dest_dir / "vision_encoder.pt"

    if out_path.exists():
        size_mb = out_path.stat().st_size / 1e6
        log.info("Vision-only checkpoint already exists: %s (%.1f MB)", out_path, size_mb)
        return out_path

    log.info("Loading full model from %s ...", src_ckpt)
    model, _, _ = open_clip.create_model_and_transforms(
        "hf-hub:Marqo/marqo-fashionSigLIP"
    )
    full_sd = torch.load(src_ckpt, map_location="cpu", weights_only=True)
    model.load_state_dict(full_sd, strict=True)

    vision_sd = {}
    for k, v in model.state_dict().items():
        if k.startswith("visual."):
            vision_sd[k] = v

    n_params = sum(v.numel() for v in vision_sd.values())
    log.info("Vision encoder: %d keys, %d params (%.1f MB fp32)",
             len(vision_sd), n_params, n_params * 4 / 1e6)

    torch.save(vision_sd, out_path)
    size_mb = out_path.stat().st_size / 1e6
    log.info("Saved vision-only checkpoint: %s (%.1f MB)", out_path, size_mb)

    meta = {
        "source": str(src_ckpt),
        "type": "vision-only-fp32",
        "params": n_params,
        "keys": len(vision_sd),
        "size_mb": round(size_mb, 1),
        "architecture": "ViT-B-16-SigLIP (vision tower only)",
    }
    with open(dest_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    return out_path


def export_vision_fp16(vision_fp32_path: Path, dest_dir: Path) -> Path:
    """Convert vision-only weights from FP32 to FP16 for 2x size reduction."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    out_path = dest_dir / "vision_encoder_fp16.pt"

    if out_path.exists():
        size_mb = out_path.stat().st_size / 1e6
        log.info("FP16 checkpoint already exists: %s (%.1f MB)", out_path, size_mb)
        return out_path

    log.info("Converting vision encoder to FP16 ...")
    vision_sd = torch.load(vision_fp32_path, map_location="cpu", weights_only=True)
    fp16_sd = {k: v.half() for k, v in vision_sd.items()}

    torch.save(fp16_sd, out_path)
    size_mb = out_path.stat().st_size / 1e6
    fp32_size = vision_fp32_path.stat().st_size / 1e6
    log.info("Saved FP16 checkpoint: %s (%.1f MB, %.1f%% of FP32)",
             out_path, size_mb, size_mb / fp32_size * 100)

    meta = {
        "source": str(vision_fp32_path),
        "type": "vision-only-fp16",
        "precision": "float16",
        "size_mb": round(size_mb, 1),
        "compression_ratio": round(fp32_size / size_mb, 2),
        "architecture": "ViT-B-16-SigLIP (vision tower only, FP16)",
    }
    with open(dest_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    return out_path


def load_vision_fp32(ckpt_path: Path, device: str):
    """Load vision-only fp32 model."""
    base_model, preprocess = _build_full_model_with_vision_weights(ckpt_path)
    encoder = VisionOnlyEncoder(base_model.visual)
    encoder = encoder.to(device).eval()

    n_params = sum(p.numel() for p in encoder.parameters())
    log.info("Loaded vision-only fp32: %d params on %s", n_params, device)
    return encoder, preprocess


def load_vision_fp16(ckpt_path: Path, device: str):
    """Load vision-only fp16 model."""
    base_model, _, preprocess = open_clip.create_model_and_transforms(
        "hf-hub:Marqo/marqo-fashionSigLIP"
    )
    fp16_sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    fp32_sd = {k: v.float() for k, v in fp16_sd.items()}
    full_sd = base_model.state_dict()
    for k, v in fp32_sd.items():
        full_sd[k] = v
    base_model.load_state_dict(full_sd, strict=True)

    encoder = VisionOnlyEncoder(base_model.visual)
    if device in ("mps", "cuda"):
        encoder = encoder.half().to(device)
    else:
        encoder = encoder.to(device)
    encoder.eval()

    n_params = sum(p.numel() for p in encoder.parameters())
    log.info("Loaded vision-only fp16: %d params on %s (dtype=%s)",
             n_params, device, next(encoder.parameters()).dtype)
    return encoder, preprocess


@torch.no_grad()
def encode_images(data, encoder, preprocess, device, batch_size=32, desc="Encoding"):
    """Encode images using vision-only encoder."""
    is_half = next(encoder.parameters()).dtype == torch.float16
    all_feats = []
    all_ids = []

    for start in tqdm(range(0, len(data), batch_size), desc=desc):
        batch = data[start:start + batch_size]
        images = batch["image"]
        ids = batch.get("item_ID", batch.get("item_id",
                        list(range(start, start + len(images)))))

        tensors = torch.stack([preprocess(img.convert("RGB")) for img in images])
        tensors = tensors.to(device)
        if is_half:
            tensors = tensors.half()

        feats = encoder.encode_image(tensors)
        feats = feats.float()
        feats = F.normalize(feats, p=2, dim=-1)

        all_feats.append(feats.cpu())
        if isinstance(ids, (list, np.ndarray)):
            all_ids.extend(ids)
        else:
            all_ids.extend(ids.tolist() if hasattr(ids, "tolist") else [ids])

    return torch.cat(all_feats, dim=0).numpy(), all_ids


def evaluate_on_lookbench(encoder, preprocess, device, batch_size, variant_name):
    """Full LookBench evaluation."""
    log.info("\n" + "=" * 70)
    log.info("EVALUATING: %s", variant_name)
    log.info("=" * 70)

    log.info("Encoding noise gallery (58K) ...")
    noise_ds = load_dataset("srpone/look-bench", "noise")["gallery"]
    noise_feats, noise_ids = encode_images(
        noise_ds, encoder, preprocess, device, batch_size, "noise"
    )
    noise_labels = [NOISE_LABEL] * len(noise_ids)
    noise_cats = [NOISE_CAT] * len(noise_ids)
    log.info("Noise encoded: %d items", len(noise_ids))

    results = {}
    for subset in SUBSETS:
        log.info("Evaluating %s ...", subset)
        ds = load_dataset("srpone/look-bench", subset)
        q_data, g_data = ds["query"], ds["gallery"]

        q_cats, _, q_labels = extract_labels(q_data)
        g_cats_sub, _, g_labels_sub = extract_labels(g_data)

        q_feats, q_ids = encode_images(
            q_data, encoder, preprocess, device, batch_size, f"{subset}-q"
        )
        g_feats_sub, g_ids_sub = encode_images(
            g_data, encoder, preprocess, device, batch_size, f"{subset}-g"
        )

        g_feats = np.concatenate([g_feats_sub, noise_feats], axis=0)
        g_ids = list(g_ids_sub) + list(noise_ids)
        g_labels = g_labels_sub + noise_labels
        g_cats = g_cats_sub + noise_cats

        metrics = compute_all_metrics(
            q_feats, q_labels, q_cats, q_ids,
            g_feats, g_labels, g_cats, g_ids,
        )
        mrr = compute_mrr(q_feats, q_ids, g_feats, g_ids)

        log.info(
            "  %s  Fine_R@1=%.2f  Coarse_R@1=%.2f  nDCG@5=%.2f  MRR=%.2f",
            subset,
            metrics["fine_recall"]["recall@1"],
            metrics["coarse_recall"]["recall@1"],
            metrics["ndcg@5"], mrr,
        )
        results[subset] = {
            "subset": subset,
            "n_queries": len(q_ids),
            "n_gallery_total": len(g_ids),
            **metrics,
            "mrr": mrr,
        }

    total_q = sum(r["n_queries"] for r in results.values())
    overall = {
        "fine_recall@1": round(sum(
            r["fine_recall"]["recall@1"] * r["n_queries"]
            for r in results.values()
        ) / total_q, 2),
        "coarse_recall@1": round(sum(
            r["coarse_recall"]["recall@1"] * r["n_queries"]
            for r in results.values()
        ) / total_q, 2),
        "ndcg@5": round(sum(
            r["ndcg@5"] * r["n_queries"]
            for r in results.values()
        ) / total_q, 2),
        "total_queries": total_q,
    }
    results["overall"] = overall

    log.info(
        "\n%s OVERALL  Fine_R@1=%.2f  Coarse_R@1=%.2f  nDCG@5=%.2f",
        variant_name, overall["fine_recall@1"],
        overall["coarse_recall@1"], overall["ndcg@5"],
    )
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-export", action="store_true",
                        help="Skip export, use cached checkpoints")
    parser.add_argument("--variants", nargs="+",
                        default=["fp32", "fp16"],
                        choices=["fp32", "fp16"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = args.device
    if not device:
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
    log.info("Device: %s", device)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Export vision-only FP32
    if not args.skip_export:
        log.info("\n=== STEP 1: Export vision-only encoder ===")
        vision_fp32_path = export_vision_only(SRC_CKPT, VISION_FP32_DIR)
    else:
        vision_fp32_path = VISION_FP32_DIR / "vision_encoder.pt"
        assert vision_fp32_path.exists(), f"No cached checkpoint at {vision_fp32_path}"

    # Step 2: FP16 conversion
    if "fp16" in args.variants:
        if not args.skip_export:
            log.info("\n=== STEP 2: FP16 half-precision conversion ===")
            fp16_path = export_vision_fp16(vision_fp32_path, VISION_FP16_DIR)
        else:
            fp16_path = VISION_FP16_DIR / "vision_encoder_fp16.pt"
            assert fp16_path.exists(), f"No cached FP16 at {fp16_path}"

    # Step 3: Evaluate
    all_results = {}

    if "fp32" in args.variants:
        log.info("\n=== STEP 3a: Evaluate vision-only FP32 ===")
        encoder_fp32, preprocess = load_vision_fp32(vision_fp32_path, device)
        t0 = time.time()
        results_fp32 = evaluate_on_lookbench(
            encoder_fp32, preprocess, device, args.batch_size,
            "MODA-Vision-Only-FP32 (93M params)"
        )
        elapsed_fp32 = time.time() - t0

        fp32_size = vision_fp32_path.stat().st_size / 1e6
        all_results["vision_only_fp32"] = {
            "variant": "MODA-Vision-Only-FP32",
            "params": "92.9M",
            "precision": "float32",
            "model_size_mb": round(fp32_size, 1),
            "device": device,
            "elapsed_sec": round(elapsed_fp32, 1),
            "subsets": results_fp32,
        }
        del encoder_fp32
        if device == "mps":
            torch.mps.empty_cache()

    if "fp16" in args.variants:
        log.info("\n=== STEP 3b: Evaluate vision-only FP16 ===")
        encoder_fp16, preprocess = load_vision_fp16(fp16_path, device)
        t0 = time.time()
        results_fp16 = evaluate_on_lookbench(
            encoder_fp16, preprocess, device, args.batch_size,
            "MODA-Vision-Only-FP16 (93M params, half)"
        )
        elapsed_fp16 = time.time() - t0

        fp16_size = fp16_path.stat().st_size / 1e6
        all_results["vision_only_fp16"] = {
            "variant": "MODA-Vision-Only-FP16",
            "params": "92.9M",
            "precision": "float16",
            "model_size_mb": round(fp16_size, 1),
            "device": device,
            "elapsed_sec": round(elapsed_fp16, 1),
            "subsets": results_fp16,
        }
        del encoder_fp16
        if device == "mps":
            torch.mps.empty_cache()

    # Step 4: Summary
    print("\n" + "=" * 100)
    print("COMPRESSION RESULTS — LookBench Fine Recall@1 (%)")
    print("=" * 100)
    hdr = f"  {'Variant':<42} {'Params':>8} {'Size':>8} {'RealStu':>8} {'AIGenSt':>8} {'RealStr':>8} {'AIGenSr':>8} {'Overall':>8}"
    print(hdr)
    print("  " + "-" * 98)
    print(f"  {'MODA-Distilled (full CLIP, baseline)':<42} {'203M':>8} {'775 MB':>8} {'70.23':>8} {'80.31':>8} {'60.24':>8} {'81.25':>8} {'67.63':>8}")

    for key, data in all_results.items():
        s = data["subsets"]
        ov = s["overall"]
        rs = s.get("real_studio_flat", {}).get("fine_recall", {}).get("recall@1", "—")
        ag = s.get("aigen_studio", {}).get("fine_recall", {}).get("recall@1", "—")
        st = s.get("real_streetlook", {}).get("fine_recall", {}).get("recall@1", "—")
        al = s.get("aigen_streetlook", {}).get("fine_recall", {}).get("recall@1", "—")
        name = data["variant"]
        params = data["params"]
        size = f"{data['model_size_mb']:.0f} MB"
        print(f"  {name:<42} {params:>8} {size:>8} {rs:>8} {ag:>8} {st:>8} {al:>8} {ov['fine_recall@1']:>8.2f}")

    print("=" * 100)

    out_path = RESULTS_DIR / "compressed_eval.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    log.info("Results saved to %s", out_path)


if __name__ == "__main__":
    main()
