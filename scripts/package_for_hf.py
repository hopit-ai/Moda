"""
Package MODA models for HuggingFace upload.

Creates self-contained directories under hf_repos/ that can be pushed
directly with `huggingface-cli upload`.

Usage:
    python scripts/package_for_hf.py                    # package all models
    python scripts/package_for_hf.py --model distilled  # package one model
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import save_file

REPO_ROOT = Path(__file__).resolve().parent.parent
HF_DIR = REPO_ROOT / "hf_repos"
FSL_DIR = REPO_ROOT / "models" / "marqo-fashionSigLIP"

TOKENIZER_FILES = [
    "spiece.model",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
]

MODELS = {
    "distilled": {
        "src": REPO_ROOT / "models" / "moda-siglip-distilled" / "best" / "model_state_dict.pt",
        "hf_name": "moda-fashion-distilled",
        "display": "MODA-Fashion-Distilled",
        "emb_dim": 768,
        "custom_class": False,
    },
    "matryoshka": {
        "src": REPO_ROOT / "models" / "moda-siglip-matryoshka" / "best" / "model_state_dict.pt",
        "hf_name": "moda-fashion-matryoshka",
        "display": "MODA-Fashion-Matryoshka",
        "emb_dim": 768,
        "custom_class": False,
    },
    "deepfashion2": {
        "src": REPO_ROOT / "models" / "moda-siglip-deepfashion2" / "best" / "model_state_dict.pt",
        "hf_name": "moda-fashion-deepfashion2",
        "display": "MODA-Fashion-DeepFashion2",
        "emb_dim": 768,
        "custom_class": False,
    },
    "distilled-512d": {
        "src_backbone": REPO_ROOT / "models" / "moda-siglip-distilled-512d" / "best" / "backbone_state_dict.pt",
        "src_proj": REPO_ROOT / "models" / "moda-siglip-distilled-512d" / "best" / "proj_state_dict.pt",
        "hf_name": "moda-fashion-distilled-512d",
        "display": "MODA-Fashion-Distilled-512d",
        "emb_dim": 512,
        "custom_class": True,
    },
}


def convert_to_safetensors(state_dict: dict, out_path: Path):
    clean = {}
    for k, v in state_dict.items():
        if v.dim() == 0:
            clean[k] = v.unsqueeze(0)
        else:
            clean[k] = v.contiguous()
    save_file(clean, str(out_path))
    mb = out_path.stat().st_size / 1e6
    print(f"  Saved {out_path.name} ({mb:.1f} MB, {len(clean)} tensors)")


def copy_tokenizer(dest: Path):
    for f in TOKENIZER_FILES:
        src = FSL_DIR / f
        if src.exists():
            shutil.copy2(src, dest / f)
    print(f"  Copied {len(TOKENIZER_FILES)} tokenizer files")


def write_open_clip_config(dest: Path):
    cfg = {
        "model_cfg": {
            "embed_dim": 768,
            "init_logit_bias": -10,
            "custom_text": True,
            "vision_cfg": {
                "image_size": 224,
                "timm_model_name": "vit_base_patch16_siglip_224",
                "timm_model_pretrained": False,
                "timm_pool": "map",
                "timm_proj": "none",
            },
            "text_cfg": {
                "context_length": 64,
                "vocab_size": 32000,
                "hf_tokenizer_name": "timm/ViT-B-16-SigLIP",
                "tokenizer_kwargs": {"clean": "canonicalize"},
                "width": 768,
                "heads": 12,
                "layers": 12,
                "no_causal_mask": True,
                "proj_bias": True,
                "pool_type": "last",
                "norm_kwargs": {"eps": 1e-6},
            },
        },
        "preprocess_cfg": {
            "mean": [0.5, 0.5, 0.5],
            "std": [0.5, 0.5, 0.5],
            "interpolation": "bicubic",
            "resize_mode": "squash",
        },
    }
    with open(dest / "open_clip_config.json", "w") as f:
        json.dump(cfg, f, indent=2)
    print("  Wrote open_clip_config.json")


def write_preprocessor_config(dest: Path):
    cfg = {
        "do_normalize": True,
        "do_rescale": True,
        "do_resize": True,
        "do_convert_rgb": True,
        "image_processor_type": "SiglipImageProcessor",
        "image_mean": [0.5, 0.5, 0.5],
        "processor_class": "SiglipImageProcessor",
        "resample": 3,
        "rescale_factor": 0.00392156862745098,
        "size": {"height": 224, "width": 224},
        "image_std": [0.5, 0.5, 0.5],
    }
    with open(dest / "preprocessor_config.json", "w") as f:
        json.dump(cfg, f, indent=2)
    print("  Wrote preprocessor_config.json")


def package_768d(key: str):
    info = MODELS[key]
    dest = HF_DIR / info["hf_name"]
    dest.mkdir(parents=True, exist_ok=True)
    print(f"\nPackaging {info['display']} -> {dest}")

    sd = torch.load(info["src"], map_location="cpu", weights_only=True)
    convert_to_safetensors(sd, dest / "open_clip_model.safetensors")

    torch.save(sd, dest / "open_clip_pytorch_model.bin")
    print(f"  Saved open_clip_pytorch_model.bin ({info['src'].stat().st_size / 1e6:.1f} MB)")

    write_open_clip_config(dest)
    write_preprocessor_config(dest)
    copy_tokenizer(dest)

    print(f"  Done! Ready at: {dest}")
    return dest


def package_512d():
    info = MODELS["distilled-512d"]
    dest = HF_DIR / info["hf_name"]
    dest.mkdir(parents=True, exist_ok=True)
    print(f"\nPackaging {info['display']} -> {dest}")

    backbone_sd = torch.load(info["src_backbone"], map_location="cpu", weights_only=True)
    proj_sd = torch.load(info["src_proj"], map_location="cpu", weights_only=True)

    combined = dict(backbone_sd)
    for k, v in proj_sd.items():
        combined[f"proj.{k}"] = v
    convert_to_safetensors(combined, dest / "model.safetensors")

    torch.save(combined, dest / "pytorch_model.bin")
    print(f"  Saved pytorch_model.bin ({len(combined)} keys)")

    write_open_clip_config(dest)
    write_preprocessor_config(dest)
    copy_tokenizer(dest)

    print(f"  Done! Ready at: {dest}")
    return dest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(MODELS.keys()) + ["all"], default="all")
    args = parser.parse_args()

    HF_DIR.mkdir(exist_ok=True)

    targets = list(MODELS.keys()) if args.model == "all" else [args.model]

    for key in targets:
        if key == "distilled-512d":
            package_512d()
        else:
            package_768d(key)

    print("\n" + "=" * 60)
    print("All models packaged under hf_repos/")
    print("=" * 60)
    print("\nTo upload, run:")
    for key in targets:
        name = MODELS[key]["hf_name"]
        print(f"  huggingface-cli upload YOUR_USERNAME/{name} hf_repos/{name}")


if __name__ == "__main__":
    main()
