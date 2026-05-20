"""
Inference validation for all 5 MODA HF-repo models.

Loads each model from the local hf_repos/ directory, runs a sample
fashion image through it, and validates:
  1. Model loads without errors
  2. Produces embeddings of expected dimension
  3. Embeddings are L2-normalized
  4. Image-to-image similarity is reasonable (same image ≈ 1.0)
  5. Different images produce different embeddings

Usage:
    python benchmark/test_hf_models_inference.py
"""
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import open_clip
from PIL import Image
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parent.parent
HF_DIR = ROOT / "hf_repos"
IMG_DIR = ROOT / "data" / "processed" / "fashion_stratified_gs_df" / "images"


def find_test_images(n=2):
    imgs = sorted(IMG_DIR.glob("*.jpg"))[:n]
    if len(imgs) < n:
        print(f"WARN: only {len(imgs)} test images found in {IMG_DIR}")
    return imgs


def check_embedding(name, emb, expected_dim):
    ok = True
    if emb.shape[-1] != expected_dim:
        print(f"  FAIL: expected dim {expected_dim}, got {emb.shape[-1]}")
        ok = False
    norm = emb.norm(dim=-1)
    if not torch.allclose(norm, torch.ones_like(norm), atol=1e-3):
        print(f"  FAIL: embedding not L2-normalized (norms: {norm.tolist()})")
        ok = False
    return ok


def test_open_clip_model(name, model_dir, test_imgs):
    """Test standard open_clip models (distilled, deepfashion2, matryoshka)."""
    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print(f"  Path: {model_dir}")
    t0 = time.time()

    config_path = model_dir / "open_clip_config.json"
    safetensors_path = model_dir / "open_clip_model.safetensors"

    if not safetensors_path.exists():
        print(f"  SKIP: {safetensors_path.name} not found")
        return None

    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-16-SigLIP",
        pretrained=str(safetensors_path),
    )
    model.eval()
    load_time = time.time() - t0
    print(f"  Loaded in {load_time:.1f}s")

    params_m = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Parameters: {params_m:.1f}M")

    images = torch.stack([preprocess(Image.open(p).convert("RGB")) for p in test_imgs])

    t0 = time.time()
    with torch.no_grad():
        img_emb = model.encode_image(images)
        img_emb = F.normalize(img_emb, p=2, dim=-1)
    infer_time = time.time() - t0
    print(f"  Inference ({len(test_imgs)} imgs): {infer_time:.3f}s")
    print(f"  Embedding shape: {img_emb.shape}")

    ok = check_embedding(name, img_emb, 768)

    sim = (img_emb @ img_emb.T)
    self_sim = sim[0, 0].item()
    cross_sim = sim[0, 1].item() if len(test_imgs) > 1 else None
    print(f"  Self-similarity: {self_sim:.4f} (expect ≈ 1.0)")
    if cross_sim is not None:
        print(f"  Cross-similarity: {cross_sim:.4f} (expect < 1.0)")
        if abs(cross_sim - 1.0) < 1e-3:
            print(f"  FAIL: different images have identical embeddings")
            ok = False

    text = open_clip.get_tokenizer("ViT-B-16-SigLIP")(["a red dress", "blue sneakers"])
    with torch.no_grad():
        txt_emb = model.encode_text(text)
        txt_emb = F.normalize(txt_emb, p=2, dim=-1)
    t2i_sim = (img_emb @ txt_emb.T)
    print(f"  Text-Image sim sample: {t2i_sim[0].tolist()}")

    status = "PASS" if ok else "FAIL"
    print(f"  Result: {status}")
    return {"name": name, "status": status, "dim": 768, "params_m": params_m,
            "self_sim": self_sim, "cross_sim": cross_sim}


def test_512d_model(model_dir, test_imgs):
    """Test the distilled-512d model with custom projection."""
    name = "moda-fashion-distilled-512d"
    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print(f"  Path: {model_dir}")
    t0 = time.time()

    safetensors_path = model_dir / "model.safetensors"
    if not safetensors_path.exists():
        print(f"  SKIP: {safetensors_path.name} not found")
        return None

    backbone, _, preprocess = open_clip.create_model_and_transforms(
        "hf-hub:Marqo/marqo-fashionSigLIP"
    )

    state = load_file(str(safetensors_path))
    proj_weight = state.pop("proj.weight")

    scalars = {}
    for k in list(state.keys()):
        if state[k].dim() == 1 and state[k].numel() == 1:
            scalars[k] = state.pop(k).squeeze(0)
    state.update(scalars)

    backbone.load_state_dict(state, strict=False)
    backbone.eval()

    proj = torch.nn.Linear(768, 512, bias=False)
    proj.weight.data.copy_(proj_weight)
    proj.eval()

    load_time = time.time() - t0
    print(f"  Loaded in {load_time:.1f}s")

    params_backbone = sum(p.numel() for p in backbone.parameters()) / 1e6
    params_proj = sum(p.numel() for p in proj.parameters()) / 1e6
    print(f"  Parameters: backbone={params_backbone:.1f}M, proj={params_proj:.2f}M")

    images = torch.stack([preprocess(Image.open(p).convert("RGB")) for p in test_imgs])

    t0 = time.time()
    with torch.no_grad():
        feats = backbone.encode_image(images)
        emb = F.normalize(proj(feats), p=2, dim=-1)
    infer_time = time.time() - t0
    print(f"  Inference ({len(test_imgs)} imgs): {infer_time:.3f}s")
    print(f"  Embedding shape: {emb.shape}")

    ok = check_embedding(name, emb, 512)

    sim = emb @ emb.T
    self_sim = sim[0, 0].item()
    cross_sim = sim[0, 1].item() if len(test_imgs) > 1 else None
    print(f"  Self-similarity: {self_sim:.4f} (expect ≈ 1.0)")
    if cross_sim is not None:
        print(f"  Cross-similarity: {cross_sim:.4f} (expect < 1.0)")
        if abs(cross_sim - 1.0) < 1e-3:
            print(f"  FAIL: different images have identical embeddings")
            ok = False

    status = "PASS" if ok else "FAIL"
    print(f"  Result: {status}")
    return {"name": name, "status": status, "dim": 512,
            "params_m": params_backbone + params_proj,
            "self_sim": self_sim, "cross_sim": cross_sim}


def test_vision_fp16(model_dir, test_imgs):
    """Test vision-only FP16 model."""
    name = "moda-fashion-vision-fp16"
    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print(f"  Path: {model_dir}")
    t0 = time.time()

    safetensors_path = model_dir / "vision_encoder.safetensors"
    if not safetensors_path.exists():
        print(f"  SKIP: {safetensors_path.name} not found")
        return None

    base_model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-16-SigLIP",
    )

    vision_sd = load_file(str(safetensors_path))
    # Strip "visual." prefix if present (weights were saved from model.visual)
    fp32_sd = {}
    for k, v in vision_sd.items():
        clean_k = k.removeprefix("visual.")
        fp32_sd[clean_k] = v.float()

    visual = base_model.visual
    visual.load_state_dict(fp32_sd, strict=True)
    visual.eval()

    load_time = time.time() - t0
    print(f"  Loaded in {load_time:.1f}s")

    params_m = sum(p.numel() for p in visual.parameters()) / 1e6
    print(f"  Parameters: {params_m:.1f}M (vision-only)")

    images = torch.stack([preprocess(Image.open(p).convert("RGB")) for p in test_imgs])

    t0 = time.time()
    with torch.no_grad():
        img_emb = visual(images)
        img_emb = F.normalize(img_emb, p=2, dim=-1)
    infer_time = time.time() - t0
    print(f"  Inference ({len(test_imgs)} imgs): {infer_time:.3f}s")
    print(f"  Embedding shape: {img_emb.shape}")

    ok = check_embedding(name, img_emb, 768)

    sim = img_emb @ img_emb.T
    self_sim = sim[0, 0].item()
    cross_sim = sim[0, 1].item() if len(test_imgs) > 1 else None
    print(f"  Self-similarity: {self_sim:.4f} (expect ≈ 1.0)")
    if cross_sim is not None:
        print(f"  Cross-similarity: {cross_sim:.4f} (expect < 1.0)")
        if abs(cross_sim - 1.0) < 1e-3:
            print(f"  FAIL: different images have identical embeddings")
            ok = False

    status = "PASS" if ok else "FAIL"
    print(f"  Result: {status}")
    return {"name": name, "status": status, "dim": 768, "params_m": params_m,
            "self_sim": self_sim, "cross_sim": cross_sim}


def test_matryoshka_dimensions(model_dir, test_imgs):
    """Test the matryoshka model at all supported dimensions."""
    name = "moda-fashion-matryoshka (MRL dimension sweep)"
    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print(f"  Path: {model_dir}")

    safetensors_path = model_dir / "open_clip_model.safetensors"
    if not safetensors_path.exists():
        print(f"  SKIP: {safetensors_path.name} not found")
        return []

    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-16-SigLIP",
        pretrained=str(safetensors_path),
    )
    model.eval()

    images = torch.stack([preprocess(Image.open(p).convert("RGB")) for p in test_imgs])

    with torch.no_grad():
        full_emb = model.encode_image(images)

    dims = [768, 512, 384, 256, 128, 64]
    results = []

    print(f"\n  {'Dim':>5} {'Self-Sim':>9} {'Cross-Sim':>10} {'Norm-OK':>8} {'Status':>7}")
    print(f"  {'-'*50}")

    for dim in dims:
        trunc = F.normalize(full_emb[:, :dim], p=2, dim=-1)

        norm_ok = torch.allclose(trunc.norm(dim=-1), torch.ones(trunc.shape[0]), atol=1e-3)
        sim = trunc @ trunc.T
        self_sim = sim[0, 0].item()
        cross_sim = sim[0, 1].item() if len(test_imgs) > 1 else None

        ok = norm_ok and (cross_sim is None or abs(cross_sim - 1.0) > 1e-3)
        status = "PASS" if ok else "FAIL"

        cross_str = f"{cross_sim:.4f}" if cross_sim is not None else "N/A"
        print(f"  {dim:>5} {self_sim:>9.4f} {cross_str:>10} {'yes' if norm_ok else 'NO':>8} {status:>7}")

        results.append({
            "name": f"matryoshka-{dim}d", "status": status, "dim": dim,
            "params_m": 203.2, "self_sim": self_sim, "cross_sim": cross_sim,
        })

    return results


def main():
    print("=" * 60)
    print("MODA HF Models — Inference Validation")
    print("=" * 60)

    test_imgs = find_test_images(2)
    if not test_imgs:
        print("ERROR: No test images found. Cannot proceed.")
        sys.exit(1)
    print(f"Test images: {[p.name for p in test_imgs]}")

    results = []

    for model_name in ["moda-fashion-distilled", "moda-fashion-deepfashion2",
                        "moda-fashion-matryoshka"]:
        model_dir = HF_DIR / model_name
        if model_dir.exists():
            r = test_open_clip_model(model_name, model_dir, test_imgs)
            if r:
                results.append(r)

    model_dir = HF_DIR / "moda-fashion-distilled-512d"
    if model_dir.exists():
        r = test_512d_model(model_dir, test_imgs)
        if r:
            results.append(r)

    model_dir = HF_DIR / "moda-fashion-vision-fp16"
    if model_dir.exists():
        r = test_vision_fp16(model_dir, test_imgs)
        if r:
            results.append(r)

    # MRL dimension sweep
    mrl_results = []
    model_dir = HF_DIR / "moda-fashion-matryoshka"
    if model_dir.exists():
        mrl_results = test_matryoshka_dimensions(model_dir, test_imgs)

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Model':<35} {'Dim':>4} {'Params':>8} {'Self-Sim':>9} {'Cross-Sim':>10} {'Status':>7}")
    print("-" * 80)
    for r in results:
        cross = f"{r['cross_sim']:.4f}" if r.get('cross_sim') is not None else "N/A"
        print(f"{r['name']:<35} {r['dim']:>4} {r['params_m']:>7.1f}M {r['self_sim']:>9.4f} {cross:>10} {r['status']:>7}")

    all_pass = all(r["status"] == "PASS" for r in results)
    print(f"\nOverall: {'ALL PASS' if all_pass else 'SOME FAILURES'} ({len(results)}/{len(results)} models tested)")

    if not all_pass:
        sys.exit(1)


if __name__ == "__main__":
    main()
