"""
Diagnostic for the v2 LoRA fine-tune (which regressed -78% MAP@10 on
fashion200k 10K). Two parts:

  PART 1 — wiring sanity check
    Load base SigLIP L/16-384, also load it with PEFT/LoRA adapters from
    step_000050. Compute image+text features for a tiny tensor and confirm
    that the LoRA-adapted features actually DIFFER from the base features
    (i.e., the LoRA path is wired correctly). If they're identical, the
    training was a no-op and the regression must be from somewhere else.

  PART 2 — merge intermediate adapters into open_clip-format state_dicts
    so the existing eval_marqo_subsample.py harness can score them.
    Writes:
      models/moda-siglip-l16-lora-v2-smoke/step_000050/best/model_state_dict.pt
      models/moda-siglip-l16-lora-v2-smoke/step_000100/best/model_state_dict.pt
      models/moda-siglip-l16-lora-v2-smoke/step_000150/best/model_state_dict.pt

  Then a separate shell wrapper invokes eval_marqo_subsample.py for each.

Usage:
    .venv/bin/python scripts/diagnose_v2_checkpoints.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("diagnose-v2")

CKPT_ROOT = REPO / "models" / "moda-siglip-l16-lora-v2-smoke"
HF_CACHE = REPO / "data" / "hf_cache"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def build_base():
    import open_clip
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-L-16-SigLIP-384", pretrained="webli", cache_dir=str(HF_CACHE),
    )
    tokenizer = open_clip.get_tokenizer("ViT-L-16-SigLIP-384")
    model.eval().to(DEVICE)
    return model, preprocess, tokenizer


def attach_lora(base_model, adapter_dir: Path):
    """Re-attach a saved LoRA adapter onto a fresh base model."""
    from peft import PeftModel
    return PeftModel.from_pretrained(base_model, str(adapter_dir), is_trainable=False).eval()


@torch.no_grad()
def features(model, images, tokens):
    if hasattr(model, "get_image_features"):
        ifeat = model.get_image_features(images)
        tfeat = model.get_text_features(tokens)
    else:
        ifeat = model.encode_image(images)
        tfeat = model.encode_text(tokens)
    return ifeat, tfeat


def part1_wiring_sanity():
    log.info("=" * 70)
    log.info("PART 1 — wiring sanity check (base vs LoRA-adapted features)")
    log.info("=" * 70)

    base_model, preprocess, tokenizer = build_base()

    from PIL import Image
    test_img = Image.new("RGB", (384, 384), (200, 100, 50))
    img_t = preprocess(test_img).unsqueeze(0).to(DEVICE)
    txt_t = tokenizer(["a red and orange dress"]).to(DEVICE)

    base_i, base_t = features(base_model, img_t, txt_t)
    log.info("BASE model: image_feat[0,:5] = %s", base_i[0, :5].cpu().tolist())
    log.info("BASE model: text_feat[0,:5]  = %s", base_t[0, :5].cpu().tolist())

    for step in (50, 100, 150, 200):
        adapter_dir = CKPT_ROOT / f"step_{step:06d}" / "adapter"
        if not adapter_dir.exists():
            log.warning("adapter missing: %s", adapter_dir)
            continue
        # Reload base fresh each time so previous adapter doesn't linger.
        fresh_base, _, _ = build_base()
        lora_model = attach_lora(fresh_base, adapter_dir).to(DEVICE)
        lora_i, lora_t = features(lora_model, img_t, txt_t)
        di = (lora_i - base_i).abs().mean().item()
        dt = (lora_t - base_t).abs().mean().item()
        cos_i = torch.nn.functional.cosine_similarity(lora_i, base_i, dim=-1).item()
        cos_t = torch.nn.functional.cosine_similarity(lora_t, base_t, dim=-1).item()
        log.info(
            "step_%03d: |Δimg|=%.4e cos(img,base)=%.4f  |Δtxt|=%.4e cos(txt,base)=%.4f",
            step, di, cos_i, dt, cos_t,
        )
        del lora_model, fresh_base
        if DEVICE == "mps":
            torch.mps.empty_cache()

    log.info("interpretation:")
    log.info("  - if |Δ| ~= 0 (and cos ~= 1.0): LoRA had no effect → wiring bug or training collapsed")
    log.info("  - if |Δ| grows monotonically with step + cos drifts away from 1.0: LoRA is wired correctly,")
    log.info("    so the regression is from training (overfitting / wrong objective).")


def part2_merge_intermediates():
    log.info("=" * 70)
    log.info("PART 2 — merge step_50/100/150 adapters into eval-ready state_dicts")
    log.info("=" * 70)

    for step in (50, 100, 150):
        adapter_dir = CKPT_ROOT / f"step_{step:06d}" / "adapter"
        out_dir = CKPT_ROOT / f"step_{step:06d}" / "best"
        out_path = out_dir / "model_state_dict.pt"
        if not adapter_dir.exists():
            log.warning("skip step %d: adapter missing", step); continue
        if out_path.exists():
            log.info("skip step %d: %s already exists", step, out_path); continue
        out_dir.mkdir(parents=True, exist_ok=True)

        log.info("merging step_%03d ...", step)
        base_model, _, _ = build_base()
        lora_model = attach_lora(base_model, adapter_dir)
        merged = lora_model.merge_and_unload()
        torch.save(merged.state_dict(), out_path)
        log.info("  -> %s (%.1f MB)", out_path, out_path.stat().st_size / 1e6)
        del lora_model, base_model, merged
        if DEVICE == "mps":
            torch.mps.empty_cache()


if __name__ == "__main__":
    part1_wiring_sanity()
    part2_merge_intermediates()
    log.info("DONE")
