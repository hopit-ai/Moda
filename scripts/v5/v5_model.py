"""
v5 student model wrapper.

Backbone: ViT-B-16-SigLIP2-384/webli (open_clip).

Trainable scope (per PLAN_V5 §B.2, with one simplification):
  - Text tower blocks 8-11           (~28M params)
  - Text projection                  (~0.6M)
  - logit_scale, logit_bias          (2 params)
  Total trainable: ~28.6M

Frozen:
  - Image tower (entire branch incl. its projection)
  - Text tower blocks 0-7
  - Token embedding, positional embedding, final norm

The image branch is fully frozen because we read pre-computed SL2 image
embeddings from disk (student_image_emb.pt) — there is no need to forward
images during training.
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

import open_clip


def _freeze_and_unfreeze(model: nn.Module, scope: str) -> None:
    """Freeze entire model, then unfreeze text tower per scope."""
    for p in model.parameters():
        p.requires_grad = False

    text_transformer = model.text.transformer
    n_blocks = len(text_transformer.resblocks)
    if scope == "text_4blocks":
        n_unfreeze = 4
    elif scope == "text_1block":
        n_unfreeze = 1
    elif scope == "heads_only":
        n_unfreeze = 0
    else:
        raise ValueError(f"unknown scope: {scope}")
    for i in range(max(0, n_blocks - n_unfreeze), n_blocks):
        for p in text_transformer.resblocks[i].parameters():
            p.requires_grad = True

    if hasattr(model.text, "text_projection") and model.text.text_projection is not None:
        try:
            model.text.text_projection.requires_grad = True
        except AttributeError:
            pass
    if hasattr(model.text, "ln_final"):
        for p in model.text.ln_final.parameters():
            p.requires_grad = True
    if hasattr(model, "logit_scale"):
        model.logit_scale.requires_grad = True
    if hasattr(model, "logit_bias"):
        model.logit_bias.requires_grad = True


def build_student(device: str = "mps", scope: str = "text_4blocks") -> tuple[nn.Module, "open_clip.SimpleTokenizer"]:
    """Construct the SigLIP-2 student model with a configurable freeze schedule.

    scope:
        "text_4blocks" — text tower last 4 blocks + ln_final + logit (~28M params).
                         Default; used by v1 and v2.
        "text_1block"  — text tower last 1 block + ln_final + logit (~7M params).
                         Iteration 3a — minimal capacity for drift.
        "heads_only"   — only ln_final + logit (~1.5K params). Most conservative.

    Always frozen: image tower (entire branch), text token embedding, positional
    embedding, all text blocks below the unfreeze cutoff.
    """
    model, _, _ = open_clip.create_model_and_transforms("ViT-B-16-SigLIP2-384", pretrained="webli")
    tokenizer = open_clip.get_tokenizer("ViT-B-16-SigLIP2-384")
    model = model.to(device)
    _freeze_and_unfreeze(model, scope)
    return model, tokenizer


def build_fsl_student(device: str = "mps", scope: str = "text_1block") -> tuple[nn.Module, "open_clip.SimpleTokenizer"]:
    """Marqo-FashionSigLIP as student — starts with FSL's fashion-domain expertise.

    Uses FSL's image tower (frozen) and fine-tunes its text tower toward the
    SL2-B teacher's atlas/polyvore semantic space.  The training image cache
    must be teacher_fsl_img_emb.pt (FSL image embeddings, not SL2-B).
    """
    model, _, _ = open_clip.create_model_and_transforms(
        "hf-hub:Marqo/marqo-fashionSigLIP"
    )
    tokenizer = open_clip.get_tokenizer("hf-hub:Marqo/marqo-fashionSigLIP")
    model = model.to(device)
    _freeze_and_unfreeze(model, scope)
    return model, tokenizer


def trainable_parameter_groups(model: nn.Module) -> list[dict]:
    """Return param groups suitable for AdamW with separate LR for logit_scale/bias."""
    big_lr_params, normal_params = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "logit_scale" in n or "logit_bias" in n:
            big_lr_params.append(p)
        else:
            normal_params.append(p)
    return [
        {"params": normal_params, "lr": 5e-6, "weight_decay": 0.01},
        {"params": big_lr_params, "lr": 1e-4, "weight_decay": 0.0},
    ]


def count_trainable(model: nn.Module) -> dict[str, int]:
    """Count trainable params by major component for sanity-checking."""
    counts = {"text_blocks": 0, "text_proj": 0, "text_ln": 0, "logit": 0, "other": 0, "total": 0}
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        n_p = p.numel()
        counts["total"] += n_p
        if "text.transformer.resblocks" in n:
            counts["text_blocks"] += n_p
        elif "text_projection" in n:
            counts["text_proj"] += n_p
        elif "text.ln_final" in n:
            counts["text_ln"] += n_p
        elif "logit_scale" in n or "logit_bias" in n:
            counts["logit"] += n_p
        else:
            counts["other"] += n_p
    return counts
