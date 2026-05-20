"""SigLIP-B/16 backbone with multi-field weighted document embedding."""
from __future__ import annotations

import open_clip
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiFieldDocHead(nn.Module):
    """
    Learnable γ weights for combining image + title + description embeddings.
    γ = softmax(logits) so they sum to 1 and are always positive.
    """
    def __init__(self, n_fields: int = 3):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(n_fields))  # uniform init

    def forward(
        self,
        img_emb: torch.Tensor,
        title_emb: torch.Tensor | None = None,
        desc_emb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        γ = F.softmax(self.logits, dim=0)
        out = γ[0] * img_emb
        if title_emb is not None:
            out = out + γ[1] * title_emb
        if desc_emb is not None:
            out = out + γ[2] * desc_emb
        return out


def build_siglip_b16(device: str, cache_dir: str = ".cache") -> tuple:
    """
    Load ViT-B-16-SigLIP (plain webli, 203.2M params).
    Returns (model, tokenizer, preprocess).
    """
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-16-SigLIP", pretrained="webli", cache_dir=cache_dir
    )
    tokenizer = open_clip.get_tokenizer("ViT-B-16-SigLIP")
    model = model.to(device)
    return model, tokenizer, preprocess


class FashionSigLIPModel(nn.Module):
    """
    Wrapper around SigLIP-B/16 with multi-field document head.
    Query (LHS): text-only.
    Document (RHS): γ_img * image_emb + γ_title * title_emb + γ_desc * desc_emb
    """
    def __init__(self, device: str, cache_dir: str = ".cache"):
        super().__init__()
        self.backbone, self.tokenizer, self.preprocess = build_siglip_b16(device, cache_dir)
        self.doc_head = MultiFieldDocHead(n_fields=3)
        self.device = device

    def encode_query(self, texts: list[str]) -> torch.Tensor:
        tokens = self.tokenizer(texts).to(self.device)
        return self.backbone.encode_text(tokens)

    def encode_doc(
        self,
        images: torch.Tensor,
        titles: list[str] | None = None,
        descs: list[str] | None = None,
        use_multifield: bool = True,
    ) -> torch.Tensor:
        img_emb = self.backbone.encode_image(images)
        if not use_multifield or (titles is None and descs is None):
            return img_emb

        title_emb = None
        if titles is not None:
            title_tokens = self.tokenizer(titles).to(self.device)
            title_emb = self.backbone.encode_text(title_tokens)

        desc_emb = None
        if descs is not None:
            nonempty = [d if d else t for d, t in zip(descs, titles or [""] * len(descs))]
            desc_tokens = self.tokenizer(nonempty).to(self.device)
            desc_emb = self.backbone.encode_text(desc_tokens)

        return self.doc_head(img_emb, title_emb, desc_emb)

    @property
    def logit_scale(self):
        return self.backbone.logit_scale

    @property
    def logit_bias(self):
        return getattr(self.backbone, "logit_bias", None)

    def forward(self, images, queries, titles=None, descs=None):
        q_emb = self.encode_query(queries)
        d_emb = self.encode_doc(images, titles, descs)
        return q_emb, d_emb


def freeze_backbone(model: FashionSigLIPModel):
    """Stage 1: freeze both towers, train only heads + logit scale/bias."""
    for p in model.backbone.parameters():
        p.requires_grad = False
    # Unfreeze logit scale, logit bias, doc_head
    if hasattr(model.backbone, "logit_scale"):
        model.backbone.logit_scale.requires_grad = True
    if hasattr(model.backbone, "logit_bias"):
        model.backbone.logit_bias.requires_grad = True
    for p in model.doc_head.parameters():
        p.requires_grad = True
    n = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"  Stage 1: frozen backbone, trainable={n:.2f}M params")


def unfreeze_all(model: FashionSigLIPModel):
    """Stage 2+: unfreeze everything."""
    for p in model.parameters():
        p.requires_grad = True
    n = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"  Stage 2: full fine-tune, trainable={n:.2f}M params")
