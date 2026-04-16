"""
MODA Phase 1 — Model Loading & Encoding Utilities

Provides a unified interface for:
  - OpenAI/Marqo CLIP models via open_clip
  - Sentence-Transformers models

MODEL_REGISTRY maps short friendly names → HuggingFace model IDs.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

MODEL_REGISTRY: dict[str, dict[str, str]] = {
    "clip": {
        "hf_id": "openai/clip-vit-base-patch32",
        "type": "clip",
    },
    "fashion-clip": {
        "hf_id": "Marqo/marqo-fashionCLIP",
        "type": "clip",
    },
    "fashion-siglip": {
        "hf_id": "Marqo/marqo-fashionSigLIP",
        "type": "clip",
    },
}


def resolve_model_name(name_or_id: str) -> dict[str, str]:
    """Return registry entry for a friendly name or pass-through HF ID.

    Args:
        name_or_id: Either a key in MODEL_REGISTRY (e.g. ``"fashion-clip"``)
            or a raw HuggingFace model ID / local path.

    Returns:
        Dict with keys ``"hf_id"`` and ``"type"``.
    """
    if name_or_id in MODEL_REGISTRY:
        return MODEL_REGISTRY[name_or_id]
    # Treat as raw HF ID; assume CLIP architecture by default
    return {"hf_id": name_or_id, "type": "clip"}


# ---------------------------------------------------------------------------
# CLIP-style model loading (open_clip)
# ---------------------------------------------------------------------------


def load_clip_model(
    model_name: str,
    device: str = "cpu",
) -> tuple[Any, Any, Any]:
    """Load a CLIP-style model via open_clip.

    Supports any model listed in MODEL_REGISTRY or a raw HuggingFace model ID
    that open_clip can load.

    Args:
        model_name: Friendly name (from MODEL_REGISTRY) or raw HF model ID.
        device: Torch device string, e.g. ``"cpu"``, ``"cuda"``, ``"mps"``.

    Returns:
        Tuple of ``(model, preprocess, tokenizer)`` ready for inference.

    Raises:
        ImportError: If open_clip is not installed.
        RuntimeError: If the model cannot be loaded.
    """
    try:
        import open_clip  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "open_clip is required for CLIP models. "
            "Install with: pip install open-clip-torch"
        ) from exc

    import torch  # type: ignore

    entry = resolve_model_name(model_name)
    hf_id = entry["hf_id"]

    logger.info("Loading CLIP model '%s' (hf_id=%s) on %s …", model_name, hf_id, device)

    try:
        model, _, preprocess = open_clip.create_model_and_transforms(
            "hf-hub:" + hf_id
        )
        tokenizer = open_clip.get_tokenizer("hf-hub:" + hf_id)
    except Exception:
        # Fallback: try the model name directly (e.g. "ViT-B-32")
        arch, pretrained = _split_openclip_name(hf_id)
        model, _, preprocess = open_clip.create_model_and_transforms(
            arch, pretrained=pretrained
        )
        tokenizer = open_clip.get_tokenizer(arch)

    model = model.to(device)
    model.eval()
    return model, preprocess, tokenizer


def load_clip_model_from_checkpoint(
    checkpoint_path: str,
    base_model: str = "fashion-clip",
    device: str = "cpu",
) -> tuple[Any, Any, Any]:
    """Load a CLIP model and overlay a local ``state_dict`` checkpoint.

    Used for fine-tuned models (e.g. Phase 4F multimodal) that share the
    same architecture as a base open_clip model but have updated weights.
    """
    import torch

    model, preprocess, tokenizer = load_clip_model(base_model, device="cpu")
    logger.info("Loading checkpoint weights from %s", checkpoint_path)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=False)
    model = model.to(device)
    model.eval()
    return model, preprocess, tokenizer


def _split_openclip_name(hf_id: str) -> tuple[str, str]:
    """Attempt to split an HF model ID into (arch, pretrained) for open_clip."""
    # Marqo models use hf-hub prefix; plain clip uses "ViT-B-32" / "laion2b_s34b_b79k"
    known = {
        "openai/clip-vit-base-patch32": ("ViT-B-32", "openai"),
        "openai/clip-vit-large-patch14": ("ViT-L-14", "openai"),
    }
    if hf_id in known:
        return known[hf_id]
    # Best-effort: use the full string as arch with no pretrained tag
    return hf_id, ""


# ---------------------------------------------------------------------------
# Sentence-Transformers loading
# ---------------------------------------------------------------------------


def load_sentence_transformer(model_name: str) -> Any:
    """Load a Sentence-Transformers model.

    Args:
        model_name: Friendly name (from MODEL_REGISTRY) or HF model ID.

    Returns:
        A ``SentenceTransformer`` instance.

    Raises:
        ImportError: If sentence-transformers is not installed.
    """
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "sentence-transformers is required. "
            "Install with: pip install sentence-transformers"
        ) from exc

    entry = resolve_model_name(model_name)
    hf_id = entry["hf_id"]
    logger.info("Loading SentenceTransformer '%s' …", hf_id)
    return SentenceTransformer(hf_id)


# ---------------------------------------------------------------------------
# Encoding utilities
# ---------------------------------------------------------------------------


def encode_texts_clip(
    texts: list[str],
    model: Any,
    tokenizer: Any,
    device: str,
    batch_size: int = 64,
    normalize: bool = True,
) -> np.ndarray:
    """Encode a list of texts with a CLIP model.

    Args:
        texts: Input strings.
        model: CLIP model (open_clip).
        tokenizer: Matching open_clip tokenizer.
        device: Torch device string.
        batch_size: Number of texts per forward pass.
        normalize: L2-normalise output embeddings (enables cosine sim via dot product).

    Returns:
        Float32 numpy array of shape ``(len(texts), embed_dim)``.
    """
    try:
        import torch  # type: ignore
        from tqdm import tqdm  # type: ignore
    except ImportError as exc:
        raise ImportError("torch and tqdm are required.") from exc

    all_embeddings: list[np.ndarray] = []

    with torch.no_grad():
        for start in tqdm(range(0, len(texts), batch_size), desc="Encoding texts (CLIP)", leave=False):
            batch = texts[start : start + batch_size]
            tokens = tokenizer(batch).to(device)
            features = model.encode_text(tokens)
            if normalize:
                features = features / features.norm(dim=-1, keepdim=True)
            all_embeddings.append(features.cpu().numpy().astype(np.float32))

    return np.vstack(all_embeddings)


def encode_images_clip(
    image_paths: list[str],
    model: Any,
    preprocess: Any,
    device: str,
    batch_size: int = 64,
    normalize: bool = True,
) -> np.ndarray:
    """Encode a list of image paths with a CLIP model.

    Args:
        image_paths: List of paths to image files.
        model: CLIP model (open_clip).
        preprocess: Preprocessing transform from open_clip.
        device: Torch device string.
        batch_size: Number of images per forward pass.
        normalize: L2-normalise output embeddings.

    Returns:
        Float32 numpy array of shape ``(len(image_paths), embed_dim)``.
    """
    try:
        import torch  # type: ignore
        from PIL import Image  # type: ignore
        from tqdm import tqdm  # type: ignore
    except ImportError as exc:
        raise ImportError("torch, Pillow, and tqdm are required.") from exc

    all_embeddings: list[np.ndarray] = []

    with torch.no_grad():
        for start in tqdm(range(0, len(image_paths), batch_size), desc="Encoding images (CLIP)", leave=False):
            batch_paths = image_paths[start : start + batch_size]
            images = torch.stack(
                [preprocess(Image.open(p).convert("RGB")) for p in batch_paths]
            ).to(device)
            features = model.encode_image(images)
            if normalize:
                features = features / features.norm(dim=-1, keepdim=True)
            all_embeddings.append(features.cpu().numpy().astype(np.float32))

    return np.vstack(all_embeddings)


def encode_pil_images_clip(
    images: list,
    model: Any,
    preprocess: Any,
    device: str,
    batch_size: int = 64,
    normalize: bool = True,
) -> np.ndarray:
    """Encode a list of PIL (or decodeable) images with a CLIP vision tower.

    Args:
        images: List of PIL.Image.Image (RGB) or objects accepted by ``preprocess``.
        model: open_clip model with ``encode_image``.
        preprocess: open_clip image transform.
        device: Torch device.
        batch_size: Batch size for forward passes.
        normalize: L2-normalize embeddings.

    Returns:
        Float32 array ``(N, D)``.
    """
    import torch
    from tqdm import tqdm

    out: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in tqdm(
            range(0, len(images), batch_size),
            desc="Encoding images (CLIP)",
            leave=False,
        ):
            batch = images[start : start + batch_size]
            t = torch.stack([preprocess(im.convert("RGB")) for im in batch]).to(device)
            feat = model.encode_image(t)
            if normalize:
                feat = feat / feat.norm(dim=-1, keepdim=True)
            out.append(feat.cpu().numpy().astype(np.float32))
    return np.vstack(out)


def encode_texts_st(
    texts: list[str],
    model: Any,
    batch_size: int = 64,
    normalize: bool = True,
) -> np.ndarray:
    """Encode texts using a Sentence-Transformers model.

    Args:
        texts: Input strings.
        model: ``SentenceTransformer`` instance.
        batch_size: Number of texts per forward pass.
        normalize: L2-normalise output embeddings.

    Returns:
        Float32 numpy array of shape ``(len(texts), embed_dim)``.
    """
    embeddings: np.ndarray = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=normalize,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    return embeddings.astype(np.float32)


# ---------------------------------------------------------------------------
# Unified encode dispatcher
# ---------------------------------------------------------------------------


def encode_texts(
    texts: list[str],
    model_name: str,
    device: str = "cpu",
    batch_size: int = 64,
) -> np.ndarray:
    """High-level dispatcher: encode texts with any supported model.

    Automatically selects the right backend (CLIP vs. SentenceTransformers)
    based on MODEL_REGISTRY or the model name heuristic.

    Args:
        texts: Input strings.
        model_name: Friendly name or HF model ID.
        device: Torch device.
        batch_size: Batch size.

    Returns:
        Float32 numpy array of shape ``(len(texts), embed_dim)``.
    """
    entry = resolve_model_name(model_name)
    model_type = entry.get("type", "clip")

    if model_type == "clip":
        model, _, tokenizer = load_clip_model(model_name, device=device)
        return encode_texts_clip(texts, model, tokenizer, device, batch_size)
    elif model_type == "st":
        model = load_sentence_transformer(model_name)
        return encode_texts_st(texts, model, batch_size)
    else:
        raise ValueError(f"Unknown model type '{model_type}' for model '{model_name}'")
