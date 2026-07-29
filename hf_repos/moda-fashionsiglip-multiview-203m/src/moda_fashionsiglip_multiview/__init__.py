"""MODA FashionSigLIP multi-view retrieval."""

from .retriever import (
    GalleryIndex,
    ModaFashionSigLIP,
    Recipe,
    SearchResult,
    auto_device,
    center_square,
    load_recipe,
    square_pad,
)

__all__ = [
    "GalleryIndex",
    "ModaFashionSigLIP",
    "Recipe",
    "SearchResult",
    "auto_device",
    "center_square",
    "load_recipe",
    "square_pad",
]

__version__ = "0.1.0"
