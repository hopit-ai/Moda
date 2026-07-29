"""Tests for the zero-extra-parameter multi-view external gate."""

from __future__ import annotations

from PIL import Image
import torch

from benchmark.eval_fashionsiglip_multiview_external import (
    CENTERING_MODES,
    EMBED_DIM,
    FSL_PARAMETERS,
    IMAGE_MIXES,
    PROMPT_MIXES,
    center_pair,
    compose,
    foreground_square,
    square_pad,
)


def test_deployment_and_grid_are_fixed() -> None:
    assert FSL_PARAMETERS == 203_155_970
    assert FSL_PARAMETERS < 300_000_000
    assert EMBED_DIM == 768
    assert len(IMAGE_MIXES) * len(PROMPT_MIXES) * len(CENTERING_MODES) == 88


def test_square_views_preserve_square_output() -> None:
    image = Image.new("RGB", (40, 80), "white")
    image.paste(Image.new("RGB", (10, 30), "black"), (15, 25))
    assert square_pad(image).size == (80, 80)
    foreground = foreground_square(image)
    assert foreground.width == foreground.height
    assert foreground.width < 80


def test_compose_and_center_return_unit_vectors() -> None:
    torch.manual_seed(11)
    features = {
        "a": torch.randn(4, 6),
        "b": torch.randn(4, 6),
    }
    value = compose(features, {"a": 1.0, "b": 0.25})
    assert torch.allclose(value.norm(dim=-1), torch.ones(4), atol=1e-6)
    query, document = center_pair(value, torch.randn(4, 6))
    assert torch.allclose(query.norm(dim=-1), torch.ones(4), atol=1e-6)
    assert torch.allclose(document.norm(dim=-1), torch.ones(4), atol=1e-6)
