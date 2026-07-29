"""Tests for the locked 203M multi-view target evaluator."""

from __future__ import annotations

from benchmark.eval_fashionsiglip_multiview_targets import (
    EVALUATION_ORDER,
    FSL_PARAMETERS,
    IMAGE_WEIGHTS,
    MINIMUM_SIGNIFICANT_WINS,
    SELECTED_ID,
    TEXT_WEIGHTS,
)


def test_locked_deployment_recipe() -> None:
    assert SELECTED_ID == "img-pad_crop025__txt-product025__raw"
    assert FSL_PARAMETERS == 203_155_970
    assert FSL_PARAMETERS < 300_000_000
    assert IMAGE_WEIGHTS == {
        "official": 1.0,
        "pad": 0.25,
        "center_crop": 0.25,
    }
    assert TEXT_WEIGHTS == {"raw": 1.0, "fashion_product": 0.25}
    assert MINIMUM_SIGNIFICANT_WINS == 4


def test_evaluation_order_is_complete_and_size_first() -> None:
    assert len(EVALUATION_ORDER) == 6
    assert set(EVALUATION_ORDER) == {
        "atlas",
        "polyvore",
        "KAGL",
        "fashion200k",
        "deepfashion_inshop",
        "deepfashion_multimodal",
    }
    assert EVALUATION_ORDER == (
        "deepfashion_multimodal",
        "KAGL",
        "deepfashion_inshop",
        "atlas",
        "polyvore",
        "fashion200k",
    )
