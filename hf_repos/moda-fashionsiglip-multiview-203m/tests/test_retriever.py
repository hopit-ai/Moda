from __future__ import annotations

import json

from PIL import Image
import pytest
import torch
import torch.nn.functional as F

from moda_fashionsiglip_multiview import (
    GalleryIndex,
    ModaFashionSigLIP,
    center_square,
    load_recipe,
    square_pad,
)


class StaticRetriever(ModaFashionSigLIP):
    def __init__(self, query: torch.Tensor):
        self.device = "cpu"
        self.recipe = load_recipe()
        self._query = F.normalize(query.float(), dim=-1).reshape(1, -1)

    def encode_queries(self, queries, *, batch_size=64):
        del queries, batch_size
        return self._query


def unit(*indices: int) -> torch.Tensor:
    value = torch.zeros(768)
    for index in indices:
        value[index] = 1.0
    return F.normalize(value, dim=-1)


def test_recipe_is_the_frozen_iteration6_candidate():
    recipe = load_recipe()
    assert recipe.candidate_id == "late-maxview-b010"
    assert recipe.base_model_parameters == 203_155_970
    assert recipe.additional_learned_parameters == 0
    assert recipe.query_prompt_weight == pytest.approx(0.25)
    assert recipe.parent_pad_weight == pytest.approx(0.25)
    assert recipe.parent_crop_weight == pytest.approx(0.25)
    assert recipe.late_fusion_max_view_weight == pytest.approx(0.10)


def test_locked_image_views_match_expected_geometry_and_fill():
    image = Image.new("RGB", (6, 2), (255, 0, 0))
    padded = square_pad(image)
    cropped = center_square(image)

    assert padded.size == (6, 6)
    assert padded.getpixel((0, 0)) == (128, 128, 128)
    assert padded.getpixel((0, 2)) == (255, 0, 0)
    assert cropped.size == (2, 2)
    assert cropped.getpixel((0, 0)) == (255, 0, 0)


def test_three_route_fusion_can_promote_a_view_match():
    query = unit(0)
    parent = torch.stack([unit(0, 1), unit(0, 2)])
    padded = torch.stack([unit(1), unit(0)])
    cropped = torch.stack([unit(1), unit(2)])
    index = GalleryIndex(
        item_ids=["stable-parent", "view-correction"],
        parent=parent,
        square_pad=padded,
        center_crop=cropped,
    )

    results = StaticRetriever(query).search("query", index, top_k=2)[0]

    assert [result.item_id for result in results] == [
        "view-correction",
        "stable-parent",
    ]
    assert results[0].score > results[1].score


def test_gallery_safetensors_round_trip(tmp_path):
    index = GalleryIndex(
        item_ids=["a", "b"],
        parent=torch.stack([unit(0), unit(1)]),
        square_pad=torch.stack([unit(2), unit(3)]),
        center_crop=torch.stack([unit(4), unit(5)]),
        source_paths=["a.jpg", "b.jpg"],
    )

    index.save_pretrained(tmp_path)
    restored = GalleryIndex.from_pretrained(tmp_path)

    assert restored.item_ids == index.item_ids
    assert restored.source_paths == index.source_paths
    assert torch.equal(restored.parent, index.parent)
    metadata = json.loads((tmp_path / "gallery.json").read_text())
    assert metadata["stored_vectors_per_item"] == 3


def test_gallery_rejects_duplicate_item_ids():
    vectors = torch.stack([unit(0), unit(1)])
    with pytest.raises(ValueError, match="unique"):
        GalleryIndex(
            item_ids=["same", "same"],
            parent=vectors,
            square_pad=vectors,
            center_crop=vectors,
        )
