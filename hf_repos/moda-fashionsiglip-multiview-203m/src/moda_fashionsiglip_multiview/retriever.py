"""Exact public implementation of the locked MODA multi-view recipe.

The neural checkpoint is the unchanged Marqo FashionSigLIP model. This module
adds deterministic query prompting, image views, and score-level late fusion.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from importlib.resources import files
from pathlib import Path
from typing import Any, Iterable, Sequence

from PIL import Image
from safetensors.torch import load_file, save_file
import torch
import torch.nn.functional as F


ImageInput = str | Path | Image.Image


@dataclass(frozen=True)
class Recipe:
    """Frozen coefficients for the benchmarked retrieval architecture."""

    schema_version: int
    candidate_id: str
    base_model: str
    base_model_parameters: int
    additional_learned_parameters: int
    embedding_dimension: int
    query_prompt_template: str
    query_raw_weight: float
    query_prompt_weight: float
    square_pad_fill: int
    parent_official_weight: float
    parent_pad_weight: float
    parent_crop_weight: float
    late_fusion_parent_weight: float
    late_fusion_max_view_weight: float
    stored_vectors_per_item: int
    ann_routes: int

    def validate(self) -> None:
        if self.schema_version != 1:
            raise ValueError("unsupported recipe schema")
        if self.embedding_dimension != 768:
            raise ValueError("the locked recipe requires 768-D embeddings")
        if self.additional_learned_parameters != 0:
            raise ValueError("the locked recipe adds no learned parameters")
        if self.stored_vectors_per_item != 3 or self.ann_routes != 3:
            raise ValueError("the locked recipe requires three gallery routes")
        if not 0 <= self.square_pad_fill <= 255:
            raise ValueError("square_pad_fill must be an RGB byte value")
        fusion_sum = (
            self.late_fusion_parent_weight
            + self.late_fusion_max_view_weight
        )
        if abs(fusion_sum - 1.0) > 1e-8:
            raise ValueError("late-fusion weights must sum to one")
        positive = (
            self.query_raw_weight,
            self.query_prompt_weight,
            self.parent_official_weight,
            self.parent_pad_weight,
            self.parent_crop_weight,
        )
        if any(weight <= 0 for weight in positive):
            raise ValueError("composition weights must be positive")


def load_recipe() -> Recipe:
    """Load and validate the immutable recipe bundled with the package."""

    recipe_path = files("moda_fashionsiglip_multiview").joinpath("recipe.json")
    payload = json.loads(recipe_path.read_text(encoding="utf-8"))
    recipe = Recipe(**payload)
    recipe.validate()
    return recipe


def auto_device() -> str:
    """Choose CUDA, then Apple MPS, then CPU."""

    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def square_pad(image: Image.Image, fill: int = 128) -> Image.Image:
    """Aspect-preserving square pad used by the locked evaluation."""

    image = image.convert("RGB")
    side = max(image.size)
    canvas = Image.new("RGB", (side, side), (fill, fill, fill))
    canvas.paste(
        image,
        ((side - image.width) // 2, (side - image.height) // 2),
    )
    return canvas


def center_square(image: Image.Image) -> Image.Image:
    """Center-square crop used by the locked evaluation."""

    image = image.convert("RGB")
    side = min(image.size)
    left = (image.width - side) // 2
    top = (image.height - side) // 2
    return image.crop((left, top, left + side, top + side))


def _open_image(value: ImageInput) -> Image.Image:
    if isinstance(value, Image.Image):
        return value.convert("RGB").copy()
    with Image.open(value) as image:
        return image.convert("RGB").copy()


def _batched(values: Sequence[Any], batch_size: int) -> Iterable[Sequence[Any]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    for start in range(0, len(values), batch_size):
        yield values[start : start + batch_size]


@dataclass(frozen=True)
class SearchResult:
    """One ranked gallery result."""

    item_id: str
    score: float
    rank: int
    source_path: str | None = None


@dataclass
class GalleryIndex:
    """The three stored 768-D gallery routes."""

    item_ids: list[str]
    parent: torch.Tensor
    square_pad: torch.Tensor
    center_crop: torch.Tensor
    source_paths: list[str] | None = None

    def __post_init__(self) -> None:
        tensors = (self.parent, self.square_pad, self.center_crop)
        if any(tensor.ndim != 2 for tensor in tensors):
            raise ValueError("gallery tensors must be rank two")
        if not (
            self.parent.shape
            == self.square_pad.shape
            == self.center_crop.shape
        ):
            raise ValueError("gallery route shapes differ")
        if self.parent.shape[0] != len(self.item_ids):
            raise ValueError("item ID count differs from gallery rows")
        if self.parent.shape[1] != 768:
            raise ValueError("gallery vectors must be 768-D")
        if self.source_paths is not None and (
            len(self.source_paths) != len(self.item_ids)
        ):
            raise ValueError("source path count differs from gallery rows")
        if len(set(self.item_ids)) != len(self.item_ids):
            raise ValueError("item IDs must be unique")

    def __len__(self) -> int:
        return len(self.item_ids)

    def save_pretrained(self, directory: str | Path) -> Path:
        """Save vectors without pickle/code-execution risk."""

        output = Path(directory)
        output.mkdir(parents=True, exist_ok=True)
        save_file(
            {
                "parent": self.parent.detach().float().cpu().contiguous(),
                "square_pad": (
                    self.square_pad.detach().float().cpu().contiguous()
                ),
                "center_crop": (
                    self.center_crop.detach().float().cpu().contiguous()
                ),
            },
            str(output / "gallery.safetensors"),
        )
        metadata = {
            "schema_version": 1,
            "architecture": "moda-fashionsiglip-multiview-203m",
            "item_ids": self.item_ids,
            "source_paths": self.source_paths,
            "rows": len(self),
            "dimension": int(self.parent.shape[1]),
            "stored_vectors_per_item": 3,
        }
        (output / "gallery.json").write_text(
            json.dumps(metadata, indent=2) + "\n",
            encoding="utf-8",
        )
        return output

    @classmethod
    def from_pretrained(cls, directory: str | Path) -> "GalleryIndex":
        """Load an index previously written by :meth:`save_pretrained`."""

        source = Path(directory)
        metadata = json.loads(
            (source / "gallery.json").read_text(encoding="utf-8")
        )
        if (
            metadata.get("schema_version") != 1
            or metadata.get("stored_vectors_per_item") != 3
            or metadata.get("dimension") != 768
        ):
            raise ValueError("unsupported or malformed gallery metadata")
        tensors = load_file(str(source / "gallery.safetensors"), device="cpu")
        return cls(
            item_ids=list(metadata["item_ids"]),
            parent=tensors["parent"],
            square_pad=tensors["square_pad"],
            center_crop=tensors["center_crop"],
            source_paths=metadata.get("source_paths"),
        )


class ModaFashionSigLIP:
    """FashionSigLIP plus the locked zero-parameter retrieval architecture."""

    def __init__(
        self,
        *,
        model: Any,
        preprocess: Any,
        tokenizer: Any,
        device: str,
        recipe: Recipe | None = None,
    ) -> None:
        self.model = model
        self.preprocess = preprocess
        self.tokenizer = tokenizer
        self.device = str(device)
        self.recipe = recipe or load_recipe()
        self.recipe.validate()

    @classmethod
    def from_pretrained(
        cls,
        base_model: str | None = None,
        *,
        device: str | None = None,
        cache_dir: str | Path | None = None,
    ) -> "ModaFashionSigLIP":
        """Load the official base model and attach the deterministic recipe."""

        import open_clip

        recipe = load_recipe()
        model_id = base_model or recipe.base_model
        load_name = (
            model_id
            if model_id.startswith("hf-hub:")
            else f"hf-hub:{model_id}"
        )
        selected_device = device or auto_device()
        kwargs: dict[str, Any] = {}
        if cache_dir is not None:
            kwargs["cache_dir"] = str(cache_dir)
        model, _, preprocess = open_clip.create_model_and_transforms(
            load_name,
            **kwargs,
        )
        tokenizer = open_clip.get_tokenizer(load_name, **kwargs)
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        if model_id == recipe.base_model and (
            parameter_count != recipe.base_model_parameters
        ):
            raise RuntimeError(
                "FashionSigLIP parameter count differs from the locked model"
            )
        model = model.to(selected_device).eval()
        for parameter in model.parameters():
            parameter.requires_grad = False
        return cls(
            model=model,
            preprocess=preprocess,
            tokenizer=tokenizer,
            device=selected_device,
            recipe=recipe,
        )

    def architecture(self) -> dict[str, Any]:
        """Return the exact public architecture recipe."""

        return asdict(self.recipe)

    def encode_queries(
        self,
        queries: str | Sequence[str],
        *,
        batch_size: int = 64,
    ) -> torch.Tensor:
        """Encode query text into the single normalized retrieval vector."""

        values = [queries] if isinstance(queries, str) else list(queries)
        if not values:
            raise ValueError("at least one query is required")
        encoded: list[torch.Tensor] = []
        with torch.inference_mode():
            for batch in _batched(values, batch_size):
                prompted = [
                    self.recipe.query_prompt_template.format(query=query)
                    for query in batch
                ]
                tokens = self.tokenizer(list(batch) + prompted).to(self.device)
                features = self.model.encode_text(tokens).float()
                raw, prompt = features.chunk(2, dim=0)
                combined = F.normalize(
                    self.recipe.query_raw_weight * raw
                    + self.recipe.query_prompt_weight * prompt,
                    dim=-1,
                )
                encoded.append(combined.cpu())
        result = torch.cat(encoded, dim=0)
        if result.shape[1] != self.recipe.embedding_dimension:
            raise RuntimeError("query embedding dimension differs")
        return result

    def build_index(
        self,
        images: Sequence[ImageInput],
        *,
        item_ids: Sequence[str] | None = None,
        batch_size: int = 32,
    ) -> GalleryIndex:
        """Encode the three gallery routes and construct an in-memory index."""

        inputs = list(images)
        if not inputs:
            raise ValueError("at least one gallery image is required")
        ids = (
            [str(value) for value in item_ids]
            if item_ids is not None
            else [str(index) for index in range(len(inputs))]
        )
        if len(ids) != len(inputs):
            raise ValueError("item ID count differs from image count")
        paths = [
            str(value) if isinstance(value, (str, Path)) else ""
            for value in inputs
        ]

        official_parts: list[torch.Tensor] = []
        pad_parts: list[torch.Tensor] = []
        crop_parts: list[torch.Tensor] = []
        with torch.inference_mode():
            for batch in _batched(inputs, batch_size):
                opened = [_open_image(value) for value in batch]
                view_tensors = {
                    "official": torch.stack(
                        [self.preprocess(image) for image in opened]
                    ),
                    "square_pad": torch.stack(
                        [
                            self.preprocess(
                                square_pad(
                                    image,
                                    fill=self.recipe.square_pad_fill,
                                )
                            )
                            for image in opened
                        ]
                    ),
                    "center_crop": torch.stack(
                        [
                            self.preprocess(center_square(image))
                            for image in opened
                        ]
                    ),
                }
                view_features: dict[str, torch.Tensor] = {}
                for name, tensors in view_tensors.items():
                    features = self.model.encode_image(
                        tensors.to(self.device)
                    ).float()
                    view_features[name] = F.normalize(features, dim=-1).cpu()
                official_parts.append(view_features["official"])
                pad_parts.append(view_features["square_pad"])
                crop_parts.append(view_features["center_crop"])

        official = torch.cat(official_parts)
        padded = torch.cat(pad_parts)
        cropped = torch.cat(crop_parts)
        parent = F.normalize(
            self.recipe.parent_official_weight * official
            + self.recipe.parent_pad_weight * padded
            + self.recipe.parent_crop_weight * cropped,
            dim=-1,
        )
        return GalleryIndex(
            item_ids=ids,
            parent=parent,
            square_pad=padded,
            center_crop=cropped,
            source_paths=paths,
        )

    def search(
        self,
        queries: str | Sequence[str],
        index: GalleryIndex,
        *,
        top_k: int = 10,
        query_batch_size: int = 32,
        document_batch_size: int = 8192,
    ) -> list[list[SearchResult]]:
        """Run exact chunked three-route retrieval.

        For large production galleries, use the three tensors as separate ANN
        indexes and apply the same score formula to the union of candidates.
        """

        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if len(index) == 0:
            raise ValueError("gallery index is empty")
        if document_batch_size <= 0:
            raise ValueError("document_batch_size must be positive")
        query_features = self.encode_queries(
            queries,
            batch_size=query_batch_size,
        )
        output: list[list[SearchResult]] = []
        k = min(top_k, len(index))
        with torch.inference_mode():
            for query in query_features:
                query = query.to(self.device)
                best_scores = torch.empty(0, dtype=torch.float32)
                best_indices = torch.empty(0, dtype=torch.long)
                for start in range(0, len(index), document_batch_size):
                    end = min(start + document_batch_size, len(index))
                    parent = F.normalize(
                        index.parent[start:end].to(self.device).float(),
                        dim=-1,
                    )
                    parent_scores = (
                        query @ parent.T
                    )
                    max_scores = parent_scores.clone()
                    for route in (index.square_pad, index.center_crop):
                        normalized_route = F.normalize(
                            route[start:end].to(self.device).float(),
                            dim=-1,
                        )
                        max_scores = torch.maximum(
                            max_scores,
                            query @ normalized_route.T,
                        )
                    fused = (
                        self.recipe.late_fusion_parent_weight * parent_scores
                        + self.recipe.late_fusion_max_view_weight * max_scores
                    ).cpu()
                    candidate_indices = torch.arange(start, end)
                    combined_scores = torch.cat((best_scores, fused))
                    combined_indices = torch.cat(
                        (best_indices, candidate_indices)
                    )
                    best_scores, positions = combined_scores.topk(
                        min(k, combined_scores.numel())
                    )
                    best_indices = combined_indices[positions]
                rows: list[SearchResult] = []
                for rank, (score, item_index) in enumerate(
                    zip(best_scores.tolist(), best_indices.tolist()),
                    start=1,
                ):
                    source_path = (
                        index.source_paths[item_index]
                        if index.source_paths is not None
                        else None
                    )
                    rows.append(
                        SearchResult(
                            item_id=index.item_ids[item_index],
                            score=float(score),
                            rank=rank,
                            source_path=source_path or None,
                        )
                    )
                output.append(rows)
        return output
