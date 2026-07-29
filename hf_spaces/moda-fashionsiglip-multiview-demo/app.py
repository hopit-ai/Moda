"""Interactive Hugging Face Space for MODA multi-view fashion retrieval."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from threading import Lock
from typing import Any

import gradio as gr
from PIL import Image

from moda_fashionsiglip_multiview import GalleryIndex, ModaFashionSigLIP


MAX_IMAGES = 24
MAX_FILE_BYTES = 15 * 1024 * 1024
MAX_IMAGE_PIXELS = 25_000_000
MODEL_LOCK = Lock()


@lru_cache(maxsize=1)
def get_retriever() -> ModaFashionSigLIP:
    """Load the upstream model once per Space process."""

    return ModaFashionSigLIP.from_pretrained()


def _file_path(value: Any) -> str:
    if isinstance(value, str):
        return value
    name = getattr(value, "name", None)
    if name:
        return str(name)
    raise ValueError("unsupported uploaded file value")


def index_gallery(files: list[Any] | None) -> tuple[GalleryIndex, str]:
    if not files:
        raise gr.Error("Upload at least one image.")
    if len(files) > MAX_IMAGES:
        raise gr.Error(f"Use at most {MAX_IMAGES} images in the public demo.")
    paths = [_file_path(value) for value in files]
    for path in paths:
        if Path(path).stat().st_size > MAX_FILE_BYTES:
            raise gr.Error(
                f"{Path(path).name} is larger than the 15 MiB demo limit."
            )
        with Image.open(path) as image:
            if image.width * image.height > MAX_IMAGE_PIXELS:
                raise gr.Error(
                    f"{Path(path).name} exceeds the 25-megapixel demo limit."
                )
    item_ids = [
        f"{position:02d}-{Path(path).name}"
        for position, path in enumerate(paths, start=1)
    ]

    with MODEL_LOCK:
        retriever = get_retriever()
        index = retriever.build_index(paths, item_ids=item_ids, batch_size=8)
    message = (
        f"Indexed {len(index)} images on {retriever.device}. "
        "The gallery now contains three 768-D vectors per image."
    )
    return index, message


def search_gallery(
    query: str,
    index: GalleryIndex | None,
    top_k: int,
) -> list[tuple[str, str]]:
    if index is None:
        raise gr.Error("Build the gallery index first.")
    query = query.strip()
    if not query:
        raise gr.Error("Enter a fashion search query.")

    with MODEL_LOCK:
        retriever = get_retriever()
        results = retriever.search(
            query,
            index,
            top_k=min(int(top_k), len(index)),
        )[0]
    return [
        (
            result.source_path or "",
            f"#{result.rank} · {result.item_id} · score {result.score:.4f}",
        )
        for result in results
    ]


with gr.Blocks(title="MODA FashionSigLIP Multi-View Search") as demo:
    gr.Markdown(
        """
        # 👗 MODA FashionSigLIP Multi-View Search

        Try the exact **203M-parameter, zero-additional-parameter** architecture:
        two query encodings, three deterministic image views, and conservative
        late fusion. Upload a small catalog, build the index, then search.
        """
    )
    gallery_state = gr.State(value=None)

    with gr.Row():
        with gr.Column(scale=1):
            uploads = gr.File(
                label=f"Catalog images (up to {MAX_IMAGES})",
                file_count="multiple",
                file_types=["image"],
                type="filepath",
            )
            build_button = gr.Button("Build multi-view index", variant="primary")
            status = gr.Textbox(
                label="Index status",
                value="No gallery indexed yet.",
                interactive=False,
            )
        with gr.Column(scale=1):
            query = gr.Textbox(
                label="Text query",
                placeholder="e.g. red floral summer dress",
            )
            top_k = gr.Slider(
                minimum=1,
                maximum=12,
                step=1,
                value=6,
                label="Top K",
            )
            search_button = gr.Button("Search", variant="primary")

    results_gallery = gr.Gallery(
        label="Ranked results",
        columns=3,
        object_fit="contain",
        height=600,
    )

    gr.Markdown(
        """
        **What is running:** unchanged Marqo FashionSigLIP weights plus the
        published MODA multi-view recipe. The demo performs exact scoring for
        the uploaded catalog; production deployments can use three ANN routes.
        """
    )

    build_button.click(
        fn=index_gallery,
        inputs=[uploads],
        outputs=[gallery_state, status],
    )
    search_button.click(
        fn=search_gallery,
        inputs=[query, gallery_state, top_k],
        outputs=[results_gallery],
    )
    query.submit(
        fn=search_gallery,
        inputs=[query, gallery_state, top_k],
        outputs=[results_gallery],
    )


if __name__ == "__main__":
    demo.launch()
