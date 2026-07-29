"""Command-line search over a local image folder."""

from __future__ import annotations

import argparse
from pathlib import Path

from .retriever import ModaFashionSigLIP


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gallery", type=Path, required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"))
    parser.add_argument("--save-index", type=Path)
    args = parser.parse_args()

    image_paths = sorted(
        path
        for path in args.gallery.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not image_paths:
        raise SystemExit(f"no supported images found in {args.gallery}")

    retriever = ModaFashionSigLIP.from_pretrained(device=args.device)
    index = retriever.build_index(
        image_paths,
        item_ids=[path.name for path in image_paths],
    )
    if args.save_index:
        index.save_pretrained(args.save_index)

    for result in retriever.search(args.query, index, top_k=args.top_k)[0]:
        print(f"{result.rank:2d}  {result.score:+.5f}  {result.item_id}")


if __name__ == "__main__":
    main()
