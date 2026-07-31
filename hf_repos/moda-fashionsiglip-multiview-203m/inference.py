"""
MODA-FashionSigLIP-MultiView-203M - Standalone Inference Script

System:     Multi-view retrieval over the frozen Marqo/marqo-fashionSigLIP
            checkpoint (ViT-B/16-SigLIP, vision + text encoders)
Parameters: 203,155,970 neural, 0 additional learned
Dimensions: 768
Weights:    none stored here; the base checkpoint is downloaded on first use
Benchmark:  4 of 6 significant full-corpus MAP@10 wins over FashionSigLIP,
            positive point estimates on all 6, zero significant losses

This is a retrieval system, not a new checkpoint. The query path runs two text
encodings and combines them into one vector. The gallery path runs three image
views per product and stores three vectors. Scoring is
0.9 * parent + 0.1 * max(parent, pad, crop).

Usage:
    # rank a folder of product images against a text query
    python inference.py --gallery ./images --query "red floral summer dress"

    # embed images (768-d parent vectors, one row per image)
    python inference.py --image img1.jpg img2.jpg

    # embed a query
    python inference.py --query "black leather ankle boots"

    # save the built gallery index for reuse
    python inference.py --gallery ./images --query "navy linen shirt" \
        --save-index ./catalog_index

    # reuse a saved index instead of re-encoding the gallery
    python inference.py --load-index ./catalog_index --query "wool coat"
"""
import argparse
import sys
from pathlib import Path

MODEL_DIR = Path(__file__).parent

# Work from a plain clone as well as an installed package.
if __package__ in (None, ""):
    src = MODEL_DIR / "src"
    if src.is_dir() and str(src) not in sys.path:
        sys.path.insert(0, str(src))

from moda_fashionsiglip_multiview import (  # noqa: E402
    GalleryIndex,
    ModaFashionSigLIP,
)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def collect_images(gallery: Path):
    """Return the sorted supported image files inside a folder."""
    paths = sorted(
        path
        for path in gallery.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not paths:
        raise SystemExit(f"no supported images found in {gallery}")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MODA multi-view FashionSigLIP retrieval",
    )
    parser.add_argument("--query", help="text query to encode or search with")
    parser.add_argument(
        "--image",
        nargs="+",
        type=Path,
        help="image files to embed (768-d parent vectors)",
    )
    parser.add_argument(
        "--gallery",
        type=Path,
        help="folder of images to index and search",
    )
    parser.add_argument("--load-index", type=Path, help="reuse a saved index")
    parser.add_argument("--save-index", type=Path, help="write the built index")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"))
    args = parser.parse_args()

    if not any((args.query, args.image, args.gallery, args.load_index)):
        parser.error("pass at least one of --query, --image, --gallery, --load-index")

    retriever = ModaFashionSigLIP.from_pretrained(device=args.device)

    index = None
    if args.load_index:
        index = GalleryIndex.from_pretrained(args.load_index)
    elif args.gallery:
        paths = collect_images(args.gallery)
        index = retriever.build_index(paths, item_ids=[p.name for p in paths])
        if args.save_index:
            index.save_pretrained(args.save_index)

    # Search: a query against a gallery.
    if index is not None and args.query:
        results = retriever.search(args.query, index, top_k=args.top_k)[0]
        print(f"query: {args.query}")
        for result in results:
            print(f"{result.rank:2d}  {result.score:+.5f}  {result.item_id}")
        return

    # Embed a list of images.
    if args.image:
        built = retriever.build_index(
            args.image,
            item_ids=[path.name for path in args.image],
        )
        vectors = built.parent
        for path, vector in zip(args.image, vectors):
            preview = ", ".join(f"{value:+.4f}" for value in vector[:8].tolist())
            print(f"{path.name}\tdim={vector.shape[0]}\t[{preview}, ...]")
        return

    # Embed a query on its own.
    if args.query:
        vector = retriever.encode_queries(args.query)[0]
        preview = ", ".join(f"{value:+.4f}" for value in vector[:8].tolist())
        print(f"query: {args.query}")
        print(f"dim={vector.shape[0]}\t[{preview}, ...]")
        return

    parser.error("--gallery or --load-index needs a --query to search")


if __name__ == "__main__":
    main()
