"""
MODA Phase 1 — H&M Article Embedding Script

Encodes all H&M articles into dense vectors and builds a FAISS index for
fast approximate nearest-neighbour retrieval.

Inputs  (from data/raw/hnm/):
  articles.csv — 105K products with text fields

Outputs (to --output_dir, default data/processed/embeddings/):
  {model_name}_embeddings.npy     — float32 array (N, D)
  {model_name}_article_ids.json   — list of article_ids in embedding order
  {model_name}_faiss.index        — FAISS IndexFlatIP (cosine after L2-norm)
  {model_name}_meta.json          — run metadata (model, timing, counts)

Usage:
  python benchmark/embed_hnm.py --model fashion-clip --batch_size 64
  python benchmark/embed_hnm.py --model clip --output_dir data/processed/embeddings
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

# Project root on sys.path so sibling modules resolve when run standalone
_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from benchmark.models import encode_texts_clip, load_clip_model, MODEL_REGISTRY

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Article text construction
# ---------------------------------------------------------------------------


def build_article_text(row: dict) -> str:
    """Concatenate product fields into a single descriptive text string.

    The order (name → type → colour → description) mirrors the visual reading
    order of a product card and gives CLIP the richest signal.
    """
    parts = [
        str(row.get("prod_name", "") or "").strip(),
        str(row.get("product_type_name", "") or "").strip(),
        str(row.get("colour_group_name", "") or "").strip(),
        str(row.get("detail_desc", "") or "").strip(),
    ]
    return " ".join(p for p in parts if p)


# ---------------------------------------------------------------------------
# FAISS index helpers
# ---------------------------------------------------------------------------


def build_faiss_index(embeddings: np.ndarray) -> "faiss.Index":
    """Build an IndexFlatIP (inner product) FAISS index.

    Assumes embeddings are already L2-normalised so inner product == cosine
    similarity.

    Args:
        embeddings: Float32 array of shape (N, D).

    Returns:
        Populated FAISS index.
    """
    try:
        import faiss  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "faiss-cpu is required. Install with: pip install faiss-cpu"
        ) from exc

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    logger.info("FAISS index built: %d vectors, dim=%d", index.ntotal, dim)
    return index


def save_faiss_index(index: "faiss.Index", path: Path) -> None:
    try:
        import faiss  # type: ignore
    except ImportError as exc:
        raise ImportError("faiss-cpu is required.") from exc
    faiss.write_index(index, str(path))
    logger.info("FAISS index saved to %s", path)


# ---------------------------------------------------------------------------
# Main embedding pipeline
# ---------------------------------------------------------------------------


def embed_articles(
    articles_csv: Path,
    model_name: str,
    output_dir: Path,
    batch_size: int = 64,
    device: str = "cpu",
) -> dict:
    """Embed all H&M articles and save artefacts.

    Args:
        articles_csv: Path to articles.csv.
        model_name: Friendly model name or HF model ID.
        output_dir: Directory to save outputs.
        batch_size: Encoding batch size.
        device: Torch device string.

    Returns:
        Metadata dict with timing and count info.
    """
    import pandas as pd  # type: ignore
    from tqdm import tqdm  # type: ignore

    if not articles_csv.exists():
        raise FileNotFoundError(
            f"articles.csv not found at {articles_csv}. "
            "Ensure Agent 1 has downloaded the H&M dataset."
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    # Sanitise model name for use in filenames
    safe_name = model_name.replace("/", "_").replace(":", "_")

    logger.info("Loading articles from %s …", articles_csv)
    df = pd.read_csv(articles_csv, dtype=str)
    logger.info("Loaded %d articles.", len(df))

    # Build text representations
    logger.info("Building article text representations …")
    texts = [build_article_text(row) for row in tqdm(df.to_dict("records"), desc="Building texts")]
    article_ids = df["article_id"].tolist()

    # Load model
    logger.info("Loading model '%s' on device '%s' …", model_name, device)
    model, _, tokenizer = load_clip_model(model_name, device=device)

    # Encode
    logger.info("Encoding %d articles (batch_size=%d) …", len(texts), batch_size)
    t0 = time.perf_counter()
    embeddings = encode_texts_clip(texts, model, tokenizer, device, batch_size=batch_size)
    elapsed = time.perf_counter() - t0

    throughput = len(texts) / elapsed
    logger.info(
        "Encoding complete: %.1f items/sec, total %.1fs, shape=%s",
        throughput, elapsed, embeddings.shape,
    )

    # Save embeddings
    emb_path = output_dir / f"{safe_name}_embeddings.npy"
    np.save(emb_path, embeddings)
    logger.info("Embeddings saved to %s", emb_path)

    # Save article_id mapping
    ids_path = output_dir / f"{safe_name}_article_ids.json"
    with open(ids_path, "w") as f:
        json.dump(article_ids, f)
    logger.info("Article ID mapping saved to %s", ids_path)

    # Build and save FAISS index
    index = build_faiss_index(embeddings)
    faiss_path = output_dir / f"{safe_name}_faiss.index"
    save_faiss_index(index, faiss_path)

    # Metadata
    meta = {
        "model_name": model_name,
        "safe_name": safe_name,
        "n_articles": len(texts),
        "embed_dim": int(embeddings.shape[1]),
        "batch_size": batch_size,
        "device": device,
        "elapsed_seconds": round(elapsed, 2),
        "throughput_items_per_sec": round(throughput, 1),
        "embeddings_path": str(emb_path),
        "article_ids_path": str(ids_path),
        "faiss_index_path": str(faiss_path),
    }
    meta_path = output_dir / f"{safe_name}_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    logger.info("Metadata saved to %s", meta_path)

    # Print summary
    print("\n" + "=" * 60)
    print(f"Embedding complete: {model_name}")
    print(f"  Articles  : {len(texts):,}")
    print(f"  Embed dim : {embeddings.shape[1]}")
    print(f"  Time      : {elapsed:.1f}s ({throughput:.0f} items/sec)")
    print(f"  Outputs   : {output_dir}/{safe_name}_*")
    print("=" * 60 + "\n")

    return meta


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Embed H&M articles for dense retrieval baselines."
    )
    parser.add_argument(
        "--model",
        default="fashion-clip",
        help=(
            "Model name or HF ID. Choices from registry: "
            + ", ".join(MODEL_REGISTRY.keys())
        ),
    )
    parser.add_argument(
        "--batch_size", type=int, default=64,
        help="Encoding batch size (default: 64)"
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("data/processed/embeddings"),
        help="Directory to save embeddings and FAISS index",
    )
    parser.add_argument(
        "--articles_csv",
        type=Path,
        default=Path("data/raw/hnm/articles.csv"),
        help="Path to H&M articles.csv",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device (cpu, cuda, mps)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    embed_articles(
        articles_csv=args.articles_csv,
        model_name=args.model,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        device=args.device,
    )
