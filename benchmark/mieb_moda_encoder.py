"""MTEB / MIEB encoder wrapper for MoDA bi-encoder checkpoints.

Implements the `mteb.EncoderProtocol` (encode + similarity + similarity_pairwise +
mteb_model_meta) on top of a `open_clip` SigLIP-style checkpoint.

Supports any of our bi-encoder variants:
  - Marqo/marqo-fashionSigLIP (zero-shot baseline)
  - MoDA-Distilled (Recipe A')
  - MoDA-Matryoshka (slice via --truncate-dim)
  - MoDA-Distilled-512d (native 512d via --base-model)
  - MoDA-Recipe-Z / Z+ (joint text+vision distillation)

Usage (smoke test on a single task):

    from benchmark.mieb_moda_encoder import build_moda_encoder
    import mteb

    enc = build_moda_encoder(checkpoint="models/moda-siglip-distilled/best/model_state_dict.pt",
                              name="MoDA-Distilled-768d")
    task = mteb.get_task("Flickr30kT2IRetrieval")
    results = mteb.MTEB(tasks=[task]).run(enc, output_folder="results/mieb/moda-distilled-768")

Direct CLI:

    python benchmark/mieb_moda_encoder.py \
        --checkpoint models/moda-siglip-distilled/best/model_state_dict.pt \
        --name MoDA-Distilled-768d \
        --tasks Flickr30kT2IRetrieval \
        --output results/mieb/moda-distilled-768
"""

from __future__ import annotations

import argparse
import datetime as _dt
import logging
import math
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import open_clip
import torch
import torch.nn.functional as F

import mteb
from mteb.models.abs_encoder import AbsEncoder
from mteb.models.model_meta import ModelMeta, ScoringFunction

logger = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parent.parent

DEFAULT_BASE = "hf-hub:Marqo/marqo-fashionSigLIP"


def _select_device(prefer: str = "auto") -> str:
    if prefer != "auto":
        return prefer
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class MoDAEncoder(AbsEncoder):
    """Wraps a MoDA / open_clip SigLIP-style model into the MTEB encoder API.

    Attributes:
        model: the loaded open_clip model
        preprocess: image preprocessing transform
        tokenizer: open_clip tokenizer for the text tower
        device: torch device string
        truncate_dim: slice each embedding to first `truncate_dim` dims (Matryoshka)
        normalize: L2-normalize embeddings before returning (default True)
        text_batch_size / image_batch_size: encoder batch sizes
    """

    model: Any

    def __init__(
        self,
        *,
        checkpoint: Optional[str] = None,
        base_model: str = DEFAULT_BASE,
        name: str = "MoDA-bi-encoder",
        embed_dim: int | list[int] = 768,
        truncate_dim: Optional[int] = None,
        device: str = "auto",
        text_batch_size: int = 64,
        image_batch_size: int = 32,
        normalize: bool = True,
        revision: str = "v1",
        license: str = "apache-2.0",
        modalities: list[str] | None = None,
    ):
        self.device = _select_device(device)
        self._base_model_name = base_model

        logger.info(
            "MoDAEncoder: loading base=%s checkpoint=%s device=%s",
            base_model, checkpoint, self.device,
        )
        model, _, preprocess = open_clip.create_model_and_transforms(base_model)
        if checkpoint is not None:
            ckpt_path = Path(checkpoint)
            if not ckpt_path.is_absolute():
                ckpt_path = REPO / ckpt_path
            if not ckpt_path.exists():
                raise FileNotFoundError(ckpt_path)
            sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            missing, unexpected = model.load_state_dict(sd, strict=False)
            logger.info(
                "  loaded %s (%d tensors), missing=%d, unexpected=%d",
                ckpt_path, len(sd), len(missing), len(unexpected),
            )
        self.model = model.to(self.device).eval()
        self.preprocess = preprocess
        self.tokenizer = open_clip.get_tokenizer(base_model)
        self.truncate_dim = truncate_dim
        self.normalize = normalize
        self.text_batch_size = text_batch_size
        self.image_batch_size = image_batch_size

        n_params = sum(p.numel() for p in self.model.parameters())
        if isinstance(embed_dim, int):
            ed: int | list[int] = truncate_dim or embed_dim
        else:
            ed = embed_dim

        self.mteb_model_meta = ModelMeta(
            loader=lambda **kw: self,        # we are already constructed
            name=name,
            revision=revision,
            release_date=_dt.date.today().isoformat(),
            languages=["eng-Latn"],
            n_parameters=n_params,
            memory_usage_mb=round(n_params * 4 / (1024 * 1024), 1),
            max_tokens=64,                    # SigLIP context length
            embed_dim=ed,
            license=license,
            open_weights=True,
            public_training_code="https://github.com/lazzyPie/MODA",
            public_training_data="H&M (anonymised public)",
            framework=["PyTorch"],
            similarity_fn_name=ScoringFunction.COSINE,
            use_instructions=False,
            training_datasets=None,           # zero-shot wrt MIEB tasks
            modalities=modalities or ["text", "image"],
        )

    # ------------------------------------------------------------------ encoding helpers

    @torch.no_grad()
    def _encode_texts(self, texts: list[str]) -> np.ndarray:
        feats = []
        for s in range(0, len(texts), self.text_batch_size):
            batch = texts[s:s + self.text_batch_size]
            toks = self.tokenizer(batch).to(self.device)
            f = self.model.encode_text(toks)
            if self.device == "mps":
                f = f.float()
            if self.normalize:
                f = F.normalize(f, p=2, dim=-1)
            feats.append(f.cpu())
        return torch.cat(feats, 0).numpy().astype(np.float32)

    @torch.no_grad()
    def _encode_images(self, images: list[Any]) -> np.ndarray:
        feats = []
        for s in range(0, len(images), self.image_batch_size):
            batch = images[s:s + self.image_batch_size]
            t = torch.stack([self.preprocess(im.convert("RGB")) for im in batch]).to(self.device)
            f = self.model.encode_image(t)
            if self.device == "mps":
                f = f.float()
            if self.normalize:
                f = F.normalize(f, p=2, dim=-1)
            feats.append(f.cpu())
        return torch.cat(feats, 0).numpy().astype(np.float32)

    # ------------------------------------------------------------------ MTEB API

    def encode(
        self,
        inputs,                 # DataLoader[BatchedInput]
        *,
        task_metadata=None,
        hf_split: str = "test",
        hf_subset: str = "default",
        prompt_type=None,
        **kwargs,
    ) -> np.ndarray:
        """MTEB calls this once per (queries) and once per (corpus) for retrieval tasks.

        Each batch is a dict with optional `text` and/or `image` keys.
        For multimodal (image + text caption together), we average-pool the two
        modality embeddings — same convention as open_clip's joint-eval scripts.
        """
        all_embeds: list[np.ndarray] = []
        for batch in inputs:
            text = batch.get("text") if isinstance(batch, dict) else None
            image = batch.get("image") if isinstance(batch, dict) else None
            # normalise empty placeholders ([""], [None]) to None
            if text is not None and not any(t for t in text):
                text = None
            if image is not None and not any(im is not None for im in image):
                image = None

            t_emb = self._encode_texts(list(text)) if text is not None else None
            i_emb = self._encode_images(list(image)) if image is not None else None

            if t_emb is not None and i_emb is not None:
                if t_emb.shape != i_emb.shape:
                    raise RuntimeError(
                        f"text emb shape {t_emb.shape} != image emb shape {i_emb.shape}"
                    )
                e = (t_emb + i_emb) / 2.0
                if self.normalize:
                    n = np.linalg.norm(e, axis=-1, keepdims=True) + 1e-12
                    e = (e / n).astype(np.float32)
            elif t_emb is not None:
                e = t_emb
            elif i_emb is not None:
                e = i_emb
            else:
                raise RuntimeError(
                    f"Empty batch — neither text nor image present (keys={list(batch.keys()) if isinstance(batch, dict) else type(batch)})"
                )

            if self.truncate_dim is not None:
                e = e[:, : self.truncate_dim]
                if self.normalize:
                    n = np.linalg.norm(e, axis=-1, keepdims=True) + 1e-12
                    e = (e / n).astype(np.float32)

            all_embeds.append(e)

        return np.concatenate(all_embeds, axis=0).astype(np.float32)

    def similarity(self, e1, e2):
        """Cosine similarity assuming embeddings are already L2-normalised."""
        if isinstance(e1, np.ndarray):
            e1 = torch.from_numpy(e1)
        if isinstance(e2, np.ndarray):
            e2 = torch.from_numpy(e2)
        return e1.float() @ e2.float().T

    def similarity_pairwise(self, e1, e2):
        if isinstance(e1, np.ndarray):
            e1 = torch.from_numpy(e1)
        if isinstance(e2, np.ndarray):
            e2 = torch.from_numpy(e2)
        return (e1.float() * e2.float()).sum(dim=-1)


# ---------------------------------------------------------------------------
# Convenience builder
# ---------------------------------------------------------------------------

def build_moda_encoder(
    *,
    checkpoint: Optional[str] = None,
    base_model: str = DEFAULT_BASE,
    name: str = "MoDA-bi-encoder",
    truncate_dim: Optional[int] = None,
    device: str = "auto",
    text_batch_size: int = 64,
    image_batch_size: int = 32,
    embed_dim: int = 768,
) -> MoDAEncoder:
    return MoDAEncoder(
        checkpoint=checkpoint,
        base_model=base_model,
        name=name,
        embed_dim=embed_dim,
        truncate_dim=truncate_dim,
        device=device,
        text_batch_size=text_batch_size,
        image_batch_size=image_batch_size,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None,
                    help="Path to MoDA state_dict; if None, runs the FashionSigLIP base zero-shot")
    ap.add_argument("--base-model", default=DEFAULT_BASE)
    ap.add_argument("--name", default="lazzyPie/MoDA-bi-encoder",
                    help="Display name in 'org/model' format (mteb requirement)")
    ap.add_argument("--tasks", nargs="+", required=True,
                    help="MIEB task names, e.g. Flickr30kT2IRetrieval")
    ap.add_argument("--truncate-dim", type=int, default=None,
                    help="Slice each embedding to first N dims (Matryoshka)")
    ap.add_argument("--device", default="auto",
                    choices=["auto", "cpu", "mps", "cuda"])
    ap.add_argument("--text-batch-size", type=int, default=64)
    ap.add_argument("--image-batch-size", type=int, default=32)
    ap.add_argument("--output", default="results/mieb/moda-default")
    ap.add_argument("--eval-splits", nargs="*", default=None,
                    help="Override task default splits (e.g. test)")
    ap.add_argument("--overwrite", action="store_true",
                    help="Overwrite cached MTEB results if they exist")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%H:%M:%S")

    enc = build_moda_encoder(
        checkpoint=args.checkpoint,
        base_model=args.base_model,
        name=args.name,
        truncate_dim=args.truncate_dim,
        device=args.device,
        text_batch_size=args.text_batch_size,
        image_batch_size=args.image_batch_size,
    )

    tasks = mteb.get_tasks(tasks=args.tasks)
    logger.info("Resolved %d task(s): %s", len(tasks),
                [t.metadata.name for t in tasks])

    runner = mteb.MTEB(tasks=tasks)
    results = runner.run(
        enc,
        output_folder=args.output,
        eval_splits=args.eval_splits,
        overwrite_results=args.overwrite,
        verbosity=2,
    )
    for r in results:
        logger.info("=== %s ===", r.task_name)
        try:
            print(r.scores)
        except Exception:
            print(r)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
