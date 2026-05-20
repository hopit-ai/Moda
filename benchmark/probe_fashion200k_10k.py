"""
In-loop collapse probe for the Path 1 distillation training.

What it does:
  Given a student `open_clip` model (already loaded, in RAM), encode the
  `fashion200k` 10K stratified subsample (seed 42), encode the GT-test
  queries, score, top-K, and return MAP@10 + Recall@10 + NDCG@10.

Why it exists:
  Every Phase 4 distill / fine-tune run collapsed within 50 training steps,
  but we only discovered the collapse at the END (running the eval harness
  on the saved checkpoint). The probe lets us catch collapse INSIDE the
  training loop and abort the run, saving hours of wasted compute.

Design notes:
  - Probe is in-process (no subprocess). The trainer hands it a live
    `model` reference and gets a metric back. Saves model-load overhead
    (~15s on every probe call).
  - Corpus image preprocessing is cached on disk in a single .pt tensor
    (the first probe call writes it; subsequent calls memory-map it).
    This makes the probe ~3x faster after the first call.
  - Probe uses the SAME stratified-subsample function as our existing
    eval (so the numbers are directly comparable to the screener
    baselines we already have on disk).

Usage (from a trainer):
    from benchmark.probe_fashion200k_10k import Fashion200kProbe
    probe = Fashion200kProbe()        # warms up corpus tensor cache
    metrics = probe.run(student_model, student_preprocess, student_tokenizer)
    print(metrics["MAP@10"])

Standalone (sanity check):
    .venv/bin/python benchmark/probe_fashion200k_10k.py \
        --model ViT-B-16-SigLIP2-384 --pretrained webli
    # expected MAP@10 ~ 0.5059 (matches Phase 4 autopsy zero-shot baseline)
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "repos" / "marqo-FashionCLIP"))
sys.path.insert(0, str(REPO / "benchmark"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("probe-f200k")

from fuse_and_eval import (  # noqa: E402
    DATASET_CONFIGS,
    reconstruct_corpus_and_queries,
    topk_dict,
    evaluate_with_beir,
)

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
PROBE_CACHE = REPO / "data/processed/probe_cache"


class Fashion200kProbe:
    """Reusable in-loop probe. Construct once, call .run() repeatedly.

    The first call materializes the corpus image list and the GT mapping;
    subsequent calls only re-encode through the (changing) student model.
    """

    def __init__(self, dataset: str = "fashion200k", corpus_size: int = 10000, seed: int = 42):
        self.dataset = dataset
        self.corpus_size = corpus_size
        self.seed = seed

        log.info("[probe init] reconstructing %s 10K subsample (seed=%d) ...",
                 dataset, seed)
        t0 = time.time()
        # item_IDs is the corpus order; we'll need the actual PIL images for
        # encoding, fetched lazily on first run.
        self.item_IDs, self.queries, self.gt = reconstruct_corpus_and_queries(
            dataset, corpus_size, seed,
        )
        log.info("[probe init] %d corpus items, %d queries — %.1fs",
                 len(self.item_IDs), len(self.queries), time.time() - t0)

        # Lazy: PIL images, loaded once on first probe call.
        self._pil_images: list | None = None
        # Memo: (preprocess_id, image_size) -> [N, 3, H, W] fp16 on CPU
        self._preprocessed_cache: dict[tuple, torch.Tensor] = {}

    def _ensure_pil_images(self):
        if self._pil_images is not None:
            return
        log.info("[probe] loading PIL images for corpus (one-time, ~30-60s)...")
        # Bypass the marqo-FashionCLIP get_dataset wrapper because it applies
        # a Transform that requires a non-None preprocess; we want raw PIL
        # objects so we can preprocess once with the student's transform.
        from datasets import load_dataset
        import json as _json
        import os as _os

        cfg_path = DATASET_CONFIGS[self.dataset]
        with open(cfg_path) as f:
            cfg = _json.load(f)

        t0 = time.time()
        ds = load_dataset(
            cfg["hf_dataset"],
            num_proc=_os.cpu_count(),
            cache_dir=str(REPO / "data/hf_cache"),
        )["data"]
        # Same item_ID extraction as repos/marqo-FashionCLIP/data/utils.py
        full_item_ID = [str(x) for x in ds.data["item_ID"].to_pylist()]
        item_to_idx = {iid: i for i, iid in enumerate(full_item_ID)}

        pils = []
        for iid in self.item_IDs:
            row = ds[item_to_idx[iid]]
            pils.append(row["image"])
        self._pil_images = pils
        log.info("[probe] %d PIL images loaded in %.1fs",
                 len(self._pil_images), time.time() - t0)

    def _preprocess_corpus(self, preprocess) -> torch.Tensor:
        """Preprocess the corpus images for one specific preprocess transform.

        Cached per-preprocess in memory (and on disk under data/processed/probe_cache).
        For this project we only ever use one student arch per training run, so
        the on-disk cache hits on every probe call after the first.
        """
        # Cache key: image size + a couple of transform attributes
        try:
            size = tuple(preprocess.transforms[1].size) if hasattr(preprocess, "transforms") else None
        except Exception:
            size = None
        key = ("default", size)
        if key in self._preprocessed_cache:
            return self._preprocessed_cache[key]

        PROBE_CACHE.mkdir(parents=True, exist_ok=True)
        size_tag = f"{size[0]}x{size[1]}" if isinstance(size, tuple) and len(size) == 2 else "unknown"
        disk_path = PROBE_CACHE / f"{self.dataset}_corpus_{self.corpus_size}_seed{self.seed}_{size_tag}.pt"
        if disk_path.exists():
            t0 = time.time()
            tens = torch.load(disk_path, map_location="cpu", weights_only=False)
            log.info("[probe] loaded preprocessed corpus from %s (%.1fs, shape=%s)",
                     disk_path, time.time() - t0, tuple(tens.shape))
            self._preprocessed_cache[key] = tens
            return tens

        self._ensure_pil_images()
        log.info("[probe] preprocessing %d images at %s (one-time)...",
                 len(self._pil_images), size_tag)
        t0 = time.time()
        chunks = []
        for img in self._pil_images:
            chunks.append(preprocess(img))
        tens = torch.stack(chunks).half()  # fp16 to halve disk + RAM
        torch.save(tens, disk_path)
        log.info("[probe] saved %s (%.1f MB, %.1fs)",
                 disk_path, disk_path.stat().st_size / 1e6, time.time() - t0)
        self._preprocessed_cache[key] = tens
        return tens

    def run(
        self,
        model,
        preprocess,
        tokenizer,
        device: str = DEVICE,
        batch_size: int = 64,
    ) -> dict:
        """Encode corpus + queries with the given student model, return metrics."""
        t_total = time.time()
        model.eval()

        corpus_tens = self._preprocess_corpus(preprocess)  # [N, 3, H, W] fp16
        N = corpus_tens.shape[0]

        # 1. Encode corpus images through student image tower
        t0 = time.time()
        img_feats: list[torch.Tensor] = []
        with torch.no_grad():
            for i in range(0, N, batch_size):
                batch = corpus_tens[i : i + batch_size].to(device).float()
                feat = model.encode_image(batch)
                feat = feat / feat.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                img_feats.append(feat.float().cpu())
        img_feats_t = torch.cat(img_feats, dim=0)
        t_img = time.time() - t0
        log.info("[probe] image encode: %d items in %.1fs (%.1f items/s)",
                 N, t_img, N / max(t_img, 1e-6))

        # 2. Encode queries through student text tower
        t0 = time.time()
        txt_feats: list[torch.Tensor] = []
        with torch.no_grad():
            for i in range(0, len(self.queries), batch_size):
                tokens = tokenizer(self.queries[i : i + batch_size]).to(device)
                feat = model.encode_text(tokens)
                feat = feat / feat.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                txt_feats.append(feat.float().cpu())
        txt_feats_t = torch.cat(txt_feats, dim=0)
        t_txt = time.time() - t0
        log.info("[probe] text encode: %d queries in %.1fs", len(self.queries), t_txt)

        # 3. Score, top-K, BEIR
        scores = txt_feats_t @ img_feats_t.T
        retrieved = topk_dict(scores, self.item_IDs, self.queries, k=100)
        metrics = evaluate_with_beir(retrieved, self.gt)
        t_total = time.time() - t_total
        log.info("[probe] DONE in %.1fs  MAP@10=%.4f  R@10=%.4f  NDCG@10=%.4f",
                 t_total, metrics["MAP@10"], metrics["Recall@10"], metrics["NDCG@10"])
        return metrics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="ViT-B-16-SigLIP2-384",
                   help="open_clip model identifier")
    p.add_argument("--pretrained", default="webli", help="open_clip pretrained tag (or empty for hf-hub)")
    p.add_argument("--dataset", default="fashion200k")
    p.add_argument("--batch-size", type=int, default=64)
    args = p.parse_args()

    import open_clip
    log.info("loading %s (pretrained=%s)...", args.model, args.pretrained)
    if args.pretrained:
        model, _, preprocess = open_clip.create_model_and_transforms(
            args.model, pretrained=args.pretrained, cache_dir=str(REPO / "data/hf_cache"),
        )
    else:
        model, _, preprocess = open_clip.create_model_and_transforms(
            args.model, cache_dir=str(REPO / "data/hf_cache"),
        )
    tokenizer = open_clip.get_tokenizer(args.model)
    model.eval().to(DEVICE)
    for p_ in model.parameters():
        p_.requires_grad = False

    probe = Fashion200kProbe(dataset=args.dataset)
    metrics = probe.run(model, preprocess, tokenizer, batch_size=args.batch_size)
    print()
    print(f"=== {args.model} {args.pretrained} on {args.dataset} 10K ===")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
