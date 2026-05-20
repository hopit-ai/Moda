"""
Fast T2I proxy eval against all 6 Marqo benchmark datasets.
Uses full corpus (same as official eval harness) — no subsampling.
Reads pre-computed image embeddings from repos/marqo-FashionCLIP/results/
and re-encodes with our model.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_dataset
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

REPO = Path(__file__).resolve().parents[1]
HARNESS = REPO / "repos" / "marqo-FashionCLIP"

DATASETS = {
    "atlas":                 {"hf": "Marqo/atlas",                    "text_col": "text",   "img_col": "image"},
    "deepfashion_inshop":    {"hf": "Marqo/deepfashion-inshop",       "text_col": "text",   "img_col": "image"},
    "deepfashion_multimodal":{"hf": "Marqo/deepfashion-multimodal",   "text_col": "text",   "img_col": "image"},
    "fashion200k":           {"hf": "Marqo/fashion200k",              "text_col": "text",   "img_col": "image"},
    "KAGL":                  {"hf": "Marqo/KAGL",                     "text_col": "text",   "img_col": "image"},
    "polyvore":              {"hf": "Marqo/polyvore",                 "text_col": "text",   "img_col": "image"},
}


class ImageListDataset(Dataset):
    def __init__(self, images: list, preprocess):
        self.images = images
        self.preprocess = preprocess

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        pil = self.images[idx]
        if not isinstance(pil, Image.Image):
            try:
                import numpy as np
                pil = Image.fromarray(pil)
            except Exception:
                return torch.zeros(3, 256, 256)
        try:
            return self.preprocess(pil.convert("RGB"))
        except Exception:
            return torch.zeros(3, 256, 256)


def _compute_metrics(query_emb: torch.Tensor, doc_emb: torch.Tensor) -> dict:
    sim = query_emb @ doc_emb.T  # (Q, D)
    Q = sim.size(0)
    # Assume query i matches doc i (1-1 correspondence in Marqo eval sets)
    ranks = (sim > sim.diagonal().unsqueeze(1)).sum(dim=1).float() + 1  # 1-indexed rank

    r1  = (ranks <= 1).float().mean().item()
    r10 = (ranks <= 10).float().mean().item()
    mrr = (1.0 / ranks).mean().item()
    avg = (r1 + r10 + mrr) / 3.0
    return {"R@1": r1, "R@10": r10, "MRR": mrr, "AvgRecall": avg, "n_queries": Q}


@torch.no_grad()
def eval_dataset(
    ds_name: str,
    encode_img_fn,   # fn(list[PIL]) -> (N, D) normalized
    encode_txt_fn,   # fn(list[str]) -> (Q, D) normalized
    batch_size: int = 256,
    max_docs: int | None = None,
    cache_dir: str = ".cache",
) -> dict:
    """Evaluate T2I on one dataset. Returns metrics dict."""
    cfg = DATASETS[ds_name]
    t0 = time.time()

    ds = load_dataset(cfg["hf"], split="test", cache_dir=cache_dir)
    if max_docs is not None:
        ds = ds.select(range(min(max_docs, len(ds))))

    # Collect docs and queries
    images = ds[cfg["img_col"]]
    texts  = ds[cfg["text_col"]]

    # Encode images in batches
    doc_embs = []
    for i in range(0, len(images), batch_size):
        batch_imgs = images[i : i + batch_size]
        emb = encode_img_fn(batch_imgs)  # (b, D)
        doc_embs.append(emb.cpu())
    doc_emb = torch.cat(doc_embs, dim=0)  # (N, D)
    doc_emb = F.normalize(doc_emb.float(), dim=-1)

    # Encode texts
    qry_embs = []
    for i in range(0, len(texts), batch_size):
        batch_txt = texts[i : i + batch_size]
        emb = encode_txt_fn(batch_txt)
        qry_embs.append(emb.cpu())
    qry_emb = torch.cat(qry_embs, dim=0)
    qry_emb = F.normalize(qry_emb.float(), dim=-1)

    metrics = _compute_metrics(qry_emb, doc_emb)
    metrics["elapsed_s"] = time.time() - t0
    return metrics


@torch.no_grad()
def run_full_eval(
    model,
    tokenizer,
    preprocess,
    device: str,
    datasets: list[str] | None = None,
    batch_size: int = 256,
    cache_dir: str = ".cache",
) -> dict:
    """Run T2I eval on all 6 (or specified) datasets. Returns summary + per-dataset."""
    model.eval()
    if datasets is None:
        datasets = list(DATASETS.keys())

    def encode_imgs(pil_list):
        tensors = []
        for p in pil_list:
            if not isinstance(p, Image.Image):
                try:
                    p = Image.fromarray(p)
                except Exception:
                    tensors.append(torch.zeros(3, 256, 256))
                    continue
            try:
                tensors.append(preprocess(p.convert("RGB")))
            except Exception:
                tensors.append(torch.zeros(3, 256, 256))
        batch = torch.stack(tensors).to(device)
        return model.encode_image(batch)

    def encode_txts(txt_list):
        tokens = tokenizer(txt_list).to(device)
        return model.encode_text(tokens)

    results = {}
    r1s, r10s, mrrs, avgs = [], [], [], []
    for ds_name in datasets:
        print(f"  Eval {ds_name} ...", end=" ", flush=True)
        m = eval_dataset(ds_name, encode_imgs, encode_txts, batch_size, cache_dir=cache_dir)
        results[ds_name] = m
        r1s.append(m["R@1"]); r10s.append(m["R@10"])
        mrrs.append(m["MRR"]); avgs.append(m["AvgRecall"])
        print(f"R@1={m['R@1']:.4f} R@10={m['R@10']:.4f} MRR={m['MRR']:.4f} avg={m['AvgRecall']:.4f} ({m['elapsed_s']:.0f}s)")

    summary = {
        "T2I_AvgRecall": sum(avgs) / len(avgs),
        "T2I_R1":        sum(r1s)  / len(r1s),
        "T2I_R10":       sum(r10s) / len(r10s),
        "T2I_MRR":       sum(mrrs) / len(mrrs),
        "per_dataset":   results,
    }
    return summary
