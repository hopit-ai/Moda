"""
MODA Phase 3.8 — Mixture-of-Encoders with Trained Field Encoders

Trains small MLP encoders for color, category, and product group fields
using LLM-judged pairwise similarity labels from PaleblueDot GPT-4o-mini.

Architecture per product:
  text_block  = FashionCLIP(prod_name + detail_desc)   512-dim
  color_block = ColorMLP(colour_group_name)              64-dim
  cat_block   = CategoryMLP(product_type_name)           64-dim
  group_block = GroupEmbed(product_group_name)            32-dim
  product_vec = concat(text, color, cat, group)         672-dim

Training data:
  - For each field, generate all unique pairwise combinations
  - Ask GPT-4o-mini to score similarity 0.0–1.0
  - Train MLP to produce embeddings where similar items are close

Usage:
  python benchmark/train_moe_encoders.py --generate-labels
  python benchmark/train_moe_encoders.py --train
  python benchmark/train_moe_encoders.py --eval
  python benchmark/train_moe_encoders.py --all
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
import os
import sys
import time
from itertools import combinations
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm

_REPO = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

HNM_DIR = _REPO / "data" / "raw" / "hnm_real"
PROCESSED_DIR = _REPO / "data" / "processed"
MODELS_DIR = _REPO / "models" / "moda-moe-encoders"
RESULTS_DIR = _REPO / "results" / "real"
LABELS_PATH = PROCESSED_DIR / "moe_pairwise_labels.json"
SPLITS_PATH = PROCESSED_DIR / "query_splits.json"

BASE_URL = "https://open.palebluedot.ai/v1"
LLM_MODEL = "openai/gpt-4o-mini"

EMBED_DIMS = {"color": 64, "category": 64, "group": 32}


# ─── 1. Extract vocabularies ─────────────────────────────────────────────────

def load_vocabularies() -> dict[str, list[str]]:
    """Load unique field values from H&M articles."""
    import pandas as pd
    df = pd.read_csv(HNM_DIR / "articles.csv", dtype=str).fillna("")

    vocabs = {}
    for field, col in [
        ("color", "colour_group_name"),
        ("category", "product_type_name"),
        ("group", "product_group_name"),
    ]:
        vals = sorted(set(v.strip() for v in df[col].unique() if v.strip()))
        vocabs[field] = vals
        log.info("  %s: %d unique values", field, len(vals))

    return vocabs


# ─── 2. Generate LLM pairwise similarity labels ──────────────────────────────

PROMPTS = {
    "color": (
        "You are a fashion color expert. Given two color names from H&M's product catalog, "
        "rate how similar they are on a 0.0-1.0 scale. Consider perceptual similarity "
        "(e.g. 'Navy' and 'Dark Blue' are very similar ~0.9, 'Black' and 'White' are opposite ~0.0, "
        "'Beige' and 'Light Beige' are close ~0.8). "
        "Return ONLY a JSON object: {{\"score\": <float>}}"
    ),
    "category": (
        "You are a fashion product expert. Given two product category names from H&M, "
        "rate how similar they are on a 0.0-1.0 scale. Consider functional similarity "
        "(e.g. 'Trousers' and 'Shorts' are similar ~0.6, 'T-shirt' and 'Vest top' ~0.7, "
        "'Boots' and 'Bracelet' ~0.0). "
        "Return ONLY a JSON object: {{\"score\": <float>}}"
    ),
    "group": (
        "You are a fashion product expert. Given two product group names from H&M, "
        "rate how similar they are on a 0.0-1.0 scale. Consider conceptual similarity "
        "(e.g. 'Garment Upper body' and 'Garment Lower body' ~0.5, 'Shoes' and 'Socks & Tights' ~0.4, "
        "'Cosmetic' and 'Garment Full body' ~0.0). "
        "Return ONLY a JSON object: {{\"score\": <float>}}"
    ),
}


async def _label_pair(
    client,
    semaphore,
    field: str,
    a: str,
    b: str,
) -> tuple[str, str, str, float]:
    """Query LLM for pairwise similarity."""
    async with semaphore:
        prompt = PROMPTS[field]
        user_msg = f'Color A: "{a}"\nColor B: "{b}"' if field == "color" else \
                   f'Category A: "{a}"\nCategory B: "{b}"' if field == "category" else \
                   f'Group A: "{a}"\nGroup B: "{b}"'
        try:
            resp = await client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.0,
                max_tokens=30,
            )
            text = resp.choices[0].message.content.strip()
            data = json.loads(text)
            score = float(data["score"])
            return field, a, b, max(0.0, min(1.0, score))
        except Exception as e:
            log.warning("LLM error for %s (%s, %s): %s", field, a, b, e)
            return field, a, b, 0.5


async def generate_labels(vocabs: dict[str, list[str]], max_concurrent: int = 20):
    """Generate pairwise similarity labels for all field vocabularies."""
    from openai import AsyncOpenAI

    api_key = os.environ.get("PALEBLUEDOT_API_KEY", "")
    if not api_key:
        raise ValueError("Set PALEBLUEDOT_API_KEY env var (from https://palebluedot.ai)")

    client = AsyncOpenAI(base_url=BASE_URL, api_key=api_key)
    semaphore = asyncio.Semaphore(max_concurrent)

    existing = {}
    if LABELS_PATH.exists():
        with open(LABELS_PATH) as f:
            existing = json.load(f)
        log.info("Loaded %d existing label sets", len(existing))

    all_labels: dict[str, list[dict]] = existing.copy()

    for field, vocab in vocabs.items():
        if field in all_labels and len(all_labels[field]) > 0:
            log.info("Skipping %s — already have %d labels", field, len(all_labels[field]))
            continue

        pairs = list(combinations(vocab, 2))
        # Also add self-pairs (similarity = 1.0)
        self_pairs = [(v, v) for v in vocab]

        log.info("Generating %d pairwise labels for %s (%d vocab)...",
                 len(pairs), field, len(vocab))

        tasks = [_label_pair(client, semaphore, field, a, b) for a, b in pairs]
        results = await asyncio.gather(*tasks)

        labels = []
        for _, a, b, score in results:
            labels.append({"a": a, "b": b, "score": score})
        for v in vocab:
            labels.append({"a": v, "b": v, "score": 1.0})

        all_labels[field] = labels
        log.info("  %s: %d labels generated (mean score: %.3f)",
                 field, len(labels),
                 sum(l["score"] for l in labels) / len(labels))

        LABELS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(LABELS_PATH, "w") as f:
            json.dump(all_labels, f, indent=2)

    total = sum(len(v) for v in all_labels.values())
    log.info("Total labels: %d across %d fields → %s", total, len(all_labels), LABELS_PATH)
    return all_labels


# ─── 3. Field Encoder Models ─────────────────────────────────────────────────

class FieldEncoder(nn.Module):
    """Small MLP that maps field values to dense embeddings."""

    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        x = self.embedding(idx)
        x = self.mlp(x)
        return F.normalize(x, dim=-1)


def train_field_encoder(
    field: str,
    vocab: list[str],
    labels: list[dict],
    embed_dim: int,
    epochs: int = 200,
    lr: float = 1e-3,
) -> tuple[FieldEncoder, dict[str, int]]:
    """Train a field encoder on pairwise similarity labels."""
    val_to_idx = {v: i for i, v in enumerate(vocab)}
    model = FieldEncoder(len(vocab), embed_dim)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # Build training pairs
    pair_a, pair_b, pair_sim = [], [], []
    for lbl in labels:
        a_idx = val_to_idx.get(lbl["a"])
        b_idx = val_to_idx.get(lbl["b"])
        if a_idx is None or b_idx is None:
            continue
        pair_a.append(a_idx)
        pair_b.append(b_idx)
        pair_sim.append(lbl["score"])

    a_t = torch.tensor(pair_a, dtype=torch.long)
    b_t = torch.tensor(pair_b, dtype=torch.long)
    sim_t = torch.tensor(pair_sim, dtype=torch.float32)

    log.info("Training %s encoder: %d vocab, %d-dim, %d pairs",
             field, len(vocab), embed_dim, len(pair_a))

    best_loss = float("inf")
    for epoch in range(epochs):
        model.train()
        emb_a = model(a_t)
        emb_b = model(b_t)

        cos_sim = (emb_a * emb_b).sum(dim=-1)
        loss = F.mse_loss(cos_sim, sim_t)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if loss.item() < best_loss:
            best_loss = loss.item()

        if (epoch + 1) % 50 == 0 or epoch == 0:
            log.info("  [%s] Epoch %d/%d: loss=%.4f (best=%.4f)",
                     field, epoch + 1, epochs, loss.item(), best_loss)

    log.info("  [%s] Final loss: %.4f", field, loss.item())
    return model, val_to_idx


def train_all_encoders(
    vocabs: dict[str, list[str]],
    labels: dict[str, list[dict]],
    epochs: int = 200,
) -> dict[str, tuple[FieldEncoder, dict[str, int]]]:
    """Train encoders for all fields."""
    encoders = {}
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    for field in ["color", "category", "group"]:
        if field not in labels or not labels[field]:
            log.warning("No labels for %s — skipping", field)
            continue

        model, val_to_idx = train_field_encoder(
            field, vocabs[field], labels[field],
            embed_dim=EMBED_DIMS[field],
            epochs=epochs,
        )

        # Save
        save_path = MODELS_DIR / f"{field}_encoder.pt"
        torch.save({
            "model_state": model.state_dict(),
            "vocab": vocabs[field],
            "val_to_idx": val_to_idx,
            "embed_dim": EMBED_DIMS[field],
        }, save_path)
        log.info("  Saved %s encoder → %s", field, save_path)

        encoders[field] = (model, val_to_idx)

    return encoders


# ─── 4. Build MoE Product Embeddings ─────────────────────────────────────────

def build_moe_embeddings(
    encoders: dict[str, tuple[FieldEncoder, dict[str, int]]],
) -> tuple[np.ndarray, list[str]]:
    """Build 672-dim product embeddings: FashionCLIP(512) + color(64) + category(64) + group(32)."""
    import pandas as pd
    from benchmark.models import load_clip_model, encode_texts_clip
    from benchmark.embed_hnm import build_article_text
    import open_clip

    base_model = "fashion-clip"
    ft_model_path = _REPO / "models" / "moda-fashionclip-finetuned" / "best"
    use_finetuned = (ft_model_path / "model_state_dict.pt").exists()
    tag = "finetuned" if use_finetuned else "baseline"
    safe_name = f"fashion-clip_{tag}"

    emb_dir = _REPO / "data" / "processed" / "embeddings"
    emb_dir.mkdir(parents=True, exist_ok=True)
    emb_path = emb_dir / f"{safe_name}_embeddings.npy"
    ids_path = emb_dir / f"{safe_name}_article_ids.json"

    if emb_path.exists():
        text_embs = np.load(str(emb_path)).astype(np.float32)
        with open(ids_path) as f:
            article_ids = json.load(f)
        log.info("Loaded cached text embeddings: %s", text_embs.shape)
    else:
        log.info("Pre-computed embeddings not found — encoding articles now …")
        df_all = pd.read_csv(HNM_DIR / "articles.csv", dtype=str).fillna("")
        texts = [build_article_text(r) for r in df_all.to_dict("records")]
        article_ids = df_all["article_id"].tolist()

        device = "mps" if torch.backends.mps.is_available() else "cpu"
        model_clip, _, tokenizer = load_clip_model(base_model, device=device)
        if use_finetuned:
            log.info("Loading fine-tuned weights from %s", ft_model_path)
            state_dict = torch.load(
                ft_model_path / "model_state_dict.pt", map_location="cpu")
            model_clip.load_state_dict(state_dict, strict=False)
            model_clip = model_clip.to(device).eval()
        text_embs = encode_texts_clip(texts, model_clip, tokenizer, device,
                                      batch_size=64)
        np.save(str(emb_path), text_embs)
        with open(ids_path, "w") as f:
            json.dump(article_ids, f)
        log.info("Saved text embeddings → %s (%s)", emb_path, text_embs.shape)
        del model_clip, tokenizer

    norms = np.linalg.norm(text_embs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    text_embs /= norms

    log.info("Loaded text embeddings: %s", text_embs.shape)

    # Load articles for field values
    df = pd.read_csv(HNM_DIR / "articles.csv", dtype=str).fillna("")
    art_map = {str(r["article_id"]): r.to_dict() for _, r in df.iterrows()}

    n = len(article_ids)

    # Encode field values for each article
    field_embs = {}
    for field, (model, val_to_idx) in encoders.items():
        col_name = {
            "color": "colour_group_name",
            "category": "product_type_name",
            "group": "product_group_name",
        }[field]

        model.eval()
        dim = EMBED_DIMS[field]
        embs = np.zeros((n, dim), dtype=np.float32)

        with torch.no_grad():
            # Pre-compute all vocab embeddings
            all_idx = torch.arange(len(val_to_idx), dtype=torch.long)
            all_emb = model(all_idx).numpy()

        for i, aid in enumerate(article_ids):
            art = art_map.get(aid, {})
            val = str(art.get(col_name, "")).strip()
            idx = val_to_idx.get(val)
            if idx is not None:
                embs[i] = all_emb[idx]

        field_embs[field] = embs
        log.info("  %s embeddings: shape=%s", field, embs.shape)

    # Concatenate: text(512) + color(64) + category(64) + group(32) = 672
    moe_embs = np.concatenate([
        text_embs,
        field_embs["color"],
        field_embs["category"],
        field_embs["group"],
    ], axis=1)

    log.info("MoE embeddings: %s (%.1f MB)", moe_embs.shape,
             moe_embs.nbytes / 1e6)

    # Save
    moe_path = emb_dir / "moe_672d_embeddings.npy"
    moe_ids_path = emb_dir / "moe_672d_article_ids.json"
    np.save(str(moe_path), moe_embs)
    with open(moe_ids_path, "w") as f:
        json.dump(article_ids, f)
    log.info("Saved → %s", moe_path)

    return moe_embs, article_ids


# ─── 5. MoE Retrieval + Evaluation ───────────────────────────────────────────

def moe_search(
    query_text_emb: np.ndarray,
    query_ner: dict[str, list[str]],
    moe_embs: np.ndarray,
    article_ids: list[str],
    encoders: dict[str, tuple[FieldEncoder, dict[str, int]]],
    top_k: int = 50,
    text_weight: float = 1.0,
    color_weight: float = 0.25,
    category_weight: float = 0.30,
    group_weight: float = 0.15,
) -> list[str]:
    """Search using MoE embeddings with adaptive per-field weighting."""
    n = moe_embs.shape[0]

    # Text similarity (first 512 dims)
    text_block = moe_embs[:, :512]
    text_scores = text_block @ query_text_emb[:512]

    scores = text_weight * text_scores

    # Color field (512:576)
    if "color" in query_ner and query_ner["color"] and "color" in encoders:
        model, val_to_idx = encoders["color"]
        color_text = " ".join(query_ner["color"])
        # Find closest vocab match
        best_idx, best_score = None, -1
        for val, idx in val_to_idx.items():
            if any(c.lower() in val.lower() or val.lower() in c.lower()
                   for c in query_ner["color"]):
                best_idx = idx
                break
        if best_idx is not None:
            model.eval()
            with torch.no_grad():
                q_emb = model(torch.tensor([best_idx])).numpy()[0]
            color_block = moe_embs[:, 512:576]
            color_scores = color_block @ q_emb
            scores += color_weight * color_scores

    # Category field (576:640)
    if "garment type" in query_ner and query_ner["garment type"] and "category" in encoders:
        model, val_to_idx = encoders["category"]
        best_idx = None
        for val, idx in val_to_idx.items():
            if any(c.lower() in val.lower() or val.lower() in c.lower()
                   for c in query_ner["garment type"]):
                best_idx = idx
                break
        if best_idx is not None:
            model.eval()
            with torch.no_grad():
                q_emb = model(torch.tensor([best_idx])).numpy()[0]
            cat_block = moe_embs[:, 576:640]
            cat_scores = cat_block @ q_emb
            scores += category_weight * cat_scores

    # Group field (640:672)
    if "category" in query_ner or "occasion" in query_ner:
        pass  # Group matching is indirect; skip for now

    top_idx = np.argpartition(-scores, min(top_k, n - 1))[:top_k]
    top_idx = top_idx[np.argsort(-scores[top_idx])]
    return [article_ids[j] for j in top_idx]


def evaluate_moe(
    moe_embs: np.ndarray,
    article_ids: list[str],
    encoders: dict[str, tuple[FieldEncoder, dict[str, int]]],
):
    """Evaluate MoE retrieval on test queries."""
    from benchmark.eval_full_pipeline import (
        load_benchmark,
        load_or_compute_ner,
        evaluate,
        dense_search_batch,
        bm25_ner_search,
        rrf_fusion,
        ce_rerank_batch,
        load_articles,
        DENSE_MODEL,
        TOP_K_RERANK,
        TOP_K_FINAL,
    )
    from benchmark.models import load_clip_model, encode_texts_clip
    from opensearchpy import OpenSearch

    # Load test data
    with open(SPLITS_PATH) as f:
        splits = json.load(f)
    test_qids = set(splits["test"])

    import pandas as pd
    queries_df = pd.read_csv(HNM_DIR / "queries.csv", dtype=str)
    all_queries = [
        (str(r["query_id"]), str(r["query_text"]))
        for _, r in queries_df.iterrows()
        if str(r["query_id"]) in test_qids
    ]
    # Sample for speed
    import random
    random.seed(42)
    queries = random.sample(all_queries, min(10000, len(all_queries)))

    qrels_raw = pd.read_csv(HNM_DIR / "qrels.csv", dtype=str)
    qrels: dict[str, dict[str, int]] = {}
    for _, r in qrels_raw.iterrows():
        qid = str(r["query_id"])
        pos_ids = str(r.get("positive_ids", "")).split()
        qrels[qid] = {aid: 1 for aid in pos_ids}

    log.info("Evaluating MoE on %d test queries...", len(queries))

    # NER cache
    from benchmark.query_expansion import FashionNER
    ner_model = FashionNER()
    ner_cache = {}
    for qid, text in tqdm(queries, desc="NER extraction", ncols=80):
        ner_cache[qid] = ner_model.extract(text)

    # Encode queries with FashionCLIP (fine-tuned if available)
    ft_model_path = _REPO / "models" / "moda-fashionclip-finetuned" / "best"
    use_ft = (ft_model_path / "model_state_dict.pt").exists()

    import torch
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model, _, tokenizer = load_clip_model("fashion-clip", device=device)
    if use_ft:
        log.info("Loading fine-tuned FashionCLIP weights for query encoding")
        sd = torch.load(ft_model_path / "model_state_dict.pt", map_location="cpu")
        model.load_state_dict(sd, strict=False)
        model = model.to(device).eval()

    query_texts = [q[1] for q in queries]
    log.info("Encoding %d query texts...", len(query_texts))
    query_embs = encode_texts_clip(query_texts, model, tokenizer, device, batch_size=128)
    # Normalize
    norms = np.linalg.norm(query_embs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    query_embs /= norms

    # MoE retrieval
    log.info("Running MoE retrieval...")
    moe_results: dict[str, list[str]] = {}
    for i, (qid, text) in enumerate(tqdm(queries, desc="MoE search", ncols=80)):
        ner_ents = ner_cache.get(qid, {})
        # Build 672-dim query embedding
        q_emb_672 = np.zeros(672, dtype=np.float32)
        q_emb_672[:512] = query_embs[i]
        moe_results[qid] = moe_search(
            query_embs[i], ner_ents, moe_embs, article_ids,
            encoders, top_k=TOP_K_RERANK,
        )

    res_moe = evaluate(moe_results, qrels, label="MoE_trained")

    # Baseline: standard dense retrieval for comparison
    log.info("Running standard dense retrieval baseline...")
    dense_results: dict[str, list[str]] = {}
    aid_to_idx = {aid: i for i, aid in enumerate(article_ids)}
    text_block = moe_embs[:, :512]
    for i, (qid, text) in enumerate(tqdm(queries, desc="Dense search", ncols=80)):
        text_scores = text_block @ query_embs[i]
        top_idx = np.argpartition(-text_scores, TOP_K_RERANK)[:TOP_K_RERANK]
        top_idx = top_idx[np.argsort(-text_scores[top_idx])]
        dense_results[qid] = [article_ids[j] for j in top_idx]

    res_dense = evaluate(dense_results, qrels, label="Dense_only")

    # Print comparison
    print("\n" + "=" * 70)
    print("MIXTURE OF ENCODERS — Phase 3.8 Results")
    print(f"  {len(queries):,} test queries")
    print("=" * 70)
    for name, res in [("Dense_only", res_dense), ("MoE_trained", res_moe)]:
        m = res["metrics"]
        print(f"  {name:<25} nDCG@10={m['ndcg@10']:.4f}  MRR={m['mrr']:.4f}  R@10={m['recall@10']:.4f}")
    print("=" * 70)

    # Save
    output = {
        "Dense_only": res_dense,
        "MoE_trained": res_moe,
        "settings": {
            "n_queries": len(queries),
            "embed_dims": EMBED_DIMS,
        },
    }
    out_path = RESULTS_DIR / "phase3_8_moe_eval.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    log.info("Results saved → %s", out_path)

    return output


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="MODA 3.8: MoE Field Encoders")
    parser.add_argument("--generate-labels", action="store_true",
                        help="Generate LLM pairwise similarity labels")
    parser.add_argument("--train", action="store_true",
                        help="Train field encoders")
    parser.add_argument("--eval", action="store_true",
                        help="Evaluate MoE retrieval")
    parser.add_argument("--all", action="store_true",
                        help="Run all steps")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--max-concurrent", type=int, default=20)
    args = parser.parse_args()

    if args.all:
        args.generate_labels = True
        args.train = True
        args.eval = True

    vocabs = load_vocabularies()

    if args.generate_labels:
        log.info("Step 1: Generating LLM pairwise labels...")
        labels = asyncio.run(generate_labels(vocabs, max_concurrent=args.max_concurrent))
    elif LABELS_PATH.exists():
        with open(LABELS_PATH) as f:
            labels = json.load(f)
    else:
        labels = {}

    encoders = {}
    if args.train:
        log.info("Step 2: Training field encoders...")
        encoders = train_all_encoders(vocabs, labels, epochs=args.epochs)
    elif args.eval:
        # Load saved encoders
        for field in ["color", "category", "group"]:
            path = MODELS_DIR / f"{field}_encoder.pt"
            if path.exists():
                data = torch.load(path, weights_only=False)
                model = FieldEncoder(
                    len(data["vocab"]), data["embed_dim"])
                model.load_state_dict(data["model_state"])
                encoders[field] = (model, data["val_to_idx"])
                log.info("Loaded %s encoder from %s", field, path)

    if args.eval:
        log.info("Step 3: Building MoE embeddings + evaluation...")
        if not encoders:
            log.error("No encoders available. Run --train first.")
            return
        moe_embs, article_ids = build_moe_embeddings(encoders)
        evaluate_moe(moe_embs, article_ids, encoders)


if __name__ == "__main__":
    main()
