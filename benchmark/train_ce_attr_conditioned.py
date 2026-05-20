"""
MODA Phase 3 — Attribute-Conditioned Cross-Encoder (Path B)

Fine-tunes cross-encoder/ms-marco-MiniLM-L-6-v2 using a restructured
product text format that explicitly tags structured fields:

  prod_name | detail_desc [COLOR] colour_group [TYPE] product_type [SEC] section [GROUP] group

This lets the cross-encoder's attention mechanism learn which attributes
matter for each query, without separate field encoders or NER at query time.

Uses the same LLM-graded labels (0-3 → 0.0-1.0) and train/val/test splits.

Output:
  models/moda-fashion-ce-attr/          final model directory
  models/moda-fashion-ce-attr-best/     best dev checkpoint

Usage:
  python benchmark/train_ce_attr_conditioned.py
  python benchmark/train_ce_attr_conditioned.py --epochs 1 --quick
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sentence_transformers import InputExample
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import (
    CECorrelationEvaluator,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).parent.parent
PROCESSED_DIR = _REPO_ROOT / "data" / "processed"
HNM_DIR = _REPO_ROOT / "data" / "raw" / "hnm_real"
MODEL_DIR = _REPO_ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

LABELS_PATH = PROCESSED_DIR / "llm_relevance_labels.jsonl"
SPLIT_PATH = PROCESSED_DIR / "query_splits.json"

BASE_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
OUTPUT_DIR = str(MODEL_DIR / "moda-fashion-ce-attr")
BEST_DIR = str(MODEL_DIR / "moda-fashion-ce-attr-best")

RANDOM_SEED = 42


def build_tagged_article_text(row: dict) -> str:
    """Build article text with explicit attribute tags.

    Format: prod_name | detail_desc [COLOR] X [TYPE] Y [SEC] Z [GROUP] W
    """
    parts = []

    name = str(row.get("prod_name", "")).strip()
    if name and name.lower() not in ("nan", "none"):
        parts.append(name)

    desc = str(row.get("detail_desc", "")).strip()
    if desc and desc.lower() not in ("nan", "none"):
        parts.append(desc[:150])

    text = " | ".join(parts)

    for tag, field in [
        ("[COLOR]", "colour_group_name"),
        ("[TYPE]", "product_type_name"),
        ("[SEC]", "section_name"),
        ("[GROUP]", "product_group_name"),
    ]:
        val = str(row.get(field, "")).strip()
        if val and val.lower() not in ("nan", "none"):
            text += f" {tag} {val}"

    return text


def load_article_map() -> dict[str, dict]:
    """Load articles.csv into a dict keyed by article_id."""
    df = pd.read_csv(HNM_DIR / "articles.csv", dtype=str).fillna("")
    art_map = {}
    for _, row in df.iterrows():
        aid = str(row.get("article_id", "")).strip()
        if aid:
            art_map[aid] = row.to_dict()
    log.info("Loaded %d articles from %s", len(art_map), HNM_DIR / "articles.csv")
    return art_map


def load_llm_labels() -> list[dict]:
    if not LABELS_PATH.exists():
        raise FileNotFoundError(
            f"No labels at {LABELS_PATH}. Run generate_llm_labels.py first."
        )
    labels = []
    with open(LABELS_PATH) as f:
        for line in f:
            if line.strip():
                labels.append(json.loads(line))
    log.info("Loaded %d LLM-judged labels", len(labels))
    return labels


def build_examples(
    labels: list[dict],
    art_map: dict[str, dict],
    max_pairs: int | None = None,
    seed: int = RANDOM_SEED,
) -> tuple[list[InputExample], list[InputExample]]:
    """Convert LLM labels to train/val examples with tagged product text."""
    rng = random.Random(seed)

    splits = json.loads(SPLIT_PATH.read_text())
    train_qids = set(splits["train"])
    val_qids = set(splits["val"])

    train_examples: list[InputExample] = []
    val_examples: list[InputExample] = []
    skipped = 0
    rebuilt = 0

    for item in labels:
        qid = item["query_id"]
        score = item["score"] / 3.0
        query_text = item["query_text"]
        article_id = item.get("article_id", "")

        if not query_text:
            skipped += 1
            continue

        art_row = art_map.get(article_id)
        if art_row:
            product_text = build_tagged_article_text(art_row)
            rebuilt += 1
        else:
            product_text = item.get("product_text", "")
            if not product_text:
                skipped += 1
                continue

        example = InputExample(texts=[query_text, product_text], label=float(score))

        if qid in train_qids:
            train_examples.append(example)
        elif qid in val_qids:
            val_examples.append(example)
        else:
            skipped += 1

    if max_pairs and len(train_examples) > max_pairs:
        rng.shuffle(train_examples)
        train_examples = train_examples[:max_pairs]

    rng.shuffle(train_examples)
    rng.shuffle(val_examples)

    if not val_examples and train_examples:
        n_val = max(100, int(len(train_examples) * 0.10))
        val_examples = train_examples[:n_val]
        train_examples = train_examples[n_val:]
        log.info("No val-split labels; split %d from train for validation", n_val)

    score_dist = {0: 0, 1: 0, 2: 0, 3: 0}
    for item in labels:
        score_dist[item["score"]] = score_dist.get(item["score"], 0) + 1

    log.info("Score distribution: %s", score_dist)
    log.info("Train: %d  Val: %d  Rebuilt from articles: %d  Skipped: %d",
             len(train_examples), len(val_examples), rebuilt, skipped)
    return train_examples, val_examples


def train(args):
    log.info("=" * 60)
    log.info("MODA — Attribute-Conditioned Cross-Encoder (Path B)")
    log.info("Base model: %s", BASE_MODEL)
    log.info("Product text format: name | desc [COLOR] X [TYPE] Y [SEC] Z [GROUP] W")
    log.info("=" * 60)

    art_map = load_article_map()
    labels = load_llm_labels()
    train_examples, val_examples = build_examples(
        labels, art_map, max_pairs=args.max_pairs, seed=RANDOM_SEED,
    )

    if not train_examples:
        log.error("No training examples! Check labels and splits.")
        return

    train_labels = [e.label for e in train_examples]
    log.info("Train label stats — mean: %.3f  std: %.3f  min: %.3f  max: %.3f",
             np.mean(train_labels), np.std(train_labels),
             np.min(train_labels), np.max(train_labels))

    log.info("\nSample tagged product texts:")
    for ex in train_examples[:3]:
        log.info("  Q: %s", ex.texts[0][:60])
        log.info("  P: %s", ex.texts[1][:120])
        log.info("  label: %.2f\n", ex.label)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    log.info("Loading cross-encoder on device: %s", device)
    model = CrossEncoder(
        BASE_MODEL,
        num_labels=1,
        max_length=512,
        device=device,
    )

    train_dataloader = DataLoader(
        train_examples,
        shuffle=True,
        batch_size=args.batch_size,
        num_workers=0,
    )

    dev_sentences1 = [e.texts[0] for e in val_examples]
    dev_sentences2 = [e.texts[1] for e in val_examples]
    dev_labels = [e.label for e in val_examples]
    evaluator = CECorrelationEvaluator(
        sentence_pairs=list(zip(dev_sentences1, dev_sentences2)),
        scores=dev_labels,
        name="attr-ce-dev",
    )

    warmup_steps = max(100, int(len(train_dataloader) * 0.06))
    eval_steps = max(500, len(train_dataloader) // 5)
    log.info("Steps/epoch: %d  Warmup: %d  Eval every: %d",
             len(train_dataloader), warmup_steps, eval_steps)

    t0 = time.time()
    log.info("Starting training: %d epochs, batch=%d, lr=%s",
             args.epochs, args.batch_size, args.lr)

    model.fit(
        train_dataloader=train_dataloader,
        evaluator=evaluator,
        epochs=args.epochs,
        optimizer_params={"lr": args.lr},
        warmup_steps=warmup_steps,
        evaluation_steps=eval_steps,
        output_path=BEST_DIR,
        save_best_model=True,
        show_progress_bar=True,
    )

    elapsed = time.time() - t0
    log.info("Training finished in %.1f min", elapsed / 60)

    model.save(OUTPUT_DIR)
    log.info("Final model saved → %s", OUTPUT_DIR)
    log.info("Best checkpoint  → %s", BEST_DIR)

    log.info("\nSanity check on 10 val examples:")
    sample = val_examples[:10]
    scores = model.predict([(e.texts[0], e.texts[1]) for e in sample])
    for e, score in zip(sample, scores):
        q_short = e.texts[0][:40]
        d_short = e.texts[1][:60]
        log.info("  label=%.2f  pred=%.3f | q='%s...' | d='%s...'",
                 e.label, score, q_short, d_short)

    return OUTPUT_DIR


def parse_args():
    p = argparse.ArgumentParser(
        description="Train attribute-conditioned CE (Path B)")
    p.add_argument("--max_pairs", type=int, default=None)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--quick", action="store_true",
                   help="Quick test: 50K pairs, 1 epoch")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.quick:
        args.max_pairs = 50_000
        args.epochs = 1
        log.info("QUICK MODE: 50K pairs, 1 epoch")
    train(args)
