"""
MODA Phase 3B — Cross-Encoder Fine-Tuning with LLM-Judged Labels

Fine-tunes cross-encoder/ms-marco-MiniLM-L-6-v2 on LLM-graded relevance
labels (0-3 → normalized 0.0-1.0).

Key differences from Phase 3A (train_cross_encoder.py):
  • Labels are graded 0-3 (normalized to 0-1) instead of binary 0/1
  • Loss: MSELoss instead of BCEWithLogitsLoss
  • Data source: llm_relevance_labels.jsonl instead of raw purchase qrels
  • Same query splits to enable fair head-to-head comparison

Output:
  models/moda-fashion-ce-llm/          final model directory
  models/moda-fashion-ce-llm-best/     best dev checkpoint

Usage:
  python benchmark/train_ce_llm_labels.py                          # full run
  python benchmark/train_ce_llm_labels.py --max_pairs 50000 --epochs 1  # quick test
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import time
from pathlib import Path

import numpy as np
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
MODEL_DIR = _REPO_ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

LABELS_PATH = PROCESSED_DIR / "llm_relevance_labels.jsonl"
SPLIT_PATH = PROCESSED_DIR / "query_splits.json"

BASE_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
OUTPUT_DIR = str(MODEL_DIR / "moda-fashion-ce-llm")
BEST_DIR = str(MODEL_DIR / "moda-fashion-ce-llm-best")

RANDOM_SEED = 42


def load_llm_labels() -> list[dict]:
    """Load LLM-judged relevance labels from JSONL file."""
    if not LABELS_PATH.exists():
        raise FileNotFoundError(
            f"No labels file at {LABELS_PATH}. "
            "Run benchmark/generate_llm_labels.py first."
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
    max_pairs: int | None = None,
    val_ratio: float = 0.10,
    seed: int = RANDOM_SEED,
) -> tuple[list[InputExample], list[InputExample]]:
    """Convert LLM labels into train/val InputExamples with normalized scores.

    Uses the same query_splits.json to ensure train labels only come from
    train-split queries, and val labels from val-split queries. This maintains
    separation from the test set used for final evaluation.
    """
    rng = random.Random(seed)

    splits = json.loads(SPLIT_PATH.read_text())
    train_qids = set(splits["train"])
    val_qids = set(splits["val"])

    train_examples: list[InputExample] = []
    val_examples: list[InputExample] = []
    skipped = 0

    for item in labels:
        qid = item["query_id"]
        score = item["score"] / 3.0  # normalize 0-3 → 0.0-1.0
        query_text = item["query_text"]
        product_text = item["product_text"]

        if not query_text or not product_text:
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
        n_val = max(100, int(len(train_examples) * val_ratio))
        val_examples = train_examples[:n_val]
        train_examples = train_examples[n_val:]
        log.info("No val-split labels found; split %d from train for validation", n_val)

    score_dist = {0: 0, 1: 0, 2: 0, 3: 0}
    for item in labels:
        score_dist[item["score"]] = score_dist.get(item["score"], 0) + 1

    log.info("Score distribution: %s", score_dist)
    log.info("Train examples: %d  Val examples: %d  (skipped %d)",
             len(train_examples), len(val_examples), skipped)
    return train_examples, val_examples


def train(args):
    log.info("=" * 60)
    log.info("MODA Phase 3B — Cross-Encoder with LLM-Judged Labels")
    log.info("Base model: %s", BASE_MODEL)
    log.info("Loss: MSE (graded 0-1 labels from LLM)")
    log.info("=" * 60)

    labels = load_llm_labels()
    train_examples, val_examples = build_examples(
        labels,
        max_pairs=args.max_pairs,
        seed=RANDOM_SEED,
    )

    if not train_examples:
        log.error("No training examples! Check labels file and query splits.")
        return

    train_labels = [e.label for e in train_examples]
    log.info("Train label stats — mean: %.3f  std: %.3f  min: %.3f  max: %.3f",
             np.mean(train_labels), np.std(train_labels),
             np.min(train_labels), np.max(train_labels))

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
        name="hnm-llm-dev",
    )

    warmup_steps = max(100, int(len(train_dataloader) * 0.06))
    eval_steps = max(500, len(train_dataloader) // 5)
    log.info("Steps per epoch: %d  |  Warmup: %d  |  Eval every: %d",
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
        d_short = e.texts[1][:40]
        log.info("  label=%.2f  pred=%.3f | q='%s...' | d='%s...'",
                 e.label, score, q_short, d_short)

    return OUTPUT_DIR


def parse_args():
    p = argparse.ArgumentParser(description="Train CE with LLM-judged labels")
    p.add_argument("--max_pairs", type=int, default=None,
                   help="Cap total training pairs (default: use all)")
    p.add_argument("--epochs", type=int, default=3,
                   help="Training epochs (default: 3)")
    p.add_argument("--batch_size", type=int, default=32,
                   help="Batch size (default: 32)")
    p.add_argument("--lr", type=float, default=2e-5,
                   help="Learning rate (default: 2e-5)")
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
