"""
MODA Phase 3A — Fashion Cross-Encoder Fine-Tuning

Fine-tunes cross-encoder/ms-marco-MiniLM-L-6-v2 on H&M purchase-log pairs.

Training data (from qrels.csv):
  • Positives:      (query, purchased_article)     label = 1.0
  • Hard negatives: (query, shown_not_purchased)   label = 0.0
  • Random negatives:(query, random_article)        label = 0.0

Article text format (same as eval_rerank.py):
  prod_name | product_type_name | colour_group_name | section_name | detail_desc[:200]

Architecture: MiniLM-L6 cross-encoder (80MB) — sentence-pair classification.
Loss: BCEWithLogitsLoss (binary cross-entropy).
Hardware: Apple MPS or CPU. A100 ~10× faster but not required.

Output:
  models/moda-fashion-ce/          final model directory
  models/moda-fashion-ce-best/     best dev checkpoint

Usage:
  python benchmark/train_cross_encoder.py              # full 2.5M pairs, 3 epochs
  python benchmark/train_cross_encoder.py --max_pairs 200000 --epochs 1  # quick test
"""

from __future__ import annotations

import argparse
import csv
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
from sentence_transformers.cross_encoder.evaluation import CEBinaryClassificationEvaluator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).parent.parent
HNM_DIR    = _REPO_ROOT / "data" / "raw" / "hnm_real"
MODEL_DIR  = _REPO_ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

BASE_MODEL   = "cross-encoder/ms-marco-MiniLM-L-6-v2"
OUTPUT_DIR   = str(MODEL_DIR / "moda-fashion-ce")
BEST_DIR     = str(MODEL_DIR / "moda-fashion-ce-best")

RANDOM_SEED  = 42


# ─── Article text builder ─────────────────────────────────────────────────────

def build_article_texts(articles_df: pd.DataFrame) -> dict[str, str]:
    """
    Build rich text representation for each article.
    Same format used in eval_rerank.py for consistency.
    """
    log.info("Building article text cache for %d articles...", len(articles_df))
    texts = {}
    for _, row in articles_df.iterrows():
        aid = str(row.get("article_id", "")).strip()
        if not aid:
            continue
        parts = []
        for field, limit in [
            ("prod_name",           None),
            ("product_type_name",   None),
            ("colour_group_name",   None),
            ("section_name",        None),
            ("garment_group_name",  None),
            ("detail_desc",         200),
        ]:
            val = str(row.get(field, "")).strip()
            if val and val.lower() not in ("nan", "none", ""):
                parts.append(val[:limit] if limit else val)
        texts[aid] = " | ".join(parts)
    log.info("Article texts built: %d entries", len(texts))
    return texts


# ─── Training data builder ────────────────────────────────────────────────────

def build_training_pairs(
    qrels_path: Path,
    queries_path: Path,
    article_texts: dict[str, str],
    max_pairs: int | None = None,
    dev_ratio: float = 0.05,
    hard_neg_per_query: int = 5,
    random_neg_per_query: int = 1,
    seed: int = RANDOM_SEED,
) -> tuple[list[InputExample], list[InputExample]]:
    """
    Build (train, dev) InputExample lists for CrossEncoder fine-tuning.

    Strategy:
      - 1 positive  per query   (purchased article)
      - up to `hard_neg_per_query` hard negatives (shown but not purchased)
      - `random_neg_per_query` random negatives (sampled uniformly)

    This gives a ~1:6 pos:neg ratio, which matches cross-encoder training norms.
    Hard negatives are the most informative signal for relevance ranking.
    """
    rng = random.Random(seed)

    # Load queries
    log.info("Loading queries...")
    queries: dict[str, str] = {}
    with open(queries_path, newline="") as f:
        for row in csv.DictReader(f):
            queries[row["query_id"].strip()] = row["query_text"].strip()

    all_article_ids = list(article_texts.keys())

    log.info("Building training pairs from qrels...")
    examples = []
    skipped = 0
    t0 = time.time()

    with open(qrels_path, newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            qid = row["query_id"].strip()
            query_text = queries.get(qid, "")
            if not query_text:
                skipped += 1
                continue

            pos_ids  = [x.strip() for x in row.get("positive_ids",  "").split() if x.strip()]
            neg_ids  = [x.strip() for x in row.get("negative_ids",  "").split() if x.strip()]

            if not pos_ids:
                skipped += 1
                continue

            # One positive (first purchased article)
            pos_text = article_texts.get(pos_ids[0], "")
            if pos_text:
                examples.append(InputExample(texts=[query_text, pos_text], label=1.0))

            # Hard negatives (shown, not purchased) — sample up to hard_neg_per_query
            hard_sample = rng.sample(neg_ids, min(hard_neg_per_query, len(neg_ids)))
            for nid in hard_sample:
                neg_text = article_texts.get(nid, "")
                if neg_text:
                    examples.append(InputExample(texts=[query_text, neg_text], label=0.0))

            # Random negatives
            for _ in range(random_neg_per_query):
                rand_id = rng.choice(all_article_ids)
                # Avoid accidentally picking a positive
                if rand_id not in pos_ids:
                    neg_text = article_texts.get(rand_id, "")
                    if neg_text:
                        examples.append(InputExample(texts=[query_text, neg_text], label=0.0))

            if (i + 1) % 50_000 == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                log.info("  built %d queries → %d pairs  (%.1f q/s)", i+1, len(examples), rate)

            # Early cap
            if max_pairs and len(examples) >= max_pairs:
                log.info("  reached max_pairs=%d cap", max_pairs)
                break

    log.info("Total pairs built: %d  (skipped %d queries)", len(examples), skipped)

    # Shuffle and split
    rng.shuffle(examples)
    n_dev = max(500, int(len(examples) * dev_ratio))
    dev   = examples[:n_dev]
    train = examples[n_dev:]
    log.info("Train: %d  |  Dev: %d", len(train), len(dev))
    return train, dev


# ─── Training ─────────────────────────────────────────────────────────────────

def train(args):
    log.info("=" * 60)
    log.info("MODA Phase 3A — Cross-Encoder Fine-Tuning")
    log.info("Base model: %s", BASE_MODEL)
    log.info("=" * 60)

    # ── Load articles ──────────────────────────────────────────────────────────
    log.info("Loading H&M articles...")
    articles_df = pd.read_csv(HNM_DIR / "articles.csv", dtype=str).fillna("")
    article_texts = build_article_texts(articles_df)

    # ── Build training pairs ───────────────────────────────────────────────────
    train_examples, dev_examples = build_training_pairs(
        qrels_path   = HNM_DIR / "qrels.csv",
        queries_path = HNM_DIR / "queries.csv",
        article_texts = article_texts,
        max_pairs     = args.max_pairs,
        dev_ratio     = 0.05,
        hard_neg_per_query = args.hard_negs,
        random_neg_per_query = 1,
    )

    # ── Report class balance ───────────────────────────────────────────────────
    pos = sum(1 for e in train_examples if e.label == 1.0)
    neg = len(train_examples) - pos
    log.info("Train class balance — pos: %d  neg: %d  ratio: 1:%.1f", pos, neg, neg/max(pos,1))

    # ── Load model ────────────────────────────────────────────────────────────
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    log.info("Loading cross-encoder on device: %s", device)
    model = CrossEncoder(
        BASE_MODEL,
        num_labels=1,
        max_length=512,
        device=device,
    )

    # ── DataLoader ────────────────────────────────────────────────────────────
    train_dataloader = DataLoader(
        train_examples,
        shuffle=True,
        batch_size=args.batch_size,
        num_workers=0,       # required for MPS
    )

    # ── Dev evaluator (AP + accuracy) ─────────────────────────────────────────
    dev_sentences1 = [e.texts[0] for e in dev_examples]
    dev_sentences2 = [e.texts[1] for e in dev_examples]
    dev_labels     = [int(e.label) for e in dev_examples]
    evaluator = CEBinaryClassificationEvaluator(
        sentence_pairs  = list(zip(dev_sentences1, dev_sentences2)),
        labels          = dev_labels,
        name            = "hnm-dev",
        show_progress_bar = False,
    )

    # ── Warmup steps (6% of first epoch) ─────────────────────────────────────
    warmup_steps = max(100, int(len(train_dataloader) * 0.06))
    log.info("Steps per epoch: %d  |  Warmup: %d", len(train_dataloader), warmup_steps)

    # ── Train ─────────────────────────────────────────────────────────────────
    t0 = time.time()
    log.info("Starting training: %d epochs, batch=%d, lr=%s",
             args.epochs, args.batch_size, args.lr)

    model.fit(
        train_dataloader    = train_dataloader,
        evaluator           = evaluator,
        epochs              = args.epochs,
        optimizer_params    = {"lr": args.lr},
        warmup_steps        = warmup_steps,
        evaluation_steps    = max(500, len(train_dataloader) // 5),
        output_path         = BEST_DIR,
        save_best_model     = True,
        show_progress_bar   = True,
    )

    elapsed = time.time() - t0
    log.info("Training finished in %.1f min", elapsed / 60)

    # ── Save final model ──────────────────────────────────────────────────────
    model.save(OUTPUT_DIR)
    log.info("Final model saved → %s", OUTPUT_DIR)
    log.info("Best checkpoint  → %s", BEST_DIR)

    # ── Quick sanity check ────────────────────────────────────────────────────
    log.info("\nSanity check on 5 dev examples:")
    sample = dev_examples[:5]
    scores = model.predict([(e.texts[0], e.texts[1]) for e in sample])
    for e, score in zip(sample, scores):
        label = "✅ pos" if e.label == 1.0 else "❌ neg"
        q_short = e.texts[0][:40]
        d_short = e.texts[1][:40]
        log.info("  [%s] score=%.3f | q='%s...' | d='%s...'",
                 label, score, q_short, d_short)

    return OUTPUT_DIR


# ─── CLI ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train MODA fashion cross-encoder")
    p.add_argument("--max_pairs", type=int, default=None,
                   help="Cap total training pairs (default: use all ~2.5M)")
    p.add_argument("--epochs",    type=int,   default=3,
                   help="Training epochs (default: 3)")
    p.add_argument("--batch_size",type=int,   default=32,
                   help="Batch size (default: 32; reduce to 16 if OOM)")
    p.add_argument("--lr",        type=float, default=2e-5,
                   help="Learning rate (default: 2e-5)")
    p.add_argument("--hard_negs", type=int,   default=5,
                   help="Hard negatives per query (default: 5)")
    p.add_argument("--quick",     action="store_true",
                   help="Quick test: 50K pairs, 1 epoch — verify pipeline works")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.quick:
        args.max_pairs = 50_000
        args.epochs    = 1
        log.info("QUICK MODE: 50K pairs, 1 epoch")
    train(args)
