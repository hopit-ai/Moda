"""
MODA Phase 3A — Fashion Cross-Encoder Fine-Tuning

Fine-tunes cross-encoder/ms-marco-MiniLM-L-6-v2 on H&M purchase-log pairs.

Training data (from qrels.csv):
  • Positives:      (query, purchased_article)     label = 1.0
  • Hard negatives: (query, shown_not_purchased)   label = 0.0
  • Random negatives:(query, random_article)        label = 0.0

Article text format (canonical, from benchmark/article_text.py):
  prod_name | product_type_name | colour_group_name | section_name | garment_group_name | detail_desc[:200]

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

from benchmark.article_text import build_article_texts_from_df as build_article_texts

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


# ─── Training data builder ────────────────────────────────────────────────────

def split_query_ids(
    qrels_path: Path,
    queries_path: Path,
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
    seed: int = RANDOM_SEED,
) -> tuple[set[str], set[str], set[str]]:
    """Split query IDs into disjoint train / val / test sets.

    Splits by *unique query text*, not query ID. All query IDs that share
    the same text (e.g. 14 users who searched "black dress") go into the
    same split. This prevents the cross-encoder from seeing test query
    texts during training — eliminating both direct and semantic leakage.

    Returns (train_qids, val_qids, test_qids).
    """
    rng = random.Random(seed)

    qid_to_text: dict[str, str] = {}
    with open(queries_path, newline="") as f:
        for row in csv.DictReader(f):
            qid_to_text[row["query_id"].strip()] = row["query_text"].strip()

    qids_with_qrels: list[str] = []
    seen: set[str] = set()
    with open(qrels_path, newline="") as f:
        for row in csv.DictReader(f):
            qid = row["query_id"].strip()
            if qid not in seen:
                qids_with_qrels.append(qid)
                seen.add(qid)
    qids_with_qrels.sort()

    text_to_qids: dict[str, list[str]] = {}
    for qid in qids_with_qrels:
        qt = qid_to_text.get(qid, "")
        text_to_qids.setdefault(qt, []).append(qid)

    unique_texts = sorted(text_to_qids.keys())
    rng.shuffle(unique_texts)

    n = len(unique_texts)
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)

    train_qids: set[str] = set()
    val_qids:   set[str] = set()
    test_qids:  set[str] = set()

    for i, text in enumerate(unique_texts):
        group = text_to_qids[text]
        if i < n_train:
            train_qids.update(group)
        elif i < n_train + n_val:
            val_qids.update(group)
        else:
            test_qids.update(group)

    log.info("Query-text split — %d unique texts → train: %d qids  val: %d  test: %d  (seed=%d)",
             n, len(train_qids), len(val_qids), len(test_qids), seed)
    return train_qids, val_qids, test_qids


SPLIT_PATH = _REPO_ROOT / "data" / "processed" / "query_splits.json"


def save_splits(train_qids: set[str], val_qids: set[str], test_qids: set[str]):
    """Persist query splits to disk so eval scripts use the same test set."""
    SPLIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    obj = {
        "train": sorted(train_qids),
        "val":   sorted(val_qids),
        "test":  sorted(test_qids),
    }
    SPLIT_PATH.write_text(json.dumps(obj))
    log.info("Saved query splits → %s", SPLIT_PATH)


def load_splits() -> tuple[set[str], set[str], set[str]]:
    """Load previously saved query splits."""
    obj = json.loads(SPLIT_PATH.read_text())
    return set(obj["train"]), set(obj["val"]), set(obj["test"])


def build_training_pairs(
    qrels_path: Path,
    queries_path: Path,
    article_texts: dict[str, str],
    train_qids: set[str],
    val_qids: set[str],
    max_pairs: int | None = None,
    hard_neg_per_query: int = 5,
    random_neg_per_query: int = 1,
    seed: int = RANDOM_SEED,
) -> tuple[list[InputExample], list[InputExample]]:
    """Build (train, dev) InputExample lists for CrossEncoder fine-tuning.

    Only queries in ``train_qids`` produce training pairs and queries in
    ``val_qids`` produce dev pairs. The held-out ``test_qids`` are never
    touched — preventing data leakage when evaluating nDCG on the test set.

    Strategy per query:
      - 1 positive  (purchased article)
      - up to ``hard_neg_per_query`` hard negatives (shown but not purchased)
      - ``random_neg_per_query`` random negatives (sampled uniformly)

    This gives a ~1:6 pos:neg ratio, which matches cross-encoder training norms.
    """
    rng = random.Random(seed)

    log.info("Loading queries...")
    queries: dict[str, str] = {}
    with open(queries_path, newline="") as f:
        for row in csv.DictReader(f):
            queries[row["query_id"].strip()] = row["query_text"].strip()

    all_article_ids = list(article_texts.keys())

    log.info("Building training pairs from qrels...")
    train_examples: list[InputExample] = []
    val_examples: list[InputExample] = []
    skipped = 0
    t0 = time.time()

    def _make_pairs(qid, query_text, pos_ids, neg_ids) -> list[InputExample]:
        pairs = []
        pos_text = article_texts.get(pos_ids[0], "")
        if pos_text:
            pairs.append(InputExample(texts=[query_text, pos_text], label=1.0))

        hard_sample = rng.sample(neg_ids, min(hard_neg_per_query, len(neg_ids)))
        for nid in hard_sample:
            neg_text = article_texts.get(nid, "")
            if neg_text:
                pairs.append(InputExample(texts=[query_text, neg_text], label=0.0))

        for _ in range(random_neg_per_query):
            rand_id = rng.choice(all_article_ids)
            if rand_id not in pos_ids:
                neg_text = article_texts.get(rand_id, "")
                if neg_text:
                    pairs.append(InputExample(texts=[query_text, neg_text], label=0.0))
        return pairs

    with open(qrels_path, newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            qid = row["query_id"].strip()
            query_text = queries.get(qid, "")
            if not query_text:
                skipped += 1
                continue

            pos_ids = [x.strip() for x in row.get("positive_ids", "").split() if x.strip()]
            neg_ids = [x.strip() for x in row.get("negative_ids", "").split() if x.strip()]
            if not pos_ids:
                skipped += 1
                continue

            if qid in train_qids:
                train_examples.extend(_make_pairs(qid, query_text, pos_ids, neg_ids))
            elif qid in val_qids:
                val_examples.extend(_make_pairs(qid, query_text, pos_ids, neg_ids))
            # test_qids are intentionally skipped — never used for training

            if (i + 1) % 50_000 == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                log.info("  processed %d queries → train=%d  val=%d  (%.1f q/s)",
                         i + 1, len(train_examples), len(val_examples), rate)

            if max_pairs and len(train_examples) >= max_pairs:
                log.info("  reached max_pairs=%d cap", max_pairs)
                break

    rng.shuffle(train_examples)
    rng.shuffle(val_examples)
    log.info("Total — train: %d pairs  val: %d pairs  (skipped %d queries)",
             len(train_examples), len(val_examples), skipped)
    return train_examples, val_examples


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

    # ── Query-level train/val/test split (prevents data leakage) ────────────
    if SPLIT_PATH.exists() and not args.resplit:
        log.info("Loading existing query splits from %s", SPLIT_PATH)
        train_qids, val_qids, test_qids = load_splits()
    else:
        train_qids, val_qids, test_qids = split_query_ids(
            HNM_DIR / "qrels.csv", HNM_DIR / "queries.csv",
            train_ratio=0.80, val_ratio=0.10)
        save_splits(train_qids, val_qids, test_qids)

    # ── Build training pairs (test queries are never touched) ─────────────
    train_examples, dev_examples = build_training_pairs(
        qrels_path   = HNM_DIR / "qrels.csv",
        queries_path = HNM_DIR / "queries.csv",
        article_texts = article_texts,
        train_qids    = train_qids,
        val_qids      = val_qids,
        max_pairs     = args.max_pairs,
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
    p.add_argument("--resplit",   action="store_true",
                   help="Re-generate train/val/test query splits (overwrites existing)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.quick:
        args.max_pairs = 50_000
        args.epochs    = 1
        log.info("QUICK MODE: 50K pairs, 1 epoch")
    train(args)
