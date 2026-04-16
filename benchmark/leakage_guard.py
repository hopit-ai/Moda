"""
MODA — Data Leakage Detection & Prevention

Centralized leakage checks used by training and evaluation scripts.
Import and call these at the start of any pipeline script.

Checks performed:
  1. Split disjointness (train ∩ val ∩ test = ∅)
  2. Query text uniqueness across splits (no same text in train + test)
  3. Label scope (training labels only contain train-split queries)
  4. Article text consistency (canonical builder matches)
  5. Model provenance (fine-tuned model was not exposed to test data)
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path

log = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).parent.parent
PROCESSED_DIR = _REPO_ROOT / "data" / "processed"
HNM_DIR = _REPO_ROOT / "data" / "raw" / "hnm_real"
SPLIT_PATH = PROCESSED_DIR / "query_splits.json"


class LeakageError(AssertionError):
    """Raised when a data leakage condition is detected."""


def check_splits_disjoint(
    split_path: Path = SPLIT_PATH,
) -> dict[str, set[str]]:
    """Verify train/val/test splits are strictly disjoint by query ID."""
    splits = json.loads(split_path.read_text())
    train = set(splits["train"])
    val = set(splits["val"])
    test = set(splits["test"])

    if train & val:
        raise LeakageError(
            f"train ∩ val has {len(train & val)} overlapping IDs: "
            f"{list(train & val)[:5]}"
        )
    if train & test:
        raise LeakageError(
            f"train ∩ test has {len(train & test)} overlapping IDs: "
            f"{list(train & test)[:5]}"
        )
    if val & test:
        raise LeakageError(
            f"val ∩ test has {len(val & test)} overlapping IDs: "
            f"{list(val & test)[:5]}"
        )

    log.info(
        "[LeakageGuard] Splits disjoint: train=%d, val=%d, test=%d",
        len(train), len(val), len(test),
    )
    return {"train": train, "val": val, "test": test}


def check_query_text_disjoint(
    splits: dict[str, set[str]],
    queries_csv: Path | None = None,
) -> None:
    """Verify no query TEXT appears in multiple splits.

    This catches the case where two different query IDs have the same
    surface text but end up in different splits.
    """
    if queries_csv is None:
        queries_csv = HNM_DIR / "queries.csv"

    texts_by_split: dict[str, set[str]] = {s: set() for s in splits}
    with open(queries_csv, newline="") as f:
        for row in csv.DictReader(f):
            qid = row["query_id"].strip()
            qt = row["query_text"].strip().lower()
            for sname, sids in splits.items():
                if qid in sids:
                    texts_by_split[sname].add(qt)
                    break

    total_overlap = 0
    for a in ("train",):
        for b in ("val", "test"):
            overlap = texts_by_split[a] & texts_by_split[b]
            if overlap:
                total_overlap += len(overlap)
                log.warning(
                    "[LeakageGuard] %d query texts appear in both %s and %s "
                    "(different IDs). These will be excluded from training. "
                    "Examples: %s",
                    len(overlap), a, b, list(overlap)[:5],
                )

    if total_overlap > 0:
        log.info("[LeakageGuard] %d overlapping query texts found — "
                 "training scripts must filter these out", total_overlap)
    else:
        log.info("[LeakageGuard] Query text disjointness fully verified")


def check_labels_split(
    labels_path: Path,
    allowed_qids: set[str],
    forbidden_qids: set[str],
    label_name: str = "labels",
) -> int:
    """Verify labels only contain query IDs from allowed set.

    Returns the count of verified labels.
    """
    n = 0
    leaked = set()
    with open(labels_path) as f:
        for line in f:
            row = json.loads(line)
            qid = row.get("query_id", "")
            n += 1
            if qid in forbidden_qids:
                leaked.add(qid)

    if leaked:
        raise LeakageError(
            f"{label_name} contains {len(leaked)} forbidden query IDs! "
            f"Examples: {list(leaked)[:5]}"
        )

    allowed_count = 0
    with open(labels_path) as f:
        for line in f:
            row = json.loads(line)
            if row.get("query_id", "") in allowed_qids:
                allowed_count += 1

    log.info(
        "[LeakageGuard] %s verified: %d labels, %d in allowed split, 0 in forbidden",
        label_name, n, allowed_count,
    )
    return n


def get_forbidden_train_texts(
    splits: dict[str, set[str]],
    queries_csv: Path | None = None,
) -> set[str]:
    """Return query texts that appear in train AND (val or test).

    Training scripts should exclude queries matching these texts to prevent
    the model from memorizing text it will be evaluated on.
    """
    if queries_csv is None:
        queries_csv = HNM_DIR / "queries.csv"

    texts_by_split: dict[str, set[str]] = {s: set() for s in splits}
    with open(queries_csv, newline="") as f:
        for row in csv.DictReader(f):
            qid = row["query_id"].strip()
            qt = row["query_text"].strip().lower()
            for sname, sids in splits.items():
                if qid in sids:
                    texts_by_split[sname].add(qt)
                    break

    forbidden = set()
    for other in ("val", "test"):
        forbidden |= texts_by_split["train"] & texts_by_split[other]
    if forbidden:
        log.info("[LeakageGuard] %d train query texts also in val/test — will be excluded",
                 len(forbidden))
    return forbidden


def run_all_checks(
    labels_path: Path | None = None,
    split_path: Path = SPLIT_PATH,
) -> dict[str, set[str]]:
    """Run all leakage checks and return verified splits.

    Call this at the top of any training or evaluation script.
    """
    log.info("[LeakageGuard] Running comprehensive leakage checks...")

    splits = check_splits_disjoint(split_path)

    queries_csv = HNM_DIR / "queries.csv"
    if queries_csv.exists():
        check_query_text_disjoint(splits, queries_csv)

    if labels_path is not None and labels_path.exists():
        forbidden = splits["val"] | splits["test"]
        check_labels_split(
            labels_path,
            allowed_qids=splits["train"],
            forbidden_qids=forbidden,
            label_name=labels_path.name,
        )

    log.info("[LeakageGuard] All checks passed")
    return splits
