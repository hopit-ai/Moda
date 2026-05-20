"""
Filter the existing Marqo-GS wfash subset down to high-confidence positives
only (position <= 10 in Google Shopping rankings).

Why this matters:
  The original 5K subset is "5000 (query, image) pairs Google's ranker
  surfaced anywhere from position 1 to 100". For our distillation task
  we want only TRUE positives — pairs Google itself ranked at the top of
  the shopping result page.

Diagnostic from the original 5K:
  - pos [1,5]:  258 rows, avg score_linear=98 → strong positive
  - pos [6,10]: 262 rows, avg score_linear=93 → mid-positive
  - pos [21+]:  noisy / weak / negative — not what we want for distillation

Filter:
  pos <= 10 → ~520 high-quality (query, image) pairs

Output:
  data/processed/marqo_gs_wfash_subset_pos10/triplets.jsonl
  (same schema as the input)
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("filter-pos10")

IN_PATH = REPO / "data/processed/marqo_gs_wfash_subset/triplets.jsonl"
OUT_DIR = REPO / "data/processed/marqo_gs_wfash_subset_pos10"
OUT_PATH = OUT_DIR / "triplets.jsonl"

POSITION_THRESHOLD = 10


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows_in = [json.loads(line) for line in open(IN_PATH)]
    rows_out = [r for r in rows_in if r["position"] <= POSITION_THRESHOLD]

    log.info("filtered %d -> %d rows (position <= %d)", len(rows_in), len(rows_out), POSITION_THRESHOLD)

    queries = set(r["query"] for r in rows_out)
    images = set(r["image_path"] for r in rows_out)
    log.info("unique queries: %d  unique images: %d", len(queries), len(images))

    if rows_out:
        avg_score = sum(r["score_linear"] for r in rows_out) / len(rows_out)
        avg_pos = sum(r["position"] for r in rows_out) / len(rows_out)
        log.info("avg score_linear=%.1f  avg position=%.1f", avg_score, avg_pos)

    with open(OUT_PATH, "w") as f:
        for r in rows_out:
            f.write(json.dumps(r) + "\n")
    log.info("wrote %s (%d rows)", OUT_PATH, len(rows_out))


if __name__ == "__main__":
    main()
