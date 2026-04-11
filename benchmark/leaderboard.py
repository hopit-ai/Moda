"""
MODA Phase 1 — Leaderboard Manager

Tracks benchmark results across all three tiers:
  Tier 1  — Marqo 7-dataset embedding benchmark
  Tier 2  — H&M full-pipeline benchmark (our contribution)
  Tier 3  — H&M embedding-only baselines (FAISS)

Results are persisted as JSON in the results/ directory and can be rendered
as markdown tables suitable for the project README.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Canonical display columns for each tier
_TIER_COLUMNS: dict[int, list[str]] = {
    1: ["model", "dataset", "t2i_r@1", "t2i_r@10", "t2i_avg_recall", "t2i_mrr",
        "cat_p@1", "cat_p@10", "cat_ap", "cat_mrr"],
    2: ["model", "retrieval_method", "ndcg@10", "mrr", "recall@20", "p@10", "mean_latency_ms"],
    3: ["model", "ndcg@10", "mrr", "recall@20", "p@10", "mean_latency_ms"],
}

_TIER_LABELS = {
    1: "Tier 1 — Marqo 7-Dataset Embedding Benchmark",
    2: "Tier 2 — H&M Full-Pipeline Benchmark",
    3: "Tier 3 — H&M Embedding-Only Baselines (FAISS)",
}


class Leaderboard:
    """Manages benchmark results across all MODA tiers.

    Results are stored in memory as a list of row dicts, each containing a
    ``"tier"`` key.  The leaderboard can be persisted to / loaded from a JSON
    file.

    Args:
        results_dir: Directory where ``leaderboard.json`` will be read/written.
    """

    def __init__(self, results_dir: str | Path = "results") -> None:
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self._rows: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    @property
    def _json_path(self) -> Path:
        return self.results_dir / "leaderboard.json"

    def load(self) -> "Leaderboard":
        """Load existing results from disk (no-op if file absent)."""
        if self._json_path.exists():
            with open(self._json_path, "r") as f:
                data = json.load(f)
            self._rows = data.get("rows", [])
            logger.info("Loaded %d leaderboard rows from %s", len(self._rows), self._json_path)
        return self

    def save(self) -> None:
        """Persist current rows to disk."""
        payload = {
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "rows": self._rows,
        }
        with open(self._json_path, "w") as f:
            json.dump(payload, f, indent=2)
        logger.info("Saved %d leaderboard rows to %s", len(self._rows), self._json_path)

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add_result(
        self,
        method_name: str,
        metrics_dict: dict[str, Any],
        tier: int,
        *,
        overwrite: bool = True,
    ) -> None:
        """Add or update a result row.

        Args:
            method_name: Unique identifier for the method / model combo.
            metrics_dict: Dict of metric names → values.
            tier: 1, 2, or 3.
            overwrite: If True, replaces any existing row with the same
                ``(tier, method_name)`` key.  If False, appends unconditionally.
        """
        row: dict[str, Any] = {
            "tier": tier,
            "method_name": method_name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **metrics_dict,
        }

        if overwrite:
            self._rows = [
                r for r in self._rows
                if not (r.get("tier") == tier and r.get("method_name") == method_name)
            ]

        self._rows.append(row)
        logger.debug("Added result for tier=%d method=%s", tier, method_name)

    def clear_tier(self, tier: int) -> None:
        """Remove all rows for a given tier."""
        self._rows = [r for r in self._rows if r.get("tier") != tier]

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def get_tier(self, tier: int) -> list[dict[str, Any]]:
        """Return all rows for a specific tier, sorted by nDCG@10 descending."""
        rows = [r for r in self._rows if r.get("tier") == tier]
        sort_key = "ndcg@10" if tier in (2, 3) else "t2i_avg_recall"
        return sorted(rows, key=lambda r: r.get(sort_key, 0), reverse=True)

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def print_leaderboard(self, tier: int) -> None:
        """Pretty-print the leaderboard table for a tier to stdout."""
        rows = self.get_tier(tier)
        title = _TIER_LABELS.get(tier, f"Tier {tier}")
        columns = _TIER_COLUMNS.get(tier, [])

        print(f"\n{'='*80}")
        print(f"  {title}")
        print(f"{'='*80}")

        if not rows:
            print("  (no results yet)")
            return

        # Determine which columns actually have data
        present_cols = [c for c in columns if any(c in r for r in rows)]
        if not present_cols:
            present_cols = [k for k in rows[0].keys() if k not in ("tier", "timestamp")]

        # Compute column widths
        widths = {c: max(len(c), max(len(_fmt(r.get(c, "-"))) for r in rows)) for c in present_cols}

        header = "  " + " | ".join(c.ljust(widths[c]) for c in present_cols)
        sep = "  " + "-+-".join("-" * widths[c] for c in present_cols)
        print(header)
        print(sep)
        for row in rows:
            line = "  " + " | ".join(_fmt(row.get(c, "-")).ljust(widths[c]) for c in present_cols)
            print(line)
        print()

    # ------------------------------------------------------------------
    # Markdown export
    # ------------------------------------------------------------------

    def save_markdown(self, output_path: str | Path | None = None) -> Path:
        """Save leaderboard as a markdown file with one table per tier.

        Args:
            output_path: Destination path. Defaults to
                ``results/leaderboard.md``.

        Returns:
            Path to the written file.
        """
        if output_path is None:
            output_path = self.results_dir / "leaderboard.md"
        output_path = Path(output_path)

        lines: list[str] = [
            "# MODA Phase 1 Leaderboard",
            "",
            f"_Last updated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}_",
            "",
        ]

        for tier in (1, 2, 3):
            rows = self.get_tier(tier)
            title = _TIER_LABELS.get(tier, f"Tier {tier}")
            lines.append(f"## {title}")
            lines.append("")

            if not rows:
                lines.append("_No results yet._")
                lines.append("")
                continue

            columns = _TIER_COLUMNS.get(tier, [])
            present_cols = [c for c in columns if any(c in r for r in rows)]
            if not present_cols:
                present_cols = [k for k in rows[0].keys() if k not in ("tier", "timestamp")]

            # Header row
            lines.append("| " + " | ".join(present_cols) + " |")
            lines.append("| " + " | ".join("---" for _ in present_cols) + " |")

            for row in rows:
                lines.append("| " + " | ".join(_fmt(row.get(c, "-")) for c in present_cols) + " |")

            lines.append("")

        output_path.write_text("\n".join(lines))
        logger.info("Leaderboard markdown saved to %s", output_path)
        return output_path


# ---------------------------------------------------------------------------
# Formatting helper
# ---------------------------------------------------------------------------


def _fmt(value: Any) -> str:
    """Format a metric value for display."""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------


def load_leaderboard(results_dir: str | Path = "results") -> Leaderboard:
    """Create and hydrate a Leaderboard from disk."""
    return Leaderboard(results_dir).load()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cli() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Print and export the MODA Phase 1 leaderboard."
    )
    parser.add_argument(
        "--results_dir", default="results", help="Path to results directory"
    )
    parser.add_argument(
        "--tier", type=int, choices=[1, 2, 3], default=None,
        help="Print a specific tier (default: all tiers)"
    )
    parser.add_argument(
        "--save_markdown", action="store_true",
        help="Also save leaderboard.md to results_dir"
    )
    args = parser.parse_args()

    lb = load_leaderboard(args.results_dir)
    tiers = [args.tier] if args.tier else [1, 2, 3]
    for tier in tiers:
        lb.print_leaderboard(tier)

    if args.save_markdown:
        path = lb.save_markdown()
        print(f"Markdown saved to {path}")


if __name__ == "__main__":
    _cli()
