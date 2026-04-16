"""
Canonical article-text builder for MODA.

Every pipeline that converts H&M article metadata into a pipe-separated text
string MUST use this module so train and eval see identical formatting.

Format:
    prod_name | product_type_name | colour_group_name | section_name
    | garment_group_name | detail_desc[:200]
"""

from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger(__name__)

_FIELDS: list[tuple[str, int | None]] = [
    ("prod_name", None),
    ("product_type_name", None),
    ("colour_group_name", None),
    ("section_name", None),
    ("garment_group_name", None),
    ("detail_desc", 200),
]


def build_article_text(row: dict[str, Any] | Any) -> str:
    """Build a pipe-separated text string from an article metadata row.

    Works with both plain dicts and pandas Series / named tuples.
    """
    parts: list[str] = []
    for field, limit in _FIELDS:
        val = str(row.get(field, "") if isinstance(row, dict) else getattr(row, field, "")).strip()
        if val and val.lower() not in ("nan", "none", ""):
            parts.append(val[:limit] if limit else val)
    return " | ".join(parts)


def build_article_texts_from_df(articles_df: Any) -> dict[str, str]:
    """Build ``{article_id: text}`` mapping from a pandas DataFrame."""
    texts: dict[str, str] = {}
    for _, row in articles_df.iterrows():
        aid = str(row.get("article_id", "")).strip()
        if not aid:
            continue
        texts[aid] = build_article_text(row)
    log.info("Article texts built: %d entries", len(texts))
    return texts
