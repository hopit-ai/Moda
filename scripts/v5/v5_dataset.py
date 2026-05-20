"""
v5 grouped-batch dataset.

Each training batch is K queries × N products/query (default 8×16 = 128).
This guarantees that products belonging to the same query are always together
in the batch — no false-negative pairing across same-query items.

The dataset reads pairs_labeled.jsonl and the precomputed student image cache
(produced by phase_a_cache_student_image_emb.py), so per-step compute is
text-only forward + cache lookups.
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import torch


@dataclass
class V5Batch:
    """A K×N grouped batch ready for the loss function."""
    query: list[str]                     # length K
    title: list[str]                     # length K (one per query — first product's title)
    category_l2: list[str]               # length K
    query_idx: torch.Tensor              # (K*N,) which query (0..K-1) each product belongs to
    image_idx: torch.Tensor              # (K*N,) row in student_image_emb cache
    score_linear: torch.Tensor           # (K*N,) score weights from GS-10M
    pair_ids: list[str]                  # K*N for diagnostic logging
    K: int
    N: int


class V5Dataset:
    """Loads labeled pairs and prepares grouped batches.

    Parameters
    ----------
    pairs_path : Path
        Path to pairs_labeled.jsonl produced by phase_a_extract_multifield.
    image_index_path : Path
        Path to student_image_index.json (maps pair_id → row in cache).
    K : int
        Queries per batch.
    N : int
        Products per query per batch.
    min_products_per_query : int
        Drop queries that have fewer than this many products available.
        Default 2 — singleton queries are useless for in-group GCL.
    """
    def __init__(
        self,
        pairs_path: Path,
        image_index_path: Path,
        K: int = 8,
        N: int = 16,
        min_products_per_query: int = 2,
        seed: int = 1337,
    ):
        self.K = K
        self.N = N
        self.rng = random.Random(seed)

        with image_index_path.open() as f:
            self.image_index: dict[str, int] = json.load(f)

        # Group pairs by query
        by_query: dict[str, list[dict]] = defaultdict(list)
        n_loaded = n_skipped_no_image = 0
        with pairs_path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                if r["pair_id"] not in self.image_index:
                    n_skipped_no_image += 1
                    continue
                by_query[r["query"]].append(r)
                n_loaded += 1

        # Drop queries with too few products
        self.queries: list[str] = []
        self.products: list[list[dict]] = []
        n_dropped_singleton = 0
        for q, prods in by_query.items():
            if len(prods) >= min_products_per_query:
                self.queries.append(q)
                self.products.append(prods)
            else:
                n_dropped_singleton += 1

        self.n_total_pairs = n_loaded
        self.n_skipped_no_image = n_skipped_no_image
        self.n_dropped_singleton = n_dropped_singleton

    def __len__(self) -> int:
        """Number of batches in one epoch (one batch per K queries)."""
        return len(self.queries) // self.K

    def stats(self) -> dict:
        prods_per_q = [len(p) for p in self.products]
        return {
            "n_pairs_loaded": self.n_total_pairs,
            "n_pairs_skipped_no_image": self.n_skipped_no_image,
            "n_queries_dropped_singleton": self.n_dropped_singleton,
            "n_queries_kept": len(self.queries),
            "products_per_query_mean": sum(prods_per_q) / max(1, len(prods_per_q)),
            "products_per_query_min": min(prods_per_q) if prods_per_q else 0,
            "products_per_query_max": max(prods_per_q) if prods_per_q else 0,
            "batches_per_epoch": len(self),
        }

    def iter_batches(self, shuffle: bool = True) -> Iterator[V5Batch]:
        """Yield V5Batch objects. One pass = one epoch."""
        order = list(range(len(self.queries)))
        if shuffle:
            self.rng.shuffle(order)

        for batch_start in range(0, len(order), self.K):
            chosen = order[batch_start : batch_start + self.K]
            if len(chosen) < self.K:
                break

            queries: list[str] = []
            titles: list[str] = []
            cats: list[str] = []
            query_idx: list[int] = []
            image_idx: list[int] = []
            scores: list[float] = []
            pair_ids: list[str] = []

            for k, qi in enumerate(chosen):
                q = self.queries[qi]
                prods = self.products[qi]
                if len(prods) >= self.N:
                    sampled = self.rng.sample(prods, self.N)
                else:
                    # Sample with replacement to fill the slot
                    sampled = list(prods) + self.rng.choices(prods, k=self.N - len(prods))

                queries.append(q)
                # Use the first sampled product's title and category for the multi-field rep
                titles.append(sampled[0].get("title") or "")
                cats.append(sampled[0].get("category_l2") or sampled[0].get("category2") or "general")

                for p in sampled:
                    query_idx.append(k)
                    image_idx.append(self.image_index[p["pair_id"]])
                    scores.append(float(p.get("score_linear", 80)))
                    pair_ids.append(p["pair_id"])

            yield V5Batch(
                query=queries,
                title=titles,
                category_l2=cats,
                query_idx=torch.tensor(query_idx, dtype=torch.long),
                image_idx=torch.tensor(image_idx, dtype=torch.long),
                score_linear=torch.tensor(scores, dtype=torch.float32),
                pair_ids=pair_ids,
                K=self.K,
                N=self.N,
            )
