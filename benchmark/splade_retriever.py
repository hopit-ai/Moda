"""
SPLADE sparse retrieval module for MODA.

Replaces BM25 (OpenSearch) with a learned sparse retriever.
Encodes articles and queries into sparse vocabulary-sized vectors via
a masked language model head (ReLU + log1p + max-pool), then retrieves
via sparse dot product.

Model: naver/splade-cocondenser-ensembledistil
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

if sys.platform != "win32":
    import fcntl

import numpy as np
import scipy.sparse as sp
import torch
from tqdm import tqdm
from transformers import AutoModelForMaskedLM, AutoTokenizer

log = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).parent.parent
DEFAULT_MODEL = "naver/splade-cocondenser-ensembledistil"
DEFAULT_CACHE_DIR = _REPO_ROOT / "data" / "processed"


class SpladeRetriever:
    """Learned sparse retriever using SPLADE max-pooling."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        cache_dir: Path | str = DEFAULT_CACHE_DIR,
        device: str | None = None,
        max_length: int = 256,
    ):
        self.model_name = model_name
        self.cache_dir = Path(cache_dir)
        self.max_length = max_length

        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device

        log.info("Loading SPLADE model: %s on %s", model_name, device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        self.article_ids: list[str] = []
        self.article_sparse: sp.csr_matrix | None = None

    def free_model(self) -> None:
        """Release the transformer model and tokenizer to free GPU/MPS memory.

        Call after search_batch() if you no longer need to encode new texts.
        The article_sparse matrix and article_ids are kept for reference.
        """
        del self.model
        del self.tokenizer
        self.model = None  # type: ignore[assignment]
        self.tokenizer = None  # type: ignore[assignment]
        if self.device == "mps":
            torch.mps.empty_cache()
        elif self.device == "cuda":
            torch.cuda.empty_cache()
        import gc
        gc.collect()
        log.info("SPLADE model freed from %s", self.device)

    # ------------------------------------------------------------------
    # Encoding helpers
    # ------------------------------------------------------------------

    def _encode_texts(
        self, texts: list[str], batch_size: int = 32, desc: str = "SPLADE encode"
    ) -> sp.csr_matrix:
        """Encode a list of texts into sparse vocab-sized vectors."""
        vocab_size = self.model.config.vocab_size
        rows, cols, vals = [], [], []

        for start in tqdm(range(0, len(texts), batch_size), desc=desc, ncols=80):
            batch = texts[start : start + batch_size]
            tokens = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            )
            tokens = {k: v.to(self.device) for k, v in tokens.items()}

            with torch.no_grad():
                logits = self.model(**tokens).logits  # (B, seq_len, vocab)
                relu_log = torch.log1p(torch.relu(logits))  # (B, seq_len, vocab)
                weighted = relu_log * tokens["attention_mask"].unsqueeze(-1)
                sparse_vecs = weighted.max(dim=1).values  # (B, vocab)

            sparse_vecs = sparse_vecs.cpu().float().numpy()

            for i, vec in enumerate(sparse_vecs):
                nz = np.nonzero(vec)[0]
                if len(nz) == 0:
                    continue
                row_idx = start + i
                rows.extend([row_idx] * len(nz))
                cols.extend(nz.tolist())
                vals.extend(vec[nz].tolist())

            if self.device == "mps" and start % (batch_size * 20) == 0:
                torch.mps.empty_cache()

        matrix = sp.csr_matrix(
            (vals, (rows, cols)), shape=(len(texts), vocab_size), dtype=np.float32
        )
        return matrix

    # ------------------------------------------------------------------
    # Article index
    # ------------------------------------------------------------------

    def _cache_path(self) -> Path:
        safe_name = self.model_name.replace("/", "_")
        return self.cache_dir / f"splade_{safe_name}_articles.npz"

    def _ids_cache_path(self) -> Path:
        safe_name = self.model_name.replace("/", "_")
        return self.cache_dir / f"splade_{safe_name}_article_ids.npy"

    def encode_articles(
        self,
        article_ids: list[str],
        article_texts: dict[str, str],
        batch_size: int = 32,
        force: bool = False,
    ) -> None:
        """Encode all articles and cache to disk. Skips if cache exists."""
        cache_path = self._cache_path()
        ids_path = self._ids_cache_path()

        if cache_path.exists() and ids_path.exists() and not force:
            log.info("Loading cached SPLADE article embeddings from %s", cache_path)
            self.article_sparse = sp.load_npz(str(cache_path))
            self.article_ids = np.load(str(ids_path), allow_pickle=True).tolist()
            log.info(
                "  Loaded %d articles, matrix shape %s, nnz=%d",
                len(self.article_ids),
                self.article_sparse.shape,
                self.article_sparse.nnz,
            )
            return

        # Serialize encode across processes (overnight parallel runs).
        lock_path = self.cache_dir / ".splade_article_encode.lock"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        lock_f = None
        if sys.platform != "win32":
            lock_f = open(lock_path, "w")
            fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX)
        try:
            if cache_path.exists() and ids_path.exists() and not force:
                log.info("Loading cached SPLADE article embeddings (after lock) from %s", cache_path)
                self.article_sparse = sp.load_npz(str(cache_path))
                self.article_ids = np.load(str(ids_path), allow_pickle=True).tolist()
                return

            log.info("Encoding %d articles with SPLADE (batch_size=%d)...", len(article_ids), batch_size)
            texts = [article_texts.get(aid, "") for aid in article_ids]
            t0 = time.time()
            self.article_sparse = self._encode_texts(texts, batch_size=batch_size, desc="SPLADE articles")
            self.article_ids = list(article_ids)
            elapsed = time.time() - t0
            log.info(
                "  Encoded %d articles in %.1fs (%.1f art/s), nnz=%d, saving cache...",
                len(article_ids), elapsed, len(article_ids) / max(elapsed, 1e-6), self.article_sparse.nnz,
            )

            sp.save_npz(str(cache_path), self.article_sparse)
            np.save(str(ids_path), np.array(self.article_ids))
            log.info("  Cache saved to %s", cache_path)
        finally:
            if lock_f is not None:
                fcntl.flock(lock_f.fileno(), fcntl.LOCK_UN)
                lock_f.close()

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search_batch(
        self, queries: list[str], top_k: int = 100, batch_size: int = 64,
        query_chunk: int = 1000,
    ) -> list[list[str]]:
        """Encode queries and retrieve top-k articles per query via sparse dot product.

        Args:
            query_chunk: number of queries to score against the article matrix at a
                time.  Keeps peak RAM bounded — the dense slice is only
                (query_chunk × n_articles) instead of (all_queries × n_articles).
        """
        if self.article_sparse is None:
            raise RuntimeError("Call encode_articles() first")

        log.info("Encoding %d queries with SPLADE...", len(queries))
        q_sparse = self._encode_texts(queries, batch_size=batch_size, desc="SPLADE queries")

        n_queries = q_sparse.shape[0]
        n_articles = self.article_sparse.shape[0]
        log.info(
            "Computing sparse dot products (%d queries x %d articles, chunk=%d)...",
            n_queries, n_articles, query_chunk,
        )
        t0 = time.time()

        art_T = self.article_sparse.T.tocsc()
        results: list[list[str]] = []

        for start in range(0, n_queries, query_chunk):
            end = min(start + query_chunk, n_queries)
            chunk_sparse = q_sparse[start:end] @ art_T  # (chunk, n_articles) sparse
            chunk_dense = chunk_sparse.toarray()

            for i in range(chunk_dense.shape[0]):
                row = chunk_dense[i]
                if top_k >= len(row):
                    top_idx = np.argsort(-row)
                else:
                    top_idx = np.argpartition(-row, top_k)[:top_k]
                    top_idx = top_idx[np.argsort(-row[top_idx])]
                results.append([self.article_ids[j] for j in top_idx if row[j] > 0])

            if start % (query_chunk * 5) == 0 and start > 0:
                log.info("  Scored %d / %d queries...", len(results), n_queries)

        elapsed = time.time() - t0
        log.info("  Retrieval done in %.1fs (%.1f q/s)", elapsed, len(queries) / max(elapsed, 0.01))
        return results
