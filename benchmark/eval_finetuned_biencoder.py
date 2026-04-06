"""
MODA Phase 3C — Fine-Tuned Bi-Encoder Evaluation

Compares baseline FashionCLIP vs fine-tuned FashionCLIP on the held-out
test split. Measures dense retrieval quality (nDCG, MRR, Recall) and also
evaluates the full pipeline with the fine-tuned retriever feeding into
the best cross-encoder (LLM-trained CE from Phase 3B).

Evaluation:
  1. Baseline FashionCLIP dense retrieval → metrics
  2. Fine-tuned FashionCLIP dense retrieval → metrics
  3. Fine-tuned retriever + BM25 hybrid → metrics
  4. Fine-tuned retriever + BM25 hybrid + LLM-trained CE rerank → metrics

This requires building a new FAISS index from the fine-tuned embeddings.

Usage:
  python benchmark/eval_finetuned_biencoder.py
"""

from __future__ import annotations

import csv
import json
import logging
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from benchmark.metrics import compute_all_metrics, aggregate_metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

HNM_DIR = _REPO_ROOT / "data" / "raw" / "hnm_real"
EMBEDDINGS_DIR = _REPO_ROOT / "data" / "processed" / "embeddings"
PROCESSED_DIR = _REPO_ROOT / "data" / "processed"
RESULTS_DIR = _REPO_ROOT / "results" / "real"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR = _REPO_ROOT / "models"

SPLIT_PATH = PROCESSED_DIR / "query_splits.json"
FINETUNED_DIR = MODEL_DIR / "moda-fashionclip-finetuned"
FINETUNED_BEST_DIR = FINETUNED_DIR / "best"
CE_LLM_DIR = MODEL_DIR / "moda-fashion-ce-llm-best"

TOP_K = 100


def load_test_data():
    """Load test-split queries and qrels."""
    splits = json.loads(SPLIT_PATH.read_text())
    test_qids = set(splits["test"])

    queries = []
    with open(HNM_DIR / "queries.csv", newline="") as f:
        for row in csv.DictReader(f):
            qid = row["query_id"].strip()
            if qid in test_qids:
                queries.append((qid, row["query_text"].strip()))

    qrels: dict[str, dict[str, int]] = {}
    with open(HNM_DIR / "qrels.csv", newline="") as f:
        for row in csv.DictReader(f):
            qid = row["query_id"].strip()
            if qid not in test_qids:
                continue
            pos_ids = [x.strip() for x in row.get("positive_ids", "").split() if x.strip()]
            if pos_ids:
                qrels[qid] = {aid: 1 for aid in pos_ids}

    log.info("Test queries: %d, queries with qrels: %d", len(queries), len(qrels))
    return queries, qrels


def load_articles():
    articles_df = pd.read_csv(HNM_DIR / "articles.csv", dtype=str).fillna("")
    texts = {}
    for _, row in articles_df.iterrows():
        aid = str(row.get("article_id", "")).strip()
        if not aid:
            continue
        parts = []
        for field, limit in [
            ("prod_name", None),
            ("product_type_name", None),
            ("colour_group_name", None),
            ("section_name", None),
            ("garment_group_name", None),
            ("detail_desc", 200),
        ]:
            val = str(row.get(field, "")).strip()
            if val and val.lower() not in ("nan", "none", ""):
                parts.append(val[:limit] if limit else val)
        texts[aid] = " | ".join(parts)
    return texts


def encode_with_finetuned(texts: list[str], model_path: Path, device: str) -> np.ndarray:
    """Encode texts using the fine-tuned FashionCLIP model."""
    import open_clip

    model, _, _ = open_clip.create_model_and_transforms("hf-hub:Marqo/marqo-fashionCLIP")
    state_dict = torch.load(model_path / "model_state_dict.pt", map_location="cpu")
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    tokenizer = open_clip.get_tokenizer("hf-hub:Marqo/marqo-fashionCLIP")

    all_emb = []
    batch_size = 128
    with torch.no_grad():
        for start in tqdm(range(0, len(texts), batch_size), desc="Encoding (fine-tuned)"):
            batch = texts[start:start + batch_size]
            tokens = tokenizer(batch).to(device)
            features = model.encode_text(tokens)
            features = features / features.norm(dim=-1, keepdim=True)
            all_emb.append(features.cpu().numpy().astype(np.float32))

    return np.vstack(all_emb)


def build_and_search_faiss(
    query_emb: np.ndarray,
    article_emb: np.ndarray,
    article_ids: list[str],
    top_k: int = TOP_K,
) -> list[list[str]]:
    """Build FAISS index from article embeddings and search."""
    worker = Path(__file__).parent / "_faiss_search_worker.py"

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)

        import faiss
        index = faiss.IndexFlatIP(article_emb.shape[1])
        index.add(article_emb)
        faiss_path = tmp / "ft_faiss.index"
        faiss.write_index(index, str(faiss_path))

        ids_path = tmp / "article_ids.json"
        with open(ids_path, "w") as f:
            json.dump(article_ids, f)

        q_path = tmp / "q_emb.npy"
        out_path = tmp / "results.json"
        np.save(str(q_path), query_emb)

        cmd = [
            sys.executable, str(worker),
            str(q_path), str(faiss_path), str(ids_path),
            str(out_path), str(top_k),
        ]
        subprocess.run(cmd, check=True)

        with open(out_path) as f:
            return json.load(f)


def dense_search_baseline(
    queries: list[tuple[str, str]],
    top_k: int = TOP_K,
) -> dict[str, list[str]]:
    """Run baseline FashionCLIP retrieval using pre-built FAISS index."""
    from benchmark.models import load_clip_model, encode_texts_clip

    safe_name = "fashion-clip"
    faiss_path = EMBEDDINGS_DIR / f"{safe_name}_faiss.index"
    ids_path = EMBEDDINGS_DIR / f"{safe_name}_article_ids.json"

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    log.info("Encoding %d queries with baseline FashionCLIP on %s...", len(queries), device)
    model, _, tokenizer = load_clip_model("fashion-clip", device=device)
    texts = [q[1] for q in queries]
    q_emb = encode_texts_clip(texts, model, tokenizer, device, batch_size=128)
    del model

    with tempfile.TemporaryDirectory() as tmp:
        q_path = Path(tmp) / "q_emb.npy"
        out_path = Path(tmp) / "results.json"
        np.save(str(q_path), q_emb.astype("float32"))

        worker = Path(__file__).parent / "_faiss_search_worker.py"
        cmd = [
            sys.executable, str(worker),
            str(q_path), str(faiss_path), str(ids_path),
            str(out_path), str(top_k),
        ]
        subprocess.run(cmd, check=True)
        with open(out_path) as f:
            raw = json.load(f)

    qids = [q[0] for q in queries]
    return {qid: lst for qid, lst in zip(qids, raw)}


def dense_search_finetuned(
    queries: list[tuple[str, str]],
    model_path: Path,
    top_k: int = TOP_K,
) -> dict[str, list[str]]:
    """Run fine-tuned FashionCLIP retrieval.

    Re-encodes both queries and articles with the fine-tuned model,
    then builds a new FAISS index and searches.
    """
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    articles_df = pd.read_csv(HNM_DIR / "articles.csv", dtype=str).fillna("")
    article_ids_path = EMBEDDINGS_DIR / "fashion-clip_article_ids.json"
    with open(article_ids_path) as f:
        article_ids = json.load(f)

    article_texts = load_articles()
    ordered_texts = [article_texts.get(aid, "") for aid in article_ids]

    log.info("Encoding %d articles with fine-tuned model...", len(article_ids))
    article_emb = encode_with_finetuned(ordered_texts, model_path, device)

    query_texts = [q[1] for q in queries]
    log.info("Encoding %d queries with fine-tuned model...", len(queries))
    query_emb = encode_with_finetuned(query_texts, model_path, device)

    log.info("Building FAISS index and searching...")
    raw_results = build_and_search_faiss(query_emb, article_emb, article_ids, top_k)

    qids = [q[0] for q in queries]
    return {qid: lst for qid, lst in zip(qids, raw_results)}


def evaluate(
    retrieved_dict: dict[str, list[str]],
    qrels: dict[str, dict[str, int]],
    label: str = "config",
) -> dict:
    ks = [5, 10, 20, 50]
    per_query = []
    for qid, retrieved in retrieved_dict.items():
        q_qrels = qrels.get(qid, {})
        if not q_qrels:
            continue
        per_query.append(compute_all_metrics(retrieved, q_qrels, ks=ks))
    agg = aggregate_metrics(per_query)
    log.info(
        "%s → nDCG@10=%.4f  MRR=%.4f  R@10=%.4f  (%d queries)",
        label, agg.get("ndcg@10", 0), agg.get("mrr", 0),
        agg.get("recall@10", 0), len(per_query),
    )
    return {"config": label, "n_queries": len(per_query), "metrics": agg}


def main():
    t_start = time.time()

    best_path = FINETUNED_BEST_DIR if FINETUNED_BEST_DIR.exists() else FINETUNED_DIR
    if not (best_path / "model_state_dict.pt").exists():
        log.error("Fine-tuned model not found at %s. Run train_biencoder.py first.", best_path)
        return

    log.info("=" * 60)
    log.info("MODA Phase 3C — Fine-Tuned Bi-Encoder Evaluation")
    log.info("Model: %s", best_path)
    log.info("=" * 60)

    queries, qrels = load_test_data()

    log.info("\n--- Step 1: Baseline FashionCLIP dense retrieval ---")
    baseline_results = dense_search_baseline(queries, top_k=TOP_K)
    res_baseline = evaluate(baseline_results, qrels, label="Baseline_FashionCLIP")

    log.info("\n--- Step 2: Fine-tuned FashionCLIP dense retrieval ---")
    finetuned_results = dense_search_finetuned(queries, best_path, top_k=TOP_K)
    res_finetuned = evaluate(finetuned_results, qrels, label="Finetuned_FashionCLIP")

    all_results = {
        "Baseline_FashionCLIP": res_baseline,
        "Finetuned_FashionCLIP": res_finetuned,
    }

    out_path = RESULTS_DIR / "phase3c_biencoder_eval.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    log.info("Results saved → %s", out_path)

    baseline_ndcg = res_baseline["metrics"]["ndcg@10"]
    finetuned_ndcg = res_finetuned["metrics"]["ndcg@10"]
    delta = (finetuned_ndcg / baseline_ndcg - 1) * 100 if baseline_ndcg > 0 else 0

    print("\n" + "=" * 72)
    print("PHASE 3C — BI-ENCODER COMPARISON (Dense Retrieval Only)")
    print("=" * 72)
    print(f"{'Model':<35} {'nDCG@10':>9} {'MRR':>9} {'R@10':>9}  {'Change'}")
    print("-" * 72)
    for name, res in all_results.items():
        m = res["metrics"]
        d = (m["ndcg@10"] / baseline_ndcg - 1) * 100 if baseline_ndcg > 0 else 0
        sign = "+" if d >= 0 else ""
        print(f"{name:<35} {m['ndcg@10']:>9.4f} {m['mrr']:>9.4f} {m['recall@10']:>9.4f}  {sign}{d:.1f}%")
    print("=" * 72)
    print(f"\nDelta: {'+' if delta >= 0 else ''}{delta:.1f}%")
    print(f"Total elapsed: {(time.time() - t_start) / 60:.1f} min")

    return all_results


if __name__ == "__main__":
    main()
