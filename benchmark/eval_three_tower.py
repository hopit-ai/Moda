"""
MODA Phase 4G — Three-Tower Retriever Evaluation

Evaluates the 3-tower model (query/text/image towers) against all
previous pipeline configurations on the same held-out test queries.

Approach:
  1. Encode 105K product texts  → t_emb (text tower)
  2. Encode 105K product images → i_emb (image tower)
  3. Combined: p_emb = α·t_emb + (1-α)·i_emb  (tunable α)
  4. Build single FAISS index on p_emb
  5. Encode test queries with query tower → q_emb
  6. Single FAISS search → candidate list
  7. Optionally: BM25 hybrid + CE reranking on top

Compared configs:
  - 3Tower-only (single-index retrieval)
  - 3Tower + BM25 hybrid
  - 3Tower + BM25 + Off-shelf CE
  - 3Tower + BM25 + LLM-trained CE
  - Text-tower-only (ablation)
  - Image-tower-only (ablation)

Usage:
  python benchmark/eval_three_tower.py
  python benchmark/eval_three_tower.py --alpha 0.6 --n_queries 2000
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from benchmark.eval_full_pipeline import (
    load_benchmark,
    load_articles,
    load_or_compute_ner,
    bm25_ner_search,
    evaluate,
    ce_rerank_batch,
    RESULTS_DIR,
    BM25_WEIGHT,
    DENSE_WEIGHT,
    TOP_K_RERANK,
    TOP_K_FINAL,
    RRF_K,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

SPLIT_PATH = _REPO_ROOT / "data" / "processed" / "query_splits.json"
MODEL_DIR = _REPO_ROOT / "models"
THREE_TOWER_DIR = MODEL_DIR / "moda-3tower" / "best"
IMAGE_DIR = _REPO_ROOT / "data" / "raw" / "hnm_images"
EMBED_DIR = _REPO_ROOT / "data" / "processed" / "embeddings"

OFFSHELF_CE = "cross-encoder/ms-marco-MiniLM-L-6-v2"
LLM_TRAINED_CE = str(MODEL_DIR / "moda-fashion-ce-llm-best")

HNM_DIR = _REPO_ROOT / "data" / "raw" / "hnm_real"

DEFAULT_ALPHA = 0.5


def load_test_query_ids() -> set[str]:
    splits = json.loads(SPLIT_PATH.read_text())
    test_qids = set(splits["test"])
    assert len(test_qids & (set(splits["train"]) | set(splits["val"]))) == 0
    return test_qids


def get_image_path(article_id: str) -> Path | None:
    aid = article_id.zfill(10)
    prefix = aid[:3]
    path = IMAGE_DIR / prefix / f"{aid}.jpg"
    if path.exists():
        return path
    return None


def encode_product_texts(model, tokenizer, article_ids: list[str],
                         article_texts: dict[str, str], device: str,
                         batch_size: int = 128) -> np.ndarray:
    """Encode all product texts using the 3-tower text encoder."""
    model.eval()
    all_emb = []
    texts = [article_texts.get(aid, "") for aid in article_ids]

    with torch.no_grad():
        for start in tqdm(range(0, len(texts), batch_size),
                          desc="Text tower encoding"):
            batch = texts[start:start + batch_size]
            tokens = tokenizer(batch).to(device)
            with torch.autocast(device_type="mps", dtype=torch.float16) if device == "mps" else torch.no_grad():
                emb = model.encode_product_text(tokens)
                emb = F.normalize(emb, dim=-1)
            all_emb.append(emb.cpu().float().numpy())

    return np.vstack(all_emb).astype(np.float32)


def encode_product_images(model, preprocess, article_ids: list[str],
                          device: str, batch_size: int = 64) -> np.ndarray:
    """Encode all product images using the 3-tower image encoder."""
    model.eval()
    all_emb = []
    zero_img = torch.zeros(3, 224, 224)

    for start in tqdm(range(0, len(article_ids), batch_size),
                      desc="Image tower encoding"):
        batch_ids = article_ids[start:start + batch_size]
        imgs = []
        for aid in batch_ids:
            ipath = get_image_path(aid)
            if ipath:
                try:
                    img = preprocess(Image.open(ipath).convert("RGB"))
                except Exception:
                    img = zero_img
            else:
                img = zero_img
            imgs.append(img)

        img_tensor = torch.stack(imgs).to(device)
        with torch.no_grad():
            with torch.autocast(device_type="mps", dtype=torch.float16) if device == "mps" else torch.no_grad():
                emb = model.encode_product_image(img_tensor)
                emb = F.normalize(emb, dim=-1)
        all_emb.append(emb.cpu().float().numpy())

        if device == "mps" and start % (batch_size * 20) == 0:
            torch.mps.empty_cache()

    return np.vstack(all_emb).astype(np.float32)


def encode_queries_3tower(model, tokenizer, queries: list[tuple[str, str]],
                          device: str, batch_size: int = 128) -> np.ndarray:
    """Encode queries using the 3-tower query encoder."""
    model.eval()
    all_emb = []
    texts = [q[1] for q in queries]

    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            tokens = tokenizer(batch).to(device)
            with torch.autocast(device_type="mps", dtype=torch.float16) if device == "mps" else torch.no_grad():
                emb = model.encode_query(tokens)
                emb = F.normalize(emb, dim=-1)
            all_emb.append(emb.cpu().float().numpy())

    return np.vstack(all_emb).astype(np.float32)


def faiss_search_subprocess(q_emb: np.ndarray, p_emb: np.ndarray,
                            article_ids: list[str],
                            top_k: int = TOP_K_RERANK) -> list[list[str]]:
    """Build FAISS index and search in subprocess to avoid BLAS conflicts."""
    import faiss

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)

        index = faiss.IndexFlatIP(p_emb.shape[1])
        index.add(p_emb)
        faiss_path = tmp / "3tower_faiss.index"
        faiss.write_index(index, str(faiss_path))
        del index

        ids_path = tmp / "article_ids.json"
        with open(ids_path, "w") as f:
            json.dump(article_ids, f)

        q_path = tmp / "q_emb.npy"
        out_path = tmp / "results.json"
        np.save(str(q_path), q_emb.astype("float32"))

        worker = Path(__file__).parent / "_faiss_search_worker.py"
        cmd = [
            sys.executable, str(worker),
            str(q_path), str(faiss_path), str(ids_path),
            str(out_path), str(top_k),
        ]
        log.info("Running FAISS search subprocess (top_k=%d)...", top_k)
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            log.error("FAISS worker failed:\nstdout: %s\nstderr: %s",
                      result.stdout[-1000:], result.stderr[-1000:])
            raise RuntimeError("FAISS search failed")

        with open(out_path) as f:
            return json.load(f)


def rrf_fusion_2way(
    list_a: list[list[str]],
    list_b: list[list[str]],
    k: int = RRF_K,
    w_a: float = BM25_WEIGHT,
    w_b: float = DENSE_WEIGHT,
    top_k: int = TOP_K_RERANK,
) -> list[list[str]]:
    fused = []
    for a, b in zip(list_a, list_b):
        scores: dict[str, float] = {}
        for rank, doc_id in enumerate(a, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + w_a / (k + rank)
        for rank, doc_id in enumerate(b, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + w_b / (k + rank)
        fused.append([d for d, _ in sorted(scores.items(), key=lambda x: -x[1])[:top_k]])
    return fused


def load_article_ids_and_texts() -> tuple[list[str], dict[str, str]]:
    """Load article IDs and build article text dict."""
    import pandas as pd
    articles_df = pd.read_csv(HNM_DIR / "articles.csv", dtype=str).fillna("")
    article_ids = []
    article_texts: dict[str, str] = {}
    for _, row in articles_df.iterrows():
        aid = str(row.get("article_id", "")).strip()
        if not aid:
            continue
        article_ids.append(aid)
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
        article_texts[aid] = " | ".join(parts)
    return article_ids, article_texts


def main():
    import open_clip
    from opensearchpy import OpenSearch
    from benchmark.train_three_tower import ThreeTowerModel, load_three_tower

    p = argparse.ArgumentParser(
        description="MODA Phase 4G — Three-Tower Retriever Evaluation")
    p.add_argument("--n_queries", type=int, default=0)
    p.add_argument("--alpha", type=float, default=DEFAULT_ALPHA,
                   help="Weight for text embeddings: p = α·text + (1-α)·image")
    p.add_argument("--model_path", type=str, default=str(THREE_TOWER_DIR))
    args = p.parse_args()

    model_path = Path(args.model_path)
    if not (model_path / "three_tower_state.pt").exists():
        log.error("3-tower model not found at %s. Run train_three_tower.py first.",
                  model_path)
        return

    t_start = time.time()

    log.info("=" * 80)
    log.info("MODA Phase 4G — Three-Tower Retriever Evaluation")
    log.info("  Model: %s", model_path)
    log.info("  Alpha (text weight): %.2f", args.alpha)
    log.info("=" * 80)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    log.info("Device: %s", device)

    # ── Load test data ──
    test_qids = load_test_query_ids()
    all_queries, all_qrels = load_benchmark(n_queries=None)
    queries = [(qid, qt) for qid, qt in all_queries if qid in test_qids]
    qrels = {qid: v for qid, v in all_qrels.items() if qid in test_qids}
    log.info("Test queries: %d", len(queries))

    if args.n_queries > 0 and len(queries) > args.n_queries:
        import random
        queries = random.Random(42).sample(queries, args.n_queries)

    qids = [q[0] for q in queries]
    texts = [q[1] for q in queries]

    # ── Load 3-tower model ──
    log.info("Loading 3-tower model...")
    model = load_three_tower(model_path, device=device)
    _, preprocess, _ = open_clip.create_model_and_transforms(
        "hf-hub:Marqo/marqo-fashionCLIP")
    tokenizer = open_clip.get_tokenizer("hf-hub:Marqo/marqo-fashionCLIP")

    # ── Encode products (offline, done once) ──
    article_ids, article_texts = load_article_ids_and_texts()
    log.info("Encoding %d products with text tower...", len(article_ids))
    t_emb = encode_product_texts(model, tokenizer, article_ids, article_texts, device)
    log.info("Text embeddings: %s", t_emb.shape)

    if device == "mps":
        torch.mps.empty_cache()

    log.info("Encoding %d products with image tower...", len(article_ids))
    i_emb = encode_product_images(model, preprocess, article_ids, device)
    log.info("Image embeddings: %s", i_emb.shape)

    if device == "mps":
        torch.mps.empty_cache()

    # ── Combined product embedding ──
    alpha = args.alpha
    p_emb_combined = alpha * t_emb + (1 - alpha) * i_emb
    norms = np.linalg.norm(p_emb_combined, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    p_emb_combined = (p_emb_combined / norms).astype(np.float32)

    log.info("Combined embeddings: α=%.2f, shape=%s", alpha, p_emb_combined.shape)

    # ── Encode queries (online) ──
    log.info("Encoding %d queries with query tower...", len(queries))
    q_emb = encode_queries_3tower(model, tokenizer, queries, device)

    del model
    if device == "mps":
        torch.mps.empty_cache()

    # ── FAISS searches ──
    all_results = {}

    # A) Text-tower-only (ablation)
    log.info("Searching: text-tower only...")
    t_emb_norm = t_emb / np.maximum(np.linalg.norm(t_emb, axis=1, keepdims=True), 1e-8)
    text_only_lists = faiss_search_subprocess(q_emb, t_emb_norm.astype(np.float32),
                                              article_ids, top_k=TOP_K_RERANK)
    text_only_dict = {qid: lst for qid, lst in zip(qids, text_only_lists)}
    all_results["3T_TextOnly"] = evaluate(text_only_dict, qrels, label="3T_TextOnly")

    # B) Image-tower-only (ablation)
    log.info("Searching: image-tower only...")
    i_emb_norm = i_emb / np.maximum(np.linalg.norm(i_emb, axis=1, keepdims=True), 1e-8)
    img_only_lists = faiss_search_subprocess(q_emb, i_emb_norm.astype(np.float32),
                                             article_ids, top_k=TOP_K_RERANK)
    img_only_dict = {qid: lst for qid, lst in zip(qids, img_only_lists)}
    all_results["3T_ImageOnly"] = evaluate(img_only_dict, qrels, label="3T_ImageOnly")

    # C) Combined (main config)
    log.info("Searching: combined (α=%.2f)...", alpha)
    combined_lists = faiss_search_subprocess(q_emb, p_emb_combined,
                                             article_ids, top_k=TOP_K_RERANK)
    combined_dict = {qid: lst for qid, lst in zip(qids, combined_lists)}
    all_results["3T_Combined"] = evaluate(combined_dict, qrels, label="3T_Combined")

    # ── BM25 hybrid ──
    articles_for_ce = load_articles()
    ner_cache = load_or_compute_ner(queries)
    client = OpenSearch(hosts=[{"host": "localhost", "port": 9200}],
                        http_compress=True, timeout=30)
    bm25_results: dict[str, list[str]] = {}
    for qid, text in tqdm(queries, desc="BM25+NER"):
        bm25_results[qid] = bm25_ner_search(
            client, text, ner_cache.get(qid, {}), top_k=TOP_K_RERANK)
    bm25_lists = [bm25_results.get(qid, []) for qid in qids]

    # D) 3Tower + BM25 hybrid
    log.info("3Tower + BM25 hybrid...")
    hybrid_lists = rrf_fusion_2way(bm25_lists, combined_lists,
                                    w_a=BM25_WEIGHT, w_b=DENSE_WEIGHT)
    hybrid_dict = {qid: lst for qid, lst in zip(qids, hybrid_lists)}
    all_results["3T_BM25Hybrid"] = evaluate(hybrid_dict, qrels, label="3T_BM25Hybrid")

    # E) 3Tower + BM25 + Off-shelf CE
    log.info("3Tower + BM25 + Off-shelf CE reranking...")
    ce_offshelf_lists = ce_rerank_batch(texts, hybrid_lists, articles_for_ce,
                                         model_name=OFFSHELF_CE)
    ce_offshelf_dict = {qid: lst for qid, lst in zip(qids, ce_offshelf_lists)}
    all_results["3T_BM25_OffshelfCE"] = evaluate(
        ce_offshelf_dict, qrels, label="3T_BM25_OffshelfCE")

    # F) 3Tower + BM25 + LLM-trained CE
    if Path(LLM_TRAINED_CE).exists():
        log.info("3Tower + BM25 + LLM-trained CE reranking...")
        ce_llm_lists = ce_rerank_batch(texts, hybrid_lists, articles_for_ce,
                                        model_name=LLM_TRAINED_CE)
        ce_llm_dict = {qid: lst for qid, lst in zip(qids, ce_llm_lists)}
        all_results["3T_BM25_LLMtrainedCE"] = evaluate(
            ce_llm_dict, qrels, label="3T_BM25_LLMtrainedCE")

    # ── Save results ──
    elapsed = time.time() - t_start
    output = {
        "configs": all_results,
        "settings": {
            "n_queries": len(queries),
            "alpha": alpha,
            "model_path": str(model_path),
            "split": "test_only",
            "elapsed_min": round(elapsed / 60, 1),
        },
    }
    out_path = RESULTS_DIR / "phase4g_three_tower_eval.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    log.info("Results saved → %s", out_path)

    # ── Print results table ──
    print(f"\n{'=' * 90}")
    print("MODA Phase 4G — Three-Tower Retriever Evaluation")
    print(f"  α = {alpha:.2f} (text weight) | {len(queries):,} test queries")
    print(f"{'=' * 90}")
    print(f"  {'Config':<35} {'nDCG@10':>9} {'MRR':>9} {'R@10':>9}")
    print(f"  {'-' * 75}")
    for name, res in all_results.items():
        m = res["metrics"]
        print(f"  {name:<35} {m['ndcg@10']:>9.4f} {m['mrr']:>9.4f} {m['recall@10']:>9.4f}")
    print(f"  {'-' * 75}")

    best_key = max(all_results, key=lambda k: all_results[k]["metrics"]["ndcg@10"])
    best_ndcg = all_results[best_key]["metrics"]["ndcg@10"]
    print(f"\n  Best: {best_key} → nDCG@10 = {best_ndcg:.4f}")

    phase3_best = 0.0757
    delta = (best_ndcg / phase3_best - 1) * 100
    print(f"  vs Phase 3 best (0.0757): {delta:+.1f}%")
    print(f"\n  Total elapsed: {elapsed / 60:.1f} min")
    print(f"{'=' * 90}\n")


if __name__ == "__main__":
    main()
