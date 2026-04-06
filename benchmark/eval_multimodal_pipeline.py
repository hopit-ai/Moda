"""
MODA Phase 4D — Zero-Shot Multimodal Pipeline Evaluation

Evaluates adding image retrieval as a third channel in the hybrid pipeline.
Compares 2-way (BM25+Dense-Text) vs 3-way (BM25+Dense-Text+Dense-Image)
hybrid fusion, with and without CE reranking.

All configurations are evaluated on the same 22,855 held-out test queries
for apples-to-apples comparison with Phase 3 results.

Usage:
  python benchmark/eval_multimodal_pipeline.py
  python benchmark/eval_multimodal_pipeline.py --n_queries 2000  # quick test
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
from opensearchpy import OpenSearch
from tqdm import tqdm

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from benchmark.eval_full_pipeline import (
    load_benchmark,
    load_articles,
    load_or_compute_ner,
    bm25_ner_search,
    dense_search_batch,
    evaluate,
    ce_rerank_batch,
    RESULTS_DIR,
    DENSE_MODEL,
    BM25_WEIGHT,
    DENSE_WEIGHT,
    TOP_K_RERANK,
    TOP_K_FINAL,
    RRF_K,
)
from benchmark.eval_finetuned_biencoder import dense_search_finetuned

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

SPLIT_PATH = _REPO_ROOT / "data" / "processed" / "query_splits.json"
MODEL_DIR = _REPO_ROOT / "models"
FINETUNED_BIENC = MODEL_DIR / "moda-fashionclip-finetuned" / "best"
OFFSHELF_CE = "cross-encoder/ms-marco-MiniLM-L-6-v2"
LLM_TRAINED_CE = str(MODEL_DIR / "moda-fashion-ce-llm-best")

EMBED_DIR = _REPO_ROOT / "data" / "processed" / "embeddings"
IMAGE_FAISS_PATH = EMBED_DIR / "fashion-clip-visual_faiss.index"
IMAGE_IDS_PATH = EMBED_DIR / "fashion-clip-visual_article_ids.json"

IMAGE_WEIGHT = 0.3


def load_test_query_ids() -> set[str]:
    splits = json.loads(SPLIT_PATH.read_text())
    test_qids = set(splits["test"])
    assert len(test_qids & (set(splits["train"]) | set(splits["val"]))) == 0
    return test_qids


def text_to_image_search(
    queries: list[tuple[str, str]],
    top_k: int = TOP_K_RERANK,
    use_finetuned: bool = False,
    finetuned_path: Path | None = None,
) -> dict[str, list[str]]:
    """Search the image FAISS index using text query embeddings.

    Uses FashionCLIP's text encoder to encode queries, then searches
    against image embeddings in FAISS via subprocess isolation
    (avoids PyTorch+FAISS BLAS library segfault on MPS).
    """
    import subprocess
    import tempfile
    import torch

    from benchmark.models import load_clip_model, encode_texts_clip

    qids = [q[0] for q in queries]
    texts = [q[1] for q in queries]
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    if use_finetuned and finetuned_path and (finetuned_path / "model_state_dict.pt").exists():
        log.info("Encoding queries with FINE-TUNED text encoder...")
        model, _, tokenizer = load_clip_model(DENSE_MODEL, device=device)
        state_dict = torch.load(finetuned_path / "model_state_dict.pt", map_location="cpu")
        model.load_state_dict(state_dict, strict=False)
        model = model.to(device).eval()
        q_embs = encode_texts_clip(texts, model, tokenizer, device, batch_size=128)
    else:
        log.info("Encoding queries with BASELINE text encoder...")
        model, _, tokenizer = load_clip_model(DENSE_MODEL, device=device)
        model = model.to(device)
        q_embs = encode_texts_clip(texts, model, tokenizer, device, batch_size=128)

    del model
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    with tempfile.TemporaryDirectory() as tmp:
        q_path = Path(tmp) / "queries.npy"
        out_path = Path(tmp) / "results.json"
        np.save(str(q_path), q_embs.astype("float32"))

        worker = Path(__file__).parent / "_faiss_search_worker.py"
        cmd = [
            sys.executable, str(worker),
            str(q_path), str(IMAGE_FAISS_PATH), str(IMAGE_IDS_PATH),
            str(out_path), str(top_k),
        ]
        log.info("Running FAISS image search subprocess (top_k=%d)...", top_k)
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            log.error("FAISS image worker failed:\nstdout: %s\nstderr: %s",
                      result.stdout[-1000:], result.stderr[-1000:])
            raise RuntimeError("FAISS image search failed")

        with open(out_path) as f:
            raw_results = json.load(f)

    return {qid: lst for qid, lst in zip(qids, raw_results)}


def rrf_fusion_3way(
    bm25_lists: list[list[str]],
    dense_text_lists: list[list[str]],
    dense_image_lists: list[list[str]],
    k: int = RRF_K,
    bm25_w: float = BM25_WEIGHT,
    text_w: float = DENSE_WEIGHT,
    image_w: float = IMAGE_WEIGHT,
    top_k: int = TOP_K_RERANK,
) -> list[list[str]]:
    """Three-way RRF fusion: BM25 + Dense-Text + Dense-Image."""
    fused = []
    for bm25, d_text, d_img in zip(bm25_lists, dense_text_lists, dense_image_lists):
        scores: dict[str, float] = {}
        for rank, doc_id in enumerate(bm25, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + bm25_w / (k + rank)
        for rank, doc_id in enumerate(d_text, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + text_w / (k + rank)
        for rank, doc_id in enumerate(d_img, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + image_w / (k + rank)
        fused.append([d for d, _ in sorted(scores.items(), key=lambda x: -x[1])[:top_k]])
    return fused


def rrf_fusion_2way(
    list_a: list[list[str]],
    list_b: list[list[str]],
    k: int = RRF_K,
    w_a: float = 0.4,
    w_b: float = 0.6,
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


def main():
    p = argparse.ArgumentParser(
        description="MODA Phase 4D — Zero-shot multimodal pipeline evaluation")
    p.add_argument("--n_queries", type=int, default=0)
    p.add_argument("--image_weight", type=float, default=IMAGE_WEIGHT)
    p.add_argument("--use_finetuned_text", action="store_true",
                   help="Use fine-tuned FashionCLIP text encoder (Phase 3C)")
    args = p.parse_args()

    if not IMAGE_FAISS_PATH.exists():
        log.error("Image FAISS index not found at %s. Run embed_hnm_images.py first.", IMAGE_FAISS_PATH)
        return

    t_start = time.time()
    log.info("=" * 80)
    log.info("MODA — Multimodal Pipeline Evaluation (Phase 4D)")
    log.info("Image weight: %.2f", args.image_weight)
    log.info("=" * 80)

    test_qids = load_test_query_ids()
    all_queries, all_qrels = load_benchmark(n_queries=None)
    queries = [(qid, qt) for qid, qt in all_queries if qid in test_qids]
    qrels = {qid: v for qid, v in all_qrels.items() if qid in test_qids}
    log.info("Test queries: %d", len(queries))

    if args.n_queries > 0 and len(queries) > args.n_queries:
        import random
        queries = random.Random(42).sample(queries, args.n_queries)

    articles = load_articles()
    qids = [q[0] for q in queries]
    texts = [q[1] for q in queries]

    # ── Shared BM25 retrieval ──
    ner_cache = load_or_compute_ner(queries)
    client = OpenSearch(hosts=[{"host": "localhost", "port": 9200}],
                        http_compress=True, timeout=30)
    bm25_results: dict[str, list[str]] = {}
    for qid, text in tqdm(queries, desc="BM25+NER"):
        bm25_results[qid] = bm25_ner_search(
            client, text, ner_cache.get(qid, {}), top_k=TOP_K_RERANK)
    bm25_lists = [bm25_results.get(qid, []) for qid in qids]

    # ── Dense-Text retrieval (use finetuned or baseline) ──
    if args.use_finetuned_text:
        log.info("Dense-text with FINE-TUNED FashionCLIP...")
        dense_text = dense_search_finetuned(queries, FINETUNED_BIENC, top_k=TOP_K_RERANK)
    else:
        log.info("Dense-text with BASELINE FashionCLIP...")
        dense_text = dense_search_batch(queries, model_name=DENSE_MODEL, top_k=TOP_K_RERANK)
    dense_text_lists = [dense_text.get(qid, []) for qid in qids]

    # ── Dense-Image retrieval (text-to-image via CLIP) ──
    log.info("Text-to-image retrieval via FashionCLIP...")
    dense_image = text_to_image_search(
        queries, top_k=TOP_K_RERANK,
        use_finetuned=args.use_finetuned_text,
        finetuned_path=FINETUNED_BIENC if args.use_finetuned_text else None,
    )
    dense_image_lists = [dense_image.get(qid, []) for qid in qids]

    # ── Image-only baseline ──
    log.info("Evaluating image-only retrieval...")
    image_only = {qid: lst for qid, lst in zip(qids, dense_image_lists)}

    all_results = {}
    all_results["ImageOnly"] = evaluate(image_only, qrels, label="ImageOnly")

    # ── 2-way hybrid: BM25 + Dense-Text (Phase 3 baseline) ──
    hybrid_2way_lists = rrf_fusion_2way(
        bm25_lists, dense_text_lists,
        w_a=BM25_WEIGHT, w_b=DENSE_WEIGHT)
    hybrid_2way = {qid: lst for qid, lst in zip(qids, hybrid_2way_lists)}
    all_results["Hybrid2Way_BM25_Text"] = evaluate(
        hybrid_2way, qrels, label="Hybrid2Way_BM25_Text")

    # ── 2-way: BM25 + Dense-Image ──
    hybrid_bm25img_lists = rrf_fusion_2way(
        bm25_lists, dense_image_lists,
        w_a=BM25_WEIGHT, w_b=args.image_weight)
    hybrid_bm25img = {qid: lst for qid, lst in zip(qids, hybrid_bm25img_lists)}
    all_results["Hybrid2Way_BM25_Image"] = evaluate(
        hybrid_bm25img, qrels, label="Hybrid2Way_BM25_Image")

    # ── 2-way: Dense-Text + Dense-Image ──
    hybrid_textimg_lists = rrf_fusion_2way(
        dense_text_lists, dense_image_lists,
        w_a=DENSE_WEIGHT, w_b=args.image_weight)
    hybrid_textimg = {qid: lst for qid, lst in zip(qids, hybrid_textimg_lists)}
    all_results["Hybrid2Way_Text_Image"] = evaluate(
        hybrid_textimg, qrels, label="Hybrid2Way_Text_Image")

    # ── 3-way hybrid: BM25 + Dense-Text + Dense-Image ──
    hybrid_3way_lists = rrf_fusion_3way(
        bm25_lists, dense_text_lists, dense_image_lists,
        bm25_w=BM25_WEIGHT, text_w=DENSE_WEIGHT,
        image_w=args.image_weight)
    hybrid_3way = {qid: lst for qid, lst in zip(qids, hybrid_3way_lists)}
    all_results["Hybrid3Way_BM25_Text_Image"] = evaluate(
        hybrid_3way, qrels, label="Hybrid3Way_BM25_Text_Image")

    # ── 3-way + Off-shelf CE reranking ──
    log.info("3-way hybrid + Off-shelf CE reranking...")
    ce_3way_lists = ce_rerank_batch(
        texts, hybrid_3way_lists, articles, model_name=OFFSHELF_CE)
    ce_3way = {qid: lst for qid, lst in zip(qids, ce_3way_lists)}
    all_results["Hybrid3Way_OffshelfCE"] = evaluate(
        ce_3way, qrels, label="Hybrid3Way_OffshelfCE")

    # ── 3-way + LLM-trained CE reranking ──
    if Path(LLM_TRAINED_CE).exists():
        log.info("3-way hybrid + LLM-trained CE reranking...")
        ce_3way_llm_lists = ce_rerank_batch(
            texts, hybrid_3way_lists, articles, model_name=LLM_TRAINED_CE)
        ce_3way_llm = {qid: lst for qid, lst in zip(qids, ce_3way_llm_lists)}
        all_results["Hybrid3Way_LLMtrainedCE"] = evaluate(
            ce_3way_llm, qrels, label="Hybrid3Way_LLMtrainedCE")

    # ── 2-way + Off-shelf CE (Phase 3 reference) ──
    log.info("2-way hybrid + Off-shelf CE reranking (reference)...")
    ce_2way_lists = ce_rerank_batch(
        texts, hybrid_2way_lists, articles, model_name=OFFSHELF_CE)
    ce_2way = {qid: lst for qid, lst in zip(qids, ce_2way_lists)}
    all_results["Hybrid2Way_OffshelfCE"] = evaluate(
        ce_2way, qrels, label="Hybrid2Way_OffshelfCE")

    # ── Save results ──
    elapsed = time.time() - t_start
    encoder_type = "finetuned" if args.use_finetuned_text else "baseline"
    output = {
        "configs": all_results,
        "settings": {
            "n_queries": len(queries),
            "text_encoder": encoder_type,
            "image_weight": args.image_weight,
            "split": "test_only",
            "elapsed_min": round(elapsed / 60, 1),
        },
    }
    out_path = RESULTS_DIR / f"phase4d_multimodal_eval_{encoder_type}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    log.info("Results saved → %s", out_path)

    # ── Print results table ──
    print(f"\n{'=' * 90}")
    print(f"MODA — Multimodal Pipeline Evaluation (Phase 4D)")
    print(f"  Text encoder: {encoder_type} | Image weight: {args.image_weight}")
    print(f"  {len(queries):,} test queries")
    print(f"{'=' * 90}")
    print(f"{'Config':<40} {'nDCG@10':>9} {'MRR':>9} {'R@10':>9}")
    print(f"{'-' * 90}")
    for name, res in all_results.items():
        m = res["metrics"]
        print(f"  {name:<38} {m['ndcg@10']:>9.4f} {m['mrr']:>9.4f} {m['recall@10']:>9.4f}")
    print(f"{'-' * 90}")

    # Highlight: 2-way vs 3-way comparison
    ndcg_2way = all_results.get("Hybrid2Way_BM25_Text", {}).get("metrics", {}).get("ndcg@10", 0)
    ndcg_3way = all_results.get("Hybrid3Way_BM25_Text_Image", {}).get("metrics", {}).get("ndcg@10", 0)
    if ndcg_2way > 0:
        delta = (ndcg_3way / ndcg_2way - 1) * 100
        print(f"\n  3-way vs 2-way hybrid (no CE): {delta:+.1f}% nDCG@10")

    ndcg_2way_ce = all_results.get("Hybrid2Way_OffshelfCE", {}).get("metrics", {}).get("ndcg@10", 0)
    ndcg_3way_ce = all_results.get("Hybrid3Way_OffshelfCE", {}).get("metrics", {}).get("ndcg@10", 0)
    if ndcg_2way_ce > 0:
        delta_ce = (ndcg_3way_ce / ndcg_2way_ce - 1) * 100
        print(f"  3-way vs 2-way hybrid (+ CE):  {delta_ce:+.1f}% nDCG@10")

    print(f"\n  Total elapsed: {elapsed / 60:.1f} min")
    print(f"{'=' * 90}\n")


if __name__ == "__main__":
    main()
