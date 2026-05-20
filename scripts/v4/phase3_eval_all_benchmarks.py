"""
Phase 3: Full evaluation on all 7 Marqo benchmarks.

Runs comprehensive evaluation using Marqo's official methodology:
- Text-to-Image retrieval
- Category-to-Product retrieval  
- Sub-Category-to-Product retrieval

Evaluates both our fine-tuned model and the baseline FashionSigLIP
for direct comparison.
"""
import os, sys, json, time, argparse, logging
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn.functional as F
from PIL import Image
import open_clip

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJ_ROOT = Path(__file__).resolve().parents[2]
CKPT_DIR = PROJ_ROOT / "checkpoints" / "v4_gcl"
RESULTS_DIR = PROJ_ROOT / "results" / "v4_gcl"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

BENCHMARKS = {
    "deepfashion_inshop": "Marqo/deepfashion-inshop",
    "deepfashion_multimodal": "Marqo/deepfashion-multimodal",
    "fashion200k": "Marqo/fashion200k",
    "KAGL": "Marqo/KAGL",
    "atlas": "Marqo/atlas",
    "polyvore": "Marqo/polyvore",
    "iMaterialist": "Marqo/iMaterialist",
}

TASK_CONFIGS = {
    "text_to_image": {
        "query_cols": ["query", "text", "caption", "description"],
        "doc_col": "image",
        "doc_weights": None,
    },
    "category_to_product": {
        "query_cols": ["category", "masterCategory", "articleType"],
        "doc_col": ["image", "title"],
        "doc_weights": [0.9, 0.1],
    },
    "subcategory_to_product": {
        "query_cols": ["subCategory", "articleType", "fine_category"],
        "doc_col": ["image", "title"],
        "doc_weights": [0.9, 0.1],
    },
}


def load_model_for_eval(checkpoint_path: str | None, device: str):
    """Load model from checkpoint or baseline."""
    model_name = "hf-hub:Marqo/marqo-fashionSigLIP"
    model, _, preprocess = open_clip.create_model_and_transforms(model_name)
    tokenizer = open_clip.get_tokenizer(model_name)
    if checkpoint_path and os.path.exists(checkpoint_path):
        try:
            sd = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        except TypeError:
            sd = torch.load(checkpoint_path, map_location="cpu")
        if isinstance(sd, dict) and "model_state_dict" in sd:
            sd = sd["model_state_dict"]
        model.load_state_dict(sd, strict=False)
        logger.info(f"Loaded checkpoint: {checkpoint_path}")
    else:
        logger.info("Using baseline FashionSigLIP")

    model = model.to(device)
    model.eval()
    return model, preprocess, tokenizer


def evaluate_benchmark(model, tokenizer, preprocess, benchmark_name: str,
                       hf_id: str, device: str, max_queries: int = 2000,
                       max_docs: int = 50000) -> dict:
    """Full evaluation on a single benchmark dataset."""
    from datasets import load_dataset

    logger.info(f"Evaluating {benchmark_name} ({hf_id})...")

    try:
        ds = load_dataset(hf_id, split="data", streaming=True)
    except Exception:
        try:
            ds = load_dataset(hf_id, split="test", streaming=True)
        except Exception as e:
            return {"error": str(e)}

    all_rows = []
    for i, row in enumerate(ds):
        if i >= max_docs:
            break
        row_data = {}
        for k, v in row.items():
            if k != "image":
                row_data[k] = v
            else:
                row_data["_has_image"] = v is not None
                if v is not None:
                    row_data["_image"] = v
        all_rows.append(row_data)

    logger.info(f"  Loaded {len(all_rows)} rows")

    available_cols = set()
    for row in all_rows[:100]:
        available_cols.update(row.keys())

    results = {}

    query_col = None
    for col in ["query", "text", "caption", "description"]:
        if col in available_cols:
            query_col = col
            break

    if query_col:
        logger.info(f"  Text-to-Image: query_col={query_col}")
        r = _run_text_to_image(model, tokenizer, preprocess, all_rows,
                               query_col, device, max_queries)
        results["text_to_image"] = r

    for col in ["category", "masterCategory", "articleType", "subCategory"]:
        if col in available_cols:
            task_name = f"{col}_to_product"
            logger.info(f"  {task_name}: col={col}")
            r = _run_category_to_product(model, tokenizer, preprocess, all_rows,
                                         col, device, max_queries)
            results[task_name] = r

    return results


def _run_text_to_image(model, tokenizer, preprocess, rows, query_col,
                        device, max_queries) -> dict:
    """Text-to-Image retrieval evaluation."""
    gt = defaultdict(set)
    queries = {}
    doc_images = {}

    for i, row in enumerate(rows):
        q = row.get(query_col, "")
        if not q or not isinstance(q, str):
            continue
        doc_id = str(row.get("product_id", row.get("id", i)))

        if q not in queries and len(queries) < max_queries:
            queries[q] = True
        if row.get("_has_image") and doc_id not in doc_images:
            doc_images[doc_id] = row.get("_image")
        if q in queries:
            gt[q].add(doc_id)

    return _compute_retrieval_metrics(model, tokenizer, preprocess,
                                       queries, doc_images, gt, device)


def _run_category_to_product(model, tokenizer, preprocess, rows, cat_col,
                              device, max_queries) -> dict:
    """Category-to-Product retrieval with 0.9*image + 0.1*title scoring."""
    gt = defaultdict(set)
    queries = {}
    doc_data = {}

    for i, row in enumerate(rows):
        cat = row.get(cat_col, "")
        if not cat or not isinstance(cat, str):
            continue
        doc_id = str(row.get("product_id", row.get("id", i)))

        if cat not in queries and len(queries) < max_queries:
            queries[cat] = True
        if row.get("_has_image") and doc_id not in doc_data:
            title = row.get("title", row.get("product_title", ""))
            doc_data[doc_id] = {"image": row.get("_image"), "title": title or ""}
        if cat in queries:
            gt[cat].add(doc_id)

    if not queries or not gt:
        return {"error": "No valid query-doc pairs"}

    query_list = list(queries.keys())
    doc_ids = list(doc_data.keys())

    with torch.no_grad():
        q_embeds = []
        for j in range(0, len(query_list), 64):
            batch = tokenizer(query_list[j:j+64]).to(device)
            emb = model.encode_text(batch)
            q_embeds.append(F.normalize(emb, dim=-1).cpu())
        q_embeds = torch.cat(q_embeds)

        d_embeds = []
        for j in range(0, len(doc_ids), 32):
            batch_ids = doc_ids[j:j+32]
            imgs, titles = [], []
            for did in batch_ids:
                try:
                    img_t = preprocess(doc_data[did]["image"].convert("RGB"))
                    imgs.append(img_t)
                except Exception:
                    imgs.append(torch.zeros(3, 224, 224))
                titles.append(doc_data[did].get("title", ""))

            img_batch = torch.stack(imgs).to(device)
            img_emb = F.normalize(model.encode_image(img_batch), dim=-1)

            title_tokens = tokenizer(titles).to(device)
            title_emb = F.normalize(model.encode_text(title_tokens), dim=-1)

            doc_emb = F.normalize(0.9 * img_emb + 0.1 * title_emb, dim=-1)
            d_embeds.append(doc_emb.cpu())
        d_embeds = torch.cat(d_embeds)

    return _compute_metrics_from_embeds(query_list, doc_ids, q_embeds, d_embeds, gt)


def _compute_retrieval_metrics(model, tokenizer, preprocess,
                                queries, doc_images, gt, device) -> dict:
    """Compute retrieval metrics for text-to-image."""
    query_list = list(queries.keys())
    doc_ids = list(doc_images.keys())

    if not query_list or not doc_ids:
        return {"error": "No data"}

    with torch.no_grad():
        q_embeds = []
        for j in range(0, len(query_list), 64):
            batch = tokenizer(query_list[j:j+64]).to(device)
            emb = model.encode_text(batch)
            q_embeds.append(F.normalize(emb, dim=-1).cpu())
        q_embeds = torch.cat(q_embeds)

        d_embeds = []
        for j in range(0, len(doc_ids), 32):
            batch_ids = doc_ids[j:j+32]
            imgs = []
            for did in batch_ids:
                try:
                    img_t = preprocess(doc_images[did].convert("RGB"))
                    imgs.append(img_t)
                except Exception:
                    imgs.append(torch.zeros(3, 224, 224))
            img_batch = torch.stack(imgs).to(device)
            emb = model.encode_image(img_batch)
            d_embeds.append(F.normalize(emb, dim=-1).cpu())
        d_embeds = torch.cat(d_embeds)

    return _compute_metrics_from_embeds(query_list, doc_ids, q_embeds, d_embeds, gt)


def _compute_metrics_from_embeds(query_list, doc_ids, q_embeds, d_embeds, gt) -> dict:
    """Compute R@1, R@10, MRR, P@1, P@10 from embeddings."""
    sims = q_embeds @ d_embeds.T
    _, top_indices = sims.topk(min(10, len(doc_ids)), dim=1)

    metrics = {"recall@1": 0, "recall@10": 0, "mrr": 0, "precision@1": 0, "precision@10": 0}
    n_queries = 0

    for qi, q_text in enumerate(query_list):
        if q_text not in gt or not gt[q_text]:
            continue
        relevant = gt[q_text]
        retrieved = [doc_ids[idx] for idx in top_indices[qi].tolist()]
        n_queries += 1

        if retrieved[0] in relevant:
            metrics["recall@1"] += 1
            metrics["precision@1"] += 1
        if any(d in relevant for d in retrieved):
            metrics["recall@10"] += 1

        hits_in_10 = sum(1 for d in retrieved if d in relevant)
        metrics["precision@10"] += hits_in_10 / len(retrieved)

        for rank, d in enumerate(retrieved, 1):
            if d in relevant:
                metrics["mrr"] += 1.0 / rank
                break

    if n_queries > 0:
        for k in metrics:
            metrics[k] /= n_queries

    metrics["n_queries"] = n_queries
    metrics["n_docs"] = len(doc_ids)
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to model checkpoint (default: baseline)")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--max-queries", type=int, default=2000)
    parser.add_argument("--max-docs", type=int, default=50000)
    parser.add_argument("--fast", action="store_true",
                        help="Smoke run: max-queries=200, max-docs=1200.")
    parser.add_argument("--medium", action="store_true",
                        help="Between fast and full: max-queries=1200, max-docs=3500.")
    parser.add_argument("--run-name", type=str, default="gcl_v4")
    args = parser.parse_args()

    if args.fast:
        args.max_queries = 200
        args.max_docs = 1200
    elif args.medium:
        args.max_queries = 1200
        args.max_docs = 3500
    if args.device is None:
        if torch.cuda.is_available():
            args.device = "cuda"
        elif torch.backends.mps.is_available():
            args.device = "mps"
        else:
            args.device = "cpu"

    model, preprocess, tokenizer = load_model_for_eval(args.checkpoint, args.device)

    all_results = {}
    for bname, hf_id in BENCHMARKS.items():
        results = evaluate_benchmark(model, tokenizer, preprocess, bname,
                                     hf_id, args.device, args.max_queries, args.max_docs)
        all_results[bname] = results

    run_dir = RESULTS_DIR / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "full_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n{'='*70}")
    print(f"Full Benchmark Results: {args.run_name}")
    print(f"{'='*70}")
    for bname, results in all_results.items():
        print(f"\n  {bname}:")
        for task_name, metrics in results.items():
            if isinstance(metrics, dict) and "mrr" in metrics:
                print(f"    {task_name}: R@1={metrics['recall@1']:.3f} "
                      f"R@10={metrics['recall@10']:.3f} MRR={metrics['mrr']:.3f} "
                      f"(queries={metrics.get('n_queries', '?')})")
            elif isinstance(metrics, dict) and "error" in metrics:
                print(f"    {task_name}: {metrics['error']}")

    logger.info(f"Results saved to {run_dir / 'full_results.json'}")


if __name__ == "__main__":
    main()
