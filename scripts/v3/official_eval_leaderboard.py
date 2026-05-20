"""Official Leaderboard Evaluation — matches Marqo's exact methodology.

Uses:
  - Full corpus (no subsampling)
  - Marqo's pre-defined ground truth files (2000 test queries per task)
  - beir metrics: Recall@1, Recall@10, MRR, Precision@1, Precision@10, NDCG, MAP
  - Same doc_col fusion weights (0.9*image + 0.1*text for category tasks)
  - All tasks per dataset as defined in their configs

Evaluates on 5 CLEAN benchmarks (no training data overlap):
  Atlas, Fashion200k, KAGL, Polyvore, iMaterialist

Usage:
  python3 -u scripts/v3/official_eval_leaderboard.py --model-source fsl
  python3 -u scripts/v3/official_eval_leaderboard.py --model-source fsl --checkpoint path/to/best.pt
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("official-eval")

REPO_ROOT = Path(__file__).resolve().parents[2]
MARQO_REPO = Path("/tmp/marqo-FashionCLIP")
RESULTS_DIR = REPO_ROOT / "results"

CLEAN_BENCHMARKS = {
    "Atlas": {
        "hf_dataset": "Marqo/atlas",
        "config": MARQO_REPO / "configs" / "atlas.json",
        "gt_dir": MARQO_REPO / "data" / "Atlas" / "gt_query_doc",
    },
    "Fashion200k": {
        "hf_dataset": "Marqo/fashion200k",
        "config": MARQO_REPO / "configs" / "fashion200k.json",
        "gt_dir": MARQO_REPO / "data" / "Fashion200k" / "gt_query_doc",
    },
    "KAGL": {
        "hf_dataset": "Marqo/KAGL",
        "config": MARQO_REPO / "configs" / "KAGL.json",
        "gt_dir": MARQO_REPO / "data" / "KAGL" / "gt_query_doc",
    },
    "Polyvore": {
        "hf_dataset": "Marqo/polyvore",
        "config": MARQO_REPO / "configs" / "polyvore.json",
        "gt_dir": MARQO_REPO / "data" / "Polyvore" / "gt_query_doc",
    },
    "iMaterialist": {
        "hf_dataset": "Marqo/iMaterialist",
        "config": MARQO_REPO / "configs" / "iMaterialist.json",
        "gt_dir": MARQO_REPO / "data" / "iMaterialist" / "gt_query_doc",
    },
}


def load_model(model_source: str, checkpoint_path: str | None, device: torch.device):
    import open_clip

    if model_source == "fsl":
        log.info("Loading Marqo-FashionSigLIP...")
        model, _, preprocess_val = open_clip.create_model_and_transforms(
            "hf-hub:Marqo/marqo-fashionSigLIP"
        )
        tokenizer = open_clip.get_tokenizer("hf-hub:Marqo/marqo-fashionSigLIP")
    else:
        log.info("Loading ViT-B-16-SigLIP/webli...")
        model, _, preprocess_val = open_clip.create_model_and_transforms(
            "ViT-B-16-SigLIP", pretrained="webli"
        )
        tokenizer = open_clip.get_tokenizer("ViT-B-16-SigLIP")

    model = model.to(device)

    if checkpoint_path and Path(checkpoint_path).exists():
        log.info("Loading checkpoint: %s", checkpoint_path)
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state_dict = ckpt.get("model_state_dict", ckpt)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        log.info("  Loaded %d keys (%d missing, %d unexpected)",
                 len(state_dict) - len(unexpected), len(missing), len(unexpected))
    elif checkpoint_path:
        log.warning("Checkpoint not found: %s", checkpoint_path)

    model.eval()
    return model, tokenizer, preprocess_val


def get_embeddings(model, tokenizer, preprocess, ds, device, batch_size=128):
    """Compute embeddings for all columns needed. Returns dict of {col_name: tensor}."""
    import io
    from torch.utils.data import DataLoader, Dataset

    text_cols = [c for c in ds.column_names if c not in ("image", "item_ID")]
    item_ids = [str(x) for x in ds["item_ID"]]

    class ImgDataset(Dataset):
        def __init__(self, hf_ds, preprocess_fn):
            self.ds = hf_ds
            self.preprocess_fn = preprocess_fn

        def __len__(self):
            return len(self.ds)

        def __getitem__(self, idx):
            item = self.ds[idx]
            img = item["image"]
            if not isinstance(img, Image.Image):
                img = Image.open(io.BytesIO(img)).convert("RGB")
            else:
                img = img.convert("RGB")
            return self.preprocess_fn(img)

    # Encode images
    log.info("  Encoding %d images...", len(ds))
    loader = DataLoader(ImgDataset(ds, preprocess), batch_size=batch_size,
                        num_workers=0, shuffle=False)
    img_embs = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="  images", leave=False):
            batch = batch.to(device)
            emb = model.encode_image(batch)
            emb = F.normalize(emb, dim=-1)
            img_embs.append(emb.cpu())
            if device.type == "mps":
                torch.mps.empty_cache()
    embeddings = {"image": torch.cat(img_embs, dim=0)}

    # Encode text columns
    for col in text_cols:
        log.info("  Encoding text column '%s'...", col)
        texts = [str(x) if x else "" for x in ds[col]]
        col_embs = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                tok = tokenizer(batch_texts).to(device)
                emb = model.encode_text(tok)
                emb = F.normalize(emb, dim=-1)
                col_embs.append(emb.cpu())
                if device.type == "mps":
                    torch.mps.empty_cache()
        embeddings[col] = torch.cat(col_embs, dim=0)

    return embeddings, item_ids


def run_retrieval(test_queries, item_ids, doc_embeddings, tokenizer, model, k, device):
    """Run retrieval for a list of query strings against doc embeddings."""
    results = {}
    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

    for query in tqdm(test_queries, desc="  retrieval", leave=False):
        tok = tokenizer([query]).to(device)
        with torch.no_grad():
            text_features = model.encode_text(tok)
            text_features = F.normalize(text_features, dim=-1)

        similarity = cos(text_features.cpu(), doc_embeddings)
        top_scores, top_inds = torch.topk(similarity, k)

        results[query] = {
            str(item_ids[idx]): float(score)
            for idx, score in zip(top_inds.tolist(), top_scores.tolist())
        }

    return results


def mrr(qrels, results):
    """MRR as computed by Marqo."""
    MRR = 0.0
    top_hits = {}
    for query_id, doc_scores in results.items():
        top_hits[query_id] = sorted(doc_scores.items(), key=lambda item: item[1], reverse=True)

    for query_id in top_hits:
        query_relevant_docs = set(
            doc_id for doc_id in qrels[query_id] if qrels[query_id][doc_id] > 0
        )
        for rank, hit in enumerate(top_hits[query_id]):
            if hit[0] in query_relevant_docs:
                MRR += 1.0 / (rank + 1)
                break

    return round(MRR / len(qrels), 5)


def evaluate_task(model, tokenizer, preprocess, embeddings, item_ids, task, gt_dir, device):
    """Evaluate a single task using Marqo's exact methodology."""
    from beir.retrieval.evaluation import EvaluateRetrieval

    query_col = task["query_col"][0]
    doc_cols = task["doc_col"]
    doc_weights = task.get("doc_weights", None)
    task_name = task["name"]

    # Build doc embeddings
    if len(doc_cols) == 1:
        doc_embs = embeddings[doc_cols[0]]
    else:
        weighted = []
        for col, w in zip(doc_cols, doc_weights):
            weighted.append(w * embeddings[col])
        doc_embs = F.normalize(torch.stack(weighted, dim=1).sum(1), dim=-1)

    # Build ground truth file name (matching Marqo's convention)
    doc_col_str = "+".join(doc_cols)
    gt_filename = f"ground_truth_{query_col}-{doc_col_str}.json"
    gt_path = gt_dir / gt_filename

    if not gt_path.exists():
        log.warning("  GT not found: %s — skipping", gt_path)
        return None

    with open(gt_path) as f:
        gt_results = json.load(f)

    test_queries = list(gt_results.keys())
    log.info("  Task '%s' [%s -> %s]: %d queries", task_name, query_col, doc_col_str, len(test_queries))

    # Run retrieval
    k = 10
    retrieval_results = run_retrieval(test_queries, item_ids, doc_embs, tokenizer, model, k, device)

    # Evaluate with beir
    evaluator = EvaluateRetrieval()
    Ks = [1, 10]
    ndcg, _map, recall, precision = evaluator.evaluate(gt_results, retrieval_results, Ks)

    MRR = mrr(gt_results, retrieval_results)

    task_results = {
        "task": task_name,
        "query_col": query_col,
        "doc_cols": doc_cols,
        "n_queries": len(test_queries),
        "recall": recall,
        "precision": precision,
        "ndcg": ndcg,
        "map": _map,
        "MRR": MRR,
    }

    # Log key metrics
    if "text" in task_name.lower() and "image" in task_name.lower():
        r1 = recall.get("Recall@1", recall.get("recall@1", 0))
        r10 = recall.get("Recall@10", recall.get("recall@10", 0))
        avg_recall = (r1 + r10) / 2
        log.info("    AvgRecall=%.3f  R@1=%.3f  R@10=%.3f  MRR=%.3f",
                 avg_recall, r1, r10, MRR)
    else:
        p1 = precision.get("P@1", precision.get("p@1", 0))
        p10 = precision.get("P@10", precision.get("p@10", 0))
        avg_p = (p1 + p10) / 2
        log.info("    AvgP=%.3f  P@1=%.3f  P@10=%.3f  MRR=%.3f",
                 avg_p, p1, p10, MRR)

    return task_results


def evaluate_benchmark(model, tokenizer, preprocess, benchmark_name, benchmark_info, device):
    """Evaluate all tasks for a single benchmark."""
    from datasets import load_dataset

    log.info("=" * 60)
    log.info("Benchmark: %s", benchmark_name)
    log.info("=" * 60)

    hf_name = benchmark_info["hf_dataset"]
    gt_dir = benchmark_info["gt_dir"]

    with open(benchmark_info["config"]) as f:
        config = json.load(f)

    log.info("Loading dataset: %s (full corpus)...", hf_name)
    ds = load_dataset(hf_name, split="data")
    log.info("  Corpus size: %d", len(ds))

    # Figure out which columns we need
    needed_cols = {"image", "item_ID"}
    for task in config["tasks"]:
        for col in task["doc_col"]:
            needed_cols.add(col)

    text_cols_needed = needed_cols - {"image", "item_ID"}
    keep_cols = [c for c in ds.column_names if c in needed_cols or c in text_cols_needed]

    log.info("  Computing embeddings...")
    embeddings, item_ids = get_embeddings(model, tokenizer, preprocess, ds, device)

    all_task_results = []
    for task in config["tasks"]:
        result = evaluate_task(model, tokenizer, preprocess, embeddings, item_ids,
                               task, gt_dir, device)
        if result:
            all_task_results.append(result)

    # Cleanup
    del embeddings, ds
    import gc
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()

    return all_task_results


def format_leaderboard(all_results: dict, model_name: str):
    """Format results in the same style as Marqo's LEADERBOARD.md."""
    lines = []
    lines.append(f"\n# Leaderboard Results — {model_name}")
    lines.append(f"- Generated: {time.strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"- Benchmarks: {', '.join(all_results.keys())}")
    lines.append("")

    for bm_name, tasks in all_results.items():
        lines.append(f"## {bm_name}")
        for task in tasks:
            task_name = task["task"]
            lines.append(f"### {task_name.replace('-', ' ').title()}")

            if "text-to-image" in task_name.lower():
                r1 = task["recall"].get("Recall@1", 0)
                r10 = task["recall"].get("Recall@10", 0)
                avg_r = round((r1 + r10) / 2, 3)
                lines.append(f"| AvgRecall | Recall@1 | Recall@10 | MRR |")
                lines.append(f"|-----------|----------|-----------|-----|")
                lines.append(f"| {avg_r:.3f} | {r1:.3f} | {r10:.3f} | {task['MRR']:.3f} |")
            else:
                p1 = task["precision"].get("P@1", 0)
                p10 = task["precision"].get("P@10", 0)
                avg_p = round((p1 + p10) / 2, 3)
                lines.append(f"| AvgP | P@1 | P@10 | MRR |")
                lines.append(f"|------|-----|------|-----|")
                lines.append(f"| {avg_p:.3f} | {p1:.3f} | {p10:.3f} | {task['MRR']:.3f} |")
            lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Official Leaderboard Evaluation")
    parser.add_argument("--model-source", type=str, required=True, choices=["phase4b", "fsl"])
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--model-name", type=str, default=None,
                        help="Label for the model in results (default: auto)")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--benchmarks", nargs="+", default=list(CLEAN_BENCHMARKS.keys()))
    parser.add_argument("--output-tag", type=str, default="official")
    args = parser.parse_args()

    t0 = time.time()

    if args.device:
        device = torch.device(args.device)
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    log.info("Device: %s", device)

    model_name = args.model_name
    if not model_name:
        if args.checkpoint:
            model_name = f"{args.model_source}+{Path(args.checkpoint).parent.name}"
        else:
            model_name = f"{args.model_source}_baseline"

    model, tokenizer, preprocess = load_model(args.model_source, args.checkpoint, device)

    all_results = {}
    for bm_name in args.benchmarks:
        if bm_name not in CLEAN_BENCHMARKS:
            log.warning("Unknown benchmark: %s — skipping", bm_name)
            continue
        try:
            task_results = evaluate_benchmark(
                model, tokenizer, preprocess, bm_name,
                CLEAN_BENCHMARKS[bm_name], device,
            )
            all_results[bm_name] = task_results
        except Exception as e:
            log.error("Failed on %s: %s", bm_name, e, exc_info=True)
            all_results[bm_name] = [{"error": str(e)}]

    # Save raw JSON
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = RESULTS_DIR / f"leaderboard_{args.output_tag}.json"
    with open(json_path, "w") as f:
        json.dump({
            "model_name": model_name,
            "model_source": args.model_source,
            "checkpoint": args.checkpoint,
            "results": {k: v for k, v in all_results.items()},
        }, f, indent=2, default=str)
    log.info("Raw results: %s", json_path)

    # Save formatted leaderboard
    md_text = format_leaderboard(all_results, model_name)
    md_path = RESULTS_DIR / f"leaderboard_{args.output_tag}.md"
    with open(md_path, "w") as f:
        f.write(md_text)
    log.info("Leaderboard: %s", md_path)

    # Print summary
    elapsed = time.time() - t0
    print(md_text)
    print(f"\nTotal time: {elapsed / 60:.1f} min")
    print(f"Results: {json_path}")


if __name__ == "__main__":
    main()
