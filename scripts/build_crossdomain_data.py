"""
Build cross-domain training data for Path 3 (overnight experiment).

Strategy: mix Marqo-GS short catalog queries with DeepFashion-Multimodal
prose-style captions to solve the training-data-domain-mismatch bottleneck
identified in AUTOPSY_PATH2.md.

Pipeline:
  1. Load existing Marqo-GS triplets (421 queries, short catalog style)
  2. Sample ~400 query-image pairs from DeepFashion-Multimodal (prose style)
  3. Save all sampled DFM images to disk (same format as Marqo-GS images)
  4. Encode ALL images+queries through both teachers (FashionSigLIP + SigLIP2-B16-384)
  5. Mine K=15 hard negatives per query from the combined pool
  6. Cache init-student (SigLIP2-B16-384 webli) embeddings for anchor loss
  7. Write hardnegs.jsonl + init_anchor_cache.pt

Output: data/processed/path3/
  - images/       (DFM images saved as jpg)
  - triplets.jsonl
  - teacher_cache.pt  (both teachers' embeddings for the combined pool)
  - hardnegs.jsonl
  - init_anchor_cache.pt

Usage:
  .venv/bin/python scripts/build_crossdomain_data.py
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import logging
import math
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("build-crossdomain")

HF_CACHE = REPO / "data" / "hf_cache"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def sample_deepfashion_multimodal(n: int, seed: int = 42) -> list[dict]:
    """Sample n unique query-image pairs from DeepFashion-Multimodal.

    Returns list of {query: str, image: PIL.Image, item_ID: str, category1: str}.
    """
    from datasets import load_dataset
    ds = load_dataset("Marqo/deepfashion-multimodal", cache_dir=str(HF_CACHE))["data"]
    log.info("DFM dataset: %d rows", len(ds))

    rng = random.Random(seed)
    indices = list(range(len(ds)))
    rng.shuffle(indices)

    seen_texts = set()
    samples = []
    for idx in indices:
        if len(samples) >= n:
            break
        row = ds[idx]
        text = row["text"].strip()
        if not text or text in seen_texts:
            continue
        if len(text) < 15:
            continue
        seen_texts.add(text)
        samples.append({
            "query": text,
            "image": row["image"],
            "item_ID": row["item_ID"],
            "category1": row.get("category1", "unknown"),
            "source": "deepfashion_multimodal",
        })

    log.info("sampled %d DFM pairs (requested %d)", len(samples), n)
    return samples


def save_dfm_images(samples: list[dict], out_dir: Path) -> list[dict]:
    """Save DFM images to disk; add image_path to each sample."""
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    for s in samples:
        img_hash = hashlib.md5(s["item_ID"].encode()).hexdigest()[:12]
        fname = f"dfm_{img_hash}.jpg"
        fpath = img_dir / fname
        if not fpath.exists():
            s["image"].convert("RGB").save(str(fpath), "JPEG", quality=95)
        s["image_path"] = str(fpath)
    log.info("saved %d DFM images to %s", len(samples), img_dir)
    return samples


def load_marqo_gs_triplets() -> list[dict]:
    """Load existing Marqo-GS triplets and return as [{query, image_path, score_linear}]."""
    triplets_path = REPO / "data/processed/marqo_gs_wfash_subset/triplets.jsonl"
    rows = []
    with open(triplets_path) as f:
        for line in f:
            rows.append(json.loads(line))
    log.info("loaded %d Marqo-GS triplets", len(rows))
    return rows


def encode_with_model(
    model, preprocess, tokenizer,
    queries: list[str], image_paths: list[str],
    device: str, batch_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode queries and images with a model. Returns (text_emb, img_emb) L2-normed."""
    import open_clip

    model.eval()
    model.to(device)

    # Text embeddings
    text_embs = []
    for i in range(0, len(queries), batch_size):
        batch_q = queries[i:i + batch_size]
        tokens = tokenizer(batch_q).to(device)
        with torch.no_grad():
            emb = model.encode_text(tokens)
            emb = emb / emb.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        text_embs.append(emb.cpu())
        if (i // batch_size) % 10 == 0:
            log.info("  text: %d/%d", min(i + batch_size, len(queries)), len(queries))
    text_embs = torch.cat(text_embs, dim=0)

    # Image embeddings
    img_embs = []
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        imgs = torch.stack([preprocess(Image.open(p).convert("RGB")) for p in batch_paths]).to(device)
        with torch.no_grad():
            emb = model.encode_image(imgs)
            emb = emb / emb.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        img_embs.append(emb.cpu())
        if (i // batch_size) % 10 == 0:
            log.info("  image: %d/%d", min(i + batch_size, len(image_paths)), len(image_paths))
    img_embs = torch.cat(img_embs, dim=0)

    return text_embs, img_embs


def mine_hard_negatives_combined(
    queries_data: list[dict],
    fsl_text: torch.Tensor, fsl_img: torch.Tensor,
    all_image_paths: list[str],
    K: int = 15,
    top_n: int = 100,
    fn_filter_factor: float = 0.95,
    peak: float = 0.75,
    sigma: float = 0.10,
    seed: int = 42,
) -> list[dict]:
    """Mine hard negatives for all queries using the combined image pool.

    queries_data: list of {query, pos_image_paths: [str], query_idx: int}
    fsl_text: [N_q, D] all text embeddings
    fsl_img: [N_img, D] all image embeddings (deduplicated pool)
    all_image_paths: paths matching fsl_img rows
    """
    rng = random.Random(seed)
    path_to_pool_idx = {p: i for i, p in enumerate(all_image_paths)}

    results = []
    cov_full = cov_partial = cov_zero = 0

    for j, qd in enumerate(queries_data):
        q = qd["query"]
        qi = qd["query_idx"]
        pos_paths = set(qd["pos_image_paths"])

        txt_q = fsl_text[qi]
        scores = (fsl_img @ txt_q).flatten()

        if not pos_paths:
            cov_zero += 1
            continue

        strongest_path = qd.get("strongest_pos_path", list(pos_paths)[0])
        strongest_pool_idx = path_to_pool_idx[strongest_path]
        strongest_fsl_text_score = float(scores[strongest_pool_idx])

        top_vals, top_idx = torch.topk(scores, k=min(top_n, scores.shape[0]))
        cand_idx = top_idx.tolist()
        cand_vals = top_vals.tolist()

        fn_threshold = fn_filter_factor * strongest_fsl_text_score
        surviving_pool_idx = []
        surviving_text_scores = []
        for ci, cv in zip(cand_idx, cand_vals):
            path = all_image_paths[ci]
            if path in pos_paths:
                continue
            if cv > fn_threshold:
                continue
            surviving_pool_idx.append(ci)
            surviving_text_scores.append(cv)

        if not surviving_pool_idx:
            results.append({
                "query": q,
                "positives": [{"image_path": p, "score_linear": qd.get("score_linear", 100.0),
                                "fsl_text_img_score": float(scores[path_to_pool_idx[p]])}
                              for p in pos_paths],
                "hard_negatives": [],
            })
            cov_zero += 1
            continue

        survivor_emb = fsl_img.index_select(0, torch.tensor(surviving_pool_idx))
        strongest_emb = fsl_img[strongest_pool_idx]
        img_img_scores = survivor_emb @ strongest_emb

        weights = torch.exp(-((img_img_scores - peak) ** 2) / (sigma ** 2))

        chosen_local = _weighted_sample_no_replace(weights, k=K, rng=rng)
        hard_negs = []
        for li in chosen_local:
            pi = surviving_pool_idx[li]
            hard_negs.append({
                "image_path": all_image_paths[pi],
                "fsl_text_img_score": float(surviving_text_scores[li]),
                "fsl_img_to_strongest_gold": float(img_img_scores[li]),
            })

        positives = [{"image_path": p, "score_linear": qd.get("score_linear", 100.0),
                       "fsl_text_img_score": float(scores[path_to_pool_idx[p]])}
                     for p in pos_paths]

        results.append({
            "query": q,
            "positives": positives,
            "hard_negatives": hard_negs,
        })

        if len(hard_negs) >= K:
            cov_full += 1
        elif len(hard_negs) > 0:
            cov_partial += 1
        else:
            cov_zero += 1

        if (j + 1) % 100 == 0:
            log.info("  mined %d/%d  full=%d partial=%d zero=%d",
                     j + 1, len(queries_data), cov_full, cov_partial, cov_zero)

    log.info("hard-neg mining: full=%d  partial=%d  zero=%d  (total=%d)",
             cov_full, cov_partial, cov_zero, len(queries_data))
    return results


def _weighted_sample_no_replace(weights: torch.Tensor, k: int, rng: random.Random) -> list[int]:
    n = weights.shape[0]
    if n == 0 or k <= 0:
        return []
    k = min(k, n)
    keys = []
    for i in range(n):
        w = float(weights[i])
        if w <= 0:
            keys.append(-math.inf)
            continue
        u = rng.random()
        keys.append(math.log(max(u, 1e-300)) / w)
    keys_t = torch.tensor(keys)
    top = torch.topk(keys_t, k=k).indices.tolist()
    return top


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-dfm", type=int, default=400,
                   help="Number of DeepFashion-Multimodal samples to add")
    p.add_argument("--K", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--out-dir", default=str(REPO / "data/processed/path3"))
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()

    # ---- Step 1: Load Marqo-GS data ----
    log.info("=" * 60)
    log.info("Step 1: Loading Marqo-GS data")
    log.info("=" * 60)
    gs_rows = load_marqo_gs_triplets()
    gs_by_query: dict[str, list[dict]] = defaultdict(list)
    for r in gs_rows:
        gs_by_query[r["query"]].append(r)
    log.info("Marqo-GS: %d unique queries", len(gs_by_query))

    # ---- Step 2: Sample DFM data ----
    log.info("=" * 60)
    log.info("Step 2: Sampling DeepFashion-Multimodal")
    log.info("=" * 60)
    dfm_samples = sample_deepfashion_multimodal(args.n_dfm, seed=args.seed)
    dfm_samples = save_dfm_images(dfm_samples, out_dir)

    for s in dfm_samples:
        del s["image"]

    # ---- Step 3: Build combined query/image lists ----
    log.info("=" * 60)
    log.info("Step 3: Building combined query/image lists")
    log.info("=" * 60)

    all_queries: list[str] = []
    all_image_paths: list[str] = []
    query_to_first_idx: dict[str, int] = {}
    path_to_first_idx: dict[str, int] = {}

    queries_meta: list[dict] = []

    for q, rows in gs_by_query.items():
        qi = len(all_queries)
        all_queries.append(q)
        query_to_first_idx[q] = qi

        pos_paths = []
        best_score = -1
        best_path = None
        for r in rows:
            ip = r["image_path"]
            if ip not in path_to_first_idx:
                path_to_first_idx[ip] = len(all_image_paths)
                all_image_paths.append(ip)
            sl = float(r.get("score_linear", 0))
            if sl > 0:
                pos_paths.append(ip)
            if sl > best_score:
                best_score = sl
                best_path = ip

        queries_meta.append({
            "query": q,
            "query_idx": qi,
            "pos_image_paths": pos_paths,
            "strongest_pos_path": best_path,
            "score_linear": best_score,
            "source": "marqo_gs",
        })

    for s in dfm_samples:
        qi = len(all_queries)
        all_queries.append(s["query"])
        query_to_first_idx[s["query"]] = qi

        ip = s["image_path"]
        if ip not in path_to_first_idx:
            path_to_first_idx[ip] = len(all_image_paths)
            all_image_paths.append(ip)

        queries_meta.append({
            "query": s["query"],
            "query_idx": qi,
            "pos_image_paths": [ip],
            "strongest_pos_path": ip,
            "score_linear": 100.0,
            "source": "deepfashion_multimodal",
        })

    n_gs_queries = len(gs_by_query)
    n_dfm_queries = len(dfm_samples)
    log.info("Combined: %d queries (%d GS + %d DFM), %d unique images",
             len(all_queries), n_gs_queries, n_dfm_queries, len(all_image_paths))

    # ---- Step 4: Encode with both teachers ----
    log.info("=" * 60)
    log.info("Step 4: Encoding with FashionSigLIP")
    log.info("=" * 60)
    import open_clip

    fsl_model, _, fsl_preprocess = open_clip.create_model_and_transforms(
        "hf-hub:Marqo/marqo-fashionSigLIP",
        cache_dir=str(HF_CACHE),
    )
    fsl_tokenizer = open_clip.get_tokenizer("hf-hub:Marqo/marqo-fashionSigLIP")

    fsl_text_emb, fsl_img_emb = encode_with_model(
        fsl_model, fsl_preprocess, fsl_tokenizer,
        all_queries, all_image_paths,
        device=DEVICE, batch_size=args.batch_size,
    )
    log.info("FSL: text=%s, img=%s", fsl_text_emb.shape, fsl_img_emb.shape)

    del fsl_model
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    log.info("=" * 60)
    log.info("Step 4b: Encoding with SigLIP2-B16-384")
    log.info("=" * 60)
    sl2_model, _, sl2_preprocess = open_clip.create_model_and_transforms(
        "ViT-B-16-SigLIP2-384",
        pretrained="webli",
        cache_dir=str(HF_CACHE),
    )
    sl2_tokenizer = open_clip.get_tokenizer("ViT-B-16-SigLIP2-384")

    sl2_text_emb, sl2_img_emb = encode_with_model(
        sl2_model, sl2_preprocess, sl2_tokenizer,
        all_queries, all_image_paths,
        device=DEVICE, batch_size=args.batch_size,
    )
    log.info("SL2: text=%s, img=%s", sl2_text_emb.shape, sl2_img_emb.shape)

    del sl2_model
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    # Save teacher cache
    teacher_cache = {
        "queries": all_queries,
        "image_paths": all_image_paths,
        "embed_dim": fsl_text_emb.shape[1],
    }
    teacher_dir = out_dir / "teacher_cache"
    teacher_dir.mkdir(parents=True, exist_ok=True)

    fsl_cache_path = teacher_dir / "fashion_siglip_embeddings.pt"
    torch.save({
        "text": fsl_text_emb, "image": fsl_img_emb,
        "queries": all_queries, "image_paths": all_image_paths,
        "embed_dim": fsl_text_emb.shape[1],
    }, fsl_cache_path)
    log.info("saved FSL teacher cache to %s", fsl_cache_path)

    sl2_cache_path = teacher_dir / "siglip2_b16_384_embeddings.pt"
    torch.save({
        "text": sl2_text_emb, "image": sl2_img_emb,
        "queries": all_queries, "image_paths": all_image_paths,
        "embed_dim": sl2_text_emb.shape[1],
    }, sl2_cache_path)
    log.info("saved SL2 teacher cache to %s", sl2_cache_path)

    # ---- Step 5: Mine hard negatives ----
    log.info("=" * 60)
    log.info("Step 5: Mining hard negatives (K=%d)", args.K)
    log.info("=" * 60)

    fused_text = 0.5 * (fsl_text_emb + sl2_text_emb)
    fused_text = fused_text / fused_text.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    fused_img = 0.5 * (fsl_img_emb + sl2_img_emb)
    fused_img = fused_img / fused_img.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    hardneg_results = mine_hard_negatives_combined(
        queries_meta,
        fused_text, fused_img,
        all_image_paths,
        K=args.K, seed=args.seed,
    )

    hardnegs_path = out_dir / "hardnegs.jsonl"
    with open(hardnegs_path, "w") as f:
        for rec in hardneg_results:
            f.write(json.dumps(rec) + "\n")
    log.info("wrote %d records to %s", len(hardneg_results), hardnegs_path)

    n_with_negs = sum(1 for r in hardneg_results if len(r.get("hard_negatives", [])) >= args.K)
    log.info("queries with >= K=%d hard negs: %d / %d", args.K, n_with_negs, len(hardneg_results))

    # ---- Step 6: Cache init-student anchor embeddings ----
    log.info("=" * 60)
    log.info("Step 6: Caching init-student anchor embeddings")
    log.info("=" * 60)

    student_model, _, student_preprocess = open_clip.create_model_and_transforms(
        "ViT-B-16-SigLIP2-384",
        pretrained="webli",
        cache_dir=str(HF_CACHE),
    )
    student_tokenizer = open_clip.get_tokenizer("ViT-B-16-SigLIP2-384")

    unique_queries_for_anchor = list(set(all_queries))
    unique_img_paths_for_anchor = list(set(all_image_paths))

    anchor_txt, anchor_img = encode_with_model(
        student_model, student_preprocess, student_tokenizer,
        unique_queries_for_anchor, unique_img_paths_for_anchor,
        device=DEVICE, batch_size=args.batch_size,
    )

    anchor_cache_path = out_dir / "init_anchor_cache.pt"
    torch.save({
        "img": anchor_img,
        "txt": anchor_txt,
        "image_paths": unique_img_paths_for_anchor,
        "queries": unique_queries_for_anchor,
    }, anchor_cache_path)
    log.info("saved anchor cache to %s (img=%s, txt=%s)",
             anchor_cache_path, anchor_img.shape, anchor_txt.shape)

    del student_model
    gc.collect()

    # ---- Summary ----
    elapsed = time.time() - t0
    log.info("=" * 60)
    log.info("DONE in %.1f min", elapsed / 60)
    log.info("  Output dir: %s", out_dir)
    log.info("  Queries: %d (%d GS + %d DFM)", len(all_queries), n_gs_queries, n_dfm_queries)
    log.info("  Images: %d unique", len(all_image_paths))
    log.info("  Hard-neg records: %d (with K>=%d negs: %d)",
             len(hardneg_results), args.K, n_with_negs)
    log.info("=" * 60)

    meta = {
        "n_gs_queries": n_gs_queries,
        "n_dfm_queries": n_dfm_queries,
        "n_total_queries": len(all_queries),
        "n_unique_images": len(all_image_paths),
        "n_with_full_negs": n_with_negs,
        "K": args.K,
        "elapsed_sec": elapsed,
    }
    (out_dir / "build_meta.json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
