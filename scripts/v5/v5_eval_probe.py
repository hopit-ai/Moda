"""
v5 in-training eval probe.

Evaluates the current student model on the 4 primary Marqo benchmarks
(fashion200k, atlas, polyvore, KAGL) at the same 200-query × 1200-doc
sample protocol used by `scripts/v4/phase3_eval_all_benchmarks.py --fast`.
This is the same eval set that produced the FSL baseline in
`results/v4_gcl/baseline_v4/full_results.json`, so probe outputs are
directly comparable.

Design:
  - Per-benchmark image embeddings are pre-computed ONCE with the frozen
    SigLIP-2 image tower (`student_image_emb` is reused — same model).
    Cached to disk; subsequent probe calls just lookup.
  - Per-probe: encode the 200 queries with the *current* student text tower
    (live forward), compute cosine vs cached image embeddings, take top-10,
    compute R@1, R@10, MRR, P@1, P@10.

This matches the existing v4 phase3 _compute_metrics_from_embeds pipeline
exactly (same metric definitions, same top-K=10).

Usage as a module (during training):
    from v5_eval_probe import EvalProbe
    probe = EvalProbe(device="mps")  # loads cached image embeds + queries
    metrics = probe.probe(model, tokenizer)
    # metrics = {"fashion200k": {"mrr": 0.51, "recall@10": 0.84, ...}, ...}

Usage as a CLI (standalone — e.g., to score a checkpoint):
    python scripts/v5/v5_eval_probe.py --build_caches            # one-time, ~10 min on MPS
    python scripts/v5/v5_eval_probe.py --baseline                # SL2 zero-shot baseline
    python scripts/v5/v5_eval_probe.py --checkpoint path/to.pt   # probe a checkpoint
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parents[2]
EVAL_CACHE_DIR = REPO / "data" / "processed" / "v5_eval_cache"
EVAL_CACHE_DIR.mkdir(parents=True, exist_ok=True)


# Match the v4 phase3 protocol — same benchmarks, same query column priority
BENCHMARKS = {
    "fashion200k": "Marqo/fashion200k",
    "atlas":       "Marqo/atlas",
    "polyvore":    "Marqo/polyvore",
    "KAGL":        "Marqo/KAGL",
}

QUERY_COLS_PRIORITY = ["query", "text", "caption", "description"]

# v4 phase3 --fast settings (matches baseline_v4 numbers in results/v4_gcl/baseline_v4/full_results.json)
DEFAULT_MAX_QUERIES = 200
DEFAULT_MAX_DOCS = 1200


@dataclass
class BenchmarkData:
    """Pre-computed eval set for one benchmark."""
    name: str
    queries: list[str]              # length Q (e.g. 200)
    doc_ids: list[str]              # length D (e.g. 1200)
    image_embeddings: torch.Tensor  # (D, 768) fp16
    gt: dict[str, set[str]]         # query → set of relevant doc_ids


# ---------------------------------------------------------------------------
# Cache build (one-time)
# ---------------------------------------------------------------------------

def _load_rows(hf_id: str, max_docs: int) -> list[dict]:
    """Load up to max_docs rows from a Marqo HF dataset, attaching the PIL image
    to row['_image']. Tries split='data' first, then 'test'."""
    from datasets import load_dataset
    ds = None
    last_err = None
    for split in ("data", "test"):
        try:
            ds = load_dataset(hf_id, split=split, streaming=True)
            break
        except Exception as e:
            last_err = e
    if ds is None:
        raise RuntimeError(f"could not load {hf_id}: {last_err}")

    rows = []
    for i, row in enumerate(ds):
        if i >= max_docs:
            break
        rd = {}
        for k, v in row.items():
            if k == "image":
                rd["_image"] = v
                rd["_has_image"] = v is not None
            else:
                rd[k] = v
        rows.append(rd)
    return rows


def _pick_query_column(rows: list[dict]) -> str | None:
    avail = set()
    for r in rows[:100]:
        avail.update(r.keys())
    for c in QUERY_COLS_PRIORITY:
        if c in avail:
            return c
    return None


def _build_query_doc_split(rows: list[dict], query_col: str,
                           max_queries: int) -> tuple[list[str], list[str], dict]:
    """Replicates v4 phase3 _run_text_to_image's query/doc/gt extraction."""
    gt: dict[str, set[str]] = defaultdict(set)
    queries: dict[str, bool] = {}
    doc_images: dict[str, object] = {}

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

    return list(queries.keys()), list(doc_images.keys()), {**doc_images}, dict(gt)


def build_eval_caches(device: str = "mps",
                      max_queries: int = DEFAULT_MAX_QUERIES,
                      max_docs: int = DEFAULT_MAX_DOCS,
                      out_dir: Path = EVAL_CACHE_DIR) -> None:
    """One-time: encode benchmark images with frozen SigLIP-2 image tower, save.

    Output per benchmark in `out_dir`:
        <name>_queries.json      — list[str]
        <name>_doc_ids.json      — list[str]
        <name>_gt.json           — dict[query, list[doc_id]]
        <name>_image_emb.pt      — (D, 768) fp16
    """
    import open_clip
    log.info("Loading SigLIP-2 base for image encoding ...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-16-SigLIP2-384", pretrained="webli"
    )
    model = model.to(device).eval()

    out_dir.mkdir(parents=True, exist_ok=True)

    for bname, hf_id in BENCHMARKS.items():
        log.info(f"\n=== {bname} ({hf_id}) ===")
        cache_emb = out_dir / f"{bname}_image_emb.pt"
        cache_queries = out_dir / f"{bname}_queries.json"
        cache_doc_ids = out_dir / f"{bname}_doc_ids.json"
        cache_gt = out_dir / f"{bname}_gt.json"
        if all(p.exists() for p in (cache_emb, cache_queries, cache_doc_ids, cache_gt)):
            log.info(f"  cache present, skipping")
            continue

        rows = _load_rows(hf_id, max_docs)
        log.info(f"  loaded {len(rows)} rows")
        query_col = _pick_query_column(rows)
        if query_col is None:
            log.warning(f"  no query column found; skipping {bname}")
            continue
        queries, doc_ids, doc_images, gt = _build_query_doc_split(
            rows, query_col, max_queries
        )
        log.info(f"  {len(queries)} queries, {len(doc_ids)} doc images")

        # Encode images
        D = len(doc_ids)
        emb = torch.zeros((D, 768), dtype=torch.float16)
        BATCH = 32
        with torch.inference_mode():
            for j in tqdm(range(0, D, BATCH), desc=f"  {bname} images"):
                batch_ids = doc_ids[j:j + BATCH]
                tensors = []
                for did in batch_ids:
                    try:
                        img = doc_images[did].convert("RGB")
                        tensors.append(preprocess(img))
                    except Exception:
                        tensors.append(torch.zeros(3, 224, 224))
                stack = torch.stack(tensors).to(device)
                e = model.encode_image(stack)
                e = F.normalize(e, dim=-1)
                emb[j:j + len(batch_ids)] = e.detach().cpu().to(torch.float16)

        torch.save(emb, cache_emb)
        cache_queries.write_text(json.dumps(queries))
        cache_doc_ids.write_text(json.dumps(doc_ids))
        cache_gt.write_text(json.dumps({q: list(s) for q, s in gt.items()}))
        log.info(f"  saved {bname}: {tuple(emb.shape)} → {cache_emb}")


# ---------------------------------------------------------------------------
# In-training probe
# ---------------------------------------------------------------------------

def _compute_metrics_from_embeds(
    queries: list[str], doc_ids: list[str],
    q_embeds: torch.Tensor, d_embeds: torch.Tensor,
    gt: dict[str, set[str]],
) -> dict:
    """Identical to v4 phase3's _compute_metrics_from_embeds — port verbatim."""
    sims = q_embeds @ d_embeds.T
    _, top_indices = sims.topk(min(10, len(doc_ids)), dim=1)

    metrics = {"recall@1": 0.0, "recall@10": 0.0, "mrr": 0.0,
               "precision@1": 0.0, "precision@10": 0.0}
    n_queries = 0

    for qi, q_text in enumerate(queries):
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


class EvalProbe:
    """Live probe that re-encodes queries with the current student each call."""

    def __init__(self, cache_dir: Path = EVAL_CACHE_DIR, device: str = "mps"):
        self.device = device
        self.benchmarks: dict[str, BenchmarkData] = {}
        for bname in BENCHMARKS:
            cache_emb = cache_dir / f"{bname}_image_emb.pt"
            cache_q = cache_dir / f"{bname}_queries.json"
            cache_d = cache_dir / f"{bname}_doc_ids.json"
            cache_gt = cache_dir / f"{bname}_gt.json"
            if not all(p.exists() for p in (cache_emb, cache_q, cache_d, cache_gt)):
                log.warning(f"missing cache for {bname}; skipping")
                continue
            queries = json.loads(cache_q.read_text())
            doc_ids = json.loads(cache_d.read_text())
            gt_raw = json.loads(cache_gt.read_text())
            gt = {q: set(v) for q, v in gt_raw.items()}
            emb = torch.load(cache_emb, map_location="cpu")
            self.benchmarks[bname] = BenchmarkData(
                name=bname, queries=queries, doc_ids=doc_ids,
                image_embeddings=emb, gt=gt,
            )
        log.info(f"EvalProbe loaded {len(self.benchmarks)} benchmarks")

    @torch.no_grad()
    def probe(self, model, tokenizer, batch_size: int = 64) -> dict[str, dict]:
        """Run the probe; returns per-benchmark metric dicts."""
        was_training = model.training
        model.eval()
        out = {}
        for bname, bdata in self.benchmarks.items():
            # Encode queries with CURRENT student text tower
            q_embeds = []
            for j in range(0, len(bdata.queries), batch_size):
                tok = tokenizer(bdata.queries[j:j + batch_size]).to(self.device)
                e = model.encode_text(tok)
                e = F.normalize(e, dim=-1)
                q_embeds.append(e.detach().cpu().float())
            q = torch.cat(q_embeds, dim=0)
            d = bdata.image_embeddings.float()
            metrics = _compute_metrics_from_embeds(
                bdata.queries, bdata.doc_ids, q, d, bdata.gt
            )
            out[bname] = metrics
        if was_training:
            model.train()
        return out

    def aggregate_score(self, metrics: dict[str, dict]) -> float:
        """Mean MRR across benchmarks — the early-stopping signal."""
        vals = [m["mrr"] for m in metrics.values() if "mrr" in m]
        return sum(vals) / max(1, len(vals))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _pick_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build_caches", action="store_true",
                    help="One-time: encode benchmark images with SigLIP-2 base")
    ap.add_argument("--baseline", action="store_true",
                    help="Probe the SL2 base model (no checkpoint loaded)")
    ap.add_argument("--checkpoint", type=Path, default=None,
                    help="Path to a Phase C/D checkpoint (state dict of trainable params)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", type=Path, default=None,
                    help="Optional JSON output for the metrics")
    args = ap.parse_args()

    device = args.device or _pick_device()

    if args.build_caches:
        build_eval_caches(device=device)
        return

    if not (args.baseline or args.checkpoint):
        ap.error("must specify --baseline, --checkpoint, or --build_caches")

    sys.path.insert(0, str(Path(__file__).parent))
    from v5_model import build_student

    log.info(f"Building student on {device} ...")
    model, tokenizer = build_student(device=device)
    if args.checkpoint:
        log.info(f"Loading checkpoint {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        if isinstance(ckpt, dict) and "model_trainable" in ckpt:
            sd = {n: p.to(device) for n, p in ckpt["model_trainable"].items()}
            missing, unexpected = model.load_state_dict(sd, strict=False)
            log.info(f"  loaded {len(sd)} tensors; "
                     f"{len(missing)} missing (expected — frozen params), "
                     f"{len(unexpected)} unexpected")
        else:
            model.load_state_dict(ckpt, strict=False)

    probe = EvalProbe(device=device)
    if not probe.benchmarks:
        sys.exit("ERROR: no eval caches found. Run with --build_caches first.")

    t0 = time.time()
    metrics = probe.probe(model, tokenizer)
    elapsed = time.time() - t0

    print("\n" + "=" * 60)
    print(f"EvalProbe ({elapsed:.1f}s)" +
          (f" — checkpoint={args.checkpoint}" if args.checkpoint else " — baseline SL2"))
    print("=" * 60)
    for bname, m in metrics.items():
        print(f"  {bname:14s}  MRR={m['mrr']:.4f}  R@10={m['recall@10']:.4f}  "
              f"R@1={m['recall@1']:.4f}  P@10={m['precision@10']:.4f}  "
              f"(q={m['n_queries']})")
    print(f"  mean MRR: {probe.aggregate_score(metrics):.4f}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(metrics, indent=2))
        print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
