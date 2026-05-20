"""
Diagnose Path 1 drift — memory-disciplined version.

Same goal as before: compare frozen-init (open_clip ViT-B-16-SigLIP2-384/webli)
vs trained Path 1 student on {f200k, atlas, polyvore} 10K subsamples.
Measure embedding drift + per-query AP@10 deltas + worst-hurt query examples.

Memory discipline:
  - Encode one model at a time. Save its [N, 768] image+text embeddings to disk.
  - Free the model + corpus-tensor handles before loading the next model.
  - Do drift + AP@10 comparison from the on-disk embeddings, not from RAM.
  - Stream the 8.8 GB corpus tensor in chunks through the model (don't keep it
    referenced after iteration — Python frees it eagerly).

Output:
  diagnostics/path1_drift/
    summary.md
    {dataset}/drift_stats.json
    {dataset}/per_query.csv
    {dataset}/worst_hurt.json
    {dataset}/_emb_cache/{init|student}_{img|txt}.pt   (intermediate)

Usage:
  .venv/bin/python benchmark/diagnose_path1_drift.py \
      --student-checkpoint models/path1-full/best/student_state_dict.pt \
      --datasets fashion200k atlas polyvore \
      --output-dir diagnostics/path1_drift
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import logging
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "benchmark"))
sys.path.insert(0, str(REPO / "repos" / "marqo-FashionCLIP"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("diagnose-p1")

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


# -----------------------------------------------------------------------------
# Model loading
# -----------------------------------------------------------------------------

def load_model(student_state_dict: Path | None, device: str):
    """Load open_clip ViT-B-16-SigLIP2-384 webli; optionally apply trained state dict.

    Returns (model, preprocess, tokenizer).
    """
    import open_clip

    cache_dir = str(REPO / "data" / "hf_cache")
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-16-SigLIP2-384", pretrained="webli", cache_dir=cache_dir,
    )
    tokenizer = open_clip.get_tokenizer("ViT-B-16-SigLIP2-384")
    if student_state_dict is not None:
        log.info("loading state dict from %s", student_state_dict)
        sd = torch.load(student_state_dict, map_location="cpu", weights_only=False)
        if isinstance(sd, dict) and "state_dict" in sd and not any(
            k.startswith("visual.") for k in sd
        ):
            sd = sd["state_dict"]
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing:
            log.warning("  missing keys: %d", len(missing))
        if unexpected:
            log.warning("  unexpected keys: %d", len(unexpected))
    model.eval().to(device)
    for p_ in model.parameters():
        p_.requires_grad = False
    return model, preprocess, tokenizer


# -----------------------------------------------------------------------------
# Streaming encode  (does NOT keep the full corpus tensor referenced)
# -----------------------------------------------------------------------------

def encode_streaming(
    model, preprocess, tokenizer, probe,
    out_img_path: Path, out_txt_path: Path,
    batch_size: int = 64, device: str = DEVICE,
):
    """Stream encode the corpus + queries with `model`, save embeddings to disk.

    The corpus tensor is opened, iterated in chunks, and the local reference
    is released before we move on to text encoding. Embeddings are written to
    disk so the caller can free `model` and reload another one without holding
    two big things in RAM at once.
    """
    # 1. Image streaming encode — tensor handle scoped to the inner block
    img_feats: list[torch.Tensor] = []
    n_img = 0
    t0 = time.time()
    {
        # Inner block ensures `corpus_tens` is dropped after the loop.
    }
    corpus_tens = probe._preprocess_corpus(preprocess)  # [N, 3, H, W] fp16
    n_img = corpus_tens.shape[0]
    log.info("  encoding %d images (chunked) ...", n_img)
    with torch.no_grad():
        for i in range(0, n_img, batch_size):
            batch = corpus_tens[i:i + batch_size].to(device).float()
            feat = model.encode_image(batch)
            feat = feat / feat.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            img_feats.append(feat.float().cpu())
            del batch, feat
            if (i // batch_size) % 20 == 0:
                log.info("    [img] %d/%d  (%.1fs)", min(i + batch_size, n_img),
                         n_img, time.time() - t0)
    img_feats_t = torch.cat(img_feats, dim=0)
    del img_feats
    # Drop the probe's preprocessed cache reference too — its 8.8GB will
    # be freed once both models that need this dataset are done. For now
    # we want it freed so the SECOND model load doesn't OOM.
    probe._preprocessed_cache.clear()
    del corpus_tens
    gc.collect()
    if torch.backends.mps.is_available():
        try:
            torch.mps.empty_cache()
        except Exception:
            pass
    log.info("  image encode done: shape=%s in %.1fs", tuple(img_feats_t.shape), time.time() - t0)
    torch.save(img_feats_t, out_img_path)
    log.info("  saved %s (%.1f MB)", out_img_path, out_img_path.stat().st_size / 1e6)
    del img_feats_t

    # 2. Text encode (small, no streaming concerns)
    t0 = time.time()
    txt_feats: list[torch.Tensor] = []
    with torch.no_grad():
        for i in range(0, len(probe.queries), batch_size):
            tokens = tokenizer(probe.queries[i:i + batch_size]).to(device)
            feat = model.encode_text(tokens)
            feat = feat / feat.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            txt_feats.append(feat.float().cpu())
    txt_feats_t = torch.cat(txt_feats, dim=0)
    log.info("  text encode: %d in %.1fs  shape=%s", len(probe.queries),
             time.time() - t0, tuple(txt_feats_t.shape))
    torch.save(txt_feats_t, out_txt_path)
    del txt_feats_t, txt_feats
    gc.collect()


# -----------------------------------------------------------------------------
# Drift + per-query analysis
# -----------------------------------------------------------------------------

def compute_drift_stats(emb_a: torch.Tensor, emb_b: torch.Tensor) -> dict:
    cos = (emb_a * emb_b).sum(dim=-1).clamp(-1.0, 1.0)
    drift = 1.0 - cos
    return {
        "n": int(drift.numel()),
        "mean": float(drift.mean()),
        "median": float(drift.median()),
        "p90": float(torch.quantile(drift, 0.9)),
        "p95": float(torch.quantile(drift, 0.95)),
        "p99": float(torch.quantile(drift, 0.99)),
        "max": float(drift.max()),
    }


def per_query_ap10(scores: torch.Tensor, item_IDs: list[str], queries: list[str],
                   gt: dict, k: int = 10) -> dict:
    top_scores, top_inds = torch.topk(scores, k=min(k, scores.shape[1]), dim=1)
    top_inds = top_inds.tolist()
    out = {}
    for qi, q in enumerate(queries):
        if q not in gt:
            continue
        gold = {d for d, r in gt[q].items() if r > 0}
        n_gold = len(gold)
        if n_gold == 0:
            continue
        top10_ids = [item_IDs[d] for d in top_inds[qi]]
        ap = 0.0
        n_hits = 0
        for r_i, doc in enumerate(top10_ids, 1):
            if doc in gold:
                n_hits += 1
                ap += n_hits / r_i
        ap /= min(k, n_gold)
        out[q] = {"ap10": ap, "n_gold": n_gold, "top10": top10_ids}
    return out


# -----------------------------------------------------------------------------
# Per-dataset orchestration
# -----------------------------------------------------------------------------

def encode_pass(
    dataset: str, which: str, ckpt: Path | None, out_dir: Path,
    batch_size: int = 64,
):
    """Encode (corpus+queries) with one model. `which` is 'init' or 'student'.

    Returns the probe handle so the caller can read item_IDs/queries/gt.
    """
    from probe_fashion200k_10k import Fashion200kProbe
    probe = Fashion200kProbe(dataset=dataset, corpus_size=10000, seed=42)

    img_path = out_dir / f"{which}_img.pt"
    txt_path = out_dir / f"{which}_txt.pt"
    if img_path.exists() and txt_path.exists():
        log.info("[%s/%s] embeddings already exist, skipping encode", dataset, which)
        return probe

    log.info("[%s/%s] loading model ...", dataset, which)
    model, preprocess, tokenizer = load_model(ckpt, DEVICE)

    log.info("[%s/%s] encoding (streaming) ...", dataset, which)
    encode_streaming(model, preprocess, tokenizer, probe, img_path, txt_path,
                     batch_size=batch_size)

    # Aggressively free model + probe caches before the next pass
    del model, preprocess, tokenizer
    probe._preprocessed_cache.clear()
    probe._pil_images = None
    gc.collect()
    if torch.backends.mps.is_available():
        try:
            torch.mps.empty_cache()
        except Exception:
            pass
    return probe


def diagnose_dataset(dataset: str, student_ckpt: Path, output_dir: Path,
                     batch_size: int = 64, n_worst: int = 20):
    log.info("=" * 70)
    log.info("DATASET: %s", dataset)
    log.info("=" * 70)

    out = output_dir / dataset
    cache = out / "_emb_cache"
    out.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)

    # Encode init pass (probe will rebuild corpus tensor cache file lazily)
    probe = encode_pass(dataset, "init", None, cache, batch_size)
    # Encode student pass — separate function call, so init model is GC'd
    probe = encode_pass(dataset, "student", student_ckpt, cache, batch_size)

    # Now compare from disk
    log.info("[%s] loading saved embeddings for analysis ...", dataset)
    img_init = torch.load(cache / "init_img.pt", map_location="cpu", weights_only=False)
    txt_init = torch.load(cache / "init_txt.pt", map_location="cpu", weights_only=False)
    img_stu = torch.load(cache / "student_img.pt", map_location="cpu", weights_only=False)
    txt_stu = torch.load(cache / "student_txt.pt", map_location="cpu", weights_only=False)

    drift_img = compute_drift_stats(img_init, img_stu)
    drift_txt = compute_drift_stats(txt_init, txt_stu)
    drift_stats = {"image": drift_img, "text": drift_txt}
    with open(out / "drift_stats.json", "w") as f:
        json.dump(drift_stats, f, indent=2)
    log.info("  drift IMG  mean=%.4f  median=%.4f  p95=%.4f  max=%.4f",
             drift_img["mean"], drift_img["median"], drift_img["p95"], drift_img["max"])
    log.info("  drift TXT  mean=%.4f  median=%.4f  p95=%.4f  max=%.4f",
             drift_txt["mean"], drift_txt["median"], drift_txt["p95"], drift_txt["max"])

    log.info("[%s] scoring init / student ...", dataset)
    s_init = txt_init @ img_init.T
    s_stu = txt_stu @ img_stu.T

    ap_init = per_query_ap10(s_init, probe.item_IDs, probe.queries, probe.gt, k=10)
    ap_stu = per_query_ap10(s_stu, probe.item_IDs, probe.queries, probe.gt, k=10)

    qs = sorted(set(ap_init) & set(ap_stu))
    rows = []
    for q in qs:
        a = ap_init[q]
        b = ap_stu[q]
        rows.append({
            "query": q, "n_gold": a["n_gold"],
            "ap10_init": a["ap10"], "ap10_student": b["ap10"],
            "delta": b["ap10"] - a["ap10"],
        })

    with open(out / "per_query.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["query", "n_gold", "ap10_init", "ap10_student", "delta"])
        w.writeheader()
        w.writerows(rows)

    helped = sum(1 for r in rows if r["delta"] > 1e-6)
    hurt = sum(1 for r in rows if r["delta"] < -1e-6)
    same = sum(1 for r in rows if abs(r["delta"]) <= 1e-6)
    mean_init = sum(r["ap10_init"] for r in rows) / max(1, len(rows))
    mean_stu = sum(r["ap10_student"] for r in rows) / max(1, len(rows))

    log.info("  per-query: %d helped  /  %d hurt  /  %d unchanged   (n=%d)",
             helped, hurt, same, len(rows))
    log.info("  MAP@10 init=%.4f  student=%.4f  delta=%+.4f",
             mean_init, mean_stu, mean_stu - mean_init)

    rows_sorted = sorted(rows, key=lambda r: r["delta"])
    worst = rows_sorted[:n_worst]
    worst_dump = []
    for r in worst:
        q = r["query"]
        worst_dump.append({
            "query": q,
            "n_gold": r["n_gold"],
            "ap10_init": round(r["ap10_init"], 4),
            "ap10_student": round(r["ap10_student"], 4),
            "delta": round(r["delta"], 4),
            "init_top10":     ap_init[q]["top10"],
            "student_top10":  ap_stu[q]["top10"],
            "gold_ids":       sorted(d for d, r_ in probe.gt[q].items() if r_ > 0),
        })
    with open(out / "worst_hurt.json", "w") as f:
        json.dump(worst_dump, f, indent=2)

    # Free big tensors
    del img_init, img_stu, txt_init, txt_stu, s_init, s_stu, probe
    gc.collect()

    return {
        "dataset": dataset,
        "n_queries": len(rows),
        "helped": helped, "hurt": hurt, "unchanged": same,
        "mean_ap10_init": mean_init,
        "mean_ap10_student": mean_stu,
        "delta_map10": mean_stu - mean_init,
        "drift_img_p95": drift_img["p95"],
        "drift_txt_p95": drift_txt["p95"],
    }


def write_summary(results: list[dict], output_dir: Path) -> None:
    md = ["# Path 1 Drift Diagnostic — Summary", ""]
    md.append("Hypothesis under test: *Path 1 student drifted off the SigLIP-2 webli "
              "manifold on atlas/polyvore but not on fashion200k, which is why it "
              "regressed there. If true, Path 2's anchor loss is the right fix.*")
    md.append("")
    md.append("## Per-dataset results")
    md.append("")
    md.append("| dataset | MAP@10 init | MAP@10 student | delta | helped | hurt | unchanged | drift_img p95 | drift_txt p95 |")
    md.append("|---|---|---|---|---|---|---|---|---|")
    for r in results:
        md.append(f"| {r['dataset']} | {r['mean_ap10_init']:.4f} | "
                  f"{r['mean_ap10_student']:.4f} | {r['delta_map10']:+.4f} | "
                  f"{r['helped']} | {r['hurt']} | {r['unchanged']} | "
                  f"{r['drift_img_p95']:.4f} | {r['drift_txt_p95']:.4f} |")
    md.append("")

    md.append("## Verdict")
    md.append("")
    f200k = next((r for r in results if r["dataset"] == "fashion200k"), None)
    atlas = next((r for r in results if r["dataset"] == "atlas"), None)
    if f200k and atlas:
        atlas_drift_higher = (
            atlas["drift_img_p95"] > f200k["drift_img_p95"] * 1.2 or
            atlas["drift_txt_p95"] > f200k["drift_txt_p95"] * 1.2
        )
        if atlas_drift_higher:
            md.append("- atlas image/text p95 drift > 1.2× f200k drift → **drift hypothesis SUPPORTED**")
            md.append("- Recommendation: proceed to Path 2 (anchor loss + MarginMSE + hard negatives).")
        else:
            md.append("- atlas drift is NOT meaningfully higher than f200k drift → **drift hypothesis NOT supported**")
            md.append("- Recommendation: investigate failure mode in `atlas/worst_hurt.json` before launching Path 2.")
    md.append("")
    md.append("## Files")
    for r in results:
        md.append(f"- `{r['dataset']}/drift_stats.json`, `per_query.csv`, `worst_hurt.json`")

    with open(output_dir / "summary.md", "w") as f:
        f.write("\n".join(md) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--student-checkpoint", default="models/path1-full/best/student_state_dict.pt")
    p.add_argument("--datasets", nargs="+", default=["fashion200k", "atlas", "polyvore"])
    p.add_argument("--output-dir", default="diagnostics/path1_drift")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--n-worst", type=int, default=20)
    args = p.parse_args()

    output_dir = REPO / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    student_ckpt = REPO / args.student_checkpoint

    results = []
    for ds in args.datasets:
        r = diagnose_dataset(ds, student_ckpt, output_dir,
                             batch_size=args.batch_size, n_worst=args.n_worst)
        results.append(r)

    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    write_summary(results, output_dir)
    log.info("DONE. Summary at %s", output_dir / "summary.md")


if __name__ == "__main__":
    main()
