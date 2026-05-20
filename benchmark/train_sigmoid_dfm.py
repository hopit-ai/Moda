"""
Path 4 — Sigmoid pair loss on DeepFashion-Multimodal (DFM) data.

KEY DIFFERENCES from all previous attempts:
  - Loss: SigLIP-native sigmoid pair loss (NOT InfoNCE, NOT distillation)
    * No in-batch negative dependency → works at batch size 1
    * Each (text, image) pair gets an independent binary cross-entropy
    * Uses pre-mined hard negatives to create explicit negative pairs
  - Data: DeepFashion-Multimodal (42K items, prose descriptions)
    * Domain-matched to fashion200k (avg 30 words, visual descriptions)
    * Zero item overlap with any eval dataset (verified)
  - No teacher model. Direct contrastive learning.
  - Image tower frozen. Only text tower trains (LoRA planned, full-params for smoke).

Loss formula (SigLIP sigmoid):
    For each query q_i with positive image p+ and K hard negatives p_k:
      L_pos = -log(σ(s(q_i, p+) * t))           # pull positive
      L_neg = Σ_k -log(σ(-s(q_i, p_k) * t))     # push negatives
    where s = cosine similarity, t = learned temperature (or fixed bias)

Usage:
    # 200-step smoke test
    .venv/bin/python benchmark/train_sigmoid_dfm.py \
        --max-steps 200 --probe-steps "100,200" \
        --output-dir models/path4-sigmoid-smoke

    # Scale up after smoke succeeds
    .venv/bin/python benchmark/train_sigmoid_dfm.py \
        --max-steps 2000 --probe-steps "200,500,1000,1500,2000" \
        --n-train 5000 --K 10 --batch-size 16 \
        --output-dir models/path4-sigmoid-full
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "benchmark"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("train-sigmoid")

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
HF_CACHE = REPO / "data" / "hf_cache"


# ---------------------------------------------------------------------------
# Data: sample from DeepFashion-Multimodal, mine hard negatives with SigLIP-2
# ---------------------------------------------------------------------------

def prepare_dfm_data(n_train: int, K: int, seed: int, cache_dir: Path) -> list[dict]:
    """Sample n_train items from DFM, encode them with SigLIP-2 zero-shot,
    mine K hard negatives per item. Cache to disk for reuse.
    """
    cache_file = cache_dir / f"dfm_sigmoid_n{n_train}_K{K}_seed{seed}.json"
    if cache_file.exists():
        log.info("[data] loading cached data from %s", cache_file)
        with open(cache_file) as f:
            return json.load(f)

    cache_dir.mkdir(parents=True, exist_ok=True)
    log.info("[data] preparing DFM data: n=%d, K=%d, seed=%d", n_train, K, seed)

    from datasets import load_dataset
    ds = load_dataset("Marqo/deepfashion-multimodal", cache_dir=str(HF_CACHE))["data"]

    rng = random.Random(seed)
    indices = list(range(len(ds)))
    rng.shuffle(indices)
    selected = indices[:n_train]

    texts = [str(ds[i]["text"]) for i in selected]
    item_ids = [str(ds[i]["item_ID"]) for i in selected]

    # Save images to disk for training
    img_dir = cache_dir / f"dfm_images_n{n_train}_seed{seed}"
    img_dir.mkdir(parents=True, exist_ok=True)
    image_paths = []
    for idx, i in enumerate(selected):
        img_path = img_dir / f"{idx:05d}.jpg"
        if not img_path.exists():
            ds[i]["image"].save(str(img_path))
        image_paths.append(str(img_path))
        if (idx + 1) % 200 == 0:
            log.info("[data] saved %d/%d images", idx + 1, n_train)

    log.info("[data] encoding %d items with SigLIP-2 for hard-neg mining...", n_train)
    import open_clip
    from PIL import Image

    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-16-SigLIP2-384", pretrained="webli", cache_dir=str(HF_CACHE),
    )
    tokenizer = open_clip.get_tokenizer("ViT-B-16-SigLIP2-384")
    model.eval().to(DEVICE)
    for p in model.parameters():
        p.requires_grad = False

    # Encode images
    img_embs = []
    batch_size = 32
    with torch.no_grad():
        for i in range(0, n_train, batch_size):
            batch_paths = image_paths[i:i + batch_size]
            imgs = torch.stack([preprocess(Image.open(p).convert("RGB")) for p in batch_paths]).to(DEVICE)
            feat = model.encode_image(imgs)
            feat = feat / feat.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            img_embs.append(feat.cpu())
    img_embs = torch.cat(img_embs, dim=0)  # [N, D]

    # Encode texts
    txt_embs = []
    with torch.no_grad():
        for i in range(0, n_train, batch_size):
            tokens = tokenizer(texts[i:i + batch_size]).to(DEVICE)
            feat = model.encode_text(tokens)
            feat = feat / feat.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            txt_embs.append(feat.cpu())
    txt_embs = torch.cat(txt_embs, dim=0)

    del model
    gc.collect()

    # Mine hard negatives: for each text, find top-(K+50) most similar images
    # excluding the positive, then sample K from the top-50 hardest
    log.info("[data] mining hard negatives...")
    scores = txt_embs @ img_embs.T  # [N, N]

    records = []
    for i in range(n_train):
        row_scores = scores[i].clone()
        row_scores[i] = -1.0  # mask out positive
        topk_vals, topk_idx = row_scores.topk(min(K + 50, n_train - 1))
        # Take K hardest (closest non-matching images)
        neg_indices = topk_idx[:K].tolist()
        records.append({
            "query": texts[i],
            "pos_image_path": image_paths[i],
            "neg_image_paths": [image_paths[j] for j in neg_indices],
            "pos_score": float(scores[i, i]),
            "hardest_neg_score": float(topk_vals[0]),
        })

    with open(cache_file, "w") as f:
        json.dump(records, f)
    log.info("[data] saved %d records to %s", len(records), cache_file)
    return records


# ---------------------------------------------------------------------------
# Loss: SigLIP sigmoid pair loss
# ---------------------------------------------------------------------------

def sigmoid_pair_loss(
    s_pos: torch.Tensor,
    s_neg: torch.Tensor,
    bias: float = -10.0,
    temperature: float = 10.0,
) -> torch.Tensor:
    """Sigmoid pair loss (SigLIP-style).

    s_pos: [B] cosine similarities for positive pairs
    s_neg: [B, K] cosine similarities for negative pairs
    bias: learnable or fixed bias (SigLIP paper uses -10)
    temperature: inverse temperature (SigLIP paper uses log_scale ~ 4.6 → t ≈ 100,
                 but we use a milder 10 for fine-tuning stability)

    L = -mean(log σ(t * s_pos + b)) - mean(log σ(-t * s_neg - b))
    """
    logit_pos = temperature * s_pos + bias    # [B]
    logit_neg = temperature * s_neg + bias    # [B, K]

    loss_pos = -F.logsigmoid(logit_pos).mean()
    loss_neg = -F.logsigmoid(-logit_neg).mean()

    return loss_pos + loss_neg


# ---------------------------------------------------------------------------
# Anchor loss (drift prevention)
# ---------------------------------------------------------------------------

def anchor_loss(txt_student: torch.Tensor, txt_init: torch.Tensor) -> torch.Tensor:
    """1 - cos(student_text, init_text). Text-only since image tower is frozen."""
    return (1.0 - (txt_student * txt_init).sum(dim=-1)).mean()


# ---------------------------------------------------------------------------
# Student forward
# ---------------------------------------------------------------------------

def student_forward(student, preprocess, tokenizer, batch: list[dict], device: str):
    """Forward B queries with (1 pos + K neg) images.

    Returns:
        s_pos: [B] cosine sim for positives
        s_neg: [B, K] cosine sim for negatives
        txt_emb: [B, D] L2-normed text embeddings
    """
    from PIL import Image
    queries = [r["query"] for r in batch]
    K = len(batch[0]["neg_image_paths"])
    B = len(batch)

    img_paths_flat = []
    for r in batch:
        img_paths_flat.append(r["pos_image_path"])
        img_paths_flat.extend(r["neg_image_paths"])

    images = [preprocess(Image.open(p).convert("RGB")) for p in img_paths_flat]
    img_tens = torch.stack(images).to(device)
    tokens = tokenizer(queries).to(device)

    with torch.no_grad():
        img_feat_flat = student.encode_image(img_tens)
    img_feat_flat = img_feat_flat / img_feat_flat.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    txt_feat = student.encode_text(tokens)
    txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    img_feat = img_feat_flat.view(B, K + 1, -1)
    sim = (txt_feat.unsqueeze(1) * img_feat).sum(dim=-1)  # [B, K+1]

    s_pos = sim[:, 0]    # [B]
    s_neg = sim[:, 1:]   # [B, K]

    return s_pos, s_neg, txt_feat


# ---------------------------------------------------------------------------
# Multi-dataset probe (reused from Path 2)
# ---------------------------------------------------------------------------

class MultiProbe:
    def __init__(self, datasets: list[str], corpus_size: int = 10000):
        from probe_fashion200k_10k import Fashion200kProbe
        self.datasets = datasets
        self.probes = {d: Fashion200kProbe(dataset=d, corpus_size=corpus_size, seed=42)
                       for d in datasets}

    def run(self, model, preprocess, tokenizer, device: str = DEVICE, batch_size: int = 64) -> dict:
        out: dict = {"per_dataset": {}}
        maps, r100s = [], []
        for d in self.datasets:
            log.info("[probe] running %s ...", d)
            m = self.probes[d].run(model, preprocess, tokenizer, device=device, batch_size=batch_size)
            out["per_dataset"][d] = {
                "MAP@10": float(m.get("MAP@10", 0)),
                "Recall@10": float(m.get("Recall@10", 0)),
                "Recall@100": float(m.get("Recall@100", 0)),
                "NDCG@10": float(m.get("NDCG@10", 0)),
            }
            maps.append(out["per_dataset"][d]["MAP@10"])
            r100s.append(out["per_dataset"][d]["Recall@100"])
            self.probes[d]._preprocessed_cache.clear()
            self.probes[d]._pil_images = None
            gc.collect()
            if torch.backends.mps.is_available():
                try:
                    torch.mps.empty_cache()
                except Exception:
                    pass
        out["mean_MAP@10"] = sum(maps) / len(maps)
        out["mean_Recall@100"] = sum(r100s) / len(r100s)
        return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-train", type=int, default=1000,
                   help="Number of DFM items to sample for training.")
    p.add_argument("-K", type=int, default=7,
                   help="Hard negatives per query.")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--grad-accum", type=int, default=2,
                   help="Effective batch = batch_size * grad_accum = 16")
    p.add_argument("--lr", type=float, default=2e-6,
                   help="Conservative LR. Recipe A collapsed at 1e-4; Paths 2/2.5 regressed at 1e-5/3e-5.")
    p.add_argument("--warmup-steps", type=int, default=20)
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--temperature", type=float, default=10.0,
                   help="Sigmoid temperature. SigLIP uses ~100 but we want stability.")
    p.add_argument("--sigmoid-bias", type=float, default=-10.0)
    p.add_argument("--w-anchor", type=float, default=0.1,
                   help="Anchor weight to prevent text-tower drift.")

    p.add_argument("--probe-steps", default="50,100,200")
    p.add_argument("--probe-datasets", default="fashion200k,atlas,polyvore")
    p.add_argument("--probe-corpus-size", type=int, default=5000,
                   help="5K screener for smoke test speed (known to be unreliable at higher fidelity).")

    p.add_argument("--output-dir", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-every", type=int, default=5)

    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "run_meta.json").write_text(json.dumps(vars(args), indent=2, default=str))

    log.info("=== Path 4: Sigmoid pair loss on DFM data ===")
    log.info("device=%s", DEVICE)

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # 1. Data
    data_cache = REPO / "data" / "processed" / "path4"
    records = prepare_dfm_data(args.n_train, args.K, args.seed, data_cache)
    log.info("[data] %d training records, %d negatives each", len(records), args.K)

    # 2. Student model — image tower frozen, text tower trainable
    import open_clip
    log.info("[student] loading ViT-B-16-SigLIP2-384 (webli)...")
    student, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-16-SigLIP2-384", pretrained="webli", cache_dir=str(HF_CACHE),
    )
    tokenizer = open_clip.get_tokenizer("ViT-B-16-SigLIP2-384")
    student.to(DEVICE)
    student.train()

    # Freeze image tower
    n_frozen = 0
    for name, param in student.named_parameters():
        if name.startswith("visual."):
            param.requires_grad = False
            n_frozen += param.numel()
    n_trainable = sum(p.numel() for p in student.parameters() if p.requires_grad)
    log.info("[student] frozen(image)=%d  trainable(text)=%d", n_frozen, n_trainable)

    # 3. Cache init text embeddings for anchor loss
    log.info("[anchor] caching init text embeddings...")
    init_txt_cache = {}
    student.eval()
    with torch.no_grad():
        all_queries = list(set(r["query"] for r in records))
        for i in range(0, len(all_queries), 32):
            batch_q = all_queries[i:i+32]
            tokens = tokenizer(batch_q).to(DEVICE)
            feat = student.encode_text(tokens)
            feat = feat / feat.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            for j, q in enumerate(batch_q):
                init_txt_cache[q] = feat[j].cpu()
    student.train()
    log.info("[anchor] cached %d init text embeddings", len(init_txt_cache))

    # 4. Optimizer
    optimizer = torch.optim.AdamW(
        [p for p in student.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=0.01, betas=(0.9, 0.98),
    )

    def lr_at_step(step: int) -> float:
        if step < args.warmup_steps:
            return args.lr * (step + 1) / args.warmup_steps
        progress = (step - args.warmup_steps) / max(1, args.max_steps - args.warmup_steps)
        return args.lr * 0.1 + (args.lr - args.lr * 0.1) * 0.5 * (1 + math.cos(math.pi * progress))

    # 5. Probe setup
    probe_steps = set()
    if args.probe_steps.strip():
        probe_steps = {int(s.strip()) for s in args.probe_steps.split(",") if s.strip()}
    probe_datasets = [d.strip() for d in args.probe_datasets.split(",") if d.strip()]
    multi_probe = MultiProbe(probe_datasets, corpus_size=args.probe_corpus_size) if probe_steps else None

    init_metrics = None
    if multi_probe:
        log.info("[probe] running INIT baseline...")
        init_metrics = multi_probe.run(student, preprocess, tokenizer,
                                       device=DEVICE, batch_size=64)
        student.train()
        for d, m in init_metrics["per_dataset"].items():
            log.info("[probe-init] %s  MAP@10=%.4f", d, m["MAP@10"])
        log.info("[probe-init] mean MAP@10=%.4f", init_metrics["mean_MAP@10"])

    # 6. Training loop
    micro_bs = max(1, args.batch_size // args.grad_accum)
    indices = list(range(len(records)))
    random.shuffle(indices)
    cursor = 0
    log_lines: list[dict] = []
    best_score = -1.0
    best_step = -1

    def next_batch() -> list[dict]:
        nonlocal cursor
        if cursor + micro_bs > len(indices):
            random.shuffle(indices)
            cursor = 0
        batch_idx = indices[cursor:cursor + micro_bs]
        cursor += micro_bs
        return [records[i] for i in batch_idx]

    t_start = time.time()
    for step in range(args.max_steps + 1):
        cur_lr = lr_at_step(step)
        for g in optimizer.param_groups:
            g["lr"] = cur_lr

        if step < args.max_steps:
            optimizer.zero_grad(set_to_none=True)
            sig_sum = anc_sum = 0.0

            for _ in range(args.grad_accum):
                batch = next_batch()
                s_pos, s_neg, txt_emb = student_forward(
                    student, preprocess, tokenizer, batch, DEVICE,
                )

                loss_sig = sigmoid_pair_loss(
                    s_pos, s_neg,
                    bias=args.sigmoid_bias,
                    temperature=args.temperature,
                )

                # Anchor: prevent text tower drift
                init_txts = torch.stack([init_txt_cache[r["query"]] for r in batch]).to(DEVICE)
                loss_anc = anchor_loss(txt_emb, init_txts)

                loss = (loss_sig + args.w_anchor * loss_anc) / args.grad_accum
                loss.backward()

                sig_sum += loss_sig.item()
                anc_sum += loss_anc.item()

            torch.nn.utils.clip_grad_norm_(
                [p for p in student.parameters() if p.requires_grad], max_norm=1.0,
            )
            optimizer.step()

            sig_avg = sig_sum / args.grad_accum
            anc_avg = anc_sum / args.grad_accum

            if step % args.log_every == 0:
                elapsed = time.time() - t_start
                log.info("[train] step=%d/%d  lr=%.2e  sig=%.4f  anc=%.4f  elapsed=%.1fs",
                         step, args.max_steps, cur_lr, sig_avg, anc_avg, elapsed)
            log_lines.append({
                "step": step, "lr": cur_lr, "sig": sig_avg, "anc": anc_avg,
                "wall": time.time() - t_start,
            })

        # Probe
        if multi_probe and step in probe_steps:
            log.info("[probe] step=%d ...", step)
            metrics = multi_probe.run(student, preprocess, tokenizer,
                                      device=DEVICE, batch_size=64)
            student.train()
            for d, m in metrics["per_dataset"].items():
                init_m = init_metrics["per_dataset"][d]
                log.info("[probe] step=%d  %s  MAP@10=%.4f (Δ%+.4f)",
                         step, d, m["MAP@10"], m["MAP@10"] - init_m["MAP@10"])
            log.info("[probe] step=%d  MEAN MAP@10=%.4f (Δ%+.4f)",
                     step, metrics["mean_MAP@10"],
                     metrics["mean_MAP@10"] - init_metrics["mean_MAP@10"])

            log_lines.append({"step": step, "probe": True, "metrics": metrics})

            ckpt_dir = out_dir / f"step_{step}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            torch.save(student.state_dict(), ckpt_dir / "student_state_dict.pt")

            score = metrics["mean_MAP@10"]
            if score > best_score:
                best_score = score
                best_step = step
                best_dir = out_dir / "best"
                best_dir.mkdir(parents=True, exist_ok=True)
                torch.save(student.state_dict(), best_dir / "student_state_dict.pt")
                log.info("[probe] NEW BEST @ step=%d  mean MAP@10=%.4f", step, score)

        with open(out_dir / "training_log.jsonl", "w") as f:
            for line in log_lines:
                f.write(json.dumps(line) + "\n")

    summary = {
        "init_metrics": init_metrics,
        "max_steps": args.max_steps,
        "best_step": best_step,
        "best_mean_map10": best_score,
        "wall_time_sec": time.time() - t_start,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    log.info("=== DONE ===")
    log.info("%s", json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
