"""Phase 13 — Residual Text Adapter on Frozen FSL.

After 16 failed attempts to modify FSL's weights, this takes a different approach:
keep FSL 100% frozen and learn a tiny residual adapter (768→256→768, ~400K params)
on top of its text encoder.

Key advantages:
  - Zero catastrophic forgetting (FSL weights never change)
  - Gated residual: output = normalize(fsl_text + sigmoid(gate) * adapter(fsl_text))
  - gate initialized to 0.0 → starts as pure FSL
  - Training entirely on cached embeddings → ~5 min per epoch
  - At inference: 203M + 0.4M = 203.4M params (negligible overhead)

Uses cached SO400M teacher embeddings (already computed in Phase 12).

Usage:
  # Stage 1: Cache FSL embeddings on training pairs
  python3 -u scripts/v3/phase13_residual_adapter.py cache-student

  # Stage 2: Train adapter (fast — all cached)
  python3 -u scripts/v3/phase13_residual_adapter.py train

  # Stage 3: Evaluate adapter on full 15K benchmarks
  python3 -u scripts/v3/phase13_residual_adapter.py eval --corpus-size 15000

  # All stages sequentially
  python3 -u scripts/v3/phase13_residual_adapter.py run-all
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import logging
import math
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("phase13")

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data" / "processed" / "v3_phase10_500k"
TEACHER_CACHE_DIR = REPO_ROOT / "data" / "processed" / "distillation_cache_so400m"
STUDENT_CACHE_DIR = REPO_ROOT / "data" / "processed" / "distillation_cache_fsl"
CHECKPOINT_DIR = REPO_ROOT / "checkpoints" / "v3_phase13"
RESULTS_DIR = REPO_ROOT / "results"

STUDENT_MODEL_HF = "hf-hub:Marqo/marqo-fashionSigLIP"

DEVICE = (
    torch.device("mps") if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
    else torch.device("cuda") if torch.cuda.is_available()
    else torch.device("cpu")
)


class ResidualAdapter(nn.Module):
    """Lightweight gated residual adapter for text embeddings.

    output = normalize(x + sigmoid(gate) * MLP(x))

    Gate initialized to 0 → starts as identity (pure FSL).
    """
    def __init__(self, dim: int = 768, hidden: int = 256, n_layers: int = 1):
        super().__init__()
        layers = []
        in_d = dim
        for i in range(n_layers):
            out_d = hidden if i < n_layers - 1 else dim
            layers.append(nn.Linear(in_d, hidden if i < n_layers else dim))
            if i < n_layers:
                layers.append(nn.GELU())
            in_d = hidden
        layers.append(nn.Linear(hidden, dim))
        self.mlp = nn.Sequential(*layers)
        self.gate = nn.Parameter(torch.tensor(-2.0))  # sigmoid(-2) ≈ 0.12

        self._init_weights()

    def _init_weights(self):
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.mlp(x)
        g = torch.sigmoid(self.gate)
        return F.normalize(x + g * residual, dim=-1)

    @property
    def effective_gate(self) -> float:
        return torch.sigmoid(self.gate).item()


# ── Stage 1: Cache FSL student embeddings ─────────────────────────────────────

def cache_student_embeddings():
    """Pre-compute FSL embeddings on the same pairs used for SO400M teacher cache."""
    import open_clip

    STUDENT_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    cache_meta_path = STUDENT_CACHE_DIR / "meta.json"
    if cache_meta_path.exists():
        meta = json.loads(cache_meta_path.read_text())
        if meta.get("model") == STUDENT_MODEL_HF and meta.get("n", 0) > 0:
            log.info("Student cache already exists with %d pairs. Skipping.", meta["n"])
            return

    with open(TEACHER_CACHE_DIR / "pairs.json") as f:
        pairs = json.load(f)
    N = len(pairs)
    log.info("Caching FSL embeddings for %d pairs...", N)

    log.info("Loading FSL model: %s ...", STUDENT_MODEL_HF)
    model, _, preprocess = open_clip.create_model_and_transforms(STUDENT_MODEL_HF)
    tokenizer = open_clip.get_tokenizer(STUDENT_MODEL_HF)
    model = model.eval().to(DEVICE)

    images_dir = DATA_DIR / "images"

    # Encode images
    log.info("Encoding images...")
    img_embs = []
    t0 = time.time()
    failed = 0

    for i in range(0, N, 64):
        batch_pairs = pairs[i:i + 64]
        imgs = []
        for p in batch_pairs:
            img_path = images_dir / Path(p["image_path"]).name
            try:
                img = preprocess(Image.open(img_path).convert("RGB"))
            except Exception:
                img = preprocess(Image.new("RGB", (224, 224)))
                failed += 1
            imgs.append(img)

        img_batch = torch.stack(imgs).to(DEVICE)
        with torch.no_grad():
            emb = model.encode_image(img_batch)
            emb = F.normalize(emb, dim=-1)
        img_embs.append(emb.cpu())
        del img_batch, emb
        if DEVICE.type == "mps":
            torch.mps.empty_cache()

        if i > 0 and (i // 64) % 50 == 0:
            elapsed = time.time() - t0
            speed = i / elapsed
            log.info("  Images: %d/%d (%.0f/s)", i, N, speed)

    img_embs = torch.cat(img_embs, dim=0)
    log.info("  Image encoding done: %s (%.0fs, %d failed)", img_embs.shape, time.time() - t0, failed)

    # Encode texts
    log.info("Encoding texts...")
    txt_embs = []
    t0 = time.time()

    texts = [p["text_used"] for p in pairs]
    for i in range(0, N, 256):
        batch_texts = texts[i:i + 256]
        tokens = tokenizer(batch_texts).to(DEVICE)
        with torch.no_grad():
            emb = model.encode_text(tokens)
            emb = F.normalize(emb, dim=-1)
        txt_embs.append(emb.cpu())
        del tokens, emb
        if DEVICE.type == "mps":
            torch.mps.empty_cache()

    txt_embs = torch.cat(txt_embs, dim=0)
    log.info("  Text encoding done: %s (%.0fs)", txt_embs.shape, time.time() - t0)

    # Save
    torch.save(img_embs, STUDENT_CACHE_DIR / "student_img_embs.pt")
    torch.save(txt_embs, STUDENT_CACHE_DIR / "student_txt_embs.pt")

    meta = {
        "n": N,
        "model": STUDENT_MODEL_HF,
        "embed_dim": int(img_embs.shape[1]),
        "n_failed_images": failed,
    }
    with open(cache_meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    log.info("Student cache saved: %s", STUDENT_CACHE_DIR)

    del model, img_embs, txt_embs
    gc.collect()
    if DEVICE.type == "mps":
        torch.mps.empty_cache()


# ── Stage 2: Train residual adapter ──────────────────────────────────────────

def train_adapter(
    hidden_dim: int = 256,
    lr: float = 1e-3,
    temperature: float = 2.0,
    batch_size: int = 256,
    max_epochs: int = 50,
    eval_every: int = 5,
    patience: int = 10,
    identity_lambda: float = 0.01,
    sweep: bool = False,
):
    """Train residual adapter on cached embeddings."""
    # Load teacher cache (SO400M)
    teacher_meta = json.loads((TEACHER_CACHE_DIR / "meta.json").read_text())
    teacher_img = torch.load(TEACHER_CACHE_DIR / "teacher_img_embs.pt", map_location="cpu", weights_only=True)
    teacher_txt = torch.load(TEACHER_CACHE_DIR / "teacher_txt_embs.pt", map_location="cpu", weights_only=True)
    log.info("Teacher: %s, %d pairs, dim=%d", teacher_meta["teacher"], teacher_meta["n"], teacher_meta["embed_dim"])

    # Load student cache (FSL)
    student_meta = json.loads((STUDENT_CACHE_DIR / "meta.json").read_text())
    student_img = torch.load(STUDENT_CACHE_DIR / "student_img_embs.pt", map_location="cpu", weights_only=True)
    student_txt = torch.load(STUDENT_CACHE_DIR / "student_txt_embs.pt", map_location="cpu", weights_only=True)
    N = student_meta["n"]
    student_dim = student_meta["embed_dim"]
    log.info("Student: FSL, %d pairs, dim=%d", N, student_dim)

    # Train/val split (90/10)
    rng = np.random.RandomState(42)
    indices = rng.permutation(N)
    n_val = max(500, N // 10)
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]
    log.info("Train: %d, Val: %d", len(train_idx), len(val_idx))

    if sweep:
        configs = [
            {"hidden": 128, "lr": 5e-4, "temp": 1.5, "id_lambda": 0.01, "tag": "small_cool"},
            {"hidden": 256, "lr": 1e-3, "temp": 2.0, "id_lambda": 0.01, "tag": "medium_default"},
            {"hidden": 256, "lr": 5e-4, "temp": 2.0, "id_lambda": 0.005, "tag": "medium_low_id"},
            {"hidden": 512, "lr": 1e-3, "temp": 2.0, "id_lambda": 0.01, "tag": "large_default"},
            {"hidden": 256, "lr": 1e-3, "temp": 4.0, "id_lambda": 0.01, "tag": "medium_hot"},
            {"hidden": 256, "lr": 2e-3, "temp": 2.0, "id_lambda": 0.0, "tag": "medium_no_id"},
        ]
    else:
        configs = [
            {"hidden": hidden_dim, "lr": lr, "temp": temperature,
             "id_lambda": identity_lambda, "tag": "single"},
        ]

    best_overall = {"val_metric": -1, "tag": None}
    all_results = []

    for cfg in configs:
        log.info("")
        log.info("=" * 60)
        log.info("Config: %s", cfg["tag"])
        log.info("  hidden=%d, lr=%.1e, temp=%.1f, id_lambda=%.3f",
                 cfg["hidden"], cfg["lr"], cfg["temp"], cfg["id_lambda"])
        log.info("=" * 60)

        result = _train_single_config(
            cfg, student_txt, student_img, teacher_txt, teacher_img,
            train_idx, val_idx, student_dim, N,
            batch_size=batch_size, max_epochs=max_epochs,
            eval_every=eval_every, patience=patience,
        )
        all_results.append(result)

        if result["best_val_metric"] > best_overall["val_metric"]:
            best_overall = {
                "val_metric": result["best_val_metric"],
                "tag": cfg["tag"],
                "config": cfg,
                "result": result,
            }
            log.info("  >>> New overall best: %s (val=%.4f)", cfg["tag"], result["best_val_metric"])

    # Save sweep results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "phase13_adapter_results.json", "w") as f:
        json.dump({
            "best": best_overall,
            "all_configs": [r["summary"] for r in all_results],
        }, f, indent=2, default=str)

    log.info("")
    log.info("=" * 60)
    log.info("ADAPTER TRAINING COMPLETE")
    log.info("=" * 60)
    log.info("  Best config: %s", best_overall["tag"])
    log.info("  Best val metric: %.4f", best_overall["val_metric"])
    log.info("=" * 60)


def _train_single_config(
    cfg, student_txt, student_img, teacher_txt, teacher_img,
    train_idx, val_idx, student_dim, N,
    batch_size, max_epochs, eval_every, patience,
):
    adapter = ResidualAdapter(dim=student_dim, hidden=cfg["hidden"]).to(DEVICE)
    n_params = sum(p.numel() for p in adapter.parameters())
    log.info("  Adapter params: %d (%.1fK)", n_params, n_params / 1e3)

    optimizer = torch.optim.AdamW(adapter.parameters(), lr=cfg["lr"], weight_decay=0.01)
    total_steps = (len(train_idx) // batch_size) * max_epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)

    save_dir = CHECKPOINT_DIR / cfg["tag"]
    save_dir.mkdir(parents=True, exist_ok=True)

    best_val = -1
    best_state = None
    patience_counter = 0
    eval_log = []
    temperature = cfg["temp"]
    id_lambda = cfg["id_lambda"]

    for epoch in range(1, max_epochs + 1):
        adapter.train()
        perm = torch.randperm(len(train_idx))
        epoch_loss = 0.0
        epoch_steps = 0
        t0 = time.time()

        for i in range(0, len(train_idx) - batch_size + 1, batch_size):
            idx = train_idx[perm[i:i + batch_size]]

            s_txt = student_txt[idx].to(DEVICE)
            s_img = student_img[idx].to(DEVICE)
            t_txt = teacher_txt[idx].to(DEVICE)
            t_img = teacher_img[idx].to(DEVICE)

            # Adapt student text embeddings
            adapted_txt = adapter(s_txt)

            # Similarity matrices
            student_sims = adapted_txt @ s_img.T
            teacher_sims = t_txt @ t_img.T

            # KL divergence on similarity distributions (ranking transfer)
            t_row = F.log_softmax(teacher_sims / temperature, dim=1)
            s_row = F.log_softmax(student_sims / temperature, dim=1)
            kl_row = F.kl_div(s_row, t_row, log_target=True, reduction="batchmean")

            t_col = F.log_softmax(teacher_sims / temperature, dim=0)
            s_col = F.log_softmax(student_sims / temperature, dim=0)
            kl_col = F.kl_div(s_col, t_col, log_target=True, reduction="batchmean")

            loss_kl = 0.5 * (kl_row + kl_col) * (temperature ** 2)

            # Identity regularization: don't stray too far from original FSL embeddings
            loss_id = F.mse_loss(adapted_txt, F.normalize(s_txt, dim=-1))

            loss = loss_kl + id_lambda * loss_id

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss_kl.item()
            epoch_steps += 1

        avg_loss = epoch_loss / max(epoch_steps, 1)
        elapsed = time.time() - t0

        # Periodic evaluation
        if epoch % eval_every == 0 or epoch == 1:
            val_metric = _eval_adapter_on_cache(
                adapter, student_txt, student_img, teacher_txt, teacher_img, val_idx
            )
            gate = adapter.effective_gate
            log.info(
                "  [E%d] loss=%.4f val_ranking=%.4f gate=%.3f lr=%.1e (%.1fs)",
                epoch, avg_loss, val_metric, gate, scheduler.get_last_lr()[0], elapsed,
            )

            eval_log.append({
                "epoch": epoch, "loss": avg_loss, "val_metric": val_metric,
                "gate": gate,
            })

            if val_metric > best_val:
                best_val = val_metric
                best_state = copy.deepcopy(adapter.state_dict())
                patience_counter = 0
                torch.save({
                    "adapter_state_dict": best_state,
                    "config": cfg,
                    "epoch": epoch,
                    "val_metric": val_metric,
                }, save_dir / "best.pt")
            else:
                patience_counter += 1

            if patience_counter >= patience:
                log.info("  Early stop at epoch %d (patience %d)", epoch, patience)
                break
        elif epoch % 5 == 0:
            log.info("  [E%d] loss=%.4f gate=%.3f (%.1fs)",
                     epoch, avg_loss, adapter.effective_gate, elapsed)

    return {
        "best_val_metric": best_val,
        "eval_log": eval_log,
        "n_params": sum(p.numel() for p in adapter.parameters()),
        "summary": {
            "tag": cfg["tag"],
            "hidden": cfg["hidden"],
            "lr": cfg["lr"],
            "temp": cfg["temp"],
            "id_lambda": cfg["id_lambda"],
            "best_val_metric": best_val,
            "best_epoch": eval_log[-1]["epoch"] if eval_log else 0,
            "final_gate": eval_log[-1]["gate"] if eval_log else 0,
        },
    }


def _eval_adapter_on_cache(adapter, student_txt, student_img, teacher_txt, teacher_img, val_idx):
    """Evaluate adapter by measuring ranking agreement with teacher on validation set."""
    adapter.eval()
    with torch.no_grad():
        s_txt = student_txt[val_idx].to(DEVICE)
        s_img = student_img[val_idx].to(DEVICE)
        t_txt = teacher_txt[val_idx].to(DEVICE)
        t_img = teacher_img[val_idx].to(DEVICE)

        adapted_txt = adapter(s_txt)

        student_sims = adapted_txt @ s_img.T
        teacher_sims = t_txt @ t_img.T

        # Ranking agreement: for each query, compare top-k rankings
        k = min(10, len(val_idx))
        agreement = 0.0
        n_queries = len(val_idx)

        for qi in range(n_queries):
            s_topk = student_sims[qi].topk(k).indices.tolist()
            t_topk = teacher_sims[qi].topk(k).indices.tolist()
            overlap = len(set(s_topk) & set(t_topk))
            agreement += overlap / k

        agreement /= n_queries

    adapter.train()
    return agreement


# ── Stage 3: Evaluate adapter on real benchmarks ─────────────────────────────

def eval_adapter(corpus_size: int = 15000, adapter_tag: str = "medium_default"):
    """Evaluate the best adapter on all 4 benchmarks."""
    import open_clip
    from datasets import load_dataset

    # Find best adapter
    best_path = None
    best_val = -1
    results_path = RESULTS_DIR / "phase13_adapter_results.json"
    if results_path.exists():
        results = json.loads(results_path.read_text())
        best_tag = results["best"]["tag"]
        best_path = CHECKPOINT_DIR / best_tag / "best.pt"
        log.info("Best adapter from sweep: %s", best_tag)
    else:
        best_path = CHECKPOINT_DIR / adapter_tag / "best.pt"

    if not best_path or not best_path.exists():
        log.error("No adapter checkpoint found at %s", best_path)
        return

    # Load adapter
    ckpt = torch.load(best_path, map_location=DEVICE, weights_only=False)
    cfg = ckpt["config"]
    adapter = ResidualAdapter(dim=768, hidden=cfg["hidden"]).to(DEVICE)
    adapter.load_state_dict(ckpt["adapter_state_dict"])
    adapter.eval()
    log.info("Loaded adapter: gate=%.3f, params=%dK",
             adapter.effective_gate,
             sum(p.numel() for p in adapter.parameters()) // 1000)

    # Load FSL model
    log.info("Loading FSL model...")
    model, _, preprocess = open_clip.create_model_and_transforms(STUDENT_MODEL_HF)
    tokenizer = open_clip.get_tokenizer(STUDENT_MODEL_HF)
    model = model.eval().to(DEVICE)

    BENCHMARKS = ["fashion200k", "atlas", "polyvore", "KAGL"]
    HF_DATASETS = {
        "fashion200k": "Marqo/fashion200k",
        "atlas": "Marqo/atlas",
        "polyvore": "Marqo/polyvore",
        "KAGL": "Marqo/KAGL",
    }

    all_results = {}

    for bm_name in BENCHMARKS:
        log.info("")
        log.info("Evaluating: %s (corpus=%d)", bm_name, corpus_size)

        ds = load_dataset(HF_DATASETS[bm_name], split="data")
        if corpus_size > 0 and corpus_size < len(ds):
            rng = np.random.RandomState(42)
            indices = rng.choice(len(ds), size=corpus_size, replace=False)
            ds = ds.select(sorted(indices.tolist()))
        log.info("  Corpus: %d items", len(ds))

        # Find text column
        text_col = None
        for col in ["text", "caption", "title", "description"]:
            if col in ds.column_names:
                text_col = col
                break
        if not text_col:
            log.warning("  No text column found in %s", bm_name)
            all_results[bm_name] = {"error": "no text column"}
            continue

        # Unique queries
        query_to_indices = defaultdict(set)
        for idx in range(len(ds)):
            text = ds[idx].get(text_col, "")
            if text:
                query_to_indices[text].add(idx)

        queries = list(query_to_indices.keys())
        if len(queries) > 2000:
            rng = np.random.RandomState(42)
            sel = rng.choice(len(queries), size=2000, replace=False)
            queries = [queries[i] for i in sel]
        log.info("  Queries: %d unique texts", len(queries))

        # Encode images (FSL, no adapter)
        import io
        img_embs = []
        with torch.no_grad():
            for i in range(0, len(ds), 64):
                batch_items = [ds[j] for j in range(i, min(i + 64, len(ds)))]
                imgs = []
                for item in batch_items:
                    img = item["image"]
                    if not isinstance(img, Image.Image):
                        img = Image.open(io.BytesIO(img)).convert("RGB")
                    else:
                        img = img.convert("RGB")
                    imgs.append(preprocess(img))
                batch = torch.stack(imgs).to(DEVICE)
                emb = model.encode_image(batch)
                emb = F.normalize(emb, dim=-1)
                img_embs.append(emb.cpu())
                del batch, emb
                if DEVICE.type == "mps":
                    torch.mps.empty_cache()
        img_embs = torch.cat(img_embs, dim=0)
        log.info("  Images encoded: %s", img_embs.shape)

        # Encode texts (FSL + adapter)
        txt_embs_raw = []
        txt_embs_adapted = []
        with torch.no_grad():
            for i in range(0, len(queries), 128):
                batch_texts = queries[i:i + 128]
                tokens = tokenizer(batch_texts).to(DEVICE)
                raw_emb = model.encode_text(tokens)
                raw_emb = F.normalize(raw_emb, dim=-1)
                adapted_emb = adapter(raw_emb)
                txt_embs_raw.append(raw_emb.cpu())
                txt_embs_adapted.append(adapted_emb.cpu())
                del tokens, raw_emb, adapted_emb
                if DEVICE.type == "mps":
                    torch.mps.empty_cache()
        txt_embs_raw = torch.cat(txt_embs_raw, dim=0)
        txt_embs_adapted = torch.cat(txt_embs_adapted, dim=0)

        # Compute metrics for both raw FSL and adapted
        gt = [query_to_indices[q] for q in queries]

        metrics_raw = _compute_all_metrics(txt_embs_raw, img_embs, gt, k=10)
        metrics_adapted = _compute_all_metrics(txt_embs_adapted, img_embs, gt, k=10)

        delta_map10 = metrics_adapted["map10"] - metrics_raw["map10"]
        delta_pct = delta_map10 / max(metrics_raw["map10"], 1e-8) * 100

        log.info("  FSL raw:     MAP@10=%.4f R@1=%.3f R@10=%.3f MRR=%.3f",
                 metrics_raw["map10"], metrics_raw["recall_1"],
                 metrics_raw["recall_10"], metrics_raw["mrr"])
        log.info("  FSL+adapter: MAP@10=%.4f R@1=%.3f R@10=%.3f MRR=%.3f",
                 metrics_adapted["map10"], metrics_adapted["recall_1"],
                 metrics_adapted["recall_10"], metrics_adapted["mrr"])
        log.info("  Delta:       MAP@10 %+.4f (%+.1f%%)", delta_map10, delta_pct)

        all_results[bm_name] = {
            "fsl_raw": metrics_raw,
            "fsl_adapted": metrics_adapted,
            "delta_map10": delta_map10,
            "delta_pct": delta_pct,
        }

    # Summary
    log.info("")
    log.info("=" * 70)
    log.info("Phase 13 — Residual Adapter Evaluation (%dK corpus)", corpus_size // 1000)
    log.info("=" * 70)
    log.info("| Benchmark | FSL raw | FSL+adapter | Delta | Direction |")
    log.info("|---|---:|---:|---:|---|")

    for bm_name in BENCHMARKS:
        if bm_name in all_results and "fsl_raw" in all_results[bm_name]:
            r = all_results[bm_name]
            direction = "BETTER" if r["delta_map10"] > 0.001 else ("SAME" if abs(r["delta_map10"]) < 0.001 else "WORSE")
            log.info("| %s | %.4f | %.4f | %+.1f%% | %s |",
                     bm_name, r["fsl_raw"]["map10"], r["fsl_adapted"]["map10"],
                     r["delta_pct"], direction)

    log.info("=" * 70)

    # Save
    with open(RESULTS_DIR / "phase13_eval_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    log.info("Results saved: %s", RESULTS_DIR / "phase13_eval_results.json")


def _compute_all_metrics(query_embs, doc_embs, gt, k=10):
    aps, recalls_1, recalls_10, prec_1, prec_10, mrrs = [], [], [], [], [], []
    for i in range(query_embs.shape[0]):
        sims = (query_embs[i:i+1] @ doc_embs.T).squeeze(0)
        _, top_indices = torch.topk(sims, min(k, sims.shape[0]))
        top_indices = top_indices.tolist()
        relevant = gt[i]
        n_rel = len(relevant)
        if n_rel == 0:
            continue
        hits = 0
        precision_sum = 0.0
        first_hit_rank = None
        for rank, idx in enumerate(top_indices[:k], 1):
            if idx in relevant:
                hits += 1
                precision_sum += hits / rank
                if first_hit_rank is None:
                    first_hit_rank = rank
        ap = precision_sum / min(k, n_rel) if n_rel > 0 else 0.0
        aps.append(ap)
        hit_at_1 = 1.0 if top_indices[0] in relevant else 0.0
        recalls_1.append(hit_at_1)
        hits_at_10 = sum(1 for idx in top_indices[:10] if idx in relevant)
        recalls_10.append(hits_at_10 / n_rel)
        prec_1.append(hit_at_1)
        prec_10.append(hits_at_10 / min(10, len(top_indices)))
        mrrs.append(1.0 / first_hit_rank if first_hit_rank else 0.0)
    return {
        "map10": float(np.mean(aps)) if aps else 0.0,
        "recall_1": float(np.mean(recalls_1)) if recalls_1 else 0.0,
        "recall_10": float(np.mean(recalls_10)) if recalls_10 else 0.0,
        "precision_1": float(np.mean(prec_1)) if prec_1 else 0.0,
        "precision_10": float(np.mean(prec_10)) if prec_10 else 0.0,
        "mrr": float(np.mean(mrrs)) if mrrs else 0.0,
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Phase 13: Residual adapter on frozen FSL")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("cache-student", help="Cache FSL embeddings")

    p_train = sub.add_parser("train", help="Train residual adapter")
    p_train.add_argument("--hidden-dim", type=int, default=256)
    p_train.add_argument("--lr", type=float, default=1e-3)
    p_train.add_argument("--temperature", type=float, default=2.0)
    p_train.add_argument("--batch-size", type=int, default=256)
    p_train.add_argument("--max-epochs", type=int, default=50)
    p_train.add_argument("--eval-every", type=int, default=5)
    p_train.add_argument("--patience", type=int, default=10)
    p_train.add_argument("--identity-lambda", type=float, default=0.01)
    p_train.add_argument("--sweep", action="store_true", help="Run hyperparameter sweep")

    p_eval = sub.add_parser("eval", help="Evaluate adapter on benchmarks")
    p_eval.add_argument("--corpus-size", type=int, default=15000)
    p_eval.add_argument("--adapter-tag", type=str, default="medium_default")

    p_all = sub.add_parser("run-all", help="Cache + sweep + eval")
    p_all.add_argument("--corpus-size", type=int, default=15000)

    args = parser.parse_args()

    if args.command == "cache-student":
        cache_student_embeddings()

    elif args.command == "train":
        train_adapter(
            hidden_dim=args.hidden_dim,
            lr=args.lr,
            temperature=args.temperature,
            batch_size=args.batch_size,
            max_epochs=args.max_epochs,
            eval_every=args.eval_every,
            patience=args.patience,
            identity_lambda=args.identity_lambda,
            sweep=args.sweep,
        )

    elif args.command == "eval":
        eval_adapter(corpus_size=args.corpus_size, adapter_tag=args.adapter_tag)

    elif args.command == "run-all":
        cache_student_embeddings()
        train_adapter(sweep=True)
        eval_adapter(corpus_size=args.corpus_size)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
