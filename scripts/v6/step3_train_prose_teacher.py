"""
v6 Step 3 — Train the prose teacher: SL2-B on FashionIQ with image tower unfrozen.

Root cause 3 fix: the fusion teacher's fashion200k score (~0.435) is below FSL's
0.4551. Pure distillation from it can never produce a student that beats FSL on
fashion200k. We need a teacher that already exceeds FSL on fashion200k.

This script fine-tunes SL2-B/16/384 on FashionIQ (77K prose pairs) with BOTH
text and image towers partially unfrozen. Direct InfoNCE supervision makes image
embeddings fashion-discriminative — the same mechanism that gives FSL its edge.

Expected: prose teacher fashion200k MRR ~0.50+ (vs FSL 0.4551, SL2-B 0.4145).

Architecture choices:
  - Unfreeze: text last 2 blocks + image last 1 block (~14M text + 7M image = 21M)
  - Loss: InfoNCE (standard contrastive) — no distillation, direct supervision
  - Data: FashionIQ + GS-10M long queries (both prose-distribution)
  - LR: 2e-6 text, 1e-6 image (image tower needs more conservative updates)
  - Batch: 64 pairs (as large as MPS VRAM allows)

Training is ~15x slower than cached-image runs (~15 sec/step instead of ~1 sec).
With 77K pairs, 2 epochs = ~2400 steps ≈ 10h on MPS.

Usage:
    python scripts/v6/step3_train_prose_teacher.py
    python scripts/v6/step3_train_prose_teacher.py --epochs 2 --batch_size 48
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

REPO = Path(__file__).resolve().parents[2]
V6_DATA = REPO / "data" / "processed" / "v6"
CKPT_DIR = REPO / "checkpoints" / "v6"
RESULTS_DIR = REPO / "results" / "v6"
LOGS_DIR = REPO / "logs"

sys.path.insert(0, str(REPO / "scripts" / "v5"))
from v5_eval_probe import EvalProbe


def pick_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class ProsePairDataset(Dataset):
    """Dataset that loads (query_text, image_path) pairs and returns live tensors."""

    def __init__(self, jsonl_paths: list[Path], img_dirs: list[Path], preprocess):
        self.preprocess = preprocess
        self.records = []
        for jpath, idir in zip(jsonl_paths, img_dirs):
            if not jpath.exists():
                print(f"  WARNING: {jpath} not found, skipping")
                continue
            for line in jpath.open():
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                img_path = idir / r["image_file"]
                if img_path.exists():
                    self.records.append({"query": r["query"], "img_path": img_path})
        print(f"  ProsePairDataset: {len(self.records):,} valid pairs")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        try:
            img = Image.open(r["img_path"]).convert("RGB")
            img_tensor = self.preprocess(img)
        except Exception:
            img_tensor = torch.zeros(3, 384, 384)
        return r["query"], img_tensor


def build_sl2b_with_image_unfrozen(device: str, text_blocks: int = 2, image_blocks: int = 1):
    """SL2-B with last N text blocks + last M image blocks unfrozen."""
    import open_clip
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-16-SigLIP2-384", pretrained="webli"
    )
    tokenizer = open_clip.get_tokenizer("ViT-B-16-SigLIP2-384")
    model = model.to(device)

    # Freeze everything
    for p in model.parameters():
        p.requires_grad = False

    # Unfreeze last N text blocks + ln_final
    text_tf = model.text.transformer
    n_text = len(text_tf.resblocks)
    for i in range(max(0, n_text - text_blocks), n_text):
        for p in text_tf.resblocks[i].parameters():
            p.requires_grad = True
    if hasattr(model.text, "ln_final"):
        for p in model.text.ln_final.parameters():
            p.requires_grad = True

    # Unfreeze last M image blocks + image ln_post
    vis_tf = model.visual.transformer
    n_vis = len(vis_tf.resblocks)
    for i in range(max(0, n_vis - image_blocks), n_vis):
        for p in vis_tf.resblocks[i].parameters():
            p.requires_grad = True
    if hasattr(model.visual, "ln_post"):
        for p in model.visual.ln_post.parameters():
            p.requires_grad = True

    # Unfreeze logit scale/bias
    if hasattr(model, "logit_scale"):
        model.logit_scale.requires_grad = True
    if hasattr(model, "logit_bias"):
        model.logit_bias.requires_grad = True

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"  Trainable: {n_train:.1f}M params "
          f"(text_last{text_blocks}_blocks + image_last{image_blocks}_block)")
    return model, tokenizer, preprocess


def infonce_loss(text_emb: torch.Tensor, img_emb: torch.Tensor,
                 logit_scale, logit_bias=None) -> torch.Tensor:
    """Standard symmetric InfoNCE (SigLIP-style with bias)."""
    scale = logit_scale.exp()
    logits = text_emb @ img_emb.T * scale
    if logit_bias is not None:
        logits = logits + logit_bias
    n = logits.shape[0]
    labels = torch.arange(n, device=logits.device)
    loss_t = F.cross_entropy(logits, labels)
    loss_i = F.cross_entropy(logits.T, labels)
    return (loss_t + loss_i) / 2.0


def collate_fn(batch, tokenizer, device):
    queries, imgs = zip(*batch)
    tokens = tokenizer(list(queries)).to(device)
    imgs = torch.stack(imgs).to(device)
    return tokens, imgs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch_size", type=int, default=48)
    ap.add_argument("--lr_text", type=float, default=2e-6)
    ap.add_argument("--lr_image", type=float, default=1e-6)
    ap.add_argument("--lr_logit", type=float, default=1e-4)
    ap.add_argument("--text_blocks", type=int, default=2,
                    help="Unfreeze last N text transformer blocks")
    ap.add_argument("--image_blocks", type=int, default=1,
                    help="Unfreeze last N image transformer blocks")
    ap.add_argument("--probe_every", type=int, default=200)
    ap.add_argument("--ckpt_every", type=int, default=500)
    ap.add_argument("--device", default=None)
    ap.add_argument("--run_tag", default="prose_teacher")
    args = ap.parse_args()

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    device = args.device or pick_device()
    print(f"Device: {device}")

    print("Building model ...")
    model, tokenizer, preprocess = build_sl2b_with_image_unfrozen(
        device, args.text_blocks, args.image_blocks
    )

    print("Loading dataset ...")
    jsonl_paths = [
        V6_DATA / "pairs_fashioniq.jsonl",
        V6_DATA / "pairs_gs10m_long_query.jsonl",
    ]
    img_dirs = [
        V6_DATA / "fashioniq_images",
        V6_DATA / "images",
    ]
    dataset = ProsePairDataset(jsonl_paths, img_dirs, preprocess)
    if len(dataset) == 0:
        sys.exit("ERROR: No training data found. Run step1 and step2 first.")

    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=0,   # MPS works best with num_workers=0
        drop_last=True,
        collate_fn=lambda b: b,
    )
    total_steps = len(loader) * args.epochs
    print(f"  {len(dataset):,} pairs | {len(loader)} batches/epoch | {total_steps:,} total steps")

    # Separate param groups for different LRs
    text_params, image_params, logit_params = [], [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "logit_scale" in n or "logit_bias" in n:
            logit_params.append(p)
        elif "visual" in n:
            image_params.append(p)
        else:
            text_params.append(p)

    optim = torch.optim.AdamW([
        {"params": text_params,  "lr": args.lr_text,  "weight_decay": 0.01},
        {"params": image_params, "lr": args.lr_image, "weight_decay": 0.01},
        {"params": logit_params, "lr": args.lr_logit, "weight_decay": 0.0},
    ])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=total_steps, eta_min=1e-7)

    print("Loading EvalProbe ...")
    probe = EvalProbe(device=device)

    print("Running initial probe ...")
    init_metrics = probe.probe(model, tokenizer)
    init_mean = probe.aggregate_score(init_metrics)
    print(f"  Initial MRR: {{{', '.join(f'{k}={v[\"mrr\"]:.4f}' for k,v in init_metrics.items())}}}")
    print(f"  Initial mean: {init_mean:.4f}")

    probe_log_path = LOGS_DIR / f"v6_prose_teacher_probes_{args.run_tag}.jsonl"
    train_log_path = LOGS_DIR / f"v6_prose_teacher_{args.run_tag}.jsonl"
    probe_log = probe_log_path.open("w")
    train_log = train_log_path.open("w")
    probe_log.write(json.dumps({"step": 0, "metrics": init_metrics, "mean_mrr": init_mean}) + "\n")

    best_mean = init_mean
    best_step = 0
    step = 0
    t0 = time.time()

    model.train()
    pbar = tqdm(total=total_steps, desc="prose_teacher")

    for epoch in range(1, args.epochs + 1):
        for raw_batch in loader:
            queries = [item[0] for item in raw_batch]
            imgs = torch.stack([item[1] for item in raw_batch]).to(device)

            tokens = tokenizer(queries).to(device)
            text_emb = F.normalize(model.encode_text(tokens), dim=-1)
            img_emb = F.normalize(model.encode_image(imgs), dim=-1)

            loss = infonce_loss(text_emb, img_emb,
                                model.logit_scale,
                                model.logit_bias if hasattr(model, "logit_bias") else None)

            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0
            )
            optim.step()
            scheduler.step()

            train_log.write(json.dumps({
                "step": step, "epoch": epoch, "loss": loss.item(),
                "elapsed_s": time.time() - t0,
            }) + "\n")
            train_log.flush()

            pbar.update(1)
            pbar.set_postfix(ep=epoch, loss=f"{loss.item():.3f}", best=f"{best_mean:.4f}@{best_step}")
            step += 1

            if step % args.probe_every == 0:
                metrics = probe.probe(model, tokenizer)
                mean_mrr = probe.aggregate_score(metrics)
                probe_log.write(json.dumps({
                    "step": step, "metrics": metrics, "mean_mrr": mean_mrr,
                    "elapsed_s": time.time() - t0,
                }) + "\n")
                probe_log.flush()
                pbar.write(f"\n[step {step}] mean_mrr={mean_mrr:.4f} "
                           f"f200k={metrics.get('fashion200k',{}).get('mrr',0):.4f} "
                           f"KAGL={metrics.get('KAGL',{}).get('mrr',0):.4f} "
                           f"atlas={metrics.get('atlas',{}).get('mrr',0):.4f}")
                if mean_mrr > best_mean:
                    best_mean = mean_mrr
                    best_step = step
                    torch.save({
                        "step": step, "epoch": epoch,
                        "metrics": metrics, "mean_mrr": mean_mrr,
                        "model_state": {n: p.detach().cpu()
                                        for n, p in model.named_parameters()
                                        if p.requires_grad},
                    }, CKPT_DIR / f"prose_teacher_best_{args.run_tag}.pt")

            if args.ckpt_every > 0 and step % args.ckpt_every == 0:
                torch.save({
                    "step": step,
                    "model_state": {n: p.detach().cpu()
                                    for n, p in model.named_parameters()
                                    if p.requires_grad},
                }, CKPT_DIR / f"prose_teacher_{args.run_tag}_step{step}.pt")

    pbar.close()
    probe_log.close()
    train_log.close()
    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s ({step} steps, {step/elapsed:.2f} step/s)")
    print(f"Best checkpoint: prose_teacher_best_{args.run_tag}.pt @ step {best_step}, mean_mrr={best_mean:.4f}")


if __name__ == "__main__":
    main()
