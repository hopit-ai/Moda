"""
v6 Step 4 — Final student training with prose teacher + image tower unfrozen.

Addresses all 3 root causes:
  RC1 (image tower): unfreezes SL2-B image last 1 block → fashion-discriminative clusters
  RC2 (data type): uses FashionIQ + GS-10M long queries (prose distribution)
  RC3 (teacher ceiling): prose teacher has fashion200k > FSL's 0.4551

Loss = InfoNCE(student) + λ_kl * KL(student || prose_teacher) + λ_anchor * anchor

The KL component aligns student to prose teacher's space.
The anchor component prevents regression on atlas/polyvore (preserves SL2-B base).

Usage:
    python scripts/v6/step4_train_v6_student.py
    python scripts/v6/step4_train_v6_student.py --run_tag v6a --epochs 3
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
from torch.utils.data import DataLoader
from tqdm import tqdm

REPO = Path(__file__).resolve().parents[2]
V6_DATA = REPO / "data" / "processed" / "v6"
V5_DATA = REPO / "data" / "processed" / "v5_multifield"
CKPT_DIR = REPO / "checkpoints" / "v6"
RESULTS_DIR = REPO / "results" / "v6"
LOGS_DIR = REPO / "logs"

sys.path.insert(0, str(REPO / "scripts" / "v5"))
from v5_eval_probe import EvalProbe
from step3_train_prose_teacher import ProsePairDataset, build_sl2b_with_image_unfrozen


def pick_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_prose_teacher(ckpt_path: Path, device: str):
    """Load prose teacher model for KL supervision."""
    import open_clip
    teacher, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-16-SigLIP2-384", pretrained="webli"
    )
    teacher = teacher.to(device).eval()
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt.get("model_state", ckpt)
    teacher.load_state_dict({k: v.to(device) for k, v in state.items()}, strict=False)
    for p in teacher.parameters():
        p.requires_grad = False
    print(f"  Prose teacher loaded from {ckpt_path}")
    return teacher


def combined_loss(
    student_text: torch.Tensor,   # (B, D) normalized
    student_img: torch.Tensor,    # (B, D) normalized
    teacher_text: torch.Tensor,   # (B, D) normalized, no grad
    teacher_img: torch.Tensor,    # (B, D) normalized, no grad
    logit_scale, logit_bias,
    lam_kl: float,
    lam_anchor: float,
    anchor_now: torch.Tensor,     # (A, D) normalized
    anchor_init: torch.Tensor,    # (A, D) normalized, frozen
    tau_kl: float = 0.05,
) -> tuple[torch.Tensor, dict]:
    scale = logit_scale.exp()
    logits = student_text @ student_img.T * scale
    if logit_bias is not None:
        logits = logits + logit_bias
    n = logits.shape[0]
    labels = torch.arange(n, device=logits.device)
    l_infonce = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2.0

    # KL toward prose teacher
    with torch.no_grad():
        t_scores = teacher_text @ teacher_img.T
    s_log = F.log_softmax(logits / tau_kl, dim=-1)
    t_p = F.softmax(t_scores / tau_kl, dim=-1)
    l_kl = F.kl_div(s_log, t_p, reduction="batchmean")

    # Anchor: preserve SL2-B init text geometry (protect atlas/polyvore)
    l_anchor = F.mse_loss(anchor_now, anchor_init)

    loss = l_infonce + lam_kl * l_kl + lam_anchor * l_anchor
    return loss, {"infonce": l_infonce.item(), "kl": l_kl.item(), "anchor": l_anchor.item()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prose_teacher_ckpt", type=Path,
                    default=CKPT_DIR / "prose_teacher_best_prose_teacher.pt")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=48)
    ap.add_argument("--lr_text", type=float, default=1e-6)
    ap.add_argument("--lr_image", type=float, default=5e-7)
    ap.add_argument("--lr_logit", type=float, default=1e-4)
    ap.add_argument("--text_blocks", type=int, default=2)
    ap.add_argument("--image_blocks", type=int, default=1)
    ap.add_argument("--lam_kl", type=float, default=0.5)
    ap.add_argument("--lam_anchor", type=float, default=0.3)
    ap.add_argument("--anchor_size", type=int, default=256)
    ap.add_argument("--probe_every", type=int, default=200)
    ap.add_argument("--ckpt_every", type=int, default=500)
    ap.add_argument("--device", default=None)
    ap.add_argument("--run_tag", default="v6a")
    args = ap.parse_args()

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    device = args.device or pick_device()
    print(f"Device: {device}")

    if not args.prose_teacher_ckpt.exists():
        sys.exit(f"ERROR: prose teacher not found at {args.prose_teacher_ckpt}. Run step3 first.")

    print("Building student (SL2-B, image tower partially unfrozen) ...")
    model, tokenizer, preprocess = build_sl2b_with_image_unfrozen(
        device, args.text_blocks, args.image_blocks
    )

    print("Loading prose teacher ...")
    teacher = load_prose_teacher(args.prose_teacher_ckpt, device)

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
        sys.exit("ERROR: No training data. Run step1 and step2 first.")

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=0, drop_last=True,
                        collate_fn=lambda b: b)
    total_steps = len(loader) * args.epochs
    print(f"  {len(dataset):,} pairs | {total_steps:,} total steps")

    # Anchor from SL2 text cache (preserve atlas/polyvore geometry)
    sl2_text_cache = torch.load(V5_DATA / "teacher_sl2_text_emb.pt",
                                map_location="cpu", weights_only=False)
    anchor_queries = list(sl2_text_cache.keys())[:args.anchor_size]
    anchor_init = torch.stack([sl2_text_cache[q] for q in anchor_queries]).to(device).float()
    anchor_init = F.normalize(anchor_init, dim=-1)

    # Param groups
    text_p, image_p, logit_p = [], [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "logit_scale" in n or "logit_bias" in n:
            logit_p.append(p)
        elif "visual" in n:
            image_p.append(p)
        else:
            text_p.append(p)
    optim = torch.optim.AdamW([
        {"params": text_p,  "lr": args.lr_text,  "weight_decay": 0.01},
        {"params": image_p, "lr": args.lr_image, "weight_decay": 0.01},
        {"params": logit_p, "lr": args.lr_logit, "weight_decay": 0.0},
    ])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=total_steps, eta_min=1e-7)

    print("Loading EvalProbe ...")
    probe = EvalProbe(device=device)
    print("Running initial probe ...")
    init_metrics = probe.probe(model, tokenizer)
    init_mean = probe.aggregate_score(init_metrics)
    print(f"  Initial mean MRR: {init_mean:.4f}")
    print(f"  {{{', '.join(f'{k}={v[\"mrr\"]:.4f}' for k, v in init_metrics.items())}}}")

    probe_log = (LOGS_DIR / f"v6_student_probes_{args.run_tag}.jsonl").open("w")
    train_log = (LOGS_DIR / f"v6_student_{args.run_tag}.jsonl").open("w")
    probe_log.write(json.dumps({"step": 0, "metrics": init_metrics, "mean_mrr": init_mean}) + "\n")

    best_mean = init_mean
    best_step = 0
    step = 0
    t0 = time.time()
    model.train()
    pbar = tqdm(total=total_steps, desc=f"v6_student_{args.run_tag}")

    for epoch in range(1, args.epochs + 1):
        for raw_batch in loader:
            queries = [item[0] for item in raw_batch]
            imgs = torch.stack([item[1] for item in raw_batch]).to(device)

            tokens = tokenizer(queries).to(device)
            s_text = F.normalize(model.encode_text(tokens), dim=-1)
            s_img = F.normalize(model.encode_image(imgs), dim=-1)

            with torch.no_grad():
                t_text = F.normalize(teacher.encode_text(tokens), dim=-1)
                t_img = F.normalize(teacher.encode_image(imgs), dim=-1)

            # Anchor: run student on anchor set
            tok_a = tokenizer(anchor_queries).to(device)
            anchor_now = F.normalize(model.encode_text(tok_a), dim=-1)

            loss, parts = combined_loss(
                s_text, s_img, t_text, t_img,
                model.logit_scale,
                model.logit_bias if hasattr(model, "logit_bias") else None,
                args.lam_kl, args.lam_anchor,
                anchor_now, anchor_init,
            )

            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0
            )
            optim.step()
            scheduler.step()

            train_log.write(json.dumps({
                "step": step, "epoch": epoch, "loss": loss.item(), **parts,
                "elapsed_s": time.time() - t0,
            }) + "\n")
            train_log.flush()

            pbar.update(1)
            pbar.set_postfix(ep=epoch, loss=f"{loss.item():.3f}",
                             infonce=f"{parts['infonce']:.3f}",
                             kl=f"{parts['kl']:.3f}", best=f"{best_mean:.4f}@{best_step}")
            step += 1

            if step % args.probe_every == 0:
                metrics = probe.probe(model, tokenizer)
                mean_mrr = probe.aggregate_score(metrics)
                probe_log.write(json.dumps({
                    "step": step, "metrics": metrics, "mean_mrr": mean_mrr,
                    "elapsed_s": time.time() - t0,
                }) + "\n")
                probe_log.flush()
                pbar.write(
                    f"\n[step {step}] mean={mean_mrr:.4f} "
                    f"f200k={metrics.get('fashion200k',{}).get('mrr',0):.4f} "
                    f"KAGL={metrics.get('KAGL',{}).get('mrr',0):.4f} "
                    f"atlas={metrics.get('atlas',{}).get('mrr',0):.4f} "
                    f"poly={metrics.get('polyvore',{}).get('mrr',0):.4f}"
                )
                if mean_mrr > best_mean:
                    best_mean = mean_mrr
                    best_step = step
                    torch.save({
                        "step": step, "epoch": epoch,
                        "metrics": metrics, "mean_mrr": mean_mrr,
                        "model_state": {n: p.detach().cpu()
                                        for n, p in model.named_parameters()
                                        if p.requires_grad},
                    }, CKPT_DIR / f"v6_student_best_{args.run_tag}.pt")

            if args.ckpt_every > 0 and step % args.ckpt_every == 0:
                torch.save({
                    "step": step,
                    "model_state": {n: p.detach().cpu()
                                    for n, p in model.named_parameters()
                                    if p.requires_grad},
                }, CKPT_DIR / f"v6_student_{args.run_tag}_step{step}.pt")

    pbar.close()
    probe_log.close()
    train_log.close()
    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s | best={best_mean:.4f}@step{best_step}")
    print(f"Checkpoint: checkpoints/v6/v6_student_best_{args.run_tag}.pt")


if __name__ == "__main__":
    main()
