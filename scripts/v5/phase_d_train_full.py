"""
Phase D — full v5 training with in-loop eval probe and best-checkpoint selection.

Same training mechanics as Phase C smoke (grouped GCL + anchor + fusion KL,
all reading from cached embeddings). Differences from smoke:

  - Runs on the FULL labeled dataset for `--epochs` epochs (default 3)
  - Calls EvalProbe every `--probe_every` steps on all 4 Marqo benchmarks
  - Tracks best checkpoint by mean MRR across the 4 benchmarks
  - Hard early stop if any benchmark regresses >2% from its INITIAL value
  - Logs per-step training loss + per-probe benchmark metrics to JSONL

Usage:
    python scripts/v5/phase_d_train_full.py
    python scripts/v5/phase_d_train_full.py --epochs 3 --probe_every 500
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from v5_dataset import V5Dataset
from v5_eval_probe import EvalProbe
from v5_loss import (
    anchor_text_loss, fusion_kl_loss, get_loss_coefficients, grouped_gcl_loss,
)
from v5_model import build_fsl_student, build_student, count_trainable, trainable_parameter_groups

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data" / "processed" / "v5_multifield"
LOGS = REPO / "logs"
CKPT_DIR = REPO / "checkpoints" / "v5"
RESULTS = REPO / "results" / "v5"


def pick_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=Path, default=None,
                    help="Defaults to pairs_50k_labeled.jsonl")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--max_steps", type=int, default=0,
                    help="Hard cap on steps; 0 = full epochs")
    ap.add_argument("--batch_K", type=int, default=8)
    ap.add_argument("--batch_N", type=int, default=16)
    ap.add_argument("--probe_every", type=int, default=500)
    ap.add_argument("--ckpt_every", type=int, default=500)
    ap.add_argument("--anchor_size", type=int, default=256)
    ap.add_argument("--regression_threshold", type=float, default=0.02,
                    help="Halt if any benchmark drops by this fraction from initial")
    ap.add_argument("--regression_patience", type=int, default=3,
                    help="Halt only if regressed for this many consecutive probes")
    ap.add_argument("--device", default=None)
    # ─── Iteration 2 recipe controls ────────────────────────────────
    ap.add_argument("--lr", type=float, default=5e-6,
                    help="Peak LR for text-tower params (default 5e-6, v2 uses 1e-6)")
    ap.add_argument("--lr_logit", type=float, default=1e-4,
                    help="LR for logit_scale + logit_bias")
    ap.add_argument("--use_multifield", type=int, default=1,
                    help="1 = 0.6q+0.3t+0.1c (v1), 0 = query only (v2)")
    ap.add_argument("--anchor_lambda_init", type=float, default=0.5)
    ap.add_argument("--anchor_lambda_final", type=float, default=0.1)
    ap.add_argument("--kl_lambda_init", type=float, default=0.3)
    ap.add_argument("--kl_lambda_final", type=float, default=0.1)
    ap.add_argument("--lambda_warmup_steps", type=int, default=500)
    ap.add_argument("--lambda_decay_end_steps", type=int, default=5000)
    ap.add_argument("--run_tag", type=str, default="v1",
                    help="Tag for log/ckpt naming (e.g. v1, v2)")
    ap.add_argument("--image_index_path", type=Path, default=None,
                    help="Override student_image_index.json path (for v2 augmented index)")
    ap.add_argument("--scope", default="text_4blocks",
                    choices=["text_4blocks", "text_1block", "heads_only"],
                    help="Trainable scope: text_4blocks (v1/v2), text_1block (v3), heads_only")
    ap.add_argument("--gcl_lambda", type=float, default=1.0,
                    help="GCL loss coefficient. Set 0 for pure-distillation iteration 3.")
    ap.add_argument("--sl2_text_cache_override", type=Path, default=None,
                    help="Path to a non-default SL2 teacher text cache (e.g. SL2-L for v4-alt)")
    ap.add_argument("--sl2_img_cache_override", type=Path, default=None,
                    help="Path to a non-default SL2 teacher image cache (e.g. SL2-L for v4-alt). "
                         "If None, the script reuses the student image cache as SL2-B teacher.")
    ap.add_argument("--fsl_student", action="store_true",
                    help="Use Marqo-FashionSigLIP as student instead of SL2-B. "
                         "Swaps image caches so FSL images are the student space and SL2-B images "
                         "become the teacher. KL teacher is SL2-only (fsl_weight=0). "
                         "Requires v5_eval_cache_fsl/ to exist (run phase_a_cache_fsl_eval.py).")
    ap.add_argument("--kl_fsl_weight", type=float, default=0.5,
                    help="FSL component weight in KL teacher. 0.5=fusion, 0.0=SL2-only "
                         "(automatically set to 0.0 when --fsl_student).")
    args = ap.parse_args()

    if args.fsl_student:
        args.kl_fsl_weight = 0.0  # FSL is the student — can't also be the teacher

    LOGS.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    tag = args.run_tag
    log_path = LOGS / f"v5_phase_d_{tag}.jsonl"
    probe_log_path = LOGS / f"v5_phase_d_probes_{tag}.jsonl"
    final_path = RESULTS / f"phase_d_final_{tag}.json"

    # ─── data path resolution ────────────────────────────────────────────
    labeled = DATA / "pairs_50k_labeled.jsonl"
    unlabeled = DATA / "pairs_50k.jsonl"
    pairs_path = args.pairs or (labeled if labeled.exists() else unlabeled)
    n_lines = sum(1 for _ in pairs_path.open())
    print(f"Pairs file: {pairs_path} ({n_lines:,} records)")

    device = args.device or pick_device()
    print(f"Device: {device}")

    # ─── caches ──────────────────────────────────────────────────────────
    print("Loading caches into RAM ...")
    fsl_img_cache_raw = torch.load(DATA / "teacher_fsl_img_emb.pt", map_location="cpu")
    sl2b_img_cache_raw = torch.load(DATA / "student_image_emb.pt", map_location="cpu")
    fsl_text_cache = torch.load(DATA / "teacher_fsl_text_emb.pt", map_location="cpu",
                                 weights_only=False)
    sl2_text_path = args.sl2_text_cache_override or (DATA / "teacher_sl2_text_emb.pt")
    sl2_text_cache = torch.load(sl2_text_path, map_location="cpu", weights_only=False)

    if args.fsl_student:
        # Student runs in FSL image space; SL2-B becomes the teacher image cache
        student_img_cache = fsl_img_cache_raw.to(device)
        fsl_img_cache = fsl_img_cache_raw.to(device)   # same tensor; not used as teacher KL
        sl2_img_cache = sl2b_img_cache_raw.to(device)  # SL2-B = KL teacher images
        print("  FSL-student mode: student_img=FSL, teacher_img=SL2-B")
    else:
        student_img_cache = sl2b_img_cache_raw.to(device)
        fsl_img_cache = fsl_img_cache_raw.to(device)
        if args.sl2_img_cache_override:
            sl2_img_cache = torch.load(args.sl2_img_cache_override, map_location="cpu").to(device)
            print(f"  sl2_img_cache (override): {tuple(sl2_img_cache.shape)} {sl2_img_cache.dtype}")
        else:
            sl2_img_cache = None  # falls back to reusing student_img_cache as SL2-B teacher image

    print(f"  student_img_cache: {tuple(student_img_cache.shape)} {student_img_cache.dtype}")
    print(f"  fsl_img_cache:     {tuple(fsl_img_cache.shape)} {fsl_img_cache.dtype}")
    print(f"  fsl_text_cache:    {len(fsl_text_cache):,} queries")
    print(f"  sl2_text_cache:    {len(sl2_text_cache):,} queries")

    # ─── model ───────────────────────────────────────────────────────────
    if args.fsl_student:
        print(f"Building FSL student (scope={args.scope}) ...")
        model, tokenizer = build_fsl_student(device=device, scope=args.scope)
    else:
        print(f"Building SL2-B student (scope={args.scope}) ...")
        model, tokenizer = build_student(device=device, scope=args.scope)
    counts = count_trainable(model)
    print(f"  trainable: {counts['total']/1e6:.1f}M params  (breakdown: {counts})")

    # ─── dataset ─────────────────────────────────────────────────────────
    image_index_path = args.image_index_path or (DATA / "student_image_index.json")
    ds = V5Dataset(pairs_path, image_index_path,
                   K=args.batch_K, N=args.batch_N, min_products_per_query=2)
    stats = ds.stats()
    print(f"  dataset: {stats}")
    batches_per_epoch = stats["batches_per_epoch"]

    if args.max_steps:
        total_steps = args.max_steps
    else:
        total_steps = batches_per_epoch * args.epochs
    print(f"  total steps planned: {total_steps:,}")

    # Anchor set — use FSL init embeddings when FSL is student (anchor = don't regress FSL)
    if args.fsl_student:
        anchor_queries = list(fsl_text_cache.keys())[: args.anchor_size]
        anchor_init = torch.stack([fsl_text_cache[q] for q in anchor_queries]).to(device).float()
    else:
        anchor_queries = list(sl2_text_cache.keys())[: args.anchor_size]
        anchor_init = torch.stack([sl2_text_cache[q] for q in anchor_queries]).to(device).float()
    anchor_init = F.normalize(anchor_init, dim=-1)

    # ─── eval probe ──────────────────────────────────────────────────────
    print("Loading EvalProbe ...")
    if args.fsl_student:
        fsl_eval_cache = REPO / "data" / "processed" / "v5_eval_cache_fsl"
        if not fsl_eval_cache.exists():
            sys.exit("ERROR: FSL eval cache missing. Run phase_a_cache_fsl_eval.py first.")
        probe = EvalProbe(cache_dir=fsl_eval_cache, device=device)
    else:
        probe = EvalProbe(device=device)
    if not probe.benchmarks:
        sys.exit("ERROR: no eval caches found. Run v5_eval_probe.py --build_caches first.")

    # Initial probe — establishes "no regression" baseline
    print("Running initial probe (step 0) ...")
    init_metrics = probe.probe(model, tokenizer)
    init_mrr = {b: m["mrr"] for b, m in init_metrics.items()}
    print(f"  initial MRR: {init_mrr}")
    print(f"  initial mean: {probe.aggregate_score(init_metrics):.4f}")

    # ─── optimizer (LR overrides from CLI) ───────────────────────────────
    param_groups = trainable_parameter_groups(model)
    # Override LRs with CLI values
    for g in param_groups:
        if any("logit_scale" in repr(type(p)) or p.numel() < 5 for p in g["params"]):
            g["lr"] = args.lr_logit
        else:
            g["lr"] = args.lr
    optim = torch.optim.AdamW(param_groups)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optim, lr_lambda=lambda step: max(0.05, 1.0 - step / max(1, total_steps))
    )

    def get_lambdas(step: int) -> tuple[float, float]:
        """Linear warmup→decay schedule for anchor and KL coefficients (per CLI)."""
        if step < args.lambda_warmup_steps:
            t = 0.0
        elif step < args.lambda_decay_end_steps:
            t = (step - args.lambda_warmup_steps) / max(
                1, args.lambda_decay_end_steps - args.lambda_warmup_steps
            )
        else:
            t = 1.0
        anchor = args.anchor_lambda_init + t * (args.anchor_lambda_final - args.anchor_lambda_init)
        kl = args.kl_lambda_init + t * (args.kl_lambda_final - args.kl_lambda_init)
        return anchor, kl

    # ─── training loop ───────────────────────────────────────────────────
    train_log = log_path.open("w")
    probe_log = probe_log_path.open("w")
    probe_log.write(json.dumps({"step": 0, "metrics": init_metrics,
                                 "mean_mrr": probe.aggregate_score(init_metrics)}) + "\n")
    probe_log.flush()

    best_mean = probe.aggregate_score(init_metrics)
    best_step = 0
    best_metrics = init_metrics
    consecutive_regressions = 0
    halt_reason = None

    model.train()
    step = 0
    t_start = time.time()
    pbar = tqdm(total=total_steps, desc="phase_d")

    for epoch in range(1, args.epochs + 1):
        if step >= total_steps or halt_reason:
            break
        for b in ds.iter_batches(shuffle=True):
            if step >= total_steps or halt_reason:
                break

            tok_q = tokenizer(b.query).to(device)
            text_q = model.encode_text(tok_q)
            if args.use_multifield:
                tok_t = tokenizer(b.title).to(device)
                tok_c = tokenizer(b.category_l2).to(device)
                text_t = model.encode_text(tok_t)
                text_c = model.encode_text(tok_c)
                text_multi = F.normalize(0.6 * text_q + 0.3 * text_t + 0.1 * text_c, dim=-1)
            else:
                # v2 recipe: query-only positive (eliminates title-distribution drift)
                text_multi = F.normalize(text_q, dim=-1)

            img_emb = F.normalize(student_img_cache[b.image_idx].float(), dim=-1)
            cos_scores = text_multi @ img_emb.T
            scale = model.logit_scale.exp()
            bias = model.logit_bias if hasattr(model, "logit_bias") else 0.0
            scores = cos_scores * scale + bias

            l_gcl = grouped_gcl_loss(
                scores, b.query_idx.to(device), b.score_linear.to(device), b.K
            )

            # Skip anchor loss entirely when both lambdas are 0 (e.g., v4-alt).
            # This also avoids dim-mismatch when sl2_text_cache_override has different
            # output dim than student (SL2-L=1024 vs SL2-B=768).
            if args.anchor_lambda_init == 0.0 and args.anchor_lambda_final == 0.0:
                l_anchor = torch.tensor(0.0, device=device)
            else:
                tok_a = tokenizer(anchor_queries).to(device)
                anchor_now = F.normalize(model.encode_text(tok_a), dim=-1)
                l_anchor = anchor_text_loss(anchor_now, anchor_init)

            try:
                fsl_t_emb = torch.stack([fsl_text_cache[q] for q in b.query]).to(device).float()
                sl2_t_emb = torch.stack([sl2_text_cache[q] for q in b.query]).to(device).float()
                fsl_t_emb = F.normalize(fsl_t_emb, dim=-1)
                sl2_t_emb = F.normalize(sl2_t_emb, dim=-1)
                fsl_img = F.normalize(fsl_img_cache[b.image_idx].float(), dim=-1)
                # SL2 teacher image: dedicated cache (v4-alt) or reuse student cache (v3, v4)
                if sl2_img_cache is not None:
                    sl2_img_for_kl = F.normalize(sl2_img_cache[b.image_idx].float(), dim=-1)
                else:
                    sl2_img_for_kl = img_emb
                l_kl = fusion_kl_loss(cos_scores, fsl_t_emb, fsl_img, sl2_t_emb, sl2_img_for_kl,
                                       fsl_weight=args.kl_fsl_weight)
            except KeyError:
                l_kl = torch.tensor(0.0, device=device)

            lam_anchor, lam_kl = get_lambdas(step)
            loss = args.gcl_lambda * l_gcl + lam_anchor * l_anchor + lam_kl * l_kl

            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], max_norm=1.0
            )
            optim.step()
            scheduler.step()

            train_log.write(json.dumps({
                "step": step, "epoch": epoch,
                "loss": loss.item(), "loss_gcl": l_gcl.item(),
                "loss_anchor": l_anchor.item(),
                "loss_kl": float(l_kl.detach().item()) if torch.is_tensor(l_kl) else float(l_kl),
                "lam_anchor": lam_anchor, "lam_kl": lam_kl,
                "lr": scheduler.get_last_lr()[0],
                "elapsed_s": time.time() - t_start,
            }) + "\n")
            train_log.flush()

            pbar.update(1)
            pbar.set_postfix(
                ep=epoch, loss=f"{loss.item():.3f}",
                gcl=f"{l_gcl.item():.3f}",
                kl=f"{float(l_kl):.3f}",
                best=f"{best_mean:.3f}@{best_step}",
            )
            step += 1

            # ─── eval probe ──────────────────────────────────────────
            if step % args.probe_every == 0:
                metrics = probe.probe(model, tokenizer)
                mean_mrr = probe.aggregate_score(metrics)
                probe_log.write(json.dumps({
                    "step": step, "epoch": epoch,
                    "metrics": metrics, "mean_mrr": mean_mrr,
                    "elapsed_s": time.time() - t_start,
                }) + "\n")
                probe_log.flush()

                # Best-checkpoint selection: highest mean MRR with NO benchmark regressed >threshold
                regressed = []
                for bname, m in metrics.items():
                    init = init_mrr[bname]
                    if init > 0 and (init - m["mrr"]) / init > args.regression_threshold:
                        regressed.append((bname, init, m["mrr"]))

                if regressed:
                    consecutive_regressions += 1
                else:
                    consecutive_regressions = 0
                    if mean_mrr > best_mean:
                        best_mean = mean_mrr
                        best_step = step
                        best_metrics = metrics
                        # Save best checkpoint
                        torch.save({
                            "step": step, "epoch": epoch,
                            "metrics": metrics, "mean_mrr": mean_mrr,
                            "model_trainable": {n: p.detach().cpu()
                                                for n, p in model.named_parameters()
                                                if p.requires_grad},
                        }, CKPT_DIR / f"phase_d_best_{tag}.pt")

                msg = (f"\n[probe step {step}] mean_mrr={mean_mrr:.4f} "
                       f"(best={best_mean:.4f}@{best_step}); "
                       f"regressed={len(regressed)}/{len(metrics)}")
                pbar.write(msg)
                if regressed:
                    pbar.write(f"  regressions: {regressed}")

                if consecutive_regressions >= args.regression_patience:
                    halt_reason = (f"halted: {consecutive_regressions} consecutive probes "
                                   f"with regression >{args.regression_threshold} on at least one benchmark")
                    pbar.write(halt_reason)

            # ─── periodic checkpoint ─────────────────────────────────
            if args.ckpt_every > 0 and step % args.ckpt_every == 0:
                torch.save({
                    "step": step, "epoch": epoch,
                    "model_trainable": {n: p.detach().cpu()
                                        for n, p in model.named_parameters()
                                        if p.requires_grad},
                }, CKPT_DIR / f"phase_d_{tag}_step{step}.pt")

    pbar.close()
    train_log.close()
    probe_log.close()

    elapsed = time.time() - t_start
    print(f"\nTraining done in {elapsed:.0f}s ({step} steps, {step/elapsed:.2f} step/s)")
    if halt_reason:
        print(f"  {halt_reason}")
    print(f"Best checkpoint: phase_d_best_{tag}.pt @ step {best_step}, mean_mrr={best_mean:.4f}")
    print(f"  initial MRR: {init_mrr}")
    best_mrr_str = ", ".join(f"{b}={m['mrr']:.4f}" for b, m in best_metrics.items())
    print(f"  best MRR:    {{ {best_mrr_str} }}")

    # Final summary
    final = {
        "total_steps": step,
        "elapsed_s": elapsed,
        "halt_reason": halt_reason,
        "initial_metrics": init_metrics,
        "initial_mean_mrr": probe.aggregate_score(init_metrics),
        "best_step": best_step,
        "best_mean_mrr": best_mean,
        "best_metrics": best_metrics,
    }
    final_path.write_text(json.dumps(final, indent=2, default=str))
    print(f"Summary: {final_path}")


if __name__ == "__main__":
    main()
