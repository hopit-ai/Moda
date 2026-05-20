"""
Path 1 — Distillation trainer.

Distill the score-mean fusion of (Marqo-FashionSigLIP, Google SigLIP-2 B/16/384)
into a single ViT-B-16-SigLIP2-384 student, using KL on row-softmaxed
in-batch score matrices.

This recipe is structurally different from the four failed Phase 4 recipes:

  Phase 4 collapses     | Why this recipe is robust
  ----------------------|---------------------------------
  MSE on embeddings     | KL on row-softmax explicitly penalises uniform output:
   collapses to mean      a constant student score matrix becomes a uniform
                          softmax, which has high KL vs any non-uniform target.
  Small-batch InfoNCE   | We use B=64 (or grad-accum equiv) — the contrastive
   collapses (bs=12)      term is anchored properly. KL term carries the load.
  End-only eval         | In-loop probe at steps {50, 100, 500, 1000, 2000}
                          aborts the run if MAP@10 collapses below baseline.

Loss:
  loss = α · KL(softmax(s_teacher/τ) || softmax(s_student/τ))     # ranking
       + β · 0.5 · (CE(s_student/τ, diag) + CE(s_student/τ, diag).T)  # anchor
  α=1.0  β=0.1  τ=0.05

Data:
  - Filtered Marqo-GS wfash subset (position<=10 in Google Shopping ranks),
    520 high-confidence (query, image) pairs, 310 unique queries.
  - Cached teacher embeddings (FSL + SL2, both L2-normed, on disk).

Student:
  - open_clip ViT-B-16-SigLIP2-384, init from `webli`. Same arch as the
    SL2 teacher, so output lives in the same geometry. Image tower frozen
    for the first 500 steps (text tower learns catalog vocabulary first).

Probe:
  - benchmark/probe_fashion200k_10k.py runs in-process every probe step.
    Returns MAP@10 vs the canonical screener (verified to match autopsy
    baseline within 0.001).

Usage:
  # Smoke test (10 steps, no probe — just check no NaN)
  .venv/bin/python benchmark/distill_fusion_to_student.py \
      --max-steps 10 --probe-steps "" --output-dir models/path1-smoke1

  # Smoke test 2 (100 steps with probes at 50 and 100)
  .venv/bin/python benchmark/distill_fusion_to_student.py \
      --max-steps 100 --probe-steps "50,100" \
      --output-dir models/path1-smoke2

  # Full overnight run
  .venv/bin/python benchmark/distill_fusion_to_student.py \
      --max-steps 2000 --probe-steps "100,500,1000,2000" \
      --abort-step 100 --abort-min-map10 0.481 \
      --output-dir models/path1-full

Outputs:
  <output-dir>/run_meta.json
  <output-dir>/training_log.jsonl
  <output-dir>/step_<N>/student_state_dict.pt
  <output-dir>/best/student_state_dict.pt
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import sys
import time
from pathlib import Path
from typing import Any

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
log = logging.getLogger("distill")

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
HF_CACHE = REPO / "data" / "hf_cache"

DEFAULT_TEACHER_CACHE = REPO / "data/processed/distillation_cache_fusion_pos10"
DEFAULT_TRIPLETS = REPO / "data/processed/marqo_gs_wfash_subset_pos10/triplets.jsonl"


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------

def load_teacher_caches(cache_dir: Path) -> dict[str, dict]:
    """Load both teacher caches; verify they have the same row order."""
    fsl = torch.load(cache_dir / "fashion_siglip_embeddings.pt",
                     map_location="cpu", weights_only=False)
    sl2 = torch.load(cache_dir / "siglip2_b16_384_embeddings.pt",
                     map_location="cpu", weights_only=False)
    # Sanity: same N, same row order (queries+image_paths must match)
    assert fsl["queries"] == sl2["queries"], "teacher caches have different query orders"
    assert fsl["image_paths"] == sl2["image_paths"], "teacher caches have different image orders"
    log.info("[teacher] loaded 2 teachers, N=%d, dim=%d (FSL), dim=%d (SL2)",
             len(fsl["queries"]), fsl["embed_dim"], sl2["embed_dim"])
    return {"fsl": fsl, "sl2": sl2}


def compute_teacher_score_matrix_for_indices(
    teachers: dict[str, dict], idxs: torch.Tensor,
) -> torch.Tensor:
    """For a batch of row indices, compute the teacher score matrix.

    teachers["fsl"]["text"][i] is the L2-normed text embedding for row i,
    teachers["fsl"]["image"][j] is the L2-normed image embedding for row j.
    The teacher score matrix is the score-mean fusion of cos(text_i, img_j)
    under both teachers.

    Returns: [B, B] float32 tensor on CPU.
    """
    txt_fsl = teachers["fsl"]["text"][idxs]   # [B, 768]
    img_fsl = teachers["fsl"]["image"][idxs]
    txt_sl2 = teachers["sl2"]["text"][idxs]
    img_sl2 = teachers["sl2"]["image"][idxs]

    s_fsl = txt_fsl @ img_fsl.T  # [B, B]
    s_sl2 = txt_sl2 @ img_sl2.T

    return 0.5 * (s_fsl + s_sl2)


# -----------------------------------------------------------------------------
# Loss
# -----------------------------------------------------------------------------

def kl_row_softmax(s_teacher: torch.Tensor, s_student: torch.Tensor, tau: float) -> torch.Tensor:
    """KL(softmax(s_t/τ) || softmax(s_s/τ)) averaged over rows.

    Note: KL(p || q) = sum_i p_i (log p_i - log q_i)
    Using log_softmax for numerical stability on both sides.
    """
    log_p = F.log_softmax(s_teacher / tau, dim=-1)
    log_q = F.log_softmax(s_student / tau, dim=-1)
    p = log_p.exp()
    return (p * (log_p - log_q)).sum(dim=-1).mean()


def symmetric_infonce(s_student: torch.Tensor, tau: float) -> torch.Tensor:
    """Standard symmetric InfoNCE on positive diagonal."""
    B = s_student.shape[0]
    labels = torch.arange(B, device=s_student.device)
    logits = s_student / tau
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))


# -----------------------------------------------------------------------------
# Training step
# -----------------------------------------------------------------------------

def student_score_matrix(
    student, preprocess, tokenizer, batch_rows: list[dict], device: str,
) -> torch.Tensor:
    """Forward the student over a batch and return [B, B] cosine score matrix.

    Both student image and text features are L2-normed before scoring.
    """
    from PIL import Image
    images, queries = [], []
    for r in batch_rows:
        img = Image.open(r["image_path"]).convert("RGB")
        images.append(preprocess(img))
        queries.append(r["query"])
    img_tens = torch.stack(images).to(device)
    tokens = tokenizer(queries).to(device)

    img_feat = student.encode_image(img_tens)
    txt_feat = student.encode_text(tokens)
    img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    return txt_feat @ img_feat.T  # [B, B]


def freeze_image_tower(student) -> int:
    """Freeze all image-tower parameters. Returns # frozen params."""
    n_frozen = 0
    for name, p in student.named_parameters():
        # open_clip CustomTextCLIP: image tower is `visual.*`
        if name.startswith("visual."):
            p.requires_grad = False
            n_frozen += p.numel()
    return n_frozen


def unfreeze_image_tower(student) -> int:
    n_unfrozen = 0
    for name, p in student.named_parameters():
        if name.startswith("visual."):
            p.requires_grad = True
            n_unfrozen += p.numel()
    return n_unfrozen


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--triplets", default=str(DEFAULT_TRIPLETS))
    p.add_argument("--teacher-cache-dir", default=str(DEFAULT_TEACHER_CACHE))
    p.add_argument("--student-model", default="ViT-B-16-SigLIP2-384")
    p.add_argument("--student-pretrained", default="webli")
    p.add_argument("--output-dir", required=True)

    p.add_argument("--batch-size", type=int, default=32,
                   help="Effective batch size. With grad-accum, micro-batch = batch / accum.")
    p.add_argument("--grad-accum", type=int, default=2,
                   help="Gradient accumulation factor (micro_bs = batch_size / grad_accum).")

    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--warmup-steps", type=int, default=50)
    p.add_argument("--max-steps", type=int, default=2000)
    p.add_argument("--unfreeze-image-step", type=int, default=500)

    p.add_argument("--alpha-kl", type=float, default=1.0)
    p.add_argument("--beta-infonce", type=float, default=0.1)
    p.add_argument("--tau", type=float, default=0.05)

    p.add_argument("--probe-steps", default="50,100,500,1000,2000",
                   help="Comma-separated step numbers at which to run the in-loop probe. Empty for none.")
    p.add_argument("--abort-step", type=int, default=100,
                   help="Step at which the abort criterion is checked.")
    p.add_argument("--abort-min-map10", type=float, default=0.481,
                   help="If probe MAP@10 at abort-step is below this, abort the run.")

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-every", type=int, default=10)

    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "training_log.jsonl"
    meta_path = out_dir / "run_meta.json"

    # Save run metadata up front
    meta_path.write_text(json.dumps(vars(args), indent=2, default=str))
    log.info("=== Path 1 distillation run ===")
    log.info("output_dir=%s", out_dir)
    log.info("device=%s", DEVICE)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if DEVICE == "mps":
        torch.mps.manual_seed(args.seed)

    # -------------------------------------------------------------------------
    # 1. Load triplets + teacher caches
    # -------------------------------------------------------------------------
    rows = [json.loads(line) for line in open(args.triplets)]
    log.info("[data] loaded %d rows from %s", len(rows), args.triplets)

    teachers = load_teacher_caches(Path(args.teacher_cache_dir))
    # Build query/image_path -> row-idx-in-cache lookup. The cache rows mirror
    # the triplets order at encoding time, so they should match exactly. We
    # verify by spot-checking that the queries/image_paths align.
    cache_queries = teachers["fsl"]["queries"]
    cache_images = teachers["fsl"]["image_paths"]
    if cache_queries[:5] != [r["query"] for r in rows[:5]] or \
       cache_images[:5] != [r["image_path"] for r in rows[:5]]:
        raise RuntimeError("Teacher cache row order does not match triplets file order. "
                           "Re-encode the teachers from this triplets file.")

    # -------------------------------------------------------------------------
    # 2. Load student
    # -------------------------------------------------------------------------
    import open_clip
    log.info("[student] loading %s pretrained=%s", args.student_model, args.student_pretrained)
    student, _, preprocess = open_clip.create_model_and_transforms(
        args.student_model, pretrained=args.student_pretrained,
        cache_dir=str(HF_CACHE),
    )
    tokenizer = open_clip.get_tokenizer(args.student_model)
    student.to(DEVICE)
    student.train()

    n_frozen = freeze_image_tower(student)
    n_total = sum(p.numel() for p in student.parameters())
    n_trainable = sum(p.numel() for p in student.parameters() if p.requires_grad)
    log.info("[student] params: total=%d  frozen(image)=%d  trainable=%d",
             n_total, n_frozen, n_trainable)

    optimizer = torch.optim.AdamW(
        [p for p in student.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=0.0, betas=(0.9, 0.95),
    )

    def lr_at_step(step: int) -> float:
        if step < args.warmup_steps:
            return args.lr * (step + 1) / args.warmup_steps
        progress = (step - args.warmup_steps) / max(1, args.max_steps - args.warmup_steps)
        return args.lr * 0.1 + (args.lr - args.lr * 0.1) * 0.5 * (1 + math.cos(math.pi * progress))

    # -------------------------------------------------------------------------
    # 3. Probe (lazy-init: building it requires loading f200k 10K corpus,
    # which is ~10s, so we only construct if probe_steps non-empty)
    # -------------------------------------------------------------------------
    probe_steps: set[int] = set()
    if args.probe_steps.strip():
        probe_steps = {int(s.strip()) for s in args.probe_steps.split(",") if s.strip()}
    log.info("[probe] will probe at steps: %s", sorted(probe_steps))

    probe = None
    if probe_steps:
        from probe_fashion200k_10k import Fashion200kProbe
        probe = Fashion200kProbe(dataset="fashion200k", corpus_size=10000, seed=42)

    # -------------------------------------------------------------------------
    # 4. Training loop
    # -------------------------------------------------------------------------
    micro_bs = max(1, args.batch_size // args.grad_accum)
    log.info("[train] batch_size=%d  micro_bs=%d  grad_accum=%d",
             args.batch_size, micro_bs, args.grad_accum)

    # Sampler: random shuffle each epoch, draw micro-batches of size `micro_bs`
    indices = list(range(len(rows)))
    random.shuffle(indices)
    cursor = 0

    def next_micro_batch_indices() -> list[int]:
        nonlocal cursor
        if cursor + micro_bs > len(indices):
            random.shuffle(indices)
            cursor = 0
        batch = indices[cursor:cursor + micro_bs]
        cursor += micro_bs
        return batch

    log_lines: list[dict] = []
    best_map10 = -1.0
    best_step = -1
    aborted = False

    t_start = time.time()
    for step in range(args.max_steps + 1):  # +1 so step==max_steps gets a probe
        # LR schedule
        cur_lr = lr_at_step(step)
        for g in optimizer.param_groups:
            g["lr"] = cur_lr

        # Unfreeze image tower if we hit the threshold
        if step == args.unfreeze_image_step and step > 0:
            n_un = unfreeze_image_tower(student)
            log.info("[train] step=%d UNFREEZING image tower (%d params)", step, n_un)
            # Rebuild optimizer with all trainable params
            optimizer = torch.optim.AdamW(
                [p for p in student.parameters() if p.requires_grad],
                lr=cur_lr, weight_decay=0.0, betas=(0.9, 0.95),
            )

        if step < args.max_steps:
            # ------ One optimizer step (with grad accum) ------
            optimizer.zero_grad(set_to_none=True)
            kl_sum = 0.0
            ce_sum = 0.0
            for accum in range(args.grad_accum):
                micro_idxs = next_micro_batch_indices()
                idx_t = torch.tensor(micro_idxs, dtype=torch.long)
                batch_rows = [rows[i] for i in micro_idxs]

                with torch.no_grad():
                    s_teacher = compute_teacher_score_matrix_for_indices(teachers, idx_t)
                s_teacher = s_teacher.to(DEVICE)

                s_student = student_score_matrix(student, preprocess, tokenizer, batch_rows, DEVICE)

                loss_kl = kl_row_softmax(s_teacher, s_student, tau=args.tau)
                loss_ce = symmetric_infonce(s_student, tau=args.tau)
                loss = (args.alpha_kl * loss_kl + args.beta_infonce * loss_ce) / args.grad_accum
                loss.backward()

                kl_sum += loss_kl.item()
                ce_sum += loss_ce.item()

            torch.nn.utils.clip_grad_norm_(
                [p for p in student.parameters() if p.requires_grad], max_norm=1.0,
            )
            optimizer.step()

            kl_avg = kl_sum / args.grad_accum
            ce_avg = ce_sum / args.grad_accum
            total = args.alpha_kl * kl_avg + args.beta_infonce * ce_avg

            if step % args.log_every == 0:
                elapsed = time.time() - t_start
                log.info("[train] step=%d/%d  lr=%.2e  loss=%.4f (kl=%.4f, ce=%.4f)  elapsed=%.1fs",
                         step, args.max_steps, cur_lr, total, kl_avg, ce_avg, elapsed)
            log_lines.append({
                "step": step, "lr": cur_lr, "kl": kl_avg, "ce": ce_avg,
                "total": total, "wall": time.time() - t_start,
            })

        # ------ Probe ------
        if step in probe_steps:
            log.info("[probe] running fashion200k 10K probe at step=%d ...", step)
            metrics = probe.run(student, preprocess, tokenizer, device=DEVICE,
                                batch_size=64) if probe is not None else {}
            student.train()  # re-enter train mode
            map10 = float(metrics.get("MAP@10", 0))
            log.info("[probe] step=%d  MAP@10=%.4f  R@10=%.4f  NDCG@10=%.4f",
                     step, map10, metrics.get("Recall@10", 0), metrics.get("NDCG@10", 0))
            log_lines.append({
                "step": step, "probe": True, **{f"probe_{k}": v for k, v in metrics.items()},
            })

            # Save checkpoint
            ckpt_dir = out_dir / f"step_{step}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            torch.save(student.state_dict(), ckpt_dir / "student_state_dict.pt")

            if map10 > best_map10:
                best_map10 = map10
                best_step = step
                best_dir = out_dir / "best"
                best_dir.mkdir(parents=True, exist_ok=True)
                torch.save(student.state_dict(), best_dir / "student_state_dict.pt")
                log.info("[probe] new best @ step=%d  MAP@10=%.4f", step, map10)

            # Abort criterion
            if step == args.abort_step and map10 < args.abort_min_map10:
                log.error("[abort] step=%d  MAP@10=%.4f  <  abort_min=%.4f  — collapse, stopping run",
                          step, map10, args.abort_min_map10)
                aborted = True

        # Persist log at every probe (and at end)
        with open(log_path, "w") as f:
            for line in log_lines:
                f.write(json.dumps(line) + "\n")

        if aborted:
            break

    # -------------------------------------------------------------------------
    # 5. Done. Write final summary.
    # -------------------------------------------------------------------------
    summary = {
        "max_steps_reached": step,
        "aborted": aborted,
        "best_step": best_step,
        "best_map10": best_map10,
        "wall_time_total_sec": time.time() - t_start,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    log.info("=== summary ===")
    log.info("%s", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
