"""
Phase B — unit tests for v5 dataset, model, loss.

Verifies:
  T1  Dataset loads, builds at least one valid grouped batch
  T2  Loss components produce finite, sane values on synthetic + real data
  T3  Model construction has the expected ~28-29M trainable params
  T4  Gradients flow only to the trainable parameters (image tower grads = 0)
  T5  Anchor loss is exactly 0 when student==teacher; > 0 otherwise
  T6  Fusion KL is small when student==teacher; large when shuffled
  T7  Coefficient schedule returns the right values per PLAN_V5 §B.3
  T8  End-to-end: one full forward + backward step on a real batch

Run:
    python scripts/v5/phase_b_unittest.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))

from v5_dataset import V5Dataset
from v5_model import build_student, count_trainable, trainable_parameter_groups
from v5_loss import (
    anchor_text_loss,
    fusion_kl_loss,
    get_loss_coefficients,
    grouped_gcl_loss,
    score_to_weight_inverse_sqrt,
)

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data" / "processed" / "v5_multifield"


# ─── helpers ────────────────────────────────────────────────────────────────

GREEN, RED, RESET = "\033[32m", "\033[31m", "\033[0m"


def passed(msg: str): print(f"  {GREEN}PASS{RESET}  {msg}")
def failed(msg: str): print(f"  {RED}FAIL{RESET}  {msg}"); sys.exit(1)


def near(a, b, tol=1e-4):
    return abs(float(a) - float(b)) < tol


# ─── tests ──────────────────────────────────────────────────────────────────

def test_dataset_construction():
    """T1 — Dataset can build batches from the 50K pairs aligned with the cache."""
    print("T1  Dataset construction")
    # Use pairs_50k.jsonl (the same set the student image cache was built from)
    # rather than validation_200.jsonl (which was sampled from the full v4 pool
    # and only partially overlaps with the cache).
    pairs = DATA / "pairs_50k.jsonl"
    image_index = DATA / "student_image_index.json"
    if not pairs.exists():
        failed(f"missing {pairs}")
    if not image_index.exists():
        failed(f"missing {image_index} — run student image cache first")

    ds = V5Dataset(pairs, image_index, K=8, N=16, min_products_per_query=2)
    stats = ds.stats()
    print(f"    stats: {stats}")
    if stats["n_pairs_loaded"] < 1000:
        failed(f"only {stats['n_pairs_loaded']} pairs loaded")
    passed(f"{stats['n_pairs_loaded']:,} pairs, {stats['n_queries_kept']:,} queries")

    batches = list(ds.iter_batches())
    if not batches:
        failed("could not produce any batches")
    b = batches[0]
    if b.query_idx.shape[0] != b.K * b.N:
        failed(f"batch shape mismatch: {b.query_idx.shape}")
    passed(f"yielded {len(batches):,} batches; first batch K={b.K} N={b.N}, "
           f"image_idx range=[{b.image_idx.min()},{b.image_idx.max()}]")
    return ds


def test_score_weight():
    """Quick check on the GCL weight transform."""
    print("T2a  score_to_weight_inverse_sqrt")
    scores = torch.tensor([1.0, 50.0, 100.0])
    w = score_to_weight_inverse_sqrt(scores)
    if not (w[2] > w[1] > w[0]):
        failed(f"expected monotone increasing, got {w}")
    passed(f"weights for [1,50,100] = {w.tolist()}")


def test_grouped_gcl_synthetic():
    """T2b — GCL loss on a controlled synthetic batch."""
    print("T2b  Grouped GCL on synthetic")
    K, N = 4, 8
    KN = K * N
    # Build a "perfect" score matrix: high score where same-query, low otherwise
    query_idx = torch.repeat_interleave(torch.arange(K), N)
    perfect = torch.full((K, KN), -10.0)
    for k in range(K):
        perfect[k, k * N : (k + 1) * N] = 10.0
    scores_real = torch.tensor([95.0] * KN)

    loss_perfect = grouped_gcl_loss(perfect, query_idx, scores_real, K)
    loss_random = grouped_gcl_loss(torch.randn(K, KN) * 0.1, query_idx, scores_real, K)

    if not torch.isfinite(loss_perfect):
        failed(f"loss not finite: {loss_perfect}")
    if loss_perfect >= loss_random:
        failed(f"perfect ({loss_perfect:.4f}) should be < random ({loss_random:.4f})")
    passed(f"perfect={loss_perfect:.4f} < random={loss_random:.4f}")


def test_anchor_loss():
    """T5 — Anchor loss is 0 when current==init; > 0 otherwise."""
    print("T5  Anchor text loss")
    a = F.normalize(torch.randn(32, 768), dim=-1)
    loss_zero = anchor_text_loss(a, a)
    if not near(loss_zero, 0.0, tol=1e-6):
        failed(f"expected 0, got {loss_zero}")
    b = F.normalize(torch.randn(32, 768), dim=-1)
    loss_drift = anchor_text_loss(a, b)
    if loss_drift <= 0:
        failed(f"expected > 0, got {loss_drift}")
    passed(f"identical→{loss_zero:.6f}, drifted→{loss_drift:.6f}")


def test_fusion_kl():
    """T6 — Fusion KL is small when student==teacher; large when shuffled."""
    print("T6  Fusion KL on synthetic")
    K, N, D = 4, 4, 768
    KN = K * N
    fsl_t = F.normalize(torch.randn(K, D), dim=-1)
    fsl_i = F.normalize(torch.randn(KN, D), dim=-1)
    sl2_t = F.normalize(torch.randn(K, D), dim=-1)
    sl2_i = F.normalize(torch.randn(KN, D), dim=-1)
    teacher_scores_match = (fsl_t @ fsl_i.T + sl2_t @ sl2_i.T) / 2

    # student_scores == teacher_scores → KL ≈ 0
    loss_match = fusion_kl_loss(teacher_scores_match, fsl_t, fsl_i, sl2_t, sl2_i)
    # student_scores random → KL > 0
    loss_random = fusion_kl_loss(torch.randn(K, KN) * 5, fsl_t, fsl_i, sl2_t, sl2_i)
    if loss_match >= loss_random:
        failed(f"matched ({loss_match}) should be < random ({loss_random})")
    if not torch.isfinite(loss_match) or not torch.isfinite(loss_random):
        failed(f"non-finite: {loss_match}, {loss_random}")
    passed(f"matched={loss_match:.6f} < random={loss_random:.6f}")


def test_coefficients():
    """T7 — Loss coefficient schedule from PLAN_V5 §B.3."""
    print("T7  Coefficient schedule")
    cases = [(0, 0.5, 0.3), (499, 0.5, 0.3), (500, 0.3, 0.2),
             (4999, 0.3, 0.2), (5000, 0.1, 0.1), (10000, 0.1, 0.1)]
    for step, exp_a, exp_kl in cases:
        a, kl = get_loss_coefficients(step)
        if not (near(a, exp_a) and near(kl, exp_kl)):
            failed(f"step {step}: got ({a},{kl}) expected ({exp_a},{exp_kl})")
    passed(f"{len(cases)} schedule points correct")


def test_model_construction():
    """T3 — Build student, check trainable param count."""
    print("T3  Model construction (CPU)")
    t0 = time.time()
    model, tokenizer = build_student(device="cpu")
    counts = count_trainable(model)
    print(f"    trainable param breakdown: {counts}")
    print(f"    build time: {time.time() - t0:.1f}s")

    total_m = counts["total"] / 1e6
    if not (20 < total_m < 50):
        failed(f"trainable params {total_m:.1f}M outside expected 20-50M range")
    if counts["text_blocks"] / 1e6 < 15:
        failed(f"text_blocks only {counts['text_blocks']/1e6:.1f}M, expected ~28M")
    passed(f"total trainable ≈ {total_m:.1f}M")
    return model, tokenizer


def test_gradient_flow(model, tokenizer):
    """T4 — Gradients flow only to trainable params."""
    print("T4  Gradient flow")
    model.train()
    text = tokenizer(["red dress", "blue jeans", "running shoes", "leather bag"])
    text_emb = model.encode_text(text)
    loss = text_emb.pow(2).sum()
    loss.backward()

    n_with_grad = n_no_grad = 0
    for n, p in model.named_parameters():
        if p.requires_grad:
            if p.grad is None or p.grad.abs().sum() == 0:
                # Some params may legitimately have zero grad on a single batch
                # (e.g. logit_bias if not used); treat as "no grad"
                n_no_grad += 1
            else:
                n_with_grad += 1
        else:
            if p.grad is not None and p.grad.abs().sum() > 0:
                failed(f"frozen param {n} has nonzero grad")
    if n_with_grad == 0:
        failed("no trainable param received a gradient")
    passed(f"{n_with_grad} trainable params have grad, no frozen param polluted")


def test_end_to_end(model, tokenizer):
    """T8 — One forward+backward step using cached image embeddings + the dataset."""
    print("T8  End-to-end forward+backward (CPU)")
    pairs = DATA / "pairs_50k.jsonl"
    image_index = DATA / "student_image_index.json"
    cache = DATA / "student_image_emb.pt"
    if not (pairs.exists() and image_index.exists() and cache.exists()):
        print("    skipped (artifacts missing — pre-Phase A)")
        return
    img_cache = torch.load(cache, map_location="cpu")  # (N, 768) fp16
    print(f"    image cache shape: {tuple(img_cache.shape)}, dtype: {img_cache.dtype}")

    ds = V5Dataset(pairs, image_index, K=2, N=4, min_products_per_query=2)
    batches = list(ds.iter_batches())
    if not batches:
        print("    skipped (no batches buildable)")
        return
    b = batches[0]

    model.train()
    text_q = model.encode_text(tokenizer(b.query))
    text_t = model.encode_text(tokenizer(b.title))
    text_c = model.encode_text(tokenizer(b.category_l2))
    text_multi = F.normalize(0.6 * text_q + 0.3 * text_t + 0.1 * text_c, dim=-1)

    img_emb = F.normalize(img_cache[b.image_idx].float(), dim=-1)
    scale = model.logit_scale.exp() if hasattr(model, "logit_scale") else 1.0
    bias = model.logit_bias if hasattr(model, "logit_bias") else 0.0
    scores = text_multi @ img_emb.T * scale + bias

    loss = grouped_gcl_loss(scores, b.query_idx, b.score_linear, b.K)
    if not torch.isfinite(loss):
        failed(f"loss not finite: {loss}")
    loss.backward()
    passed(f"loss={loss.item():.4f}, gradient flowed end-to-end")


# ─── runner ─────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Phase B unit tests")
    print("=" * 60)
    test_dataset_construction()
    test_score_weight()
    test_grouped_gcl_synthetic()
    test_anchor_loss()
    test_fusion_kl()
    test_coefficients()
    model, tok = test_model_construction()
    test_gradient_flow(model, tok)
    test_end_to_end(model, tok)
    print()
    print(f"{GREEN}All Phase B tests passed.{RESET}")


if __name__ == "__main__":
    main()
