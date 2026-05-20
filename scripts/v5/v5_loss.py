"""
v5 loss components.

Three losses combined at training time:

  L_total = L_gcl + λ_anchor · L_anchor_text + λ_kl · L_kl_fusion

where:
  L_gcl          — grouped GCL with multi-positive sigmoid CE
  L_anchor_text  — MSE between current text embedding and frozen SL2 init
                   (text-only because the image branch is frozen)
  L_kl_fusion    — KL on row-softmaxed scores against (FSL+SL2)/2 teacher

Coefficient schedule from PLAN_V5 §B.3:
  step 0–500       λ_anchor=0.5  λ_kl=0.3
  step 500–5000    λ_anchor=0.3  λ_kl=0.2
  step 5000+       λ_anchor=0.1  λ_kl=0.1
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def score_to_weight_inverse_sqrt(scores: torch.Tensor, max_score: float = 100.0) -> torch.Tensor:
    """Convert linear scores ∈ [1, max_score] to in-batch positive weights.

    Same recipe used in v4; higher score → higher weight, with diminishing returns.
    Returns weights in roughly [0.1, 1.0].
    """
    return 1.0 / torch.sqrt(1.0 + max_score - scores.clamp(min=1, max=max_score))


def grouped_gcl_loss(
    scores: torch.Tensor,        # (K, K*N) student score matrix
    query_idx: torch.Tensor,     # (K*N,) which query each product belongs to
    score_linear: torch.Tensor,  # (K*N,) GS-10M score values
    K: int,
) -> torch.Tensor:
    """Multi-positive sigmoid-CE GCL loss.

    Positive when (i, j) belong to the same query group (i.e. query_idx[j] == i).
    Weighted by inverse-sqrt of score_linear so high-confidence positives count more.
    """
    device = scores.device
    KN = query_idx.shape[0]
    # Labels: (K, KN) bool — True where product j's group matches query i
    labels = (torch.arange(K, device=device).unsqueeze(1) == query_idx.unsqueeze(0)).float()

    # Positive weights (per-product, broadcast across queries)
    pos_weights = score_to_weight_inverse_sqrt(score_linear).to(device)  # (KN,)
    weights = labels * pos_weights.unsqueeze(0) + (1.0 - labels) * 1.0    # (K, KN)

    # Sigmoid CE — same form SigLIP uses
    bce = F.binary_cross_entropy_with_logits(
        scores, labels, weight=weights, reduction="mean"
    )
    return bce


def anchor_text_loss(
    student_text_emb: torch.Tensor,  # (B, D), L2-normalized
    teacher_text_emb: torch.Tensor,  # (B, D), L2-normalized, frozen SL2 init
) -> torch.Tensor:
    """MSE between current and frozen-init text embeddings on a held-out anchor set.

    Penalizes drift from the SL2 zero-shot text geometry — protects the
    atlas/polyvore/KAGL wins that base SL2 already has.
    """
    return F.mse_loss(student_text_emb, teacher_text_emb)


def fusion_kl_loss(
    student_scores: torch.Tensor,    # (K, K*N)
    fsl_text_emb: torch.Tensor,      # (K, D), normalized fp32
    fsl_img_emb: torch.Tensor,       # (K*N, D), normalized fp32
    sl2_text_emb: torch.Tensor,      # (K, D), normalized fp32 (from teacher_sl2_text_emb cache)
    sl2_img_emb: torch.Tensor,       # (K*N, D), normalized fp32 (= student image cache reused)
    tau: float = 0.05,
    fsl_weight: float = 0.5,
) -> torch.Tensor:
    """Soft KL on row-softmaxed in-batch scores.

    Teacher = weighted combination of FSL and SL2 score matrices.
    fsl_weight=0.5  → standard (FSL + SL2) / 2 fusion teacher
    fsl_weight=0.0  → SL2-only teacher (used when FSL is the student)
    fsl_weight=1.0  → FSL-only teacher
    """
    with torch.no_grad():
        s_sl2 = sl2_text_emb @ sl2_img_emb.T
        if fsl_weight > 0.0:
            s_fsl = fsl_text_emb @ fsl_img_emb.T
            teacher_scores = fsl_weight * s_fsl + (1.0 - fsl_weight) * s_sl2
        else:
            teacher_scores = s_sl2

    student_log = F.log_softmax(student_scores / tau, dim=-1)
    teacher_p = F.softmax(teacher_scores / tau, dim=-1)
    return F.kl_div(student_log, teacher_p, reduction="batchmean")


def get_loss_coefficients(step: int) -> tuple[float, float]:
    """λ_anchor and λ_kl at a given step (per PLAN_V5 §B.3 schedule)."""
    if step < 500:
        return 0.5, 0.3
    if step < 5000:
        return 0.3, 0.2
    return 0.1, 0.1
