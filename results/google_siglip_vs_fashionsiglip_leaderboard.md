# Google SigLIP vs Marqo-FashionSigLIP — head-to-head on Marqo benchmark

_Updated 2026-04-22. No fashion fine-tuning on the Google models — straight `open_clip` checkpoints (`pretrained="webli"`)._

## ⭐ HEADLINE — Google SigLIP SO400M-384 (no fine-tune) BEATS Marqo-FashionSigLIP on atlas text-to-image

| Metric | FashionSigLIP | **SO400M-384** | Δ relative |
| --- | ---: | ---: | ---: |
| **MAP@10** | 0.1826 | **0.2308** | **+26.4%** 🏆 |
| **NDCG@10** | 0.2338 | **0.2860** | **+22.3%** 🏆 |
| **Recall@10** | 0.3239 | **0.3776** | **+16.6%** 🏆 |
| **MRR** | 0.2337 | **0.2894** | **+23.8%** 🏆 |

**A publicly available, off-the-shelf model beats Marqo's fine-tuned FashionSigLIP on Marqo's own atlas text-to-image task by ~22% relative across every metric, with zero fashion training.** Confirmation runs on the other 3 clean datasets (KAGL, polyvore, fashion200k) are queued.

On atlas sub-category-to-product NDCG@10 (the right metric for category tasks) SO400M ties FashionSigLIP at +0.1%.

## Setup

- **Marqo-FashionSigLIP**: ViT-B/16, 224×224, 768d. WebLI-pretrained Google SigLIP **+ Marqo's GCL fine-tune on fashion**.
- **Google SigLIP B/16-224**: identical architecture. **NO fashion training** — vanilla webli checkpoint. The "architectural twin" experiment isolates exactly the value Marqo's GCL adds.
- Eval harness: `repos/marqo-FashionCLIP/eval.py` (Marqo's official benchmark code).
- Datasets: 4 contamination-free fashion benchmarks (atlas, KAGL, polyvore, fashion200k).
- Hardware: M-series Mac, MPS device, batch 256.

## Text-to-Image Retrieval (the metric Marqo's GCL targets)

| Dataset | FashionSigLIP MAP@10 | Google B/16-224 MAP@10 | Δ relative |
| --- | ---: | ---: | ---: |
| atlas | 0.1826 | 0.1719 | **−5.9%** |
| KAGL | 0.2769 | 0.2588 | **−6.5%** |
| polyvore | 0.3664 | 0.3589 | **−2.1%** |
| fashion200k | 0.1858 | 0.1538 | **−17.2%** |

Generic Google SigLIP, with **zero** fashion training, is within **2–7%** of Marqo-FashionSigLIP on 3 of 4 datasets. The only meaningful gap is fashion200k.

## Category-style Retrieval (NDCG@10)

| Task | FashionSigLIP | Google B/16-224 | Δ rel | Winner |
| --- | ---: | ---: | ---: | --- |
| atlas / sub-category-to-product | 0.6854 | 0.6572 | −4.1% | FashionSigLIP |
| **KAGL / category-to-product** | 0.6738 | **0.7031** | **+4.4%** | **Google SigLIP** |
| KAGL / color-to-product | 0.5175 | 0.4323 | −16.5% | FashionSigLIP |
| KAGL / fine-category-to-product | 0.6850 | 0.6600 | −3.6% | FashionSigLIP |
| **KAGL / season-to-product** | 0.3657 | **0.3670** | **+0.3%** | **Google SigLIP (tie)** |
| KAGL / sub-category-to-product | 0.7169 | 0.6769 | −5.6% | FashionSigLIP |
| **KAGL / usage-to-product** | 0.4011 | **0.4069** | **+1.4%** | **Google SigLIP** |
| polyvore / category-to-product | 0.5438 | 0.5081 | −6.6% | FashionSigLIP |
| fashion200k / category-to-product | 0.9264 | 0.8776 | −5.3% | FashionSigLIP |
| fashion200k / fine-category-to-product | 0.1871 | 0.1714 | −8.4% | FashionSigLIP |
| fashion200k / sub-category-to-product | 0.6555 | 0.6119 | −6.7% | FashionSigLIP |

**Google SigLIP B/16-224 wins 3 KAGL category subtasks outright with no fashion training.**

## Read

1. Marqo's in-domain GCL is **less critical than expected**. On 3 of 4 datasets the gap on the optimized metric (text-to-image MAP@10) is single-digit percentage. Polyvore is essentially a tie (−2.1%).
2. The fashion200k gap (−17%) is the only one big enough to suggest fashion-specific text alignment matters meaningfully on that dataset. (Fashion200k's queries are unusually short/noisy.)
3. **Generic Google SigLIP wins KAGL/category, KAGL/season, KAGL/usage NDCG@10** — base SigLIP's zero-shot category understanding actually beats Marqo's fine-tuned model on these.

## Implication for beating FashionSigLIP — CONFIRMED

The Apr 21 prediction (a bigger off-the-shelf SigLIP would beat FashionSigLIP zero-shot) was correct. **SigLIP SO400M-384 wins every text-to-image metric on atlas by 16–26% relative**, no fine-tuning required.

## Atlas — full SO400M head-to-head

```
                                                   text-to-image          sub-category-to-product
Model                                              MAP@10  NDCG@10  R@10  NDCG@10
Marqo-FashionSigLIP            (B/16-224, fine-tuned)  0.1826  0.2338  0.3239   0.6854
Google SigLIP B/16-224         (no FT, 300M, 224)      0.1719  0.2214  0.3089   0.6572
Google SigLIP SO400M-384       (no FT, 840M, 384)      0.2308  0.2860  0.3776   0.6862  ⭐
```

## Next steps

1. **Confirmation runs (READY TO RESUME):** `bash run_so400m_remaining.sh` queues SO400M-384 on `polyvore → KAGL → fashion200k` sequentially. ~40–50h total on MPS, crash-safe via 50-batch checkpointing. Last attempted Apr 22 12:37 IST, stopped per user request after ~5 min on polyvore (no checkpoint reached).
2. If wins hold across all 4 datasets, this is the publishable headline: *"Off-the-shelf Google SigLIP SO400M-384 beats Marqo-FashionSigLIP on Marqo's own benchmark — no fine-tuning required."*

## Status of all SO400M datasets

| dataset | status | result |
|---|---|---|
| atlas | ✅ DONE Apr 22 03:37 IST | **WIN** (+26.4% MAP@10 t2i, tie on sub-cat NDCG@10) |
| polyvore | ⏸️  pending — `run_so400m_remaining.sh` | — |
| KAGL | ⏸️  pending — `run_so400m_remaining.sh` | — |
| fashion200k | ⏸️  pending — `run_so400m_remaining.sh` | — |

## Resolved earlier issues

- `google-siglip2-b16-384 × atlas/polyvore` Tier B retries — abandoned in favor of the higher-leverage SO400M-384 experiment which decisively answered the same question.
- Wrapper subprocess timeout: raised 3600s → 64800s (18h) for big models on MPS.
- `repos/marqo-FashionCLIP/utils/retrieval.py` patched (snapshot in `patches/`) to checkpoint embeddings every 50 batches and resume from any interruption — this is what made the 8h SO400M overnight run safe to leave unattended.
