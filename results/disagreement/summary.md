# Disagreement analysis — FashionSigLIP ⊕ SigLIP-2 B/16/384 (full corpus)

Per-query bucketisation of the rank of the gold doc under each model.

Notation: A = Marqo-FashionSigLIP, B = Google SigLIP-2 B/16/384, F = score-mean fusion.

## 1. Hit@10 and the perfect-router ceiling

| Dataset | A hit@10 | B hit@10 | Fusion hit@10 | Oracle (best of A,B) | Headroom for fusion |
| --- | ---: | ---: | ---: | ---: | ---: |
| fashion200k | 0.3830 | 0.3400 | **0.4060** | 0.4605 | +0.0545 |
| atlas | 0.4200 | 0.4365 | **0.4465** | 0.5160 | +0.0695 |
| polyvore | 0.5840 | 0.6225 | **0.6285** | 0.6725 | +0.0440 |
| KAGL | 0.5220 | 0.5505 | **0.5665** | 0.6320 | +0.0655 |

**Reading:** `Oracle - Fusion` is the residual headroom available to a *better* fusion of the same two models. 
If it's small, score-mean is already near-optimal. If it's large, a learned router could help.

## 2. Failure mode buckets

| Dataset | BOTH_HIT@10 | A_only@10 | B_only@10 | BOTH_MISS@10 | …of which RECALL_CLIFF (miss@100) | …of which RERANKER_REACHABLE | FUSION_RECOVERS@10 | FUSION_HURTS@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fashion200k | 26.2% | 12.1% | 7.8% | 54.0% | **22.4%** | 31.6% | 1.2% | 6.7% |
| atlas | 34.0% | 8.0% | 9.6% | 48.4% | **19.2%** | 29.1% | 0.8% | 7.7% |
| polyvore | 53.4% | 5.0% | 8.8% | 32.8% | **11.9%** | 20.9% | 0.7% | 5.1% |
| KAGL | 44.0% | 8.2% | 11.0% | 36.8% | **8.7%** | 28.1% | 0.9% | 7.5% |

**How to read this table:**

- **BOTH_HIT@10**: easy queries; fusion neither helps nor hurts.
- **A_only@10 / B_only@10**: complementary wins. These are the slices where each model already captures something the other misses; fusion preserves them.
- **BOTH_MISS@10**: the population we still need to fix.
  - **RECALL_CLIFF (miss@100)**: gold doc is not in either model's top-100. **A re-ranker cannot fix these.** This is the *upper bound* on the value of any rank-stage improvement on these two encoders. To shrink this we need either (a) a third tower with different geometry (e.g. DINOv2), (b) data work (synthetic captions, query expansion), or (c) a sparse retriever (BM25/SPLADE) added to the recall pool.
  - **RERANKER_REACHABLE**: both miss top-10 but at least one has gold in top-100. A learned re-ranker over the union of top-100s *could* recover these.
- **FUSION_RECOVERS@10**: queries where *neither* solo model put gold in top-10 but the fusion did. This is the slice that justifies score-mean fusion existing at all.
- **FUSION_HURTS@10**: queries where one of the solo models had gold in top-10 but fusion pushed it out. The cost of averaging.

## 3. What this means for the next step

**If RECALL_CLIFF is large (e.g. >10%)**: model-side work has a hard ceiling. Highest-EV moves are data-side or 3rd-tower (different geometry).

**If RERANKER_REACHABLE is large**: a small cross-encoder or a learned re-ranker on the fusion's top-100 could capture meaningful headroom.

**If FUSION_HURTS is large relative to FUSION_RECOVERS**: the score-mean fusion is destabilising the strong-model wins; a calibrated or learned router beats vanilla mean.
