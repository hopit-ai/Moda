# MoDA Phase 1/2 Screener Leaderboard — POLYVORE corpus=10000

_Generated: 2026-04-27 15:28 UTC_

Stratified subsampling: every test query keeps all of its positives, 
remainder filled with random non-positives (fixed seed). Absolute Recall@K 
values are inflated vs full-corpus eval; the **relative ordering** between 
models is what matters for screening.

## Text-to-Image (the metric Marqo's GCL targets)

| Rank | Model | MAP@10 | NDCG@10 | Recall@10 | MRR | Queries used |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | MoDA Path 1: SigLIP-2 B/16/384 distilled from fusion teacher (KL on scores, 2000 steps) | 0.6150 | 0.6649 | 0.8220 | 0.6150 | dropped 0 |
