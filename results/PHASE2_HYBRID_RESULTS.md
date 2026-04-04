# MODA Phase 2 — Hybrid Retrieval Results
_Generated: 2026-04-02 20:02 UTC_

## Method Comparison (H&M Benchmark, 4,078 queries × 105,542 articles)

| Method | nDCG@5 | nDCG@10 | MRR | Recall@10 | P@10 | vs Phase 1 Best |
| --- | --- | --- | --- | --- | --- | --- |
| BM25 only | 0.1158 | 0.1432 | 0.1675 | 0.0209 | 0.0801 | +48.2% ✅ |
| Dense (fashion-siglip) | 0.0579 | 0.0765 | 0.0904 | 0.0100 | 0.0377 | -20.8% 🔴 |
| Hybrid BM25+Dense (fashion-siglip) | 0.0983 | 0.1243 | 0.1526 | 0.0173 | 0.0658 | +28.7% ✅ |
| Dense (clip) | 0.0751 | 0.0966 | 0.1154 | 0.0125 | 0.0485 | -0.0% 🔴 |
| Hybrid BM25+Dense (clip) | 0.1068 | 0.1342 | 0.1599 | 0.0189 | 0.0724 | +38.9% ✅ |

### Phase 1 Baselines (for reference)

| Model | nDCG@10 | MRR | Recall@10 | P@10 |
| --- | --- | --- | --- | --- |
| Dense (clip) | 0.0966 | 0.1154 | 0.0125 | 0.0485 |
| Dense (fashion-clip) | 0.0886 | 0.1088 | 0.0125 | 0.0477 |
| Dense (fashion-siglip) | 0.0765 | 0.0904 | 0.0100 | 0.0377 |