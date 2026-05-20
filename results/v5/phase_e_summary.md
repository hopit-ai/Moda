# v5 Phase E — Final Evaluation

_Generated: 2026-05-16 23:45:13_

## Per-benchmark MRR (1K-subsample protocol)

| Benchmark | SL2 base | v5 trained | Δ vs SL2 | FSL (target) | Δ vs FSL |
|---|---:|---:|---:|---:|---:|
| fashion200k | 0.4145 | **0.0000** | -100.0% | 0.4551 | -100.0% |
| atlas | 0.4910 | **0.0000** | -100.0% | 0.4226 | -100.0% |
| polyvore | 0.7745 | **0.0000** | -100.0% | 0.7402 | -100.0% |
| KAGL | 0.5570 | **0.0000** | -100.0% | 0.5805 | -100.0% |

## Decision

Per PLAN_V5 §6.3 decision matrix:
- Beat FSL on all 4 by ≥3%, p<0.0125 → **ship**
- Beat FSL on 3/4 by ≥3% → **Phase F surgical fix**
- Beat FSL on ≤2/4 → **diagnose**
- Match or worse → **Plan v6 (need scale or larger backbone)**