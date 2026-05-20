# v5 Phase E — Final Evaluation

_Generated: 2026-05-18 10:16:20_

## Per-benchmark MRR (1K-subsample protocol)

| Benchmark | SL2 base | v5 trained | Δ vs SL2 | FSL (target) | Δ vs FSL |
|---|---:|---:|---:|---:|---:|
| fashion200k | 0.4145 | **0.4201** | +1.3% | 0.4551 | -7.7% |
| atlas | 0.4910 | **0.4884** | -0.5% | 0.4226 | +15.6% |
| polyvore | 0.7745 | **0.7783** | +0.5% | 0.7402 | +5.1% |
| KAGL | 0.5570 | **0.5585** | +0.3% | 0.5805 | -3.8% |
| **mean** | 0.5593 | **0.5613** | +0.4% | 0.5496 | +2.1% |

## Decision

Per PLAN_V5 §6.3 decision matrix:
- Beat FSL on all 4 by ≥3%, p<0.0125 → **ship**
- Beat FSL on 3/4 by ≥3% → **Phase F surgical fix**
- Beat FSL on ≤2/4 → **diagnose**
- Match or worse → **Plan v6 (need scale or larger backbone)**