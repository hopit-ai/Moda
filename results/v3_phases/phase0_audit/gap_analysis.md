# Phase 0 — Gap Analysis

_Generated: 2026-05-02 17:14 IST_
_Corpus size: 3000 per benchmark, seed=42_

## 1. Aggregate MAP@10 by dataset

| Dataset | FSL MAP@10 | Base SigLIP MAP@10 | Delta% | FSL wins? |
|---|---:|---:|---:|---|
| fashion200k | 0.3859 | 0.3809 | +1.3% | **Yes** |
| atlas | 0.6919 | 0.6982 | -0.9% | No |
| polyvore | 0.5783 | 0.5125 | +12.8% | **Yes** |
| KAGL | 0.6779 | 0.6268 | +8.2% | **Yes** |

## 2. Strata where FSL beats base SigLIP by >5% (must close these gaps)

| Dataset | L1 category | FSL MAP@10 | Base MAP@10 | Delta% | N queries |
|---|---|---:|---:|---:|---:|
| polyvore | bottoms | 0.5318 | 0.4245 | +25.3% | 15 |
| polyvore | bags | 0.4782 | 0.3877 | +23.3% | 10 |
| KAGL | tops | 0.7380 | 0.6088 | +21.2% | 6 |
| polyvore | beauty | 0.4646 | 0.3899 | +19.2% | 17 |
| polyvore | other | 0.4506 | 0.3798 | +18.7% | 31 |
| polyvore | tops | 0.6489 | 0.5478 | +18.5% | 9 |
| KAGL | bags | 0.8098 | 0.6926 | +16.9% | 5 |
| polyvore | home | 0.8167 | 0.7222 | +13.1% | 21 |
| polyvore | shoes | 0.6179 | 0.5580 | +10.7% | 14 |
| KAGL | bottoms | 0.7767 | 0.7207 | +7.8% | 9 |
| polyvore | dresses | 0.5476 | 0.5115 | +7.1% | 5 |

## 3. Strata where base SigLIP beats FSL (must preserve these)

| Dataset | L1 category | Base MAP@10 | FSL MAP@10 | Delta% | N queries |
|---|---|---:|---:|---:|---:|
| _(none)_ | | | | | |

## 4. Strata where both models are weak (MAP@10 < 0.3)

| Dataset | L1 category | FSL MAP@10 | Base MAP@10 | N queries |
|---|---|---:|---:|---:|
| _(none)_ | | | | |

## 5. L1 category distribution across benchmarks (query count)

| L1 category | fashion200k | atlas | polyvore | KAGL | Total |
|---|---:|---:|---:|---:|---:|
| accessories | 0 | 0 | 17 | 9 | 26 |
| bags | 0 | 0 | 10 | 5 | 15 |
| beauty | 0 | 0 | 17 | 0 | 17 |
| bottoms | 1 | 0 | 15 | 9 | 25 |
| dresses | 489 | 4 | 5 | 4 | 502 |
| home | 0 | 0 | 21 | 0 | 21 |
| intimates | 0 | 2 | 3 | 0 | 5 |
| other | 2 | 0 | 31 | 1 | 34 |
| outerwear | 6 | 3 | 5 | 1 | 15 |
| shoes | 0 | 0 | 14 | 8 | 22 |
| swimwear | 0 | 0 | 3 | 0 | 3 |
| tops | 2 | 0 | 9 | 6 | 17 |

## 6. Recommended oversampling weights for Phase 1

| L1 category | Avg FSL advantage over base | Recommended weight | Rationale |
|---|---:|---:|---|
| accessories | -2.5% | 1.0x | Parity — maintain |
| bags | +20.1% | 2.5x | Large FSL advantage — priority target |
| beauty | +19.2% | 2.5x | Large FSL advantage — priority target |
| bottoms | +6.6% | 2.0x | Moderate FSL advantage — oversample |
| dresses | +13.8% | 2.5x | Large FSL advantage — priority target |
| home | +13.1% | 2.5x | Large FSL advantage — priority target |
| intimates | +2.6% | 1.5x | Slight FSL advantage |
| other | +9.3% | 2.0x | Moderate FSL advantage — oversample |
| outerwear | -15.0% | 0.8x | Base SigLIP already better — light touch |
| shoes | +7.6% | 2.0x | Moderate FSL advantage — oversample |
| swimwear | +1.9% | 1.5x | Slight FSL advantage |
| tops | +19.9% | 2.5x | Large FSL advantage — priority target |
