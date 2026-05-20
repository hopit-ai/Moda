# Phase 7 — Category Embedding Alignment Report

_Generated: 2026-05-03 03:13 IST_

## 1. Category-Label-to-Image Alignment (higher = better category retrieval)

For each L1 category, we encode the category name and compute mean cosine similarity
to all images in that category. FSL should have higher alignment on categories it wins.

### fashion200k

| Category | FSL Align | Base Align | Delta | FSL Advantage? | N images |
|---|---:|---:|---:|---|---:|
| bottoms | 0.0330 | 0.0444 | -0.0114 | NO | 6 |
| dresses | 0.0521 | 0.0580 | -0.0060 | NO | 2943 |
| other | -0.0196 | -0.0246 | +0.0050 | ~tie | 18 |
| outerwear | 0.0147 | 0.0236 | -0.0089 | NO | 15 |
| tops | 0.0265 | 0.0416 | -0.0151 | NO | 18 |

### atlas

| Category | FSL Align | Base Align | Delta | FSL Advantage? | N images |
|---|---:|---:|---:|---|---:|
| dresses | 0.0019 | 0.0144 | -0.0124 | NO | 724 |
| intimates | 0.0172 | 0.0179 | -0.0008 | ~tie | 337 |
| outerwear | 0.0390 | 0.0278 | +0.0112 | YES | 1939 |

### polyvore

| Category | FSL Align | Base Align | Delta | FSL Advantage? | N images |
|---|---:|---:|---:|---|---:|
| accessories | 0.0359 | 0.0348 | +0.0011 | ~tie | 641 |
| bags | 0.0634 | 0.0496 | +0.0139 | YES | 307 |
| beauty | 0.0484 | 0.0360 | +0.0124 | YES | 238 |
| bottoms | 0.0540 | 0.0548 | -0.0008 | ~tie | 323 |
| dresses | 0.0449 | 0.0497 | -0.0048 | ~tie | 116 |
| home | 0.0137 | 0.0164 | -0.0027 | ~tie | 159 |
| intimates | 0.0763 | 0.0691 | +0.0072 | YES | 10 |
| other | -0.0013 | -0.0119 | +0.0106 | YES | 282 |
| outerwear | 0.0593 | 0.0572 | +0.0021 | ~tie | 176 |
| shoes | 0.0609 | 0.0501 | +0.0109 | YES | 363 |
| swimwear | 0.0739 | 0.0691 | +0.0048 | ~tie | 10 |
| tops | 0.0444 | 0.0496 | -0.0052 | NO | 374 |

### KAGL

| Category | FSL Align | Base Align | Delta | FSL Advantage? | N images |
|---|---:|---:|---:|---|---:|
| accessories | 0.0251 | 0.0275 | -0.0023 | ~tie | 269 |
| bags | 0.0606 | 0.0596 | +0.0010 | ~tie | 95 |
| bottoms | 0.0483 | 0.0606 | -0.0123 | NO | 264 |
| dresses | 0.0399 | 0.0468 | -0.0069 | NO | 251 |
| other | -0.0056 | -0.0113 | +0.0056 | YES | 3 |
| outerwear | 0.0569 | 0.0597 | -0.0027 | ~tie | 16 |
| shoes | 0.0575 | 0.0510 | +0.0065 | YES | 749 |
| tops | 0.0295 | 0.0470 | -0.0174 | NO | 1352 |

## 2. Per-Query Alignment (specific benchmark queries, not just L1 labels)

Mean cosine similarity between each query and its own positive images.

| Benchmark | FSL Mean Align | Base Mean Align | Delta | FSL Better? |
|---|---:|---:|---:|---|
| fashion200k | 0.0982 | 0.0996 | -0.0014 | ~tie |
| atlas | 0.0897 | 0.0849 | +0.0048 | ~tie |
| polyvore | 0.0705 | 0.0619 | +0.0086 | YES |
| KAGL | 0.0741 | 0.0743 | -0.0002 | ~tie |

### Per-L1 Query Alignment (for key failing strata)

**polyvore** (key strata):

| Stratum | FSL Align | Base Align | Delta | N queries |
|---|---:|---:|---:|---:|
| bottoms | 0.0778 | 0.0773 | +0.0006 | 15 |
| tops | 0.0679 | 0.0663 | +0.0016 | 9 |
| dresses | 0.0531 | 0.0584 | -0.0054 | 5 |
| bags | 0.0728 | 0.0561 | +0.0167 | 10 |
| beauty | 0.0883 | 0.0722 | +0.0161 | 14 |
| home | 0.0770 | 0.0656 | +0.0114 | 20 |

**KAGL** (key strata):

| Stratum | FSL Align | Base Align | Delta | N queries |
|---|---:|---:|---:|---:|
| bottoms | 0.0783 | 0.0826 | -0.0043 | 9 |
| tops | 0.0692 | 0.0772 | -0.0080 | 6 |
| dresses | 0.0773 | 0.0751 | +0.0022 | 4 |
| bags | 0.0832 | 0.0760 | +0.0071 | 5 |

**fashion200k** (key strata):

| Stratum | FSL Align | Base Align | Delta | N queries |
|---|---:|---:|---:|---:|
| bottoms | 0.0873 | 0.0995 | -0.0122 | 1 |
| tops | 0.0570 | 0.0677 | -0.0107 | 2 |
| dresses | 0.0984 | 0.0997 | -0.0013 | 489 |

## 3. Text Embedding Drift (FSL vs Base)

How far FSL moved text embeddings from the base model (cosine distance).
Higher = more change from GCL training.

| Benchmark | Mean Drift | Median Drift | Max Drift |
|---|---:|---:|---:|
| fashion200k | 0.0216 | 0.0193 | 0.1134 |
| atlas | 0.0070 | 0.0054 | 0.0144 |
| polyvore | 0.0046 | 0.0037 | 0.0157 |
| KAGL | 0.0047 | 0.0041 | 0.0110 |

### Per-L1 Category Drift

Categories where FSL moved embeddings the most (these are where GCL had most effect):

**polyvore:**

| Category | Mean Drift | N queries |
|---|---:|---:|
| swimwear | 0.0103 | 3 |
| dresses | 0.0076 | 5 |
| bottoms | 0.0063 | 15 |
| tops | 0.0056 | 9 |
| outerwear | 0.0056 | 5 |
| shoes | 0.0052 | 14 |
| intimates | 0.0050 | 3 |
| bags | 0.0043 | 10 |
| accessories | 0.0040 | 17 |
| other | 0.0038 | 35 |
| home | 0.0035 | 20 |
| beauty | 0.0034 | 14 |

**KAGL:**

| Category | Mean Drift | N queries |
|---|---:|---:|
| shoes | 0.0062 | 8 |
| tops | 0.0054 | 6 |
| other | 0.0049 | 1 |
| bottoms | 0.0046 | 9 |
| dresses | 0.0043 | 4 |
| bags | 0.0038 | 5 |
| outerwear | 0.0037 | 1 |
| accessories | 0.0036 | 9 |

## 4. Interpretation

### What this tells us about the gap:

- If FSL has much higher category alignment on the strata it wins, the gap is **text→category grounding** (our category-loss approach is correct but insufficient).
- If alignment is similar but FSL still wins on retrieval, the gap is **image-side discrimination** (frozen image tower is limiting).
- If drift is large on the winning strata, FSL's GCL moved text embeddings significantly — we need either more training signal or different scope to match.
