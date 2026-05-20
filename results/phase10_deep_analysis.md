# Phase 10 — Deep Error Analysis & Next Steps

## Results Summary

| Benchmark | FSL Baseline | Our Best (FSL-FT) | Delta | Target (+10%) | Gap to Target |
|---|---:|---:|---:|---:|---:|
| fashion200k | 0.3859 | **0.6543** | **+69.6%** | 0.4245 | PASSED |
| polyvore | 0.5783 | **0.7341** | **+26.9%** | 0.6361 | PASSED |
| atlas | 0.6919 | 0.3411 | -50.7% | 0.7611 | -0.4200 |
| KAGL | 0.6779 | 0.5524 | -18.5% | 0.7457 | -0.1933 |

**Status**: Beats +10% on 2/4 benchmarks. Catastrophic forgetting on atlas and KAGL.

---

## Root Cause Analysis

### Why we WIN on fashion200k and polyvore:
- Our 500K training data heavily overlaps with these benchmark categories (dresses, tops, outfits)
- Multi-field GCL gives us precise attribute-based retrieval (color + material + style)
- These benchmarks reward fine-grained clothing discrimination — exactly what we trained for

### Why we LOSE on atlas (-51%):

Atlas is **90% Western Wear** (shirts, shorts, jeans, blazers, jackets) — categories we DO have in our training data. So the issue is NOT a category gap.

**Real cause: TEXT STYLE MISMATCH**
- Atlas texts: `"Plain Dupion Silk Dhoti Kurta in Fawn"` — structured as [adjective] [material] [garment] in [color]
- Our training texts: `"Women's Red Floral Maxi Dress"` — more natural language style
- Fine-tuning the text encoder on our style **broke alignment** for atlas-style structured descriptions
- Also: 34 unique sub-categories many of which are Indian ethnic wear (Dhoti, Kurta, Lehenga, Saree, Salwar) — our training has minimal coverage here

### Why we LOSE on KAGL (-18.5%):

**Partial category gap + text style mismatch:**
- KAGL has 7 category1 types: Apparel (48%), **Accessories (25%)**, **Footwear (21%)**, **Personal Care (5%)**
- Key missing categories: Watches (2542), Perfume (609), Lipstick (315), Nail Polish (329), Earrings (417)
- KAGL texts are brand-heavy: `"Nike Men As 7 Sw Temp Grey Shorts"` — abbreviated, brand-first style
- Our training texts don't have this brand-abbreviated pattern

### The Fundamental Problem:

**We're only training the TEXT encoder** (`text-only` scope), and our training data has a DIFFERENT text distribution than atlas/KAGL. When we fine-tune the text encoder to produce good embeddings for OUR text style, we destroy its ability to produce good embeddings for atlas/KAGL text styles.

The anchor regularization (lambda 0.3-0.5) was too weak to prevent this drift because our 500K training pairs are so numerous that they overwhelm the regularization signal.

---

## Our Training Data Profile

| L1 Category | Count | % |
|---|---:|---:|
| tops | 109,721 | 21.9% |
| other | 106,718 | 21.3% |
| accessories | 51,925 | 10.4% |
| home | 43,146 | 8.6% |
| shoes | 40,785 | 8.2% |
| bottoms | 40,464 | 8.1% |
| dresses | 35,784 | 7.2% |
| outerwear | 29,914 | 6.0% |
| bags | 16,174 | 3.2% |
| intimates | 14,652 | 2.9% |
| swimwear | 7,191 | 1.4% |
| activewear | 2,014 | 0.4% |
| beauty | 1,555 | 0.3% |

**Total: 500,043 pairs + 15,000 enriched descriptions**

Key gaps vs benchmarks:
- No watches, jewelry, perfume, makeup data
- No Indian ethnic wear (saree, kurta, lehenga) in significant quantity
- No brand-abbreviated text patterns
- No structured "[material] [garment] in [color]" patterns

---

## Potential Strategies (ranked by likelihood of success)

### Strategy A: Much Stronger Anchor + Lower LR (Quick, low risk)
- Lambda anchor: 0.8-0.9 (was 0.3)
- LR: 5e-7 (was 1e-6)
- Rationale: Preserve FSL's atlas/KAGL knowledge while making smaller, targeted improvements
- Risk: May not improve fashion200k/polyvore enough
- ETA: ~3.5 hours

### Strategy B: Mix benchmark-style data into training (Medium effort)
- Add atlas/KAGL-style text patterns to training WITHOUT using benchmark data
- Source: Stream from GS-10M filtering for watches, ethnic wear, brand products
- Generate synthetic descriptions in atlas format ("[material] [garment] in [color]")
- Risk: Still text-only — may not be enough
- ETA: ~5 hours (data prep + training)

### Strategy C: Train BOTH encoders — `text-image-light` scope (Higher risk, higher reward)
- Unlock vision encoder with very low LR (1e-7 for vision, 1e-6 for text)
- This allows the IMAGE embeddings to adapt to our loss without breaking text alignment
- Rationale: If both encoders move together, neither drifts far from the original space
- Risk: Vision encoder catastrophic forgetting, higher memory
- ETA: ~4 hours

### Strategy D: Two-stage curriculum training (Best theoretical approach)
- Stage 1: Train on atlas+KAGL-style data (ethnic wear, accessories, brands) with low LR
- Stage 2: Then train on fashion200k/polyvore-style data (clothing, outfits)
- Use strong anchor in both stages
- Risk: Complex, takes 2x training time
- ETA: ~7 hours

### Strategy E: Freeze & only train projection heads (Safest)
- Add small MLP projection heads on top of FSL's frozen embeddings
- Train only these heads (~2M params) on our multi-field loss
- Cannot degrade FSL's base representations AT ALL
- Risk: Limited improvement capacity
- ETA: ~1 hour

---

## Recommended Path: Strategy A first, then E as fallback

**Strategy A** is the fastest test — if we can keep atlas/KAGL close to FSL while still gaining on fashion200k/polyvore, we win. The key insight is that our current improvement on fashion200k (+69.6%) is WAY more than we need (+10%), so we have massive headroom to sacrifice some fashion200k performance in exchange for preserving atlas/KAGL.

**If A fails**: Strategy E (projection heads) guarantees no degradation and just needs to add +10% on top.

---

_Generated: 2026-05-03 20:30_
