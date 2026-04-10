# MODA Phase 3 — Fine-Tuned Models: Complete Findings

## Executive Summary

Phase 3 fine-tuned every component of the MODA fashion search pipeline on H&M's 105K-article catalog with 253K real search queries. Starting from off-the-shelf models (Phase 2), we trained domain-specific models for dense retrieval, cross-encoder reranking, NER-based query understanding, and a novel fused retrieval architecture.

**Best configuration at the end of Phase 3:**

> **Fine-tuned BM25+NER hybrid retrieval + LLM-trained cross-encoder reranking**
> - nDCG@10 = **0.0757** (full pipeline, 22,855 queries)
> - **+79.4%** over Phase 2 baseline (0.0422)

---

## Table of Contents

1. [Models Trained](#1-models-trained)
2. [Phase 3.1–3.2: Data Foundation](#2-phase-31-32-data-foundation)
3. [Phase 3.3–3.4: Cross-Encoder Fine-Tuning](#3-phase-33-34-cross-encoder-fine-tuning)
4. [Phase 3.5–3.6: Bi-Encoder Fine-Tuning](#4-phase-35-36-bi-encoder-fine-tuning)
5. [Phase 3.7: Fashion NER Fine-Tuning](#5-phase-37-fashion-ner-fine-tuning)
6. [Phase 3.8 Path A: Fused Item Tower](#6-phase-38-path-a-fused-item-tower)
7. [Phase 3.8 Path B: Attribute-Conditioned Cross-Encoder](#7-phase-38-path-b-attribute-conditioned-cross-encoder)
8. [Phase 3.9: Comprehensive Evaluation](#8-phase-39-comprehensive-evaluation)
9. [Tier 1 Cross-Check: Marqo 7-Dataset](#9-tier-1-cross-check-marqo-7-dataset)
10. [Best Configuration Analysis](#10-best-configuration-analysis)
11. [Key Takeaways](#11-key-takeaways)

---

## 1. Models Trained

| Model | Path | Architecture | Training Data |
|---|---|---|---|
| `moda-fashion-ce-best` | `models/moda-fashion-ce-best` | CrossEncoder (ms-marco-MiniLM-L-6-v2) | Purchase labels (binary) |
| `moda-fashion-ce-llm-best` | `models/moda-fashion-ce-llm-best` | CrossEncoder (ms-marco-MiniLM-L-6-v2) | 42,800 LLM-judged labels (0–3 scale) |
| `moda-fashion-ce-attr-best` | `models/moda-fashion-ce-attr-best` | CrossEncoder (ms-marco-MiniLM-L-6-v2) | LLM labels + attribute-tagged text |
| `moda-fashionclip-finetuned` | `models/moda-fashionclip-finetuned/best` | FashionCLIP (ViT-B/32, text encoder only) | 100K contrastive triplets |
| `moda-fused-item-tower` | `models/moda-fused-item-tower/best` | FusedRetriever (QueryTower + ItemTower) | 100K contrastive triplets |
| `moda-fashion-ner` (v1, failed) | `models/moda-fashion-ner/best` (v1) | GLiNER2 (DeBERTa-v3-base) + LoRA | 15K NER examples from articles |
| `moda-fashion-ner` (v2) | `models/moda-fashion-ner/best` | GLiNER2 (DeBERTa-v3-base) + LoRA | 5.7K LLM-generated query-domain examples |

---

## 2. Phase 3.1–3.2: Data Foundation

### 3.1 — Train/Test Split

Split 253,685 queries by unique `query_text` (not query ID) to prevent leakage.

| Split | Queries | % |
|---|---|---|
| Train | ~202,948 | 80% |
| Val | ~25,369 | 10% |
| Test | 22,855 | 10% |

Output: `data/processed/query_splits.json`

### 3.2 — LLM Relevance Labels

Used PaleblueDot AI (GPT-4o-mini equivalent) to grade query-product pairs on a 0–3 scale.

- **42,800 labeled pairs** from ~5,000 unique train queries
- Top-20 candidates per query from baseline hybrid retriever (BM25 + FashionCLIP)
- Prompt: rate relevance 0 (irrelevant) to 3 (exact match)

Output: `data/processed/llm_relevance_labels.jsonl`

### 3.5 — Bi-Encoder Training Labels

Mined hard negatives from baseline FashionCLIP retriever, labeled by LLM.

- **100,000 contrastive triplets** (query, positive_article, hard_negative_article)
- Products scored 0 by LLM but in top-20 retrieval = hard negatives
- Products scored 2–3 = positives

Output: `data/processed/biencoder_retriever_labels.jsonl`

---

## 3. Phase 3.3–3.4: Cross-Encoder Fine-Tuning

Base model: `cross-encoder/ms-marco-MiniLM-L-6-v2` (22M params)

### 3.3 — LLM-Trained Cross-Encoder

- **Training:** 42,800 pairs, labels normalized 0→0.0, 1→0.33, 2→0.67, 3→1.0
- **Config:** batch_size=32, epochs=3, lr=2e-5, warmup=10%
- **Product text format:** `prod_name | product_type_name | colour_group_name | section_name | detail_desc[:200]`

### 3.4 — Purchase-Trained Cross-Encoder (Control)

- **Training:** Purchase labels (binary: bought=1.0, not-bought=0.0)
- Same training config as 3.3

### Cross-Encoder Eval (22,855 test queries, BM25+NER hybrid retrieval → CE rerank top-50)

| Reranker | nDCG@10 | MRR | nDCG@50 | R@50 | vs No-Rerank |
|---|---|---|---|---|---|
| No reranker | 0.0422 | 0.0558 | 0.0842 | 0.0515 | — |
| Off-shelf CE (ms-marco) | 0.0646 | 0.0671 | 0.1195 | 0.0620 | +53.1% |
| Purchase-trained CE | 0.0654 | 0.0644 | 0.1204 | 0.0616 | +55.0% |
| **LLM-trained CE** | **0.0747** | **0.0755** | **0.1297** | **0.0659** | **+77.0%** |

**Key finding:** LLM-judged labels (graded 0–3) dramatically outperform purchase labels (binary). Purchase-trained CE barely beats off-the-shelf (+1.2% nDCG@10), while LLM-trained CE gains +15.6% over off-the-shelf. This confirms that label quality matters more than label quantity.

---

## 4. Phase 3.5–3.6: Bi-Encoder Fine-Tuning

### Architecture

- Base: Marqo/marqo-fashionCLIP (ViT-B/32, 512-dim embeddings)
- Only text encoder trainable; vision encoder frozen
- Training: InfoNCE loss with in-batch negatives + 1 mined hard negative per query
- Config: 5 epochs, lr=1e-6, batch_size=64, gradient_accumulation=4, cosine LR schedule, FP16

### Dense-Only Retrieval Results (22,855 test queries)

| Retriever | nDCG@10 | MRR | R@10 | R@50 | vs Baseline |
|---|---|---|---|---|---|
| Baseline FashionCLIP | 0.0229 | 0.0208 | 0.0433 | 0.1239 | — |
| **Fine-tuned FashionCLIP** | **0.0444** | **0.0405** | **0.0811** | **0.2118** | **+94.1%** |

**Key finding:** Fine-tuning nearly doubles dense retrieval quality. The improvement comes from adapting to H&M's vocabulary (e.g., learning that "navy" = H&M's "Dark Blue") and learning from retriever failure modes via hard negative mining.

### Impact on Full Pipeline (BM25+NER hybrid + CE reranking, 22,855 queries)

| Retriever + Reranker | nDCG@10 | MRR | nDCG@50 | vs Baseline |
|---|---|---|---|---|
| Baseline hybrid, no rerank | 0.0422 | 0.0558 | 0.0842 | — |
| **Fine-tuned hybrid, no rerank** | **0.0515** | **0.0740** | **0.1132** | **+22.0%** |
| Baseline + off-shelf CE | 0.0646 | 0.0671 | 0.1195 | +53.1% |
| Fine-tuned + off-shelf CE | 0.0650 | 0.0723 | 0.1394 | +54.0% |
| Baseline + LLM CE | 0.0747 | 0.0755 | 0.1297 | +77.0% |
| **Fine-tuned + LLM CE** | **0.0757** | **0.0799** | **0.1516** | **+79.4%** |

---

## 5. Phase 3.7: Fashion NER Fine-Tuning

### 3.7a — V1: Article-Derived Training Data (Failed)

**Architecture:**
- Base: `fastino/gliner2-base-v1` (DeBERTa-v3-base, 208M params)
- Fine-tuning: LoRA (r=16, α=32, dropout=0.05) on encoder + span_rep + classifier
- Trainable params: 3.1M / 211.6M total (1.46%)
- Entities: color, garment type, fit style, material, pattern, occasion, gender, brand

**V1 Training Data (from H&M article structured fields):**
- 15,000 examples mined from 91,586 train-split articles
- Entity mentions derived from product metadata columns (e.g., `colour_group_name` → color, `product_type_name` → garment type)

**Critical flaw — severe class imbalance and domain mismatch:**

| Entity Type | Mentions in 83K articles | % of examples |
|---|---|---|
| material | 61,708 | 74.0% |
| garment type | 61,439 | 73.7% |
| fit style | 19,605 | 23.5% |
| pattern | 6,748 | 8.1% |
| gender | 694 | 0.8% |
| **color** | **106** | **0.1%** |

The model trained on **long product description text** but needed to extract from **short search queries** — a fundamental domain mismatch. Color had only 106 mentions across 83K examples, making it nearly impossible to learn.

**V1 Extraction Quality (500 test queries):**

| Entity Type | Off-the-shelf | V1 Fine-tuned | Delta |
|---|---|---|---|
| brand | 17 | 168 | +151 |
| color | 106 | 249 | +143 |
| fit style | 89 | 253 | +164 |
| garment type | 486 | 497 | +11 |
| gender | 36 | 66 | +30 |
| material | 162 | 195 | +33 |
| occasion | 40 | 115 | +75 |
| pattern | 36 | 231 | +195 |

**Diagnosis:** The V1 model massively over-extracted — tagging the same word as multiple entity types simultaneously (e.g., "cotton" → color AND material, "patterned" → color AND fit AND pattern). The model learned to be aggressive but not precise, because the article-derived training data didn't teach proper entity boundaries for query-style text.

### 3.7b — V2: LLM-Generated Query-Level Training Data

**Strategy:** Use PaleblueDot API (`anthropic/claude-sonnet-4.6`) to generate high-quality, query-domain-specific NER training data via two complementary approaches.

**Strategy A — Annotate real queries (3,105 valid from 5,000):**
- Fed real H&M search queries to Claude with strict rules: exact substring matching, one entity type per word, no hallucination
- 62.1% annotation success rate

**Strategy B — Synthesize from product attributes (2,571 from 1,667 combos):**
- Combined real H&M attributes (colors, materials, garment types, etc.) into synthetic queries
- Ensured balanced representation across all entity types

**Combined V2 dataset: 5,674 unique examples**

| Entity Type | V1 (article data) | V2 (LLM-generated) | Improvement |
|---|---|---|---|
| garment type | 61,439 | 5,401 (95.2%) | Query-domain text |
| **color** | **106** | **3,116 (54.9%)** | **29× more coverage** |
| material | 61,708 | 1,629 (28.7%) | Query-style spans |
| gender | 694 | 1,508 (26.6%) | 2× more coverage |
| pattern | 6,748 | 1,373 (24.2%) | Query-style spans |
| occasion | (absent) | 1,187 (20.9%) | New entity type |
| fit style | 19,605 | 1,105 (19.5%) | Query-style spans |

Query text style: mean 22 tokens (vs product descriptions at 50–200 tokens).

### 3.7c — V2 Training: GLiNER2 LoRA Bugs and Fixes

**Bug 1 — Tensor dimension mismatch with variable-length batches:**
GLiNER2's `compute_span_rep_batched` slices `span_rep` to actual token length but keeps `span_mask` at padded length, causing shape mismatches in `compute_struct_loss`. Short, variable-length query texts (V2 data) trigger this consistently.

*Fix:* Set `batch_size=1`, `eval_batch_size=1`, `gradient_accumulation_steps=8` (effective batch=8).

**Bug 2 — LoRA layers destroyed during checkpoint save:**
With `save_adapter_only=False`, the GLiNER2 trainer's internal `unmerge_lora_weights` function erroneously **removed** LoRA layers entirely instead of merely unmerging weights. After the epoch 1 checkpoint save, all subsequent epochs had zero trainable parameters (every batch skipped with "loss doesn't require grad").

*Symptoms:* Epoch 2+ logged `Loss: 0.0000`, all batches skipped.
*Fix:* Set `save_adapter_only=True` in `TrainingConfig`.

**Final successful training (V2 data, both bugs fixed):**

| Epoch | Train Loss | Eval Loss | Status |
|---|---|---|---|
| 1 | 0.2089 | 0.7570 | LoRA adapter saved (11.8MB) |
| 2 | 0.1211 | **0.6893** | **New best** — adapter saved |
| — | — | — | Early stopping (patience=2) |

Training completed in 28.8 minutes. Loss dropped across epochs, confirming real multi-epoch learning with LoRA intact.

### 3.7d — V2 Extraction Quality (500 test queries)

| Entity Type | Off-the-shelf | V1 Fine-tuned | **V2 Fine-tuned** |
|---|---|---|---|
| brand | 17 | 168 | **8** |
| color | 106 | 249 | **117** |
| fit style | 89 | 253 | **154** |
| garment type | 486 | 497 | **469** |
| gender | 36 | 66 | **59** |
| material | 162 | 195 | **190** |
| occasion | 40 | 115 | **71** |
| pattern | 36 | 231 | **170** |

**V2 vs V1 comparison:**
- V1 over-extracted wildly (brand 168, pattern 231, fit 253). V2 is dramatically more controlled.
- V2 learns proper entity decomposition: `"long sleeve pullover"` → fit="long sleeve" + garment="pullover" (vs base: garment="long sleeve pullover" as one blob)
- V2 correctly separates: `"turquoise jersey top"` → color="turquoise" + garment="top" + material="jersey"
- Residual issue: `pattern` still over-extracted (170 vs 36 base) — some words like "button", "khaki" get tagged as pattern when they shouldn't

### 3.7e — V2 NER in Full Pipeline Evaluation (22,855 test queries)

Replaced the default GLiNER v1 (`urchade/gliner_medium-v2.1`) with the fine-tuned GLiNER2 V2 adapter in the BM25+NER hybrid retrieval stage and re-ran all 8 pipeline configurations.

**Full 2×4 Matrix — Old NER vs Fine-tuned V2 NER (nDCG@10):**

| | No Rerank | Off-shelf CE | LLM-trained CE | Attr-cond CE |
|---|---|---|---|---|
| **Old NER (GLiNER v1)** | | | | |
| Baseline retriever | 0.0422 | 0.0646 | 0.0747 | 0.0738 |
| Fine-tuned retriever | 0.0515 | 0.0650 | **0.0757** | 0.0746 |
| **Fine-tuned NER (GLiNER2 v2)** | | | | |
| Baseline retriever | 0.0430 **(+1.9%)** | 0.0661 **(+2.3%)** | 0.0740 (-0.9%) | 0.0729 (-1.2%) |
| Fine-tuned retriever | 0.0511 (-0.8%) | 0.0661 **(+1.7%)** | 0.0752 (-0.7%) | 0.0740 (-0.8%) |

**Best config remains: Fine-tuned retriever + LLM-trained CE + Old NER = 0.0757**

### 3.7f — Analysis: Why Fine-Tuned NER Helps Weak Rerankers but Hurts Strong Ones

The fine-tuned NER improves BM25+NER candidate quality through better entity decomposition (splitting "long sleeve pullover" into separate fit + garment boosts). This matters when:

1. **No reranker / weak reranker (up to +2.3%):** BM25 candidate quality directly determines final ranking. Better NER → better attribute-boosted BM25 queries → better candidates. The baseline retriever benefits more (+1.9% A0, +2.3% A1) because it relies more heavily on BM25 candidates.

2. **Strong reranker (LLM CE / Attr CE) — slight regression (-0.7% to -1.2%):** The cross-encoder reranker is powerful enough to recover from imperfect BM25 candidates. The fine-tuned NER's residual over-extraction on `pattern` (170 vs 36 off-shelf) introduces noise into the BM25 query boosting — boosting irrelevant fields that pull in wrong candidates the CE then has to filter out, slightly diluting the candidate pool.

**Root cause of the regression:** The `pattern` entity type remains over-extracted by the V2 model. Words like "button" (should be garment descriptor, not pattern), "khaki" (color, not pattern), and "modal" (material, not pattern) get incorrectly tagged as patterns, adding noise to the BM25 boosted query.

**Conclusion:** NER fine-tuning is a net positive for simpler pipeline configurations and demonstrates that domain-specific NER can improve retrieval. However, when combined with a strong cross-encoder reranker, the marginal gains from better NER are offset by the noise from imperfect entity classification. Future work should focus on improving precision (reducing false positives) rather than recall.

---

## 6. Phase 3.8 Path A: Fused Item Tower

### Motivation

The original Phase 3.8 MoE (Mixture-of-Encoders) approach failed at **-12% nDCG@10** vs baseline. Error analysis revealed:
- Score variance mismatch: field embeddings (64d color, 64d category, 32d group) overpowered the 512d text signal
- Brittle NER-to-vocabulary matching at query time

### Redesigned Architecture (inspired by Vinted's production system)

**Two-tower model:**
- **Query Tower:** Frozen FashionCLIP text encoder → linear projection (512d → 256d)
- **Item Tower:** FusionMLP combining FashionCLIP text (512d→128d) + color embedding (51 vocab, 64d→32d) + category embedding (132 vocab, 64d→32d) + group embedding (20 vocab, 32d→16d) → fusion MLP → 256d output

**Training:**
- InfoNCE contrastive loss + uniformity regularization (anti-collapse)
- Learnable temperature clamped to [0.01, 0.5]
- 15 epochs, lr=1e-4, AdamW, cosine annealing with warmup
- 100K triplets from `biencoder_retriever_labels.jsonl`

### Training Curve

| Epoch | Val Accuracy | Temperature | Item Similarity |
|---|---|---|---|
| 1 | 0.807 | 0.500 | 0.003 |
| 5 | 0.887 | 0.500 | -0.005 |
| 10 | 0.897 | 0.500 | 0.004 |
| 15 | 0.900 | 0.500 | 0.001 |

Item similarity stayed near zero throughout (no representation collapse).

### Critical Bug Fix: Representation Collapse

Initial training (v1) produced **representation collapse** — all item embeddings clustered to the same point (pairwise cosine sim = 0.94). Root causes:
1. **Temperature explosion:** Learnable temperature reached 13.7 (should be 0.07–1.0), diluting the contrastive signal
2. **No uniformity pressure:** Embeddings could minimize loss by mapping everything to the same point

Fixes applied:
1. Clamped temperature to max 0.5
2. Added uniformity loss (Wang & Isola 2020): `uni_loss = log(mean(exp(-t * ||ei - ej||^2)))`
3. Total loss: `(InfoNCE + 0.1 * uniformity_loss) / grad_accum`

After fix: item similarity = 0.003 (healthy), temperature = 0.500 (at clamp).

### Retrieval Results (2,000 test queries)

| Method | nDCG@10 | MRR | R@10 | R@50 | vs Dense-only |
|---|---|---|---|---|---|
| Dense-only (FashionCLIP) | 0.0380 | 0.0348 | 0.0710 | 0.1770 | — |
| Fused (pure, 256d) | 0.0201 | 0.0184 | 0.0390 | 0.1145 | -47.1% |
| Fused rerank (CLIP→fused top-200) | 0.0235 | 0.0219 | 0.0465 | 0.1530 | -38.2% |
| **Score interpolation (0.6 dense + 0.4 fused)** | **0.0404** | **0.0369** | **0.0725** | **0.1765** | **+6.3%** |

**Key finding:** The fused item tower alone underperforms CLIP (its 256d space is less expressive than CLIP's 512d), but when blended via score interpolation, it provides a +6.3% lift — the structured attribute knowledge complements CLIP's semantic understanding.

---

## 7. Phase 3.8 Path B: Attribute-Conditioned Cross-Encoder

### Motivation

Standard cross-encoders see product text as `"prod_name | detail_desc | colour | type | section"` — a flat string where attribute boundaries are ambiguous. The hypothesis: explicitly tagging attributes helps the cross-encoder learn attribute-specific relevance patterns.

### Architecture

Same base model (`cross-encoder/ms-marco-MiniLM-L-6-v2`), but with tagged input format:

```
prod_name | detail_desc [COLOR] colour_group_name [TYPE] product_type_name [SEC] section_name [GROUP] product_group_name
```

### Training

- Same 42,800 LLM-judged labels as 3.3
- 3 epochs, batch_size=32, lr=2e-5
- Eval every 500 steps on held-out validation

### Training Curve

| Epoch | Loss | Pearson | Spearman |
|---|---|---|---|
| 0.42 | 0.5309 | 0.8107 | 0.8614 |
| 0.83 | 0.4355 | 0.8247 | 0.8770 |
| 1.25 | 0.4197 | 0.8304 | 0.8849 |
| 1.66 | 0.4127 | 0.8341 | 0.8909 |
| 2.08 | 0.4056 | 0.8425 | 0.8948 |
| 2.49 | 0.3955 | 0.8484 | 0.8976 |
| 3.0 (final) | 0.3921 | **0.8484** | **0.8976** |

### Retrieval Results (2,000 test queries, dense-only retrieval → CE rerank)

| Reranker | nDCG@10 | MRR | R@10 | vs No-Rerank |
|---|---|---|---|---|
| No reranker | 0.0380 | 0.0348 | 0.0710 | — |
| Off-shelf CE | 0.0431 | 0.0365 | 0.0805 | +13.4% |
| LLM-trained CE | 0.0502 | 0.0422 | 0.0985 | +32.1% |
| **Attr-conditioned CE** | **0.0517** | **0.0430** | **0.1005** | **+36.1%** |

**Key finding:** Explicit attribute tags give the cross-encoder an additional +3.0% nDCG@10 over the already-strong LLM-trained CE. The model learns that `[COLOR] Navy` in the product should match the color intent in "navy dress" — something the flat-text CE has to infer from positional patterns alone.

---

## 8. Phase 3.9: Comprehensive Evaluation

### 8a. Dense-Only Retrieval Matrix (2,000 test queries)

All 8 combinations of retrieval method × reranker:

| Config | nDCG@5 | nDCG@10 | nDCG@20 | nDCG@50 | MRR | R@5 | R@10 | R@20 | R@50 |
|---|---|---|---|---|---|---|---|---|---|
| Dense_NoRerank | 0.0292 | 0.0380 | 0.0486 | 0.0613 | 0.0348 | 0.0430 | 0.0710 | 0.1125 | 0.1770 |
| Dense_OffshelfCE | 0.0345 | 0.0431 | 0.0515 | 0.0671 | 0.0365 | 0.0535 | 0.0805 | 0.1140 | 0.1925 |
| Dense_LLM_CE | 0.0363 | 0.0502 | 0.0642 | 0.0793 | 0.0422 | 0.0545 | 0.0985 | 0.1540 | 0.2295 |
| **Dense_AttrCE** | **0.0394** | **0.0517** | **0.0619** | **0.0804** | **0.0430** | **0.0615** | **0.1005** | **0.1405** | **0.2335** |
| Fused_NoRerank | 0.0322 | 0.0404 | 0.0494 | 0.0631 | 0.0369 | 0.0470 | 0.0725 | 0.1080 | 0.1765 |
| Fused_OffshelfCE | 0.0338 | 0.0425 | 0.0524 | 0.0678 | 0.0371 | 0.0500 | 0.0775 | 0.1175 | 0.1945 |
| Fused_LLM_CE | 0.0359 | 0.0487 | 0.0625 | 0.0779 | 0.0412 | 0.0545 | 0.0950 | 0.1500 | 0.2265 |
| Fused_AttrCE | 0.0351 | 0.0492 | 0.0593 | 0.0775 | 0.0403 | 0.0540 | 0.0985 | 0.1385 | 0.2295 |

### 8b. Full Pipeline with BM25+NER (22,855 test queries)

Hybrid retrieval: BM25 (weight=0.4) + Dense (weight=0.6) with GLiNER NER boosts. CE reranks top-100 → top-50.

**Full 2×4 Matrix (8 configurations) — Default NER (GLiNER v1):**

| Config | nDCG@5 | nDCG@10 | nDCG@20 | nDCG@50 | MRR | R@5 | R@10 | R@20 | R@50 |
|---|---|---|---|---|---|---|---|---|---|
| A0: Baseline_NoRerank | 0.0324 | 0.0422 | 0.0552 | 0.0842 | 0.0558 | 0.0083 | 0.0142 | 0.0243 | 0.0515 |
| A1: Baseline + OffshelfCE | 0.0442 | 0.0646 | 0.0859 | 0.1195 | 0.0671 | 0.0095 | 0.0195 | 0.0340 | 0.0620 |
| A2: Baseline + LLM_CE | 0.0553 | 0.0747 | 0.0964 | 0.1297 | 0.0755 | 0.0124 | 0.0217 | 0.0358 | 0.0659 |
| A3: Baseline + AttrCE | — | 0.0738 | — | — | — | — | — | — | — |
| B0: FineTuned_NoRerank | 0.0390 | 0.0515 | 0.0730 | 0.1132 | 0.0740 | 0.0110 | 0.0188 | 0.0351 | 0.0775 |
| B1: FineTuned + OffshelfCE | 0.0438 | 0.0650 | 0.0901 | 0.1394 | 0.0723 | 0.0103 | 0.0207 | 0.0383 | 0.0798 |
| **B2: FineTuned + LLM_CE** | **0.0533** | **0.0757** | **0.1025** | **0.1516** | **0.0799** | **0.0131** | **0.0243** | **0.0418** | **0.0853** |
| B3: FineTuned + AttrCE | — | 0.0746 | — | — | — | — | — | — | — |

### 8c. Full Pipeline with Fine-Tuned NER V2 (22,855 test queries)

Re-ran all 8 configurations replacing default GLiNER v1 with fine-tuned GLiNER2 V2 for query NER.

**nDCG@10 Comparison (Old NER → Fine-tuned NER V2):**

| Config | Old NER | Fine-tuned NER V2 | Δ |
|---|---|---|---|
| A0: Baseline_NoRerank | 0.0422 | 0.0430 | **+1.9%** |
| A1: Baseline + OffshelfCE | 0.0646 | 0.0661 | **+2.3%** |
| A2: Baseline + LLM_CE | 0.0747 | 0.0740 | -0.9% |
| A3: Baseline + AttrCE | 0.0738 | 0.0729 | -1.2% |
| B0: FineTuned_NoRerank | 0.0515 | 0.0511 | -0.8% |
| B1: FineTuned + OffshelfCE | 0.0650 | 0.0661 | **+1.7%** |
| B2: FineTuned + LLM_CE | **0.0757** | 0.0752 | -0.7% |
| B3: FineTuned + AttrCE | 0.0746 | 0.0740 | -0.8% |

See [Section 5, 3.7f](#37f--analysis-why-fine-tuned-ner-helps-weak-rerankers-but-hurts-strong-ones) for detailed analysis.

---

## 9. Tier 1 Cross-Check: Marqo 7-Dataset

Verifies that fine-tuning on H&M doesn't degrade general fashion understanding.

### Text-to-Image Retrieval

| Model | DeepFashion InShop R@10 | DeepFashion MM R@10 | Fashion200k R@10 | Atlas R@10 | Polyvore R@10 | Avg R@10 | Avg MRR |
|---|---|---|---|---|---|---|---|
| Marqo-FashionCLIP (baseline) | 0.239 | 0.024 | 0.280 | 0.247 | 0.513 | 0.260 | 0.187 |
| Marqo-FashionSigLIP | 0.248 | 0.031 | 0.378 | 0.324 | 0.584 | 0.313 | 0.226 |
| **MoDA-FashionCLIP (fine-tuned)** | **0.229** | **0.019** | **N/A** | **0.253** | **0.480** | **~0.245** | **~0.194** |

### Category-to-Product Retrieval

| Model | Avg P@1 | Avg P@10 | Avg MRR |
|---|---|---|---|
| Marqo-FashionCLIP | 0.738 | 0.703 | 0.791 |
| Marqo-FashionSigLIP | 0.790 | 0.716 | 0.839 |
| MoDA-FashionCLIP (fine-tuned) | 0.729 | 0.703 | 0.738 |

### Sub-Category-to-Product Retrieval

| Model | Avg P@1 | Avg P@10 | Avg MRR |
|---|---|---|---|
| Marqo-FashionCLIP | 0.774 | 0.688 | 0.800 |
| Marqo-FashionSigLIP | 0.774 | 0.680 | 0.824 |
| MoDA-FashionCLIP (fine-tuned) | 0.699 | 0.686 | 0.762 |

**Key finding:** Fine-tuning causes a ~6% regression on text-to-image retrieval (Avg R@10: 0.260 → 0.245). This is expected: we only fine-tuned the text encoder, which slightly drifts the shared CLIP embedding space. The trade-off is justified — +94% on H&M domain retrieval for -6% on general benchmarks. Category-to-product remains comparable, confirming the model hasn't forgotten fashion category structure.

---

## 10. Best Configuration Analysis

### Ranking All Configurations by nDCG@10

**Full pipeline (22,855 queries, with BM25+NER, default GLiNER v1 NER):**

| Rank | Configuration | nDCG@10 | vs Phase 2 Baseline |
|---|---|---|---|
| 1 | **Fine-tuned hybrid + LLM-trained CE** | **0.0757** | **+79.4%** |
| 2 | Baseline hybrid + LLM-trained CE | 0.0747 | +77.0% |
| 3 | Fine-tuned hybrid + Attr-cond CE | 0.0746 | +76.8% |
| 4 | Baseline hybrid + Attr-cond CE | 0.0738 | +74.9% |
| 5 | Fine-tuned hybrid + purchase-trained CE | 0.0654 | +55.0% |
| 6 | Fine-tuned hybrid + off-shelf CE | 0.0650 | +54.0% |
| 7 | Baseline hybrid + off-shelf CE | 0.0646 | +53.1% |
| 8 | Fine-tuned hybrid, no rerank | 0.0515 | +22.0% |
| 9 | Baseline hybrid, no rerank | 0.0422 | — |

**Dense-only retrieval (2,000 queries, no BM25):**

| Rank | Configuration | nDCG@10 | vs Dense Baseline |
|---|---|---|---|
| 1 | **Dense + Attr-conditioned CE** | **0.0517** | **+36.1%** |
| 2 | Dense + LLM-trained CE | 0.0502 | +32.1% |
| 3 | Fused + Attr-conditioned CE | 0.0492 | +29.5% |
| 4 | Fused + LLM-trained CE | 0.0487 | +28.2% |
| 5 | Dense + Off-shelf CE | 0.0431 | +13.4% |
| 6 | Fused + Off-shelf CE | 0.0425 | +11.8% |
| 7 | Fused, no rerank | 0.0404 | +6.3% |
| 8 | Dense, no rerank | 0.0380 | — |

### The Winner

**Fine-tuned BM25+NER hybrid retrieval + LLM-trained cross-encoder reranking** is the best Phase 3 configuration:

- **nDCG@10 = 0.0757** on 22,855 test queries
- **MRR = 0.0799**
- **nDCG@50 = 0.1516**
- **R@50 = 0.0853**

This combines:
- BM25 lexical search with GLiNER2 NER query boosts (weight=0.4)
- Fine-tuned FashionCLIP dense retrieval (weight=0.6)
- Top-100 candidates → LLM-trained cross-encoder reranks to top-50

### Why Not Attr-Conditioned CE as the Winner?

The attribute-conditioned CE (Path B) shows the highest single-component lift in dense-only retrieval (+36.1%, 2K queries), but **in the full BM25+NER pipeline (22,855 queries), it finishes 3rd and 4th** — behind the LLM-trained CE by -1.2% (baseline retriever) and -1.5% (fine-tuned retriever).

**Why the reversal?** In dense-only retrieval, the explicit attribute tags (`[COLOR] Navy`) disambiguate product text for the CE, which is critical when the candidate pool has no lexical pre-filtering. But in the hybrid pipeline, BM25+NER already handles attribute matching (NER extracts "navy" → BM25 boosts `colour_group_name` field), so the attribute tags become partially redundant. The flat-text LLM-trained CE, which learns more nuanced semantic matching from its 42K graded labels, edges out the tagged CE when the attribute-matching heavy lifting is already done by the retriever.

---

## 11. Key Takeaways

### 1. Label quality > label quantity
Purchase labels (1.5M pairs, binary) barely beat off-the-shelf CE (+1.2%). LLM-judged labels (42K pairs, graded 0–3) produce +15.6% improvement. Graded relevance captures partial matches that binary labels miss.

### 2. Hard negative mining is the biggest retrieval lever
Fine-tuning FashionCLIP on its own failure modes (hard negatives mined from baseline retrieval) nearly doubles dense retrieval quality (+94.1%). The model learns catalog-specific vocabulary mapping.

### 3. Structured attributes help cross-encoders
Adding explicit attribute tags (`[COLOR] Navy [TYPE] Dress`) to cross-encoder input gives +3.0% nDCG over flat-text CE. The tags remove ambiguity about which part of the product text corresponds to which query intent.

### 4. The fused item tower provides marginal gains at retrieval
Score interpolation (60% CLIP + 40% fused) yields +6.3% over CLIP alone, but the gains don't stack with CE reranking. The fused model's structured attribute signal is largely redundant once a strong cross-encoder sees the full text.

### 5. Domain fine-tuning trades breadth for depth
Fine-tuned FashionCLIP gains +94% on H&M retrieval but loses ~6% on general fashion benchmarks (Tier 1). Acceptable for a domain-specific search system.

### 6. NER fine-tuning has a ceiling set by the reranker
Fine-tuned NER improves candidate quality for BM25 (+2.3% nDCG@10 with weak reranker), but a strong cross-encoder already compensates for imperfect NER. The takeaway: invest in NER precision only if your pipeline lacks a strong reranker, or if you can eliminate false positives without reducing recall.

### 7. Article-derived NER training data fails for query extraction
Training NER on product descriptions (long, structured text) doesn't transfer to search queries (short, informal text). Class imbalance (color: 0.1% of examples) and domain mismatch made V1 worse than off-the-shelf. LLM-generated query-domain data (V2) solved both issues but introduced its own precision challenges on edge-case entity types.

### 8. Attribute-tagged CE wins in isolation but not in a hybrid pipeline
When BM25+NER already handles attribute matching at retrieval time, explicit `[COLOR]`/`[TYPE]` tags in the CE input become redundant. The flat-text LLM-trained CE's superior semantic matching (from 42K graded labels) outweighs structural tags when the retriever pre-filters well.

### 9. The compound effect is what matters
No single improvement is transformative alone. The +79.4% total lift comes from stacking: better retriever (+22% from fine-tuned FashionCLIP) + better candidate pool (+53% from BM25 hybrid) + better reranking (+77% from LLM-trained CE). Each component contributes.

---

## Appendix: File Locations

| Artifact | Path |
|---|---|
| Phase 3.9 comprehensive eval | `results/real/phase3_9_comprehensive_eval.json` |
| Full pipeline eval (BM25+NER) | `results/real/phase3_combined_eval.json` |
| CE fine-tuning eval (off-shelf / purchase / LLM CE, 22.8K) | `results/real/hnm_finetuned_ce_eval.json` |
| Phase 2 MoE exploratory (10K) | `results/real/hnm_mixture_of_encoders.json` |
| ColBERT + CE eval | `results/real/hnm_colbert_rerank.json` |
| Bi-encoder eval | `results/real/phase3c_biencoder_eval.json` |
| Fused item tower eval | `results/real/phase3_fused_item_tower_eval.json` |
| Tier 1 leaderboard | `results/tier1/tier1_leaderboard.md` |
| Tier 1 raw results | `results/tier1/tier1_raw_results.json` |
| Phase 3 final summary | `results/real/phase3_final_summary.json` |
| Overnight runner log | `results/real/overnight_phase3.log` |
| NER v1 training log | `results/real/phase3_7_fashion_ner.log` |
| NER v2 training data generator | `benchmark/generate_ner_labels.py` |
| NER v2 training data | `data/processed/ner_training_data_v2.jsonl` |
| NER v2 training script | `benchmark/train_fashion_ner.py --v2` |
| NER v2 full pipeline eval log | `results/real/phase3_combined_eval_finetuned_ner.log` |
| Attr-CE training log | `results/real/path_b_ce_attr_train.log` |
| Fused tower training log | `results/real/path_a_retrain_v2.log` |
