# MODA Phase 3, 4, 5 Plan

## Context

Phase 1-2 is done and being published. Phase 3 has partial work (LLM-label CE, fine-tuned bi-encoder, some multimodal embedding) but we're willing to restructure it.

New input: LookBench paper (arXiv:2601.14706v2) gives us a live, attribute-supervised image retrieval benchmark for fashion with ~2,500 queries across 4 subsets (~60K corpus images each). This becomes our Tier 3 benchmark alongside Tier 1 (Marqo 7-dataset) and Tier 2 (H&M 253K queries).

The existing Phase 3/4 work in the codebase is partially done (see README). We restructure into three clean phases below. Writing the paper is not a phase; it happens alongside.

---

## Phase 3: Trained models

Everything in Phase 2 was off-the-shelf. Phase 3 trains models on our data. We keep the work that proved out (LLM labels, contrastive bi-encoder) and add MoE properly.

### 3A: LLM-judged relevance labels (keep, already done)

Generate graded relevance labels (0-3) using GPT-4o-mini for query-product pairs. This was the single biggest unlock in the existing Phase 3 work. 42.8K labels, ~$2-3.

Input: H&M queries + candidate products from retriever
Output: `llm_relevance_labels.jsonl`

### 3B: Cross-encoder trained on LLM labels (keep, already done)

Fine-tune ms-marco-MiniLM-L-6-v2 on LLM-graded labels instead of purchase labels.

Input: `llm_relevance_labels.jsonl`
Output: `moda-fashion-ce/` model
Expected: nDCG@10 ~0.0747 (+15.7% over off-shelf CE)

### 3C: Bi-encoder trained on retriever-mined hard negatives (keep, already done)

Fine-tune FashionCLIP text encoder using hard negatives mined from its own top-20 results, scored by GPT-4o-mini. 24K contrastive triplets.

Input: `biencoder_retriever_labels.jsonl`
Output: `moda-fashion-embed/` model
Expected: dense nDCG@10 ~0.0444 (+94% over baseline FashionCLIP)

### 3D: Mixture-of-encoders with trained field encoders (new)

The Phase 2 MoE attempt failed (-12%) because we used the same FashionCLIP for all four fields. This time we train proper per-field encoders.

Approach:
- Color encoder: Small MLP trained on H&M color vocabulary. Training signal: LLM judges "is query color X same as product color Y?" (e.g. navy = dark blue, coral = light orange). H&M has ~50 unique colour_group_name values. Train on pairwise similarity.
- Category encoder: Same approach for product_type_name (~100 values). "Is jeans similar to trousers?" scored by LLM.
- Group encoder: product_group_name (~20 values). Simpler, possibly just a learned embedding table.
- Text encoder: Fine-tuned FashionCLIP from 3C.

Concatenate into one vector per product: FashionCLIP(512) + color(64) + category(64) + group(32) = 672-dim. Build FAISS index. Query-time: extract attributes via NER, encode each with the matching field encoder, weight and search.

Cost: LLM labels for color/category similarity pairs ~$1. Training: minutes on CPU (small MLPs).
This is the proper test of Superlinked's idea.

### 3E: Full evaluation (2x3 factorial + MoE)

Retriever variants: baseline FashionCLIP, fine-tuned FashionCLIP (3C), MoE (3D)
Reranker variants: no reranker, off-shelf CE, LLM-trained CE (3B)
9 combinations on 22,855 held-out test queries.

Plus: run all retriever variants on Marqo's Tier 1 benchmark (7 datasets) to verify that fine-tuning on H&M doesn't degrade general fashion retrieval.

Deliverables:
- Trained models on HuggingFace: moda-fashion-ce, moda-fashion-embed
- Updated Tier 2 leaderboard with all 9 combinations
- Tier 1 cross-check (fine-tuned models on Marqo's benchmark)
- MoE results (the proper version)

Timeline: ~3-4 days
Cost: ~$3-5 (LLM labels) + $0 compute (Apple MPS)

---

## Phase 4: Multimodal

Fashion is visual. Phase 4 adds image retrieval and benchmarks on LookBench.

### 4A: LookBench integration (new)

Download LookBench dataset and eval code. Run our models on their benchmark.

LookBench has 4 subsets:
- RealStudioFlat: 1,011 queries, ~62K corpus. Easy. Single-item, studio photos.
- AIGen-Studio: 192 queries, ~59K corpus. Medium. AI-generated product images.
- RealStreetLook: 1,000 queries, ~61K corpus. Hard. Real street-style outfit photos, multi-item.
- AIGen-StreetLook: 160 queries, ~59K corpus. Hard. AI-generated street outfits.

Metrics: Fine Recall@1 (category + all attributes match), Coarse Recall@1, nDCG@5.

Run:
- FashionCLIP (our Phase 1 baseline)
- Fine-tuned FashionCLIP (Phase 3C)
- GR-Lite (LookBench's open model, DINOv3-based, our external baseline)
- Marqo-FashionSigLIP / FashionCLIP

Published SOTA numbers for reference:
| Model | Overall Fine R@1 |
|-------|-----------------|
| GR-Pro (proprietary) | 67.38% |
| GR-Lite (open) | 65.71% |
| Marqo-FashionCLIP | 63.24% |
| Marqo-FashionSigLIP | 62.77% |

This establishes our Tier 3 (visual retrieval on a contamination-aware benchmark).

### 4B: Image embedding for H&M (keep, already done per README)

Embed 105K H&M product images with FashionCLIP vision encoder. Build FAISS image index.

### 4C: Text-to-image retrieval channel (keep, already done)

Add text-to-image as a retrieval signal: query text encoded with FashionCLIP text encoder, searched against product image vectors. This catches products where the image shows attributes not in the text (e.g. a "floral pattern" visible in the photo but not mentioned in the title).

### 4D: Three-way hybrid retrieval

Fuse three signals via RRF:
1. BM25 (lexical)
2. Text-to-text dense (FashionCLIP text encoder)
3. Text-to-image dense (FashionCLIP text encoder vs image vectors)

Evaluate all weight combinations on H&M test set. Does adding images improve text search results?

### 4E: Joint text+image fine-tuning (already in progress per README)

Contrastive + alignment regularization training on both FashionCLIP encoders. Uses LLM labels from image hard negative mining (generate_image_labels.py).

### 4F: Three-Tower Fashion Retriever

Novel architecture: dedicated query tower (trainable), frozen text tower, frozen image tower, all projecting into shared 512-dim space. Product embeddings (text + image) precomputed offline. Only query tower trains at serving time.

This is the architecture experiment. Does a dedicated query encoder that projects into a shared product space outperform symmetric text-to-text and text-to-image retrieval?

### 4G: Visual search demo

Upload an image, find similar products. Image-to-image retrieval using the vision encoder.
Integrate into Flask UI from the original Moda codebase.

### 4H: LookBench leaderboard submission

Run our best Phase 4 model (joint fine-tuned or Three-Tower) on all 4 LookBench subsets. Submit to their leaderboard. This is our public Tier 3 result.

Deliverables:
- LookBench numbers for all our models (Tier 3 leaderboard)
- Three-way hybrid results on H&M (does image help text search?)
- Three-Tower architecture evaluation
- Visual search demo
- Trained multimodal models on HuggingFace

Timeline: ~5-7 days
Cost: ~$1-2 (LLM image labels) + $5-10 if using cloud GPU for image training

---

## Phase 5: Search experience + data augmentation

Phases 3-4 optimize retrieval quality. Phase 5 makes it usable for real shoppers and extends the data.

### 5A: Data augmentation

- LLM-generated query variants: For each real query, generate 3-5 paraphrases using GPT-4o-mini. "navy summer dress" becomes "dark blue casual dress for summer", "lightweight navy dress", etc. Expands training data without new human labels.
- Synthetic hard negatives: Use the fine-tuned retriever to mine harder negatives on the augmented query set. Iterative hard negative mining (ANCE-style): retrieve, label, train, repeat.
- Catalog enrichment: Generate detailed product descriptions from product images using a VLM (Qwen2.5-VL or GPT-4o). H&M's detail_desc field is often sparse. Enriched descriptions improve both BM25 and dense retrieval.

### 5B: Faceted navigation

Fashion shoppers filter by color, size, brand, price range, category, occasion. Build structured facets from H&M metadata:
- Color facets (from colour_group_name)
- Category hierarchy (product_group → product_type → garment_group)
- Gender (index_group_name: Menswear, Ladieswear, etc.)
- Season/collection
- Price range

Implement in OpenSearch as aggregations alongside search results. Post-retrieval filtering that doesn't hurt ranking quality.

### 5C: Partitioned indexes

Split the OpenSearch index by high-level category (Womenswear, Menswear, Kids, etc.). When NER detects gender in the query, route to the appropriate partition. Reduces corpus size per query and improves precision.

Alternative: use filtered search with pre-filter on gender/category. Compare partitioned vs filtered approaches on latency and quality.

### 5D: Auto-suggest and query relaxation

- Auto-suggest: As the user types, suggest completions from the query log (253K real queries give us a strong suggestion corpus). Trie-based or prefix-matching approach.
- Query relaxation: When a query returns few results (< 5), automatically relax constraints. Drop the least important NER attribute (e.g. keep color and type, drop fit). Re-query with relaxed terms. Report to user: "Showing results for navy dress (relaxed from navy slim fit dress)".

### 5E: End-to-end search demo

Bring it all together in a polished Flask/React UI:
- Text search with hybrid retrieval + reranking
- Image search (upload photo, find similar)
- Faceted filters (color, category, gender, price)
- Auto-suggest
- "More like this" from any result
- Query relaxation with transparency

Deliverables:
- Augmented training data (query variants, enriched catalog)
- Faceted search on H&M catalog
- Partitioned/filtered index comparison
- Auto-suggest from real query log
- Polished search demo

Timeline: ~5-7 days
Cost: ~$5-10 (LLM for query augmentation + catalog enrichment)

---

## Summary

| Phase | Focus | Key deliverable | Cost | Time |
|-------|-------|----------------|------|------|
| 3 | Trained models + MoE | Fine-tuned retriever, CE, and proper MoE with trained field encoders | ~$3-5 | 3-4 days |
| 4 | Multimodal + LookBench | Image retrieval, Three-Tower, LookBench Tier 3 numbers | ~$5-12 | 5-7 days |
| 5 | Search experience | Data augmentation, facets, partitioned indexes, auto-suggest, demo | ~$5-10 | 5-7 days |
| **Total** | | | **~$15-27** | **~2-3 weeks** |

Three benchmarks maintained throughout:
- Tier 1: Marqo 7-dataset (embedding quality, cross-check after training)
- Tier 2: H&M 253K queries (full-pipeline search quality)
- Tier 3: LookBench (visual retrieval, contamination-aware, attribute-supervised)
