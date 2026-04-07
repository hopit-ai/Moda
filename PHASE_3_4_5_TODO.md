# MODA Phase 3, 4, 5 — Detailed TODO

Starting assumption: Phase 1-2 is done. We have:
- Marqo 7-dataset benchmark reproduced (Tier 1)
- H&M 253K query full-pipeline benchmark with 11 configs (Tier 2)
- OpenSearch index with 105K articles, FAISS indexes for FashionCLIP/SigLIP/CLIP
- GLiNER v1 + GLiNER2 NER caches (10K + 253K queries)
- ColBERT cascade results (10K)
- Working eval harness (metrics.py, eval_full_253k.py, etc.)

Everything below is from scratch.

---

## PHASE 3: Fine-tuned models

Goal: Train domain-specific models for each pipeline component. Show that targeted training with the right data beats off-the-shelf models.

---

### 3.1 — Generate train/test split

**What:** Create a clean query split ensuring no query text appears in both train and test.

**Why:** Every training and evaluation step depends on this. Without it, we risk data leakage.

**How:**
1. Load all 253,685 queries from `queries.csv`
2. Deduplicate by `query_text` (many queries repeat). Get unique query texts.
3. Split unique query texts 80/10/10 into train/val/test
4. Map back to query IDs. All IDs sharing a query text go into the same split.
5. Save to `data/processed/query_splits.json` as `{"train": [qid, ...], "val": [qid, ...], "test": [qid, ...]}`

**Input:** `data/raw/hnm_real/queries.csv`
**Output:** `data/processed/query_splits.json`
**Verify:** No query text appears in multiple splits. Print split sizes.

> **Watch out:** Split by unique `query_text`, not by `query_id`. The same text "black dress" can have dozens of query IDs (different users searching the same thing). If you split by ID, the same query text leaks across splits. After splitting, assert: `set(train_texts) & set(test_texts) == empty`. Also check that the positive article IDs in test aren't disproportionately concentrated in a few popular products — if they are, the test set is measuring popularity, not retrieval quality.

---

### 3.2 — Generate LLM relevance labels for cross-encoder training

**What:** Use GPT-4o-mini to grade query-product pairs on a 0-3 scale (irrelevant / partial / good / exact match).

**Why:** Phase 2 showed purchase labels are noisy (buying one black dress doesn't make others irrelevant). LLM-judged labels give graded relevance that captures partial matches.

**How:**
1. Sample ~5,000 unique queries from the **train split only**
2. For each query, get the top-20 products from the Phase 2 hybrid retriever (BM25 + FashionCLIP)
3. For each (query, product) pair, call GPT-4o-mini with a prompt like:
   ```
   Query: "navy slim fit jeans mens"
   Product: "Slim Fit Stretch Jeans | Trousers | Dark Blue | Menswear"
   Rate relevance 0-3:
   0 = not relevant, 1 = partially relevant, 2 = good match, 3 = exact match
   ```
4. Parse the response, validate score is 0-3
5. Save to `data/processed/llm_relevance_labels.jsonl` as `{"query_id": ..., "article_id": ..., "query_text": ..., "product_text": ..., "score": 2}`

**Input:** Train split queries + hybrid retrieval results + GPT-4o-mini API
**Output:** `data/processed/llm_relevance_labels.jsonl` (~42K-100K labeled pairs)
**Cost:** ~$2-3 for 100K pairs via GPT-4o-mini
**Verify:** Check score distribution (should be roughly balanced, not all 0s or all 3s). Spot-check 50 labels manually.

> **Watch out:** Use the **off-the-shelf Phase 2 retriever** (baseline FashionCLIP) to mine candidates, not a fine-tuned one. If you mine from a model trained on this data, your hard negatives are easy negatives. The product text sent to the LLM must be the same concatenation used at eval time: `prod_name | product_type_name | colour_group_name | section_name | detail_desc[:200]`. If you send different fields to the LLM vs the cross-encoder, the labels won't transfer. Also: never include test-split queries in the labeling batch. Filter strictly by `query_splits.json` train IDs before sampling.

---

### 3.3 — Fine-tune cross-encoder on LLM labels

**What:** Train ms-marco-MiniLM-L-6-v2 on the LLM-graded labels from 3.2.

**Why:** The off-the-shelf CE was trained on MS MARCO web search. Fashion queries have different vocabulary and intent. LLM labels capture fashion-specific relevance grades.

**How:**
1. Load `llm_relevance_labels.jsonl`, filter to train split queries
2. Create `InputExample` pairs: `(query_text, product_text)` with `label = score / 3.0` (normalize to 0-1)
3. Use `sentence_transformers.CrossEncoder` class
4. Training config: batch_size=32, epochs=3, learning_rate=2e-5, warmup=10%
5. Evaluate on val split after each epoch, save best checkpoint
6. Final eval on held-out test split

**Input:** `llm_relevance_labels.jsonl` + `query_splits.json`
**Output:** `models/moda-fashion-ce/` (saved checkpoint)
**Hardware:** T4 GPU (Colab free) or Apple MPS. ~2-3 hours.
**Verify:** Val loss decreasing. Test nDCG@10 should be meaningfully above off-shelf CE (target: +10-15%).

> **Watch out:** The label normalization matters. Score 0→0.0, 1→0.33, 2→0.67, 3→1.0. If you use raw integers as labels with BCEWithLogitsLoss, the loss function interprets 0 and 3 on different scales than 0.0 and 1.0. Validate on the **val split**, never touch test during training. If val loss plateaus or increases, stop early. Don't fall back to using train data as validation if val set is small — raise an error instead (the existing code has a fallback that creates val from train, which is a leakage risk we identified in the audit).

---

### 3.4 — Fine-tune cross-encoder on purchase labels (comparison)

**What:** Same as 3.3 but using purchase labels instead of LLM labels. This is the control experiment.

**Why:** To prove that label quality matters more than label quantity. Purchase data gives us 1.5M pairs (more data) but noisier labels.

**How:**
1. Load `qrels.csv`, filter to train split
2. For each query: positive = purchased article (label=1.0), hard negatives = shown-but-not-bought (label=0.0), random negatives = sampled from catalog (label=0.0)
3. ~3 hard negatives + ~2 random negatives per query
4. Same training config as 3.3
5. Eval on same test split

**Input:** `qrels.csv` + `query_splits.json` + `articles.csv`
**Output:** `models/moda-fashion-ce-purchase/`
**Verify:** Compare test nDCG@10 vs 3.3. Expected: purchase CE barely beats off-shelf (+1-2%), LLM CE beats both (+10-15%).

> **Watch out:** The H&M `negative_ids` in qrels are products **shown but not bought** — these are grade=1 in our eval scheme (partially relevant), not grade=0 (irrelevant). But here we're training them as label=0.0 negatives. This is the core noise problem: many negative_ids are actually decent results the user just didn't purchase. Use all `positive_ids` for each query, not just the first one (the existing `train_cross_encoder.py` only uses `pos_ids[0]` which throws away data). Random negatives should be sampled from articles **not** in either positive_ids or negative_ids for that query.

---

### 3.5 — Generate LLM labels for bi-encoder hard negative mining

**What:** Use the current FashionCLIP retriever to find its own mistakes, then have GPT-4o-mini label them.

**Why:** The best training data for a retriever is examples of where it currently fails. Products it ranks highly but are actually irrelevant become hard negatives. Products it misses but are actually relevant become hard positives.

**How:**
1. Sample ~5,000 unique queries from **train split**
2. For each query, run FashionCLIP dense retrieval, take top-20 results
3. Send each (query, product) pair to GPT-4o-mini for 0-3 scoring (same prompt as 3.2, can reuse labels if overlap)
4. Products scored 0 by LLM but ranked in top-20 by retriever = **hard negatives** (retriever was wrong)
5. Products scored 2-3 = **positives**
6. Build contrastive triplets: (query, positive_product, hard_negative_product)
7. Save to `data/processed/biencoder_retriever_labels.jsonl`

**Input:** Train split queries + FashionCLIP FAISS index + GPT-4o-mini API
**Output:** `data/processed/biencoder_retriever_labels.jsonl` (~20-30K triplets)
**Cost:** ~$1 (if reusing labels from 3.2, otherwise ~$1-2 additional)
**Verify:** Check that hard negatives are genuinely hard (visually or textually plausible but wrong). Spot-check 30 triplets.

> **Watch out:** Mine from the **baseline off-the-shelf FashionCLIP**, not the fine-tuned one from 3.6. You're mining the baseline model's mistakes as training data for itself. If you accidentally mine from an already-fine-tuned model, the negatives are too easy and training won't help. Also verify that FAISS embeddings are L2-normalized before search (cosine similarity requires this). If normalization is missing, rankings are wrong and your "hard negatives" are random noise. The query embeddings and article embeddings must use the same `encode_texts_clip()` call with `normalize=True`.

---

### 3.6 — Fine-tune FashionCLIP bi-encoder

**What:** Fine-tune FashionCLIP's text encoder using contrastive learning on the hard negatives from 3.5.

**Why:** FashionCLIP was trained on general fashion product descriptions. Fine-tuning on H&M-specific retriever failures teaches it exactly where it goes wrong on this catalog.

**How:**
1. Load triplets from `biencoder_retriever_labels.jsonl`
2. Use InfoNCE loss with in-batch negatives + one mined hard negative per query
3. Only the text encoder is trainable. Vision encoder frozen (we fine-tune that in Phase 4).
4. Training config: 5 epochs, learning_rate=1e-6, batch_size=64, gradient_accumulation=4, cosine LR schedule
5. Mixed precision (FP16) on MPS or T4
6. Eval: after each epoch, compute nDCG@10 on val split using FAISS
7. Save best checkpoint

**Input:** `biencoder_retriever_labels.jsonl` + FashionCLIP model
**Output:** `models/moda-fashion-embed/` (fine-tuned text encoder)
**Hardware:** Apple MPS (~4-6 hours) or T4 (~2-3 hours)
**Verify:** Dense-only nDCG@10 on test split. Target: +50-100% over baseline FashionCLIP.

> **Watch out:** InfoNCE with in-batch negatives assumes that for query_i in the batch, every other positive product_j (j≠i) is a negative. This breaks if two queries in the same batch share the same positive product (e.g. two users both bought article X). Shuffle triplets but verify no article_id appears as both positive and in-batch-negative within a batch. Keep the vision encoder fully frozen (`requires_grad=False`) — if you accidentally train it here, the FAISS image index from Phase 4 will be misaligned. After training, rebuild the text FAISS index with the new encoder and re-evaluate. Don't compare the fine-tuned model against the old FAISS index — the embedding space has changed.

---

### 3.7 — Fine-tune GLiNER2 for fashion NER

**What:** Fine-tune GLiNER2 on fashion-specific entity types using H&M product data as weak supervision.

**Why:** Off-the-shelf GLiNER2 improved BM25+NER by +16% over v1 but still makes mistakes on fashion-specific terms (misses "slim fit" as a fit type, confuses "rose" as color vs pattern). Fine-tuning on fashion data should improve extraction quality, which feeds into better MoE training data, better LLM prompts, and better faceted filtering.

**How:**
1. Build training data from H&M articles: each product has structured fields (colour_group_name, product_type_name, etc.)
2. Create NER training examples: product descriptions with entity spans labeled using the structured fields as ground truth
   - "Navy Slim Fit Stretch Jeans" → {color: "Navy", fit: "Slim Fit", type: "Jeans"}
   - Use `prod_name` + `detail_desc` as text, structured fields as labels
3. ~10K training examples from train split products
4. Fine-tune `fastino/gliner2-base-v1` using GLiNER2's training API
5. Eval: compare entity extraction F1 on a held-out set of 500 manually verified examples

**Input:** `articles.csv` (structured fields as weak supervision)
**Output:** `models/moda-fashion-ner/` (fine-tuned GLiNER2)
**Hardware:** T4 or MPS, ~1-2 hours
**Verify:** Run on 100 test queries, compare extracted entities vs off-the-shelf GLiNER2. Should get more precise color/type/fit extraction.

> **Watch out:** The structured fields aren't perfect NER labels. `prod_name` = "Ben zip hoodie" and `product_type_name` = "Hoodie" — but the NER span in the text should be "zip hoodie", not "Hoodie". The structured field gives you the category, not the exact span boundary. You need a span-matching step: find where in `prod_name` the `product_type_name` value (or a synonym) appears and mark that span. If the value doesn't appear in the text (e.g. `product_type_name` = "Vest top" but `prod_name` = "Lina tank"), skip that example rather than forcing a wrong span. Also: `colour_group_name` = "Dark Blue" but the text might say "Navy" — use the COLOR_MAP from `query_expansion.py` to map between H&M color names and common query terms. Train on **articles from train-split queries only** (articles that appear as positives/negatives in train queries) to prevent test data from influencing NER training.

---

### 3.8 — Mixture-of-Encoders with trained field encoders

**What:** Build proper per-field encoders for color, category, and group. Concatenate with fine-tuned FashionCLIP into a single product vector.

**Why:** Phase 2 MoE failed (-12%) because we used FashionCLIP for all fields. Encoding "Dark Blue" through a model trained on full sentences doesn't produce a useful color representation. We need encoders designed for each data type.

**How:**

**Color encoder (64-dim):**
1. Get all unique `colour_group_name` values from H&M (~50 values)
2. Generate pairwise similarity labels using GPT-4o-mini: "Is navy similar to dark blue?" → score 0-1. ~1,225 pairs for 50 colors.
3. Train a small embedding model: Embedding(50, 64) → L2 normalize. Loss: cosine similarity should match LLM score.

**Category encoder (64-dim):**
1. Get all unique `product_type_name` values (~100 values)
2. Generate pairwise similarity: "Is jeans similar to trousers?" ~4,950 pairs.
3. Same architecture: Embedding(100, 64) → L2 normalize.

**Group encoder (32-dim):**
1. `product_group_name` has ~20 values. Learned embedding table.
2. Train with LLM pairwise similarity.

**Concatenation:**
1. Per product: `concat(FashionCLIP_text(512), color_enc(64), category_enc(64), group_enc(32))` = 672-dim
2. Build FAISS index on 672-dim vectors for all 105K products
3. Per query: extract attributes via fine-tuned GLiNER2 (from 3.7), encode each with matching field encoder
4. Query vector: `concat(FashionCLIP_query(512), color_query(64), category_query(64), group_query(32))`
5. Cosine search against product FAISS index

**Input:** H&M structured fields + GPT-4o-mini for pairwise similarity + fine-tuned FashionCLIP from 3.6
**Output:** `models/moda-color-encoder/`, `models/moda-category-encoder/`, `models/moda-group-encoder/`, + FAISS index
**Cost:** ~$1 for LLM similarity labels (small vocabularies). Training: minutes on CPU.
**Verify:** MoE retrieval nDCG@10 should beat single-encoder FashionCLIP (unlike Phase 2 which lost -12%). Run full pipeline with MoE retriever + LLM-trained CE.

> **Watch out:** When NER doesn't detect an attribute (e.g. query "summer dress" has no color), do NOT zero out the color block in the query vector. Zeroing a 64-dim block in a 672-dim vector changes the cosine geometry — products with strong color signals get penalized relative to products with weak ones. Instead, use a **default vector** for missing attributes: the mean of all color embeddings (represents "any color"). Same for category and group. The LLM pairwise similarity labels must be symmetric: if "navy" ~ "dark blue" = 0.9, then "dark blue" ~ "navy" must also = 0.9. Verify this. The NER entity text needs to be mapped to H&M vocabulary values before encoding: user says "navy" but H&M calls it "Dark Blue". Use the existing `COLOR_MAP` and `GARMENT_TYPE_MAP` from `query_expansion.py` for this mapping. If a query entity doesn't map to any H&M vocab value, fall back to the default vector, don't crash.

---

### 3.9 — Full Phase 3 evaluation

**What:** Run all retriever x reranker combinations on the test split. Cross-check on Marqo Tier 1.

**How:**

**Tier 2 (H&M test split, ~22K queries):**

Retriever variants:
- Baseline FashionCLIP (Phase 2)
- Fine-tuned FashionCLIP (3.6)
- MoE with trained encoders (3.8)

Reranker variants:
- No reranker
- Off-shelf CE (Phase 2)
- LLM-trained CE (3.3)
- Purchase-trained CE (3.4, for comparison)

12 combinations. All with hybrid BM25 fusion (using fine-tuned GLiNER2 NER from 3.7).

**Tier 1 (Marqo 7-dataset):**
Run fine-tuned FashionCLIP (3.6) on all 7 datasets to verify it doesn't degrade on general fashion retrieval.

**Output:**
- Updated Tier 2 leaderboard with 12+ configs
- Tier 1 cross-check results
- Upload models to HuggingFace: moda-fashion-ce, moda-fashion-embed, moda-fashion-ner, moda-color-encoder, moda-category-encoder

> **Watch out:** Every retriever variant needs its own FAISS index. Don't run the fine-tuned FashionCLIP queries against the baseline FashionCLIP FAISS index — the embedding space changed after fine-tuning. Re-embed all 105K articles with each retriever variant and build separate indexes. For the MoE retriever, the FAISS index is 672-dim, not 512-dim — make sure the index dimensionality matches. The BM25 component and NER boosts stay the same across all configs (same OpenSearch index, same GLiNER2 NER). Only the dense retrieval and reranking change. All 12 configs must be evaluated on the **same test split** — don't resplit between experiments. For the Tier 1 cross-check: the fine-tuned text encoder must work with FashionCLIP's vision encoder (frozen). If the text encoder drifted too far during training, text-to-image tasks on Marqo's benchmark will degrade. Report both improvements and regressions honestly.

---

## PHASE 4: Multimodal

Goal: Add image understanding. Fashion is visual. Text search alone can't capture pattern, silhouette, or style.

---

### 4.1 — Download and prepare H&M product images

**What:** Download product images for all 105K articles.

**Why:** Needed for image embedding, visual search, and multimodal training.

**How:**
1. H&M images are available via the HuggingFace dataset (microsoft/hnm-search-data, images subset)
2. Images are organized by article_id (first 3 digits as directory)
3. Download to `data/raw/hnm_images/`
4. Verify: count images, check for missing articles, resize any oversized images to 512x512 max

**Input:** HuggingFace dataset
**Output:** `data/raw/hnm_images/` (~105K images)
**Verify:** Match article count against articles.csv. Report coverage.

> **Watch out:** Not all 105,542 articles will have images. Some products in the catalog may have been delisted or images may be missing from the dataset. Track coverage: how many articles have images vs total. Products without images must be excluded from image-based retrieval and MoE (they'd have zero image vectors) but should remain in BM25 and text-dense indexes. Report the coverage number (e.g. "98,234 of 105,542 articles have images") rather than silently ignoring missing ones. Also check for corrupt or tiny images (< 10x10 pixels) — these produce garbage embeddings.

---

### 4.2 — Embed product images with FashionCLIP vision encoder

**What:** Encode all product images into 512-dim vectors using FashionCLIP's vision encoder. Build FAISS index.

**Why:** Enables text-to-image retrieval (query text vs product images) and image-to-image retrieval (visual search).

**How:**
1. Load FashionCLIP vision encoder via open_clip
2. Preprocess images: resize to 224x224, normalize per CLIP spec
3. Batch encode all images (batch_size=128 on GPU)
4. L2-normalize embeddings
5. Build FAISS IndexFlatIP on 512-dim image vectors
6. Save: `data/processed/embeddings/fashion-clip-images_{embeddings.npy, article_ids.json, faiss.index}`

**Input:** Product images + FashionCLIP model
**Output:** Image FAISS index + embeddings
**Hardware:** T4 (~1-2 hours) or MPS (~3-4 hours)
**Verify:** Query a few text embeddings against image index. "red dress" should return images of red dresses.

> **Watch out:** The image preprocessing MUST match what FashionCLIP was trained with — use `open_clip.create_model_and_transforms()` and apply the returned `preprocess_val` transform, don't roll your own resize/normalize. The text encoder and vision encoder must be from the **same** FashionCLIP checkpoint. If you mix text embeddings from FashionCLIP with image embeddings from SigLIP, they're in different vector spaces and cosine similarity is meaningless. L2-normalize image embeddings the same way as text embeddings (`normalize=True`). The article_ids.json for the image index must match the article_ids.json for the text index — same order, same IDs — or your retrieval results will map to wrong products.

---

### 4.3 — Download and prepare LookBench

**What:** Download LookBench dataset and evaluation code. Set up as Tier 3 benchmark.

**Why:** LookBench is a live, contamination-aware, attribute-supervised fashion image retrieval benchmark. 4 subsets from easy (studio flat-lay) to hard (real street outfits). Published SOTA: GR-Pro 67.38% Fine R@1.

**How:**
1. Clone LookBench repo (code + evaluation scripts)
2. Download all 4 subsets:
   - RealStudioFlat: 1,011 queries, ~62K corpus images
   - AIGen-Studio: 192 queries, ~59K corpus images
   - RealStreetLook: 1,000 queries, ~61K corpus images
   - AIGen-StreetLook: 160 queries, ~59K corpus images
3. Download corpus images for each subset
4. Verify dataset integrity: check query counts, corpus sizes, attribute annotations
5. Run their eval code with CLIP ViT-L/14 to reproduce their published baseline

**Input:** LookBench release (GitHub + data download)
**Output:** `data/raw/lookbench/` with all 4 subsets, working eval pipeline
**Verify:** Reproduce CLIP ViT-L/14 baseline from paper (Fine R@1 ~39.79% overall)

> **Watch out:** LookBench is **image-to-image retrieval**, not text-to-image. Queries are image crops (detected garments from photos), not text strings. This is fundamentally different from our Tier 2 (text queries). Use their exact eval code and metrics — Fine Recall@1 requires matching both the garment category AND all annotated attributes. Coarse Recall@1 only requires category match. Don't reimplementing their metrics; subtle differences in attribute matching logic will produce incomparable numbers. Their corpus includes ~58K Fashion200K distractor images mixed with ranked gallery images — this is intentional, don't filter them out. Check their preprocessing (image resolution, crop handling) and match it exactly.

---

### 4.4 — Run baseline models on LookBench

**What:** Evaluate all our existing models on LookBench to establish Tier 3 baselines.

**How:**
Run LookBench eval on:
1. CLIP ViT-B/32 (generic baseline)
2. Marqo-FashionCLIP (our Phase 2 text backbone)
3. Marqo-FashionSigLIP
4. Fine-tuned FashionCLIP from Phase 3 (text encoder fine-tuned, vision encoder still off-the-shelf)
5. GR-Lite (LookBench's own open model, DINOv3-based)

Metrics: Fine Recall@1, Coarse Recall@1, nDCG@5 per subset + overall.

**Input:** LookBench subsets + our models
**Output:** Tier 3 leaderboard with 5 models x 4 subsets
**Verify:** Our Marqo-FashionCLIP numbers should be close to LookBench's published 63.24%.

> **Watch out:** LookBench evaluates the **vision encoder only** (image query → image corpus). For CLIP-based models, this means encoding both query and corpus images with the vision encoder, not the text encoder. Our Phase 3 fine-tuning only touched the text encoder, so the vision encoder is still baseline FashionCLIP — expect Phase 3 models to perform identically to baseline on LookBench. This is expected and not a failure. The improvement will come in Phase 4 after joint training. GR-Lite uses DINOv3 (vision-only, no text encoder at all) — it's a different architecture family, not directly comparable on Tier 2 but fair game on Tier 3.

---

### 4.5 — Text-to-image retrieval on H&M

**What:** Add a text-to-image retrieval channel: query text encoded by FashionCLIP text encoder, searched against product image vectors.

**Why:** Products sometimes have attributes visible in the image but not mentioned in the text. A "floral pattern" might be in the photo but the title just says "Summer Dress."

**How:**
1. For each query, encode with FashionCLIP text encoder (same as Phase 2)
2. Search against image FAISS index (from 4.2) instead of text FAISS index
3. Return top-100 image-matched products
4. Evaluate standalone text-to-image nDCG@10 on H&M test split

**Input:** Query embeddings + image FAISS index
**Output:** Text-to-image retrieval results + metrics
**Verify:** Compare vs text-to-text retrieval. Image channel may be weaker standalone but complementary.

> **Watch out:** Text-to-image retrieval through CLIP works because text and image encoders project into the same space. But this assumes the text encoder and vision encoder are from the **same model and checkpoint**. If you fine-tuned the text encoder in Phase 3 but not the vision encoder, the shared space is partially broken — the text encoder has drifted. Test both: baseline text encoder → baseline image index (proper alignment), and fine-tuned text encoder → baseline image index (may be misaligned). If the fine-tuned text encoder performs worse on text-to-image, that's expected drift and motivates the joint fine-tuning in 4.8.

---

### 4.6 — Three-way hybrid retrieval

**What:** Fuse three retrieval signals via RRF: BM25 + text-to-text dense + text-to-image dense.

**Why:** The core Phase 4 experiment. Does adding image retrieval improve text search results?

**How:**
1. For each query, get top-100 from:
   - BM25 (with NER boosts from fine-tuned GLiNER2)
   - Text-to-text dense (fine-tuned FashionCLIP)
   - Text-to-image dense (FashionCLIP text → product images)
2. RRF fusion with weight grid search:
   - BM25: 0.3-0.4
   - Text-to-text: 0.3-0.5
   - Text-to-image: 0.1-0.3
3. Cross-encoder rerank top-100 → top-50
4. Evaluate all weight combos on val split, pick best for test

**Input:** Three retrieval channels + LLM-trained CE
**Output:** Best 3-way hybrid weights + nDCG@10/MRR/Recall on test split
**Verify:** Does 3-way beat 2-way? How much does the image channel add?

> **Watch out:** Tune RRF weights on the **val split only**. Pick the best weights, then run once on test. If you tune on test, you're overfitting to the test set and the numbers aren't real. The three retrieval channels will have different article coverage (some products have images, some don't). The image channel will return only image-available products. RRF handles this naturally (products not returned by a channel just don't get that channel's score), but be aware that the image channel is systematically missing ~7K products. This could hurt Recall if the purchased product happens to be one without an image. Report this coverage gap.

---

### 4.7 — Generate LLM labels for image hard negatives

**What:** Mine hard negatives from the image retrieval channel and label them.

**Why:** Needed for fine-tuning the vision encoder. Same approach as 3.5 but for images.

**How:**
1. For each of ~5K train queries, run text-to-image retrieval, take top-20 image matches
2. For scoring, build product text from the matched article's fields and send (query_text, product_text) to GPT-4o-mini for 0-3 scoring
3. Products scored 0 by LLM but ranked in top-20 by image retriever = image hard negatives
4. Save to `data/processed/image_retriever_labels.jsonl`

**Input:** Text-to-image retrieval results + GPT-4o-mini API
**Output:** `data/processed/image_retriever_labels.jsonl`
**Cost:** ~$1-2

> **Watch out:** We're labeling the **image retriever's** mistakes, not the text retriever's. The hard negatives here are products whose images look relevant to the query but whose actual content doesn't match. These are different from the text hard negatives in 3.5 — don't merge the two files. Using a VLM (GPT-4o with vision) to directly judge query text vs product image would be more accurate than text-only LLM judging, but also ~10x more expensive. If budget allows, label a 500-pair VLM subset and compare quality against text-only LLM labels. If they agree >90%, stick with text-only.

---

### 4.8 — Joint text+image fine-tuning

**What:** Fine-tune both FashionCLIP text and vision encoders jointly.

**Why:** Phase 3 only fine-tuned the text encoder. The vision encoder is still generic. Joint training aligns both encoders for this specific catalog.

**How:**
1. Contrastive loss: text embedding should be close to image embedding for the same product, far for negatives
2. Use triplets from both text labels (3.5) and image labels (4.7)
3. Alignment regularization: penalize drift from pretrained weights to prevent catastrophic forgetting
4. Both encoders trainable, learning rate for vision encoder lower than text (1e-7 vs 1e-6)
5. 5 epochs, mixed precision

**Input:** Text + image labels + FashionCLIP both encoders
**Output:** `models/moda-fashion-multimodal/` (both encoders fine-tuned)
**Hardware:** T4 recommended (~6-8 hours), MPS possible but slow (~15-20 hours)
**Verify:** Re-embed all images with fine-tuned vision encoder. Rebuild FAISS index. Re-run 3-way hybrid + CE.

> **Watch out:** Alignment regularization is critical. Without it, the text encoder drifts toward the training data and text-to-text retrieval on queries outside the training set degrades. Specifically: compute cosine similarity between fine-tuned and pretrained embeddings for a held-out sample — if it drops below 0.8, the model has drifted too far and you need to increase the regularization weight. After training, you MUST re-embed both text and images for all 105K products and rebuild both FAISS indexes. The old indexes are invalid because the embedding space changed. Check that text-to-text retrieval (Tier 2) doesn't regress — if it does, the joint training hurt more than it helped. Vision encoder learning rate should be 5-10x lower than text encoder because it has more parameters and is more prone to catastrophic forgetting.

---

### 4.9 — Three-Tower Fashion Retriever

**What:** Novel architecture: dedicated query tower (trainable MLP), frozen text product tower, frozen image product tower. All project into shared 512-dim space.

**Why:** Product embeddings can be precomputed offline. Only the query tower runs at serving time. Practical for production.

**How:**
1. Architecture:
   - Query tower: FashionCLIP text encoder + 2-layer MLP → 512-dim
   - Text product tower: FashionCLIP text encoder (frozen) → 512-dim
   - Image product tower: FashionCLIP vision encoder (frozen) → 512-dim
   - Product embedding = mean(text_embedding, image_embedding)
2. Training: only query tower MLP is trainable
3. Loss: query embedding close to product embedding for relevant products
4. Use same triplets as 4.8

**Input:** Triplets + FashionCLIP encoders
**Output:** `models/moda-three-tower/` (query tower MLP weights)
**Hardware:** Fast — only training a small MLP. ~1-2 hours on MPS.
**Verify:** Compare vs symmetric retrieval (4.8). Is the asymmetric query tower competitive?

> **Watch out:** The product embedding `mean(text_emb, image_emb)` only works if both are in the same space, same scale, and L2-normalized. Verify that `||text_emb|| ≈ ||image_emb|| ≈ 1.0` before averaging. If one is larger, it dominates the mean. For products without images, the product embedding is just the text embedding — don't average with a zero vector (that halves the magnitude). The query tower MLP must output L2-normalized vectors to match the product embeddings. The product embeddings are precomputed and frozen during training — don't accidentally backpropagate through them. Check `requires_grad=False` on both product towers.

---

### 4.10 — LookBench evaluation with fine-tuned models

**What:** Run all Phase 4 models on LookBench. Submit to leaderboard.

**How:**
1. Run LookBench eval with:
   - Joint fine-tuned model (4.8)
   - Three-Tower model (4.9)
2. Compare against baselines from 4.4
3. Submit best result to LookBench leaderboard

**Output:** Updated Tier 3 leaderboard. Public submission.

> **Watch out:** LookBench is image-to-image. The Three-Tower model has an asymmetric query tower that encodes text, not images. It doesn't apply to LookBench directly. For LookBench, use the joint fine-tuned vision encoder from 4.8. The Three-Tower is only evaluated on Tier 2 (H&M text queries). Don't claim Three-Tower results on LookBench — it wasn't designed for image queries.

---

### 4.11 — Visual search demo

**What:** Image upload → find similar products. Image-to-image retrieval.

**How:**
1. User uploads image
2. Encode with fine-tuned FashionCLIP vision encoder
3. kNN search against product image FAISS index
4. Return top-10 similar products with thumbnails
5. Integrate into Flask UI

**Output:** Working visual search endpoint in the demo app.

> **Watch out:** The uploaded image needs the exact same preprocessing as the indexed images (same `preprocess_val` transform from open_clip). If a user uploads a phone photo (variable size, EXIF rotation, different aspect ratio), handle that gracefully before passing to the model. Strip EXIF, convert to RGB, apply the standard CLIP transform. Don't normalize twice.

---

## PHASE 5: Search experience + data augmentation

Goal: Make the search system usable for actual shoppers. Improve data quality.

---

### 5.1 — LLM query augmentation

**What:** Generate 3-5 paraphrases per real query using GPT-4o-mini.

**Why:** Expands training data without new human labels. "navy dress" becomes "dark blue dress", "navy colored dress", "dress in navy blue."

**How:**
1. Sample 10K unique queries from train split
2. Prompt: "Generate 3-5 search query paraphrases for: {query}. Keep the same intent, vary the wording."
3. Deduplicate against existing queries
4. Use augmented queries for iterative hard negative mining

**Input:** Train queries + GPT-4o-mini
**Output:** `data/processed/augmented_queries.jsonl` (~30-50K additional queries)
**Cost:** ~$1-2

> **Watch out:** Paraphrases must preserve intent. "black dress" → "dark colored dress" is fine. "black dress" → "black shoes" changes the category entirely. Validate a random sample of 100 augmented queries manually. Also: paraphrases should generate varied vocabulary, not just reword the same terms. "hoodie" should produce "sweatshirt with hood", "pullover hoodie", "hooded top" — not "hoodie garment", "hoodie clothing", "hoodie item" which are the lazy LLM outputs. Check for this. Don't use augmented queries for evaluation, only for training. They go into the train split only.

---

### 5.2 — Catalog enrichment with VLM

**What:** Generate detailed product descriptions from product images using a vision-language model.

**Why:** H&M's `detail_desc` field is often sparse. "Jersey top" doesn't mention the neckline, pattern, or fit visible in the product image.

**How:**
1. For each product image, call a VLM: "Describe this clothing product in detail: color, pattern, neckline, sleeve length, fit, material if visible."
2. Append generated description to existing product text
3. Re-index in OpenSearch (improves BM25)
4. Re-embed with FashionCLIP (improves dense retrieval)
5. Evaluate: do enriched descriptions improve nDCG@10?

**Input:** Product images + VLM API
**Output:** `data/processed/enriched_articles.csv` with `vlm_description` column
**Cost:** ~$3-5 for 105K images via GPT-4o-mini with vision
**Verify:** Compare search quality before/after enrichment on test split.

> **Watch out:** VLM descriptions supplement existing text, they don't replace it. Concatenate: `original_text + " | " + vlm_description`. If you overwrite, you lose the original product name which is critical for exact-match queries. The VLM may hallucinate attributes not visible in the image (e.g. "cotton blend" when the material isn't visible). This is acceptable for retrieval (false enrichment is better than missing description) but would be a problem for structured faceting. Don't use VLM-generated attributes for faceted filters without validation. Re-indexing in OpenSearch means rebuilding the entire index — not just updating documents — because the analyzer and field mappings need to accommodate the new text. Budget ~20 minutes for full re-index of 105K enriched articles.

---

### 5.3 — Iterative hard negative mining (ANCE-style)

**What:** Retrieve → label → train → re-retrieve → re-label → re-train. Multiple rounds.

**Why:** After one round of fine-tuning, the retriever's mistakes change. Mining new hard negatives from the improved retriever pushes quality further.

**How:**
1. Round 1: already done in Phase 3 (3.5 → 3.6)
2. Round 2: use the fine-tuned retriever from 3.6, retrieve top-20 for 5K train queries, label with GPT-4o-mini, build new triplets, retrain
3. Round 3: repeat with round-2 model
4. Typically 2-3 rounds before diminishing returns

**Input:** Fine-tuned retriever + GPT-4o-mini
**Output:** Iteratively improved retriever models
**Cost:** ~$1 per round for LLM labels
**Verify:** nDCG@10 should improve with each round, flattening by round 3.

> **Watch out:** Each round MUST re-embed all 105K articles with the latest model and rebuild the FAISS index before mining. If you mine from the round-1 model's index but using round-2 model's query encoding, the rankings are wrong because the index and query are in different embedding spaces. This is the most common bug in iterative training. Also: the new hard negatives from round 2 should genuinely be different from round 1. If >80% of the hard negatives are the same products, the model isn't learning new failure modes and another round won't help. Track overlap between rounds.

---

### 5.4 — Faceted navigation

**What:** Add structured filters alongside search results: color, category, gender, price.

**How:**
1. OpenSearch aggregations on structured fields
2. Return aggregations alongside search results
3. When user clicks a facet, add as `post_filter`
4. Build UI: filter sidebar with checkboxes and counts

**Input:** Existing OpenSearch index
**Output:** Faceted search API + UI
**Verify:** Filter "Ladieswear" + "Black" should show only black women's products.

> **Watch out:** Use `post_filter`, not `filter`, for faceted search. Regular `filter` affects both results AND aggregation counts, so clicking "Black" would hide the count for "Blue" — the user can't see other color options. `post_filter` filters results but keeps aggregation counts for the full result set, which is what shoppers expect. The `colour_group_name` values in H&M data are high-level groups ("Dark Blue", "Light Orange"), not user-friendly names ("Navy", "Coral"). Map to display names in the UI using the COLOR_MAP from `query_expansion.py`.

---

### 5.5 — Partitioned indexes

**What:** Split the OpenSearch index by gender. Route queries to the right partition based on NER.

**How:**
1. Create separate indexes: `moda_menswear`, `moda_ladieswear`, `moda_kids`, `moda_general`
2. NER detects gender → route to partition
3. No gender detected → search general (all products)
4. Compare: partitioned vs single-index-with-filter

**Input:** NER output + articles with `index_group_name`
**Output:** Partitioned indexes + routing logic

> **Watch out:** Some products have ambiguous or missing `index_group_name`. "Divided" is an H&M section that spans genders. Unisex items (accessories, some basics) need to exist in all partitions or in a dedicated "general" partition. If you put them in only one partition, queries without gender terms won't find them in the other partitions. Also: the FAISS dense indexes need to be partitioned the same way as OpenSearch. If BM25 searches the menswear partition but dense searches the full 105K catalog, the RRF fusion is comparing different candidate pools. Either partition both or partition neither.

---

### 5.6 — Auto-suggest

**What:** Query completion from the 253K real query log as the user types.

**How:**
1. Build prefix trie from unique query texts, weighted by frequency
2. API endpoint: `GET /suggest?q=nav` → `["navy dress", "navy hoodie", ...]`

**Input:** `queries.csv` (253K queries)
**Output:** Suggest API endpoint

> **Watch out:** The query log may contain PII (people sometimes type names or addresses into search bars), offensive terms, or garbled text (keyboard mashing). Filter the query corpus before building the trie: remove queries shorter than 2 characters, queries containing digits (likely order IDs), and queries that appear only once (likely typos). A frequency threshold of ≥3 occurrences is a reasonable starting point.

---

### 5.7 — Query relaxation

**What:** When a query returns too few results, automatically drop constraints and notify the user.

**How:**
1. If search returns < 5 results:
2. NER identifies query attributes
3. Drop attributes in priority order: pattern → material → fit → gender → color → type
4. Re-search with relaxed query
5. Show user: "Showing results for navy dress (relaxed from navy slim fit v-neck cashmere sweater mens)"

**Input:** NER output + search results
**Output:** Query relaxation logic + UI messaging

> **Watch out:** Don't drop "type" (the garment category) unless absolutely necessary — relaxing "navy slim fit jeans" to "navy slim fit" removes the most important constraint and returns random navy products. The priority order should keep type and color last. Also: relaxation should be transparent. If you silently drop "slim fit" without telling the user, they'll see regular-fit jeans and think the search is broken. Always show what was relaxed. Set a floor: if even the most relaxed query (just the garment type) returns < 5 results, show what you have rather than relaxing further.

---

### 5.8 — End-to-end search demo

**What:** Polished web UI with everything integrated.

**Features:**
- Text search with hybrid retrieval + reranking
- Image search (upload photo → similar products)
- Faceted filters (color, category, gender, price)
- Auto-suggest as you type
- "More like this" from any result
- Query relaxation with transparent messaging

**How:**
1. Flask or FastAPI backend
2. React or simple HTML/JS frontend
3. Docker Compose: OpenSearch + FAISS service + API server + frontend
4. One command to start everything

**Output:** `docker-compose up` → working fashion search demo at localhost:3000

> **Watch out:** Docker Compose startup order matters. OpenSearch takes 15-30 seconds to initialize. The API server needs to wait for OpenSearch to be healthy before indexing or searching. Use `depends_on` with a healthcheck, or a startup script with a retry loop. The FAISS index lives in memory — for 105K products at 512-dim, that's ~200MB, fine for a demo. But if you load multiple FAISS indexes (text, image, MoE at 672-dim), total memory grows. Keep the demo container's memory limit above 4GB. Serve product images from disk or a CDN, not from OpenSearch — don't store base64 images in the search index.

---

## Summary

| # | Task | Phase | Depends on | Hardware | Cost | Time |
|---|------|-------|-----------|----------|------|------|
| 3.1 | Train/test split | 3 | — | CPU | $0 | 5 min |
| 3.2 | LLM labels for CE | 3 | 3.1 | CPU + API | $2-3 | 2-3 hrs |
| 3.3 | Fine-tune CE (LLM labels) | 3 | 3.2 | T4/MPS | $0 | 2-3 hrs |
| 3.4 | Fine-tune CE (purchase labels) | 3 | 3.1 | T4/MPS | $0 | 2-3 hrs |
| 3.5 | LLM labels for bi-encoder | 3 | 3.1 | CPU + API | $1-2 | 2-3 hrs |
| 3.6 | Fine-tune FashionCLIP | 3 | 3.5 | T4/MPS | $0 | 4-6 hrs |
| 3.7 | Fine-tune GLiNER2 NER | 3 | 3.1 | T4/MPS | $0 | 1-2 hrs |
| 3.8 | MoE with trained encoders | 3 | 3.6, 3.7 | CPU + API | $1 | 2-3 hrs |
| 3.9 | Full Phase 3 eval | 3 | 3.3-3.8 | MPS | $0 | 4-6 hrs |
| 4.1 | Download H&M images | 4 | — | CPU | $0 | 1-2 hrs |
| 4.2 | Embed images (FAISS) | 4 | 4.1 | T4/MPS | $0 | 1-3 hrs |
| 4.3 | Download LookBench | 4 | — | CPU | $0 | 1 hr |
| 4.4 | LookBench baselines | 4 | 4.3 | T4/MPS | $0 | 2-4 hrs |
| 4.5 | Text-to-image retrieval | 4 | 4.2 | MPS | $0 | 1 hr |
| 4.6 | Three-way hybrid | 4 | 4.5 | MPS | $0 | 2-3 hrs |
| 4.7 | LLM labels for image negatives | 4 | 4.5 | CPU + API | $1-2 | 2-3 hrs |
| 4.8 | Joint text+image fine-tuning | 4 | 4.7 | T4 | $0-5 | 6-20 hrs |
| 4.9 | Three-Tower architecture | 4 | 4.8 | MPS | $0 | 1-2 hrs |
| 4.10 | LookBench final eval | 4 | 4.8, 4.9 | T4/MPS | $0 | 2-4 hrs |
| 4.11 | Visual search demo | 4 | 4.2 | CPU | $0 | 4-6 hrs |
| 5.1 | Query augmentation | 5 | 3.1 | CPU + API | $1-2 | 2 hrs |
| 5.2 | Catalog enrichment (VLM) | 5 | 4.1 | API | $3-5 | 4-6 hrs |
| 5.3 | Iterative hard neg mining | 5 | 3.6 | T4/MPS + API | $2-3 | 1-2 days |
| 5.4 | Faceted navigation | 5 | — | CPU | $0 | 4-6 hrs |
| 5.5 | Partitioned indexes | 5 | 3.7 | CPU | $0 | 2-3 hrs |
| 5.6 | Auto-suggest | 5 | — | CPU | $0 | 2-3 hrs |
| 5.7 | Query relaxation | 5 | 3.7 | CPU | $0 | 2-3 hrs |
| 5.8 | End-to-end demo | 5 | all above | CPU | $0 | 2-3 days |

**Total cost: ~$12-22**
**Total time: ~3-4 weeks**

Three benchmark tiers maintained throughout:
- **Tier 1:** Marqo 7-dataset (embedding quality cross-check)
- **Tier 2:** H&M 253K queries (full-pipeline search quality)
- **Tier 3:** LookBench (visual retrieval, contamination-aware)
