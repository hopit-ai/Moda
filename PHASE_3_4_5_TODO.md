# MODA Phase 3, 4, 5 — Detailed TODO

Starting assumption: Phase 1-2 is done. We have:
- Marqo 7-dataset benchmark reproduced (Tier 1)
- H&M 253K query full-pipeline benchmark with 11 configs (Tier 2)
- OpenSearch index with 105K articles, FAISS indexes for FashionCLIP/SigLIP/CLIP
- GLiNER v1 + GLiNER2 NER caches (10K + 253K queries)
- ColBERT cascade results (10K)
- `query_splits.json` with train/val/test split by unique query text
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

---

### 3.8 — Mixture-of-Encoders with trained field encoders

**What:** Build proper per-field encoders for color, category, and group. Concatenate with fine-tuned FashionCLIP into a single product vector.

**Why:** Phase 2 MoE failed (-12%) because we used FashionCLIP for all fields. Encoding "Dark Blue" through a model trained on full sentences doesn't produce a useful color representation. We need encoders designed for each data type.

**How:**

**Color encoder (64-dim):**
1. Get all unique `colour_group_name` values from H&M (~50 values)
2. Generate pairwise similarity labels using GPT-4o-mini: "Is navy similar to dark blue?" → score 0-1. ~1,225 pairs for 50 colors.
3. Train a small embedding model: Embedding(50, 64) → L2 normalize. Loss: cosine similarity should match LLM score.
4. Alternative: use LLM to cluster colors into groups (blues, reds, neutrals, etc.) and train with cluster-based contrastive loss.

**Category encoder (64-dim):**
1. Get all unique `product_type_name` values (~100 values)
2. Generate pairwise similarity: "Is jeans similar to trousers?" ~4,950 pairs.
3. Same architecture: Embedding(100, 64) → L2 normalize.

**Group encoder (32-dim):**
1. `product_group_name` has ~20 values. Small enough for a learned embedding table.
2. Train with product co-occurrence: groups that frequently appear together in outfits should be closer.
3. Or use LLM: "Is Garment Upper body related to Garment Lower body?" (yes, they form outfits).

**Concatenation:**
1. Per product: `concat(FashionCLIP_text(512), color_enc(64), category_enc(64), group_enc(32))` = 672-dim
2. Build FAISS index on 672-dim vectors for all 105K products
3. Per query: extract attributes via fine-tuned GLiNER2 (from 3.7), encode each with matching field encoder
4. Query vector: `concat(FashionCLIP_query(512), color_query(64), category_query(64), group_query(32))`
5. If NER doesn't detect an attribute (e.g. no color in query), zero out that block in the query vector
6. Cosine search against product FAISS index

**Input:** H&M structured fields + GPT-4o-mini for pairwise similarity + fine-tuned FashionCLIP from 3.6
**Output:** `models/moda-color-encoder/`, `models/moda-category-encoder/`, `models/moda-group-encoder/`, + FAISS index
**Cost:** ~$1 for LLM similarity labels (small vocabularies). Training: minutes on CPU.
**Verify:** MoE retrieval nDCG@10 should beat single-encoder FashionCLIP (unlike Phase 2 which lost -12%). Run full pipeline with MoE retriever + LLM-trained CE.

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

---

### 4.3 — Download and prepare LookBench

**What:** Download LookBench dataset and evaluation code. Set up as Tier 3 benchmark.

**Why:** LookBench is a live, contamination-aware, attribute-supervised fashion image retrieval benchmark. It tests visual retrieval quality, which our Tier 1 and Tier 2 don't cover. 4 subsets from easy (studio flat-lay) to hard (real street outfits).

**How:**
1. Clone LookBench repo (code + evaluation scripts)
2. Download all 4 subsets from their data release:
   - RealStudioFlat: 1,011 queries, ~62K corpus images
   - AIGen-Studio: 192 queries, ~59K corpus images
   - RealStreetLook: 1,000 queries, ~61K corpus images
   - AIGen-StreetLook: 160 queries, ~59K corpus images
3. Download corpus images for each subset
4. Verify dataset integrity: check query counts, corpus sizes, attribute annotations
5. Run their eval code with CLIP ViT-L/14 to reproduce their published baseline numbers

**Input:** LookBench release (GitHub + data download)
**Output:** `data/raw/lookbench/` with all 4 subsets, working eval pipeline
**Verify:** Reproduce CLIP ViT-L/14 baseline from paper (Fine R@1 ~39.79% overall)

---

### 4.4 — Run baseline models on LookBench

**What:** Evaluate all our existing models on LookBench to establish Tier 3 baselines.

**Why:** We need to know where our models stand on visual retrieval before fine-tuning for it.

**How:**
Run LookBench eval on:
1. CLIP ViT-B/32 (generic baseline)
2. Marqo-FashionCLIP (our Phase 2 text backbone)
3. Marqo-FashionSigLIP
4. Fine-tuned FashionCLIP from Phase 3 (text encoder only, vision encoder still off-the-shelf)
5. GR-Lite (LookBench's own open model, DINOv3-based) — external SOTA reference

Metrics: Fine Recall@1, Coarse Recall@1, nDCG@5 per subset + overall.

**Input:** LookBench subsets + our models
**Output:** Tier 3 leaderboard with 5 models x 4 subsets
**Verify:** Our Marqo-FashionCLIP numbers should be close to LookBench's published 63.24%.

---

### 4.5 — Text-to-image retrieval on H&M

**What:** Add a text-to-image retrieval channel: query text encoded by FashionCLIP text encoder, searched against product image vectors.

**Why:** Products sometimes have attributes visible in the image but not mentioned in the text. A "floral pattern" might be in the photo but the title just says "Summer Dress." Text-to-image catches these.

**How:**
1. For each query, encode with FashionCLIP text encoder (same as Phase 2)
2. Search against image FAISS index (from 4.2) instead of text FAISS index
3. Return top-100 image-matched products
4. Evaluate standalone text-to-image nDCG@10 on H&M test split

**Input:** Query embeddings + image FAISS index
**Output:** Text-to-image retrieval results + metrics
**Verify:** Compare vs text-to-text retrieval. Image channel may be weaker standalone but complementary.

---

### 4.6 — Three-way hybrid retrieval

**What:** Fuse three retrieval signals via RRF: BM25 + text-to-text dense + text-to-image dense.

**Why:** The core experiment. Does adding image retrieval improve text search results?

**How:**
1. For each query, get top-100 from:
   - BM25 (with NER boosts from fine-tuned GLiNER2)
   - Text-to-text dense (fine-tuned FashionCLIP)
   - Text-to-image dense (fine-tuned FashionCLIP text → product images)
2. RRF fusion with weight grid search:
   - BM25: 0.3-0.4
   - Text-to-text: 0.3-0.5
   - Text-to-image: 0.1-0.3
3. Cross-encoder rerank top-100 → top-50
4. Evaluate all weight combos on val split, pick best for test

**Input:** Three retrieval channels + LLM-trained CE
**Output:** Best 3-way hybrid weights + nDCG@10/MRR/Recall on test split
**Verify:** Does 3-way beat 2-way? How much does the image channel add?

---

### 4.7 — Generate LLM labels for image hard negatives

**What:** Mine hard negatives from the image retrieval channel and label with GPT-4o-mini.

**Why:** Needed for training the vision encoder. Same approach as 3.5 but for images.

**How:**
1. For each of ~5K train queries, run text-to-image retrieval, take top-20 image matches
2. Send (query_text, product_image_description) pairs to GPT-4o-mini for 0-3 scoring
3. Or: use a VLM (GPT-4o with vision) to directly judge query vs product image
4. Products scored 0 by LLM but ranked in top-20 = image hard negatives

**Input:** Text-to-image retrieval results + GPT-4o-mini API
**Output:** `data/processed/image_retriever_labels.jsonl`
**Cost:** ~$1-2

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

---

### 4.9 — Three-Tower Fashion Retriever

**What:** Novel architecture: dedicated query tower (trainable MLP on top of CLIP text encoder), frozen text product tower, frozen image product tower. All project into shared 512-dim space.

**Why:** Product embeddings (text + image) can be precomputed offline. Only the query tower runs at serving time. This is more practical for production where you don't want to re-embed 105K products every time the query model changes.

**How:**
1. Architecture:
   - Query tower: FashionCLIP text encoder + 2-layer MLP → 512-dim
   - Text product tower: FashionCLIP text encoder (frozen) → 512-dim
   - Image product tower: FashionCLIP vision encoder (frozen) → 512-dim
   - Product embedding = mean(text_embedding, image_embedding)
2. Training: only query tower MLP is trainable
3. Loss: query embedding should be close to product embedding for relevant products
4. Use same triplets as 4.8

**Input:** Triplets + FashionCLIP encoders
**Output:** `models/moda-three-tower/` (query tower MLP weights)
**Hardware:** Fast — only training a small MLP. ~1-2 hours on MPS.
**Verify:** Compare vs symmetric retrieval (4.8). Is the asymmetric query tower competitive?

---

### 4.10 — LookBench evaluation with fine-tuned models

**What:** Run all Phase 4 models on LookBench. Submit to leaderboard.

**How:**
1. Re-run LookBench eval with:
   - Joint fine-tuned model (4.8)
   - Three-Tower model (4.9)
2. Compare against baselines from 4.4
3. Submit best result to LookBench leaderboard

**Output:** Updated Tier 3 leaderboard. Public submission.

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

---

## PHASE 5: Search experience + data augmentation

Goal: Make the search system usable for actual shoppers. Improve data quality.

---

### 5.1 — LLM query augmentation

**What:** Generate 3-5 paraphrases per real query using GPT-4o-mini.

**Why:** 253K queries is a lot, but many are short and ambiguous. Paraphrases expand training data without new human labels. "navy dress" becomes "dark blue dress", "navy colored dress", "dress in navy blue."

**How:**
1. Sample 10K unique queries from train split
2. Prompt: "Generate 3-5 search query paraphrases for: {query}. Keep the same intent, vary the wording."
3. Deduplicate against existing queries
4. Use augmented queries for iterative hard negative mining (retrieve → label → train → repeat)

**Input:** Train queries + GPT-4o-mini
**Output:** `data/processed/augmented_queries.jsonl` (~30-50K additional queries)
**Cost:** ~$1-2

---

### 5.2 — Catalog enrichment with VLM

**What:** Generate detailed product descriptions from product images using a vision-language model.

**Why:** H&M's `detail_desc` field is often sparse or missing. "Jersey top" tells you nothing about the neckline, pattern, or fit. A VLM can describe what's visible in the product image: "crew neck, short sleeve, navy and white horizontal stripes, relaxed fit."

**How:**
1. For each product image, call a VLM (Qwen2.5-VL or GPT-4o-mini with vision): "Describe this clothing product in detail: color, pattern, neckline, sleeve length, fit, material if visible."
2. Append generated description to existing product text
3. Re-index in OpenSearch (improves BM25)
4. Re-embed with FashionCLIP (improves dense retrieval)
5. Evaluate: do enriched descriptions improve nDCG@10?

**Input:** Product images + VLM API
**Output:** `data/processed/enriched_articles.csv` with `vlm_description` column
**Cost:** ~$3-5 for 105K images via GPT-4o-mini with vision
**Verify:** Compare search quality before/after enrichment on test split.

---

### 5.3 — Iterative hard negative mining (ANCE-style)

**What:** Retrieve → label → train → re-retrieve → re-label → re-train. Multiple rounds.

**Why:** After one round of fine-tuning (Phase 3), the retriever's mistakes change. The hard negatives from round 1 may no longer be hard. Mining new hard negatives from the improved retriever and retraining should push quality further.

**How:**
1. Round 1: already done in Phase 3 (3.5 → 3.6)
2. Round 2: use the fine-tuned retriever from 3.6, retrieve top-20 for 5K train queries, label with GPT-4o-mini, build new triplets, retrain
3. Round 3: repeat with round-2 model
4. Typically 2-3 rounds is enough before diminishing returns

**Input:** Fine-tuned retriever + GPT-4o-mini
**Output:** Iteratively improved retriever models
**Cost:** ~$1 per round for LLM labels
**Verify:** nDCG@10 should improve with each round, flattening by round 3.

---

### 5.4 — Faceted navigation

**What:** Add structured filters alongside search results: color, category, gender, price range.

**Why:** Fashion shoppers filter. "Show me dresses" then filter by "blue" and "under $50." Facets don't replace ranking but they let users narrow results efficiently.

**How:**
1. OpenSearch aggregations on structured fields:
   - `colour_group_name` → color facets with counts
   - `product_type_name` → category facets
   - `index_group_name` → gender facets (Menswear, Ladieswear, etc.)
   - `price` → range buckets ($0-25, $25-50, $50-100, $100+)
2. Return aggregations alongside search results in API response
3. When user clicks a facet, add it as a `post_filter` in OpenSearch (doesn't affect aggregation counts)
4. Build UI: filter sidebar with checkboxes and counts

**Input:** Existing OpenSearch index (already has all fields)
**Output:** Faceted search API + UI
**Verify:** Filter "Ladieswear" + "Black" should show only black women's products. Counts should be accurate.

---

### 5.5 — Partitioned indexes

**What:** Split the OpenSearch index by gender (Menswear, Ladieswear, Kids, etc.). Route queries to the right partition based on NER-detected gender.

**Why:** Searching 105K products when NER detects "mens" means 70K women's/kids products are noise. Partitioning reduces corpus size and improves precision.

**How:**
1. Create separate OpenSearch indexes: `moda_menswear`, `moda_ladieswear`, `moda_kids`, `moda_general`
2. When NER detects gender in query, route to that partition
3. When no gender detected, search across all (or search `moda_general` which has everything)
4. Compare: partitioned search vs single-index with filter. Measure latency + nDCG.

**Input:** NER output + articles with `index_group_name`
**Output:** Partitioned indexes + routing logic
**Verify:** "mens hoodie" should only search menswear index. Latency should drop.

---

### 5.6 — Auto-suggest

**What:** As the user types, suggest query completions from the real query log.

**Why:** 253K real queries is a strong suggestion corpus. If a user types "nav", suggest "navy dress", "navy hoodie", "navy slim fit jeans."

**How:**
1. Build a prefix trie from all unique query texts
2. For each prefix, rank suggestions by frequency (how many times that query appeared)
3. API endpoint: `GET /suggest?q=nav` → `["navy dress", "navy hoodie", "navy slim fit jeans"]`
4. Alternative: OpenSearch completion suggester on a dedicated field

**Input:** `queries.csv` (253K queries)
**Output:** Suggest API endpoint
**Verify:** Type "bl" → should suggest "black dress", "black hoodie", "blazer", etc.

---

### 5.7 — Query relaxation

**What:** When a query returns too few results, automatically relax constraints and notify the user.

**Why:** "navy slim fit v-neck cashmere sweater mens" might return 0 results. Dropping "v-neck" or "cashmere" may surface relevant products.

**How:**
1. If search returns < 5 results:
2. Use NER to identify query attributes
3. Drop attributes in priority order (least important first): pattern → material → fit → gender → color → type
4. Re-search with relaxed query
5. Show user: "Showing results for navy slim fit sweater mens (relaxed from navy slim fit v-neck cashmere sweater mens)"

**Input:** NER output + search results
**Output:** Query relaxation logic + UI messaging
**Verify:** Queries that return 0 results should now return something reasonable. Check 50 zero-result queries.

---

### 5.8 — End-to-end search demo

**What:** Polished web UI with everything integrated.

**Features:**
- Text search with hybrid retrieval + reranking
- Image search (upload photo → similar products)
- Faceted filters (color, category, gender, price)
- Auto-suggest as you type
- "More like this" from any result (image similarity)
- Query relaxation with transparent messaging

**How:**
1. Flask or FastAPI backend serving all search endpoints
2. React or simple HTML/JS frontend
3. Docker Compose: OpenSearch + FAISS service + API server + frontend
4. One command to start everything

**Output:** `docker-compose up` → working fashion search demo at localhost:3000
**Verify:** End-to-end walkthrough: search "navy summer dress", filter by color, click a result, try "more like this", upload a photo.

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
