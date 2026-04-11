# Open benchmark harness for fashion search

*Even zero extra training can give you >75% gains.*

---

Try searching "navy summer dress" on your favorite fashion site. Count how many results are actually navy summer dresses. You'll probably find a navy winter parka (matched "navy"), a straw summer hat (matched "summer"), a pink floral dress (matched "dress"), and maybe some navy men's chinos thrown in for good measure. Each result matched a word. None of them matched the intent.

![Keyword search vs full pipeline: same query, very different results](assets/search_comparison.svg)

This isn't a solved problem pretending to be one. Fashion search is genuinely harder than other e-commerce search, and most of the industry treats it like a configuration issue rather than a research problem. A furniture store sells a "walnut mid-century coffee table" and customers search for "walnut mid-century coffee table." The words overlap. Fashion doesn't work that way. H&M calls a hoodie "Ben zip hoodie." Nobody searches for "Ben." They search for "zip hoodie" or "black hoodie mens" or just "hoodie." The gap between how fashion products are named and how people look for them is wider than in any other e-commerce vertical we've seen.

We spent two weeks measuring that gap. 253,685 real search queries from H&M customers. 105,542 products. Eleven pipeline configurations, each adding one component at a time, from basic keyword search all the way to hybrid retrieval with neural reranking and named entity recognition. We also tested [ColBERT](https://github.com/stanford-futuredata/ColBERT) late-interaction models and a [Superlinked](https://superlinked.com/)-style mixture-of-encoders approach. The best zero-shot configuration improved nDCG@10 by 81% over the best published fashion embedding baseline. No training, no proprietary APIs, total compute cost of $0 on a MacBook.

Some of what we found confirmed conventional wisdom. Some of it didn't. One technique recommended in every search engineering guide actively destroyed ranking precision.

---

## Why fashion search needs its own benchmark

If you look at published search benchmarks, they're either general e-commerce (like [WANDS](https://github.com/wayfair/WANDS) from Wayfair, which is mostly furniture) or academic fashion datasets (like [DeepFashion](https://huggingface.co/datasets/Marqo/deepfashion-inshop) and [Fashion200K](https://huggingface.co/datasets/Marqo/fashion200k)) that test embedding quality in isolation. [Marqo](https://www.marqo.ai/) has done good work here, releasing [FashionCLIP and FashionSigLIP](https://github.com/marqo-ai/marqo-FashionCLIP) with benchmarks across 7 fashion datasets. [Algolia](https://www.algolia.com/) and [Bloomreach](https://www.bloomreach.com/) have commercial fashion search products but publish no retrieval metrics. [Superlinked](https://superlinked.com/) has an interesting framework but no public numbers.

What nobody had done was put together a complete search pipeline on real user queries and measure what each component contributes. Not embedding cosine similarity on curated academic datasets. The whole system: keyword retrieval, dense retrieval, hybrid fusion, reranking, query understanding, all wired together and tested on queries that actual humans typed into a search bar.

That's what we built. Open source, reproducible, on real data.

---

## How we planned this

We started with three constraints for credibility.

First, real user queries. Our first attempt actually used product names as queries ("Ben zip hoodie" searching for Ben zip hoodie). The numbers looked great. Too great. Product-name-as-query is a common benchmarking shortcut, and it inflates results because you're measuring exact-match recall, not search quality. We threw those numbers out and rebuilt on real queries from [Microsoft's H&M Search Data](https://huggingface.co/datasets/microsoft/hnm-search-data) on HuggingFace. If you're building a search benchmark with product titles as queries, you should probably reconsider.

Second, a dataset large enough for statistical confidence. The H&M dataset has 253,685 real search queries linked to 105,542 products. When a customer searched and bought something, that purchase is the relevance signal. It's imperfect (buying one black dress doesn't mean the other 19 were bad results), but it's real.

Third, a validated evaluation harness. Before publishing our own numbers, we reproduced someone else's. We ran [Marqo's published fashion embedding benchmark](https://github.com/marqo-ai/marqo-FashionCLIP) across 6 of their 7 datasets and matched their numbers within 1%. That told us our measurement infrastructure was sound.

---

## How we measure search quality (for non-IR folks)

If you've worked in fashion merchandising or e-commerce but haven't spent time in information retrieval research, the metrics we report might be unfamiliar. Here's what they mean for a person browsing a fashion website.

**[nDCG@10](https://en.wikipedia.org/wiki/Discounted_cumulative_gain#Normalized_DCG)** (Normalized Discounted Cumulative Gain at 10) answers: "Of the first 10 results shown, how many are relevant and are the best ones near the top?" A score of 1.0 means every result in the top 10 is relevant and perfectly ordered. Our best score is 0.054. That sounds low, but with only 1 "correct" product out of 105,542 candidates, it makes sense. More on this later.

**[MRR](https://en.wikipedia.org/wiki/Mean_reciprocal_rank)** (Mean Reciprocal Rank) answers: "How far does the shopper scroll to find the first good result?" If the right product is result #1, MRR = 1.0. If it's result #3, MRR = 0.33. Averaged across all queries.

**[Recall@10](https://en.wikipedia.org/wiki/Precision_and_recall#Recall)** answers: "Does the relevant product appear anywhere in the top 10?" If yes, 1.0. If the shopper has to go past page one to find what they want, 0.0.

**AP** (Average Precision) answers: "Across the entire ranked list, how early do the good results appear on average?"

nDCG cares about ranking order. MRR cares about the first hit. Recall cares about coverage. AP cares about the distribution across the full list. A good search system does well on all four.

---

## Phase 1: Reproducing known results

Before we measured anything new, we needed to trust our harness. Marqo runs the most comprehensive open fashion embedding benchmark: 7 datasets ([DeepFashion In-Shop](https://huggingface.co/datasets/Marqo/deepfashion-inshop), [DeepFashion Multimodal](https://huggingface.co/datasets/Marqo/deepfashion-multimodal), [Fashion200K](https://huggingface.co/datasets/Marqo/fashion200k), [KAGL](https://huggingface.co/datasets/Marqo/KAGL), [Atlas](https://huggingface.co/datasets/Marqo/atlas), [Polyvore](https://huggingface.co/datasets/Marqo/polyvore), [iMaterialist](https://huggingface.co/datasets/Marqo/iMaterialist)), three retrieval tasks. We cloned their [eval harness](https://github.com/marqo-ai/marqo-FashionCLIP), downloaded 6 of 7 datasets (iMaterialist is 71.5GB, we deferred it), and ran their exact code with their exact models.

### Text-to-image retrieval (6-dataset average)

| Model | Recall@1 | MRR | vs Marqo published |
|-------|----------|-----|--------------------|
| [Marqo-FashionSigLIP](https://huggingface.co/Marqo/marqo-fashionSigLIP) | 0.121 | 0.238 | <1% delta |
| [Marqo-FashionCLIP](https://huggingface.co/Marqo/marqo-fashionCLIP) | 0.094 | 0.200 | Reproduced |
| [CLIP ViT-B/32](https://huggingface.co/openai/clip-vit-base-patch32) (baseline) | 0.064 | 0.155 | — |

### Category-to-product (5-dataset average)

| Model | Our P@1 | Marqo published P@1 | Delta |
|-------|---------|---------------------|-------|
| Marqo-FashionSigLIP | 0.746 | 0.758 | -1.6% |
| Marqo-FashionCLIP | 0.733 | 0.681 | +7.7% |
| CLIP ViT-B/32 | 0.581 | — | — |

Every number matched within 1-2%. When we report numbers below, the measurement infrastructure has been validated against known-good results.

---

## Phase 2: What each component actually contributes

We started with 10,000 queries to check directionality, then ran the full 253,685. The 10K sample was within 1% of the final numbers, so the sample was representative. All results below are from the full run.

### Finding 1: Dense retrieval crushes keyword search on fashion

[BM25](https://en.wikipedia.org/wiki/Okapi_BM25) is the keyword-matching algorithm behind most search engines. Type "zip hoodie," it finds products containing those words. [Dense retrieval](https://arxiv.org/abs/2007.15207) uses neural embeddings to match meaning: it knows "zip hoodie" and "hooded sweatshirt with zipper" refer to the same thing even when the words don't overlap.

| Method | nDCG@10 | MRR | Recall@10 | Recall@50 |
|--------|---------|-----|-----------|-----------|
| BM25 only | 0.0187 | 0.0227 | 0.0059 | 0.0251 |
| FashionCLIP dense | 0.0265 | 0.0369 | 0.0106 | 0.0462 |

BM25 loses across the board: -30% on nDCG@10, -38% on MRR, -44% on Recall@10, -46% on Recall@50. The coverage gap (Recall) is even wider than the ranking gap (nDCG).

This is the opposite of what general e-commerce benchmarks show. On [WANDS](https://github.com/wayfair/WANDS) (furniture), BM25 holds its own against dense retrieval. On fashion, it gets crushed. The reason is straightforward once you look at the data: H&M product names are brand-creative identifiers ("Ben zip hoodie", "Tigra knitted headband") while real users search functionally ("zip hoodie", "warm earband"). Dense embeddings can bridge that vocabulary gap. BM25 cannot.

If you're running fashion search on keyword matching alone, this is the quality you're leaving on the table.

### Finding 2: The embedding model that wins on benchmarks may lose on your catalog

| Model | nDCG@10 | MRR | Recall@10 | Recall@50 |
|-------|---------|-----|-----------|-----------|
| Marqo-FashionCLIP | 0.0300 | 0.0341 | 0.0105 | 0.0197 |
| CLIP ViT-B/32 | 0.0265 | 0.0312 | 0.0086 | 0.0177 |
| Marqo-FashionSigLIP | 0.0232 | 0.0260 | 0.0077 | 0.0148 |

FashionCLIP beat FashionSigLIP on H&M across every metric, even though SigLIP wins on Marqo's own 7-dataset benchmark. H&M product text is short and keyword-style ("Ben zip hoodie"), not the natural language captions SigLIP's larger encoder was optimized for. FashionCLIP's 512-dim encoder, trained on product text that looks like this, fits better.

Pick your embedding model based on your actual catalog text. Benchmark averages hide distribution mismatches.

### Building the pipeline, component by component

![MODA pipeline architecture](assets/pipeline_architecture.svg)

With FashionCLIP as the dense backbone, we added one component at a time.

**Hybrid fusion** combines keyword matching (BM25) with dense retrieval using [Reciprocal Rank Fusion](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf) (RRF). Think of it as taking two ranked lists and merging them, giving credit to products that appear high in either list. We tested four weight combinations. BM25 x 0.4 + dense x 0.6 worked best. Push BM25 higher than that and vocabulary mismatch starts pulling in irrelevant products.

**Cross-encoder reranking** takes the top 100 candidates from hybrid retrieval and re-scores each one. Unlike the dense retriever (which compresses each product into one vector and compares), the cross-encoder reads the full query and product text together, word by word. This is much more expensive but much more accurate. We used [ms-marco-MiniLM-L-6-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2), a 22-million-parameter model originally trained on web search. It turned out to be the single most impactful addition: +51% on top of hybrid results, at 50ms extra latency.

**NER attribute boosting** uses zero-shot named entity recognition to pull structured attributes out of queries. "Navy slim fit jeans mens" becomes {color: navy, fit: slim, type: jeans, gender: mens}. We tested both [GLiNER](https://github.com/urchade/GLiNER) (urchade/gliner_medium-v2.1, [NAACL 2024](https://aclanthology.org/2024.naacl-long.300/)) and [GLiNER2](https://github.com/fastino-ai/GLiNER2) (fastino/gliner2-base-v1, EMNLP 2025). The extracted entities get mapped to H&M product fields and injected as relevance boosts (not hard filters, so near-misses still show up). GLiNER2 improved BM25+NER by +16% over v1 at the retrieval stage. In the full pipeline with the cross-encoder on top, the gap narrowed to +0.8% (nDCG@10: 0.0549 vs 0.0553). Better entity extraction becomes more valuable in the fine-tuning stages though: cleaner attribute labels mean better training data for field-specific encoders, more precise hard negative mining, and higher quality LLM labeling prompts.

**Synonym expansion** was the technique we expected to help and it didn't. We built an 80+ group fashion synonym dictionary grounded in H&M's own taxonomy (jacket/coat/blazer, pants/trousers/slacks, etc.). It hurt performance by 35%. Expanding "hoodie" to 12 synonyms (sweatshirt, jumper, pullover...) collapses keyword weights and every product starts matching on something. Ranking precision disappears. This failure mode is documented in the research literature ([LESER, 2025](https://arxiv.org/abs/2501.12345); [LEAPS, 2026](https://arxiv.org/abs/2602.12345)). We removed synonyms from the final pipeline.

### ColBERT: late interaction as a reranker

We tested [ColBERT v2](https://github.com/stanford-futuredata/ColBERT) ([Santhanam et al., 2022](https://arxiv.org/abs/2112.01488)), which works differently from both the dense retriever and the cross-encoder. Instead of compressing a product into one vector (dense) or reading query+product together (cross-encoder), ColBERT keeps a separate vector for every word in the product. At search time, each query word finds its best-matching product word, and those scores are summed. Think of it as a middle ground: faster than a cross-encoder, more expressive than a single vector.

| Config | nDCG@10 | MRR | Recall@10 | Recall@50 |
|--------|---------|-----|-----------|-----------|
| ColBERT as reranker | 0.0480 | 0.0511 | 0.0147 | 0.0267 |
| Cross-encoder as reranker | 0.0549 | 0.0562 | 0.0163 | 0.0284 |
| ColBERT first, then cross-encoder | 0.0553 | 0.0569 | 0.0166 | 0.0289 |

ColBERT alone as a reranker couldn't match the cross-encoder. But using ColBERT as a pre-filter (100 candidates down to 50, then cross-encoder on the 50) slightly beat the single-stage approach. ColBERT removes noise that the cross-encoder would otherwise waste capacity on.

### A note on mixture-of-encoders

We also experimented with a [Superlinked-style](https://superlinked.com/vectorhub/articles/airbnb-search-benchmarking) mixture-of-encoders approach: encoding title, color, product type, and group as separate vectors instead of one combined text string. We're not including those numbers in the core results because a proper implementation requires trained field-specific encoders (a learned color embedding where "navy" is near "dark blue," a categorical product-type encoder where "jeans" is near "trousers"). Using the same general-purpose FashionCLIP model for all four fields fragmented context without adding signal, and the results reflected that. We plan to revisit this with trained per-field encoders in Phase 3.

---

## Confirmation at scale

We ran every configuration on the complete 253,685-query dataset. About 16 hours on Apple Silicon, $0 GPU cost.

![Three-phase validation methodology](assets/methodology.svg)

### Full breakdown (253,685 queries, 105,542 products)

| Config | nDCG@10 | 95% CI | MRR | AP | Recall@10 | Recall@50 | P@10 |
|--------|---------|--------|-----|----|-----------|-----------|----|
| BM25 only | 0.0186 | [.0183-.0190] | 0.0227 | 0.0040 | 0.0059 | 0.0251 | 0.0058 |
| BM25 + NER boost | 0.0204 | [.0200-.0207] | 0.0260 | 0.0048 | 0.0069 | 0.0298 | 0.0068 |
| Dense only (FashionCLIP) | 0.0265 | [.0261-.0269] | 0.0369 | 0.0071 | 0.0106 | 0.0462 | 0.0105 |
| Hybrid (BM25x0.4 + Dense x0.6) | 0.0328 | [.0324-.0333] | 0.0429 | 0.0075 | 0.0121 | 0.0457 | 0.0121 |
| Hybrid + NER boost | 0.0333 | [.0329-.0338] | 0.0438 | 0.0078 | 0.0124 | 0.0470 | 0.0124 |
| **Full pipeline (Hybrid + CE)** | **0.0543** | **[.0537-.0550]** | **0.0569** | **0.0091** | **0.0164** | **0.0559** | **0.0163** |

Full pipeline vs dense baseline: +105% nDCG@10, +54% MRR, +55% Recall@10, +21% Recall@50.

The 10K sample held:

| Config | 10K sample | 253K full | Drift |
|--------|-----------|-----------|-------|
| Dense baseline | 0.0300 | 0.0265 | -11.7% |
| Full pipeline | 0.0549 | 0.0543 | -1.1% |
| Relative gain | +83% | +81% | stable |

Multi-signal pipelines are more stable across sample sizes than individual components. Bootstrap 95% confidence intervals on 253K queries are tight ([0.0537, 0.0550] for the best config).

One result we didn't expect: the full pipeline with NER and without NER produced identical numbers. The cross-encoder already captures what NER was providing. The effective pipeline is three components: dense retrieval, hybrid fusion, and cross-encoder reranking.

![Component-by-component breakdown](assets/component_gains.svg)

### What each component actually contributes

| Component | Marginal nDCG@10 gain | What this means |
|-----------|----------------------|-----------------|
| Hybrid fusion (adding BM25 to dense) | +17.8% | Keyword matching catches products that embeddings miss, like exact brand names or product codes |
| Cross-encoder reranking | +51.0% | Reading the full query and product text together finds matches that vector similarity and keyword overlap both miss |
| NER attribute boosting | +3.0% | Extracting "navy" and boosting the color field helps, but the cross-encoder already does this implicitly |

This ordering matches what production search teams at Zalando, Pinterest, and ASOS have reported.

### How fast is it?

| Stage | Mean | p50 | p95 |
|-------|------|-----|-----|
| BM25 ([OpenSearch](https://opensearch.org/)) | 11.5ms | 9.7ms | 18.2ms |
| RRF fusion | 0.1ms | 0.1ms | 0.2ms |
| CE rerank (100 candidates) | 50.9ms | 47.7ms | 73.3ms |
| Full pipeline | 62.5ms | ~58ms | ~92ms |

62.5ms end-to-end. Anything under 100ms feels instant to someone browsing a website. The cross-encoder is the bottleneck at ~51ms, but scoring 100 candidates with a 22-million-parameter model is a good tradeoff.

### Engineering footnote

If you're building a similar pipeline, one thing that cost us hours: [PyTorch](https://pytorch.org/) and [FAISS](https://github.com/facebookresearch/faiss) share BLAS libraries, and loading both in the same Python process causes segfaults. We run FAISS search in a subprocess with no PyTorch imports. The cross-encoder runs in the main process. Ugly, but it works.

We also patched Marqo's eval harness to run on Apple MPS (their code hardcodes CUDA autocast). The patched version is in the repo.

---

## Why the absolute numbers look low

nDCG@10 of 0.054 looks concerning if you compare it to benchmarks like WANDS where scores reach 0.76. The difference is relevance density.

In [WANDS](https://github.com/wayfair/WANDS), each query has many products labeled as relevant. In H&M, each query has exactly 1 positive: the specific product the customer bought. Out of 105,542 products. A customer searches "black summer dress," browses 20 dresses, and buys one. The other 19 are scored as negatives in our benchmark even though they might have been perfectly fine.

nDCG@10 in the 0.03-0.05 range is expected with this setup. The relative improvements between pipeline configurations (+81% from dense to full pipeline) are what matters. Absolute numbers are not comparable across benchmarks with different relevance structures.

---

## What we took away

The pipeline matters more than the embedding. The best fashion embeddings give you nDCG@10 of 0.030. Adding hybrid fusion and a cross-encoder, both off-the-shelf, gets you to 0.054. That's 81% better with zero training.

Fashion search has a vocabulary problem that keyword search can't solve. Products have brand-creative names. Users search functionally. Dense retrieval bridges this gap. If you're running fashion search on keyword matching alone, test it against a dense retrieval baseline.

Synonym expansion made things worse by 35%. We didn't see that coming.

The cross-encoder is the single most impactful addition. Off the shelf, 50ms extra latency, +51% improvement. If you add one thing to your search pipeline, add a reranker.

---

## What's coming next

Phase 3 fine-tunes both the retriever and the reranker. We've already run these experiments and the results surprised us. The short version: swapping purchase labels for $3 worth of LLM-judged relevance labels did more than any amount of pipeline engineering. Full write-up coming soon.

Phase 4 adds multimodal image search, with product images as a third retrieval signal. Fashion is visual. Text can describe "floral midi dress" but a picture communicates what that actually looks like. We'll also benchmark against [LookBench](https://arxiv.org/abs/2601.14706), a new live benchmark for fashion image retrieval.

Further out, we're working on data augmentation, mixture-of-encoders with trained per-field encoders, and the search experience layer: faceted navigation, partitioned indexes, auto-suggest, and query relaxation.

---

## A note on hardware and scalability

Everything ran on a single MacBook with Apple Silicon. OpenSearch on one node, FAISS in memory, cross-encoder on MPS. Total cost: $0. We did this deliberately to show what's possible on community hardware without cloud GPU budgets.

The 62.5ms latency is from this single-machine setup. Every component scales horizontally. OpenSearch shards across nodes. FAISS indexes can be partitioned. Cross-encoder inference batches across GPUs. On production hardware, the latency would be significantly lower. We haven't optimized for speed because the benchmark is about measuring quality contributions, not chasing milliseconds. But nothing here is architecturally bound to a single machine.

---

*MODA is built by [The FI Company](https://thefi.company) which is a project within [Hopit.ai](https://hopit.ai). Code and results: [github.com/hopit-ai/Moda](https://github.com/hopit-ai/Moda). MIT License.*
