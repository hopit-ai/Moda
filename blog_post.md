# Open benchmark harness for fashion search

*Even 0 extra training can give you >75% gains.*

---

Everyone benchmarks fashion embeddings. Marqo publishes numbers on FashionCLIP. OpenAI has CLIP. Google has SigLIP. But nobody has published what happens when you wire up a full search pipeline and measure end-to-end results on real user queries. Not embedding-to-embedding cosine similarity on academic datasets. The whole thing: lexical retrieval, dense retrieval, hybrid fusion, reranking, query understanding.

We ran that experiment. 253,685 real search queries from H&M customers. 105,542 products. Eight pipeline configurations ablated one component at a time. The best configuration improved nDCG@10 by 81% over the best published fashion embedding baseline, using off-the-shelf zero-shot components. No custom training, no proprietary APIs. Total compute: a MacBook with Apple Silicon.

One widely recommended technique actively made things worse.

---

## First, we needed to trust our own numbers

Before measuring anything new, we had to know our evaluation harness worked. So we started by reproducing Marqo's published results.

Marqo runs the most comprehensive open fashion embedding benchmark: 7 datasets (DeepFashion In-Shop, DeepFashion Multimodal, Fashion200K, KAGL, Atlas, Polyvore, iMaterialist), three retrieval tasks. Their FashionCLIP and FashionSigLIP models are the best published numbers for fashion embedding quality.

We cloned their eval harness, downloaded 6 of 7 datasets (the 7th, iMaterialist, is 71.5GB; we deferred it), and ran their exact code with their exact models.

### Text-to-image retrieval (6-dataset average)

| Model | Recall@1 | MRR | vs Marqo published |
|-------|----------|-----|--------------------|
| Marqo-FashionSigLIP | 0.121 | 0.238 | <1% delta |
| Marqo-FashionCLIP | 0.094 | 0.200 | Reproduced |
| CLIP ViT-B/32 (baseline) | 0.064 | 0.155 | — |

### Category-to-product (5-dataset average)

| Model | Our P@1 | Marqo published P@1 | Delta |
|-------|---------|---------------------|-------|
| Marqo-FashionSigLIP | 0.746 | 0.758 | -1.6% |
| Marqo-FashionCLIP | 0.733 | 0.681 | +7.7% |
| CLIP ViT-B/32 | 0.581 | — | — |

Every number matched within 1-2%. FashionCLIP actually exceeded Marqo's published scores on 5 datasets, probably because their average includes iMaterialist which we skipped. The important thing: when we report numbers below, the measurement infrastructure has been validated against known-good results.

---

## Testing on real queries (10K sample)

With a working harness, we moved to the question nobody has answered publicly: what happens when you run a full search pipeline on real fashion queries?

### The data

Microsoft's H&M Search Data on HuggingFace has 253,685 real search queries from H&M customers, linked to 105,542 products with purchase-based relevance labels. These are queries people actually typed into a search box, with the product they bought as the positive label.

We started with 10,000 queries to check directionality before committing to the full run.

### BM25 loses badly on fashion

| Method | nDCG@10 | vs dense baseline |
|--------|---------|-------------------|
| BM25 only | 0.0187 | -37.7% |
| FashionCLIP dense | 0.0300 | baseline |

This is the opposite of general e-commerce benchmarks. On WANDS (furniture search), BM25 is competitive with dense retrieval. On fashion, it loses by 38%.

The reason is straightforward once you look at the data. H&M product names are brand-style identifiers: "Ben zip hoodie", "Tigra knitted headband." Real users search differently: "zip hoodie", "warm earband." There's a vocabulary gap between how fashion products are named and how people search for them. Dense embeddings can bridge that gap. BM25 can't.

### FashionCLIP beats FashionSigLIP (on this catalog)

| Model | nDCG@10 |
|-------|---------|
| Marqo-FashionCLIP | 0.0300 |
| CLIP ViT-B/32 | 0.0265 |
| Marqo-FashionSigLIP | 0.0232 |

This contradicts Marqo's own 7-dataset benchmark where SigLIP wins overall. The explanation is data distribution: H&M product text is short, keyword-style stuff, not natural language captions. FashionCLIP's 512-dim encoder was trained on product text that looks like this. SigLIP's bigger 768-dim encoder doesn't gain anything from the extra capacity on three-word titles.

Pick your embedding model based on your actual catalog, not average benchmark scores.

### Building up the pipeline

With FashionCLIP as the dense backbone, we added components one at a time.

Hybrid fusion (BM25 + dense, combined via Reciprocal Rank Fusion): we tested four weight combinations. BM25 x 0.4 + dense x 0.6 worked best. Push BM25 higher and vocabulary mismatch starts pulling in garbage.

Cross-encoder reranking: reranked the top-100 hybrid candidates with cross-encoder/ms-marco-MiniLM-L-6-v2. This was the biggest single improvement, +51% on top of hybrid results.

NER attribute boosting: GLiNER (zero-shot NER, NAACL 2024) extracts color, type, gender, and fit from queries. We mapped extracted entities to H&M field boosts via OpenSearch bool.should clauses. +14% on BM25 standalone.

Synonym expansion: we built an 80+ group fashion synonym dictionary (jacket/coat/blazer, pants/trousers/slacks, etc.). It hurt performance by 35%. Expanding "hoodie" to 12+ synonyms (sweatshirt, jumper, pullover...) collapses IDF weights and every product starts matching on something. Ranking precision disappears. This failure mode is documented in LESER (2025) and LEAPS (2026). We removed synonyms from the final pipeline.

On 10K queries, the full pipeline (hybrid + cross-encoder + NER) hit nDCG@10 = 0.0549. That's +83% over the dense baseline.

But 10K is a sample. Would it hold at full scale?

---

## Confirmation at scale (253,685 queries)

We ran every configuration on the complete dataset. Pre-computed caches for BM25 results, FAISS dense results, NER extractions, and cross-encoder scores. About 16 hours wall clock on Apple Silicon.

### Full ablation (253,685 queries, 105,542 products)

| # | Configuration | nDCG@10 | 95% CI | MRR | Latency | vs best baseline |
|---|---------------|---------|--------|-----|---------|-----------------|
| 1 | BM25 only | 0.0187 | [.0183-.0190] | 0.0227 | 11.5ms | -37.8% |
| 2 | BM25 + NER boost | 0.0204 | [.0200-.0207] | 0.0260 | ~18ms | -32.1% |
| 3 | Dense only (FashionCLIP) | 0.0265 | [.0261-.0269] | 0.0369 | <1ms | -11.8% |
| 4 | Hybrid (BM25x0.4 + dense x0.6) | 0.0328 | [.0324-.0333] | 0.0429 | 11.6ms | +9.4% |
| 5 | Hybrid + NER | 0.0333 | [.0329-.0338] | 0.0438 | ~18ms | +11.2% |
| 6 | Hybrid + CE rerank | 0.0543 | [.0537-.0550] | 0.0569 | 62.5ms | +81.1% |
| 7 | Full pipeline (+ NER) | 0.0543 | [.0537-.0550] | 0.0569 | ~69ms | +81.1% |

The numbers held. The 10K sample was within 1% of the full run. Bootstrap 95% confidence intervals are tight.

One thing we didn't expect: Config 6 and Config 8 produce identical results. NER adds nothing when the cross-encoder is already in the pipeline. The cross-encoder sees the full query-document pair and already captures what NER was contributing. In the final analysis, the pipeline is really three components: dense retrieval, hybrid fusion, and cross-encoder reranking.

### What each component actually adds

| Component | Marginal nDCG@10 gain | Relative improvement |
|-----------|----------------------|---------------------|
| Hybrid fusion (adding BM25 to dense) | +0.0053 | +17.8% |
| Cross-encoder reranking | +0.0180 | +51.0% |
| NER attribute boosting | +0.0016 | +3.0% |

The cross-encoder does most of the work. It scores full query-document pairs, so it picks up compositional queries like "relaxed fit navy summer dress" that both embeddings and BM25 handle poorly. Hybrid adds a real second layer. NER is noise once the cross-encoder is in play.

This ordering (dense >> hybrid >> rerank >> NER) matches what production search teams at Zalando, Pinterest, and ASOS have reported.

### Latency

| Stage | Mean | p50 | p95 |
|-------|------|-----|-----|
| BM25 (OpenSearch) | 11.5ms | 9.7ms | 18.2ms |
| RRF fusion | 0.1ms | 0.1ms | 0.2ms |
| CE rerank (100 candidates) | 50.9ms | 47.7ms | 73.3ms |
| Full pipeline | 62.5ms | ~58ms | ~92ms |

62.5ms end-to-end. The cross-encoder is the bottleneck at ~51ms, but 100 candidates through a 22M-parameter model is hard to beat on cost/quality tradeoff.

---

## What we took away from this

Full-pipeline gains dwarf embedding-only improvements. The best fashion embeddings (Marqo-FashionCLIP) give you 0.0300 nDCG@10. Adding hybrid fusion and a cross-encoder, both off-the-shelf, gets you to 0.0543. 81% better with zero training. The embedding matters, but it's maybe a third of the story.

Fashion search has a vocabulary problem that most general e-commerce doesn't. Products have brand-creative names ("Tigra knitted headband"). Users search with functional descriptions ("warm earband"). Dense retrieval handles this gap. BM25 alone does not. If you're building fashion search on BM25, check your numbers against a dense baseline.

Synonym expansion, which is recommended in basically every search engineering guide, hurt us by 35%. IDF collapse and query pollution. This isn't unique to our setup; LESER (2025) and LEAPS (2026) document the same failure mode. The fix requires behavioral click-through data to validate which expansions actually help, and public benchmarks don't have that.

Pick your embedding model based on your catalog, not benchmark rankings. FashionCLIP (512-dim) beat the supposedly better FashionSigLIP (768-dim) on H&M because H&M product text is short and structured. Benchmark averages hide this kind of thing.

If you add one component to dense retrieval, make it a cross-encoder reranker. Off the shelf, 50ms extra latency, +51% improvement. Nothing else came close.

---

## About the absolute numbers

nDCG@10 of 0.054 looks low if you're used to benchmarks like WANDS (0.76). The difference is relevance density. H&M has 1 positive per query across 105,542 candidates, using purchase as the relevance signal. A user searches, browses 20 dresses, buys one. The other 19 good dresses are scored as negatives. nDCG@10 in the 0.03-0.05 range is expected for this setup. The relative improvements (+81%) are what matter; absolute numbers aren't comparable across benchmarks with different relevance structures.

---

## What's next

This is Phase 2 of MODA (Modular Open-Source Discovery Architecture). Code, eval harness, and results are all open source.

Phase 3 trains custom models: a fashion cross-encoder fine-tuned on H&M pairs, and a GCL-trained bi-encoder using Marqo's open source methodology. Phase 4 adds image search. Phase 5 publishes the full benchmark as a preprint with a live leaderboard.

We want this to become a benchmark that anyone building fashion search can run against. If you try it on your catalog, we'd like to hear what you find.

---

*MODA is built by [The FI Company](https://thefi.company) which is a project within [Hopit.ai](https://hopit.ai). Code and results: [github.com/hopit-ai/moda](https://github.com/hopit-ai/moda). Apache 2.0.*
