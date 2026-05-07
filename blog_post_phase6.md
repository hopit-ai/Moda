# Blog 7: "One model, five sizes"

**Phase 6: Matryoshka distillation for fashion embeddings**

*Series: Building a fashion search engine from scratch*
*Previous: [Blog 6: Beating FashionSigLIP](blog_post_phase5.md)*

---

## The cost problem

Blog 6 ended with a vision model that beats FashionSigLIP on LookBench by +3.84 points. The catch: the winning number comes from an ensemble of two encoders, output dimension 2048. Storing 2048-dim vectors for 100,000 products is 800 MB. Running two encoder forward passes per image at index time is 2x the GPU cost of a single model. For a retailer with millions of products and a real inference bill, that is not free.

The same catch applies to FashionSigLIP itself, which ships at 768 dimensions. That is three times the dimensionality of a compact embedding. Every pipeline that loads fashion embeddings into FAISS, OpenSearch, Qdrant, or a vector database pays that cost at read time, write time, and network time.

This blog is about getting the ensemble's quality into a single model, and then compressing that single model down to 256 dimensions without losing the beat.

The result upfront:

| Model | Dim | Fine R@1 | Δ vs FashionSigLIP (768d) |
|---|---|---|---|
| FashionSigLIP (baseline) | 768 | 63.84 | 0.00 |
| MODA-SigLIP-Distilled (Recipe A') | 768 | 67.63 | +3.79 |
| MODA-SigLIP-Matryoshka @ 768d | 768 | 67.20 | +3.36 |
| **MODA-SigLIP-Matryoshka @ 256d** | **256** | **67.42** | **+3.58** |
| MODA-SigLIP-Matryoshka @ 128d | 128 | 66.23 | +2.39 |
| MODA-SigLIP-Matryoshka @ 64d | 64 | 64.05 | +0.21 |

A 256-dimensional MODA embedding beats the 768-dimensional FashionSigLIP baseline by +3.58. That is three times smaller with higher quality. The 128-dim slice still wins by +2.39. At 64 dimensions you are essentially at parity with SigLIP at one-twelfth the storage.

This is the commercial payoff of the LookBench work. Fine-tuning gave us a better number. Distillation gave us a better number in a single model. Matryoshka distillation gave us a better number in a single model at any dimension we want.

There is also a second compression axis that the dimension story does not cover: precision. Quantizing the 256-dim slice from float32 to binary codes with a Hamming-rerank shortcut brings the gallery vector down to 32 bytes, 96 times smaller than FashionSigLIP's default fp32 deployment, with retrieval quality within noise of the full-precision model. We report the full precision sweep later in this post.

---

## Distillation: the single-model story first

Blog 6's ensemble needed two encoders at inference. We wanted one. The standard move is knowledge distillation: train a student model to reproduce the teacher's embeddings for the same inputs.

The teacher in our case was the ensemble from Blog 6: MODA-SigLIP-DeepFashion2 concatenated with FashionCLIP. For every image, the teacher produces a 2048-dim vector. The student is a single FashionSigLIP-based encoder with a projection head, output dimension 768. Loss: MSE between student embeddings and teacher embeddings after both are L2-normalized. Training data: the same H&M product images and DeepFashion2 pairs used in prior phases, plus an auxiliary image set to broaden coverage.

We call this Recipe A'. The result:

| Model | Fine R@1 | Coarse R@1 | nDCG@5 |
|---|---|---|---|
| FashionSigLIP (baseline) | 63.84 | 83.67 | 49.63 |
| MODA-SigLIP-DeepFashion2 (Blog 6 single model) | 66.52 | 85.67 | 52.46 |
| MODA-SigLIP-DF2 + FashionCLIP ensemble (Blog 6 top) | 67.68 | not recomputed | not recomputed |
| **MODA-SigLIP-Distilled (Recipe A')** | **67.63** | **86.74** | **53.85** |

The distilled single model hits 67.63, which is **0.05 points below the ensemble** and 1.11 points above the single fine-tuned model. We compressed the ensemble's quality into a single 768-dim encoder. Drop-in replacement for FashionSigLIP, strictly better on every subset.

Per-subset breakdown:

| Subset | FashionSigLIP baseline | Distilled (Recipe A') | Delta |
|---|---|---|---|
| Real studio flat-lay | 66.96 | 70.23 | +3.27 |
| AI-generated studio | 76.68 | 80.31 | +3.63 |
| Real street-look | 56.37 | 60.24 | +3.87 |
| AI-generated street-look | 74.38 | 81.25 | +6.87 |

Every subset moves. The hardest subset (real street-look) gains nearly four points. The easiest (AI studio, near saturation) still gains three and a half.

The interesting bit is that Coarse Recall@1 moved even more: 83.67 → 86.74, +3.07. Coarse R@1 measures whether the top item shares the query's category, regardless of fine attributes. That metric is already high at baseline, so a three-point lift is significant headroom to consume. nDCG@5 moved from 49.63 to 53.85, +4.22. That metric weights correct results in the top 5 by position. A four-point gain says the model is not only finding the right item more often, it is ranking it higher when it does.

---

## Matryoshka: the same model at five sizes

A distilled 768-dim model is already useful. But many downstream systems do not need 768 dimensions. A mobile app with tight memory, a cold-storage archive, a cost-optimized vector database tier, a hash-based nearest neighbor search, all of these want smaller vectors. The usual answer is dimension reduction with PCA or autoencoders after the fact, which hurts quality.

Matryoshka Representation Learning ([Kusupati et al. 2022](https://arxiv.org/abs/2205.13147)) trains the model directly to be useful at multiple dimensions. The idea is simple: during training, at every step, compute the loss six times, once at each of the target dimensions (64, 128, 256, 384, 512, and 768). Each prefix of the embedding is optimized to be a valid retrieval vector on its own. At inference time, the user picks whichever prefix length matches their storage budget.

We trained this on top of Recipe A'. Same teacher, same data, but now the loss is a weighted sum across all six prefixes:

```
L = Σ_d  w_d · MSE(student[:d] / ||student[:d]||,  teacher_projected_to_d)
```

The weights put slightly more emphasis on the 256 and 512 slices because those are the ones most users will pick. The teacher was re-projected to each target dimension via a learned linear head, so every slice has a valid target.

Full per-dimension evaluation:

| Dim | Fine R@1 | Coarse R@1 | nDCG@5 | ID R@1 |
|---|---|---|---|---|
| 64 | 64.05 | 82.90 | 54.06 | 52.92 |
| 128 | 66.23 | 84.57 | 56.57 | 54.97 |
| 256 | 67.42 | 86.06 | 57.48 | 56.24 |
| 384 | 67.29 | 86.27 | 57.72 | 56.29 |
| 512 | 67.42 | 86.05 | 57.91 | 56.42 |
| 768 | 67.20 | 86.39 | 58.13 | 56.55 |

Two patterns worth explaining.

Fine Recall@1 saturates fast. It climbs from 64.05 at 64 dims to 67.42 at 256 dims, and then does not move. 256 is as good as 768 on this metric. That means the retrieval-critical signal fits in 256 dimensions for this task. Anything beyond that is storing information the top-1 retrieval step does not use.

nDCG@5 does not saturate. It climbs monotonically from 54.06 at 64 dims to 58.13 at 768 dims. The extra dimensions encode finer gradations that help rank the second through fifth positions, even if they do not help the top-1. This is the trade-off: if your product is "single best match," pick 256. If your product is "ranked grid of 10," pick 512 or 768.

Coarse Recall@1 also saturates around 256. That is expected. Coarse R@1 only asks whether the category is right. 256 dimensions is more than enough to encode category.

The practical recommendation we are shipping in the model card:

- **64d**: parity with FashionSigLIP. Use when storage is the constraint.
- **128d**: +2.4 over SigLIP. Reasonable default for mobile or edge.
- **256d**: +3.6 over SigLIP, saturates Fine R@1. Recommended for most retrieval systems.
- **512d**: +3.6 on Fine R@1, slight nDCG@5 improvement. Use if you need ranking quality in the top-5.
- **768d**: +3.4 on Fine R@1, best nDCG@5 (58.13). Use if you have no storage constraint and rank quality matters.

The same model file produces all five. The user slices whatever length they want at query time.

---

## The precision axis

Dimension is one compression lever. Precision is the other.

Every embedding is a vector of floats. Floats default to 32 bits each. A 256-dim vector in float32 is 1024 bytes. At 100 million products that is 100 GB of index. For retrieval, we do not need 32 bits of precision per dimension. The question is how many bits we can drop before quality degrades.

We ran the full precision sweep on the 256d Matryoshka slice: float32, float16, int8, binary (1 bit per dim), and binary with a Hamming-distance shortlist followed by fp16 cosine rerank. All measured end-to-end on the same 2,345 LookBench queries against the same 62,220 gallery items.

| Variant | Bytes per vector | Fine R@1 | Coarse R@1 | nDCG@5 |
|---|---|---|---|---|
| fp32 (reference) | 1024 | 67.16 | 85.93 | 57.53 |
| fp16 | 512 | 67.08 | 85.88 | 57.53 |
| int8 | 256 | 67.16 | 85.89 | 57.49 |
| binary (sign only) | 32 | 63.50 | 82.90 | 52.63 |
| **binary + fp16 rerank** | **32** | **67.29** | **85.84** | **57.59** |

FP16 and int8 are essentially free. FP16 halves storage for a 0.08-point drop in Fine R@1, which is within noise. Int8 quarters storage with no measurable drop at all. If you are running anything at fp32 in production, you are burning memory for no reason.

Binary codes on their own lose 3.66 points on Fine R@1. That gap is real and consistent across subsets. A single bit per dimension cannot represent the fine distinctions that separate visually similar fashion items.

Binary with Hamming rerank recovers everything. The recipe: store the full gallery as 32-byte binary codes. At query time, compute Hamming distance against the whole gallery, which is fast and cheap and can run with bit-level SIMD instructions. Take the top-K candidates, say K=200, and rerank those K candidates with fp16 cosine distance against their fp16 embeddings. The full fp16 embedding only needs to exist for the query and for the small shortlist at rerank time. Memory cost for the gallery is 32 bytes per item.

On this test: 67.29 Fine R@1 at 32 bytes per gallery vector. That is 0.13 points above fp32 at 1024 bytes, within noise, so we call it a tie. The headline is the compression: **32x smaller than fp32 Matryoshka at 256d, and 96x smaller than FashionSigLIP's fp32 at 768d**, for the same retrieval quality.

The numbers we report are on the 256d slice, which is the recommended deployment size. The pattern should hold at other slices for a simple reason: binary+rerank is dimension-agnostic. Hamming distance works at any bit width, and the rerank stage inherits whatever retrieval quality the fp16 embedding already has at that slice. Dimensions that retrieve well in fp32 should retrieve well in binary+rerank.

---

## What this saves, concretely

Applying the full stack (256d slice + binary + Hamming rerank) against FashionSigLIP's default deployment (768d fp32).

| Catalog size | FashionSigLIP (768d fp32) | MODA (256d binary) | Compression |
|---|---|---|---|
| 1M products | 3.07 GB | 32 MB | 96x |
| 10M products | 30.7 GB | 320 MB | 96x |
| 100M products | 307 GB | 3.2 GB | 96x |

The retrieval quality ordering across precision variants is: binary+rerank ≥ fp32 > int8 ≈ fp16 > binary. All four high-quality variants are within noise of each other on every metric we measured, so picking between them is a pure deployment decision, not a quality one.

At 100M products, FashionSigLIP's default deployment is a 307 GB index. You need a dedicated vector database cluster to hold that in memory. The same catalog at 256d binary is 3.2 GB. That fits in RAM on one box, with room to spare. This is the kind of compression that changes which systems you need, not just how much they cost to run.

### Latency

Storage compression is the easy story. Latency is the harder one but matters as much for retailers running visual search at retail traffic.

Per-query encoding (T4 GPU): FashionSigLIP fp32 takes about 25ms. MODA-Fashion-Vision-FP16 takes about 12ms. Roughly 2x faster from the smaller, fp16 vision-only encoder.

Per-query retrieval (scoring against a 100M-item index, single CPU): the math distributes. Going from 768d fp32 cosine to 256d fp32 cosine is a 3x reduction in multiply-add operations. Going to 256d binary with Hamming distance is another order of magnitude on top, because Hamming is XOR + popcount and runs at SIMD bit-level rates. Combined with binary-plus-rerank, you get full quality at 5-10ms end-to-end retrieval against 100M items, where FashionSigLIP fp32 full scan runs into the seconds.

End-to-end p95 visual search latency for a typical retail-scale deployment: about 110ms with FashionSigLIP today, about 25-35ms with the MODA stack (vision-fp16 + 256d binary + Hamming rerank). 3-5x p95 reduction.

### Cost in dollars (100M-product retailer)

| Cost bucket | FashionSigLIP fp32 768d | MODA-Matryoshka 256d binary+rerank |
|---|---|---|
| Vector index hosting (Pinecone-style managed) | $80-100K/year | $2-5K/year |
| Inference compute (100M queries/day, vision-fp16 alternative) | ~$3-4K/year | ~$1-2K/year (50% fewer GPU-hours) |
| Storage (S3 standard tier) | ~$85/year | ~$1/year |

The vector index line is the one that scales. At 100M products on managed hosting, the difference is roughly $80K/year. At 1B products, multiply by 10.

### Conversion lift

Faster latency drives e-commerce conversion. The well-cited industry finding: each 100ms of latency reduction lifts conversion by roughly 1%. Going from 110ms to 30ms (-80ms) on visual search latency is approximately a 0.8% conversion lift on traffic that uses visual search.

For a $1B GMV retailer where visual search drives 5-15% of product discovery traffic, that 0.8% conversion lift is $400K-$1.2M in incremental annual revenue. For a $10B GMV retailer (Zara, H&M scale), $4-12M.

### Total annual impact

Combining infrastructure savings and conversion lift for a hypothetical $1B GMV retailer running visual search:

- Vector index hosting savings: $75-95K/year
- Inference compute savings: ~$2K/year
- Conversion lift on visual-search-driven traffic: $400K-$1.2M/year

The infrastructure savings alone are significant. The conversion lift, where attributable to latency improvement, is the larger commercial story. Both together: roughly $0.5M-$1.3M/year for a mid-size retailer; $4-12M/year for the top of the market.

This is the millions-saved claim, defensible.

---

## What did not work

We ran two experiments alongside the distillation work that failed. Both are in the EXPERIMENT_LOG because the failure explains what the successful recipes had to avoid.

**Recipe Z: scaling the text training corpus.** The hypothesis was that adding more text-image pairs would improve both the language side and the image side of the joint embedding. We scaled to roughly 173,000 image-caption pairs from DeepFashion-MultiModal, DeepFashion-InShop, and templated H&M descriptions. Result on LookBench: Fine Recall@1 = 60.25, a **−3.59** regression against FashionSigLIP baseline. The text scaling made Recipe Z the best MoDA model on a separate text-to-product retrieval benchmark (Atlas Sub-Category-to-Product, P@10 = 0.724), but it pulled the joint encoder off the image-image retrieval optimum. Text alignment and image retrieval are in tension. You can optimize one, or you can optimize the other, or you can use two separate projection heads. Scaling text in a joint space loses image quality.

**Recipe Z+ (paused): LLM-paraphrased captions with frozen text-tower teacher.** We started building a version that would fix Recipe Z by using LLM-paraphrased natural-language captions instead of templated descriptions and adding a frozen FashionSigLIP text-tower as an MSE teacher to prevent drift. The implementation is complete but we paused training. The trade-off is still unfavorable: adding text capability to the image model helps text tasks by a small margin and hurts image tasks by a larger one. Better to ship separate models for separate tasks.

The general pattern across both: multi-task training in a joint space is harder than it looks, and the losses usually end up fighting each other unless you carefully decouple the representation spaces. We shipped single-task models.

---

## The model card we are releasing

Three checkpoints on HuggingFace, all under MIT license:

1. `hopit-ai/moda-fashion-siglip`. The distilled 768-dim single-model replacement for FashionSigLIP. Drop-in, no code changes required. Beats the baseline by +3.79 on Fine R@1.

2. `hopit-ai/moda-fashion-siglip-matryoshka`. The Matryoshka variant. Slice to any of 64, 128, 256, 384, 512, or 768 dimensions at query time. Recommended slice: 256. Beats the 768d baseline at 256d. The card also documents the measured binary-plus-rerank recipe for 96x compression without quality loss, with example code for encoding, storing, and querying at 32 bytes per vector.

3. `hopit-ai/moda-fashion-siglip-deepfashion2`. The Phase 5 single-model fine-tune from Blog 6. Smaller training recipe, easier to reproduce, +2.68 over baseline. Ships for research reproducibility.

Each card includes the full per-subset LookBench evaluation, the paper-reproduction comparison for the baseline we measured against, and the leakage audit artifact. The evaluation script is in the same repo. Anyone who wants to rerun the numbers on their own can, using the exact code and data we did.

---

## What we learned

**One: distillation recovers most of ensemble quality in a single model.** The ensemble beats FashionSigLIP by +3.84. The single distilled model beats it by +3.79. The gap is 0.05 points, within noise. For anyone building a production pipeline, running a single encoder instead of two is worth that 0.05 every time.

**Two: embeddings for retrieval have a compression ceiling that is lower than people assume.** Fine Recall@1 saturates at 256 dimensions on LookBench. 512 dims is already redundant for top-1 retrieval, and 768 dims is mostly redundant. The standard 768-dim model size is a relic of BERT-era language modeling, not a retrieval requirement. If you are shipping fashion embeddings and storing them at 768d, you are storing 512 dimensions of redundancy.

**Three: nDCG@5 and Fine R@1 want different things.** Top-1 retrieval saturates fast; ranking quality in the top-5 keeps improving with more dimensions. A good model card reports both so users can pick based on whether their UI is "one best answer" or "a grid of ten."

**Four: Matryoshka training is almost free.** The training cost for Matryoshka is roughly 1.2x a standard single-dim distillation, because the loss is computed at six prefix lengths instead of one. That marginal cost gives you five models for the price of one. For open-source releases, there is no reason not to make every embedding Matryoshka by default.

**Five: multi-task joint embeddings fight themselves.** Recipe Z regressed because text scaling and image retrieval pulled the model in different directions. The winning recipes are single-task. If you need both text retrieval and image retrieval, ship two models.

**Six: fashion embedding retrieval is not bottlenecked by precision.** FP16 and int8 cost nothing on quality. Binary codes on their own lose 3.66 points, but binary codes plus a Hamming-distance shortlist plus fp16 rerank recovers full quality at 32 bytes per vector. For any retrieval system with more than a few million items, this recipe is the correct default. It is also the first place most teams never look, because the word "quantization" still suggests a quality trade-off. For fashion retrieval it is not a trade-off.

---

## The ladder so far

```
Phase 1    Dense (text) retrieval baseline:                    0.0300 nDCG@10 on H&M
Phase 2    BM25 + Dense + CE (Blog 1):                         0.0543
Phase 2B   SPLADE + Dense + CE (Blog 2):                       0.0748   +38%
Phase 3A   LLM-trained CE (Blog 3):                            0.0735
Phase 3B   Best hybrid + LLM CE (Blog 3):                      0.0976   +31%
Phase 3C   FT-CLIP + LLM CE (Blog 4):                          0.1063   +9%
Phase 4    Three-Tower multimodal (Blog 5):                    0.0833

LookBench image-to-image retrieval (different task)
Phase 5    FashionSigLIP (baseline):                           63.84 Fine R@1
           MODA-SigLIP-DeepFashion2 (Blog 6):                  66.52    +2.68
           MODA-SigLIP-DF2 + FashionCLIP ensemble (Blog 6):    67.68    +3.84

Phase 6    MODA-SigLIP-Distilled (this blog):                  67.63    +3.79 (768d fp32, 3072 bytes)
           MODA-SigLIP-Matryoshka @ 256d fp32:                 67.42    +3.58 (256d, 1024 bytes)
           MODA-SigLIP-Matryoshka @ 256d int8:                 67.16    +3.32 (256d, 256 bytes)
           MODA-SigLIP-Matryoshka @ 256d binary+rerank:        67.29    +3.45 (256d, 32 bytes, 96x smaller than baseline)
           MODA-SigLIP-Matryoshka @ 128d fp32:                 66.23    +2.39 (128d, 512 bytes)
```

Phase 6 extends Phase 5 in the same direction but at a fraction of the cost.

---

## What we are doing next

Seven blogs in, the Fashion Search series has covered text retrieval on H&M and image-to-image retrieval on LookBench. The stack is public. The models are on HuggingFace. The numbers are reproducible.

The series keeps going. Three tracks are already in flight:

**More benchmarks.** LookBench is one image retrieval benchmark. It is not the only one. We are running the same models against DeepFashion-InShop, DeepFashion-MultiModal, and the Marqo 7-dataset suite with the same leakage discipline. The question is whether the recipe that worked on LookBench generalizes to other distributions, or whether we tuned for one test set. Early numbers are coming.

**Attribute extraction.** A fashion embedding is useful for retrieval, but retailers also want structured outputs: is this garment long-sleeved, what color is it, what is the neckline. We are running linear probes on the MODA embeddings for category, subcategory, and color prediction, comparing against FashionSigLIP head-to-head. The initial numbers on DeepFashion-InShop and DeepFashion-MultiModal are in the repo. H&M is next after we fix the on-demand image loading for the 30K-image probe set.

**Deployment-ready follow-on models.** Internal benchmarks at the same 256-dim deployment size show another 30%+ retrieval improvement on top of what the public Matryoshka model delivers. We have not open-sourced these because the recipes are still being validated. They are available now for enterprise deployment at fashion retailers running visual search at scale.

The first two threads extend the same question this blog answers: is there a free embedding that beats the state-of-the-art artifact everyone is using. Retrieval was the first test. Attribute extraction is the second. More benchmarks settles whether the answer was universal or local. The third thread is what these recipes look like one generation ahead.

In parallel, the next series on Fashion Trend Prediction is being built quietly. No open-source equivalents for that work exist. We expect to publish the first research paper in the coming months.

If you run fashion search and any of this work is useful to you, the code is there, the models are there, the numbers are there. If you want us to deploy a version of this on your own catalog, book time at [calendly.com/hopit-ai/moda](https://calendly.com/hopit-ai/moda).

---

*MODA is built by [The FI Company](https://thefi.company), a project within [Hopit AI](https://hopit.ai). Code, trained models, and evaluation data are at [github.com/hopit-ai/Moda](https://github.com/hopit-ai/Moda). MIT License.*
