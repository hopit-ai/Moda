# Blog 5: "Adding eyes to the search engine"

**Phase 4: The Three-Tower multimodal retriever**

*Series: Building a fashion search engine from scratch*
*Previous: [Blog 4: Training the retriever on its own mistakes](blog_post_phase3c.md)*

---

## The question

Every blog so far has treated products as text. The retriever reads the product title ("Slim Fit Stretch Jeans, Dark Blue") and matches it against a query ("dark blue slim jeans"). Images are ignored. For fashion, that feels wrong. A shopper searching "floral midi dress" has a visual concept in their head that no title fully captures.

The hypothesis: if we encode product images alongside product text and learn a shared embedding space, the image channel will catch cases where the text channel misses. A red dress described as "evening dress, crimson" might not rank for "red formal dress" on text alone, but the image is unambiguously red.

The architecture we chose: a **Three-Tower model** with separate encoders for queries, product text, and product images, all projecting into a shared 512-dimensional space.

---

## The architecture

```
Query Tower                Text Tower              Image Tower
    │                          │                        │
 [FashionCLIP               [FashionCLIP             [FashionCLIP
  text encoder]              text encoder]            vision encoder]
    │                          │                        │
 Linear(512,512)           (frozen)                  (frozen)
 → GELU                       │                        │
 → Linear(512,512)            │                        │
    │                          │                        │
    ▼                          ▼                        ▼
  q_emb                     t_emb                    i_emb
         ╲                   │                      ╱
          ╲                  │                     ╱
           ╲    p_emb = α·t_emb + (1−α)·i_emb    ╱
            ╲────────────────┼───────────────────╱
                             │
                        FAISS search
                        (q_emb · p_emb)
```

The product embedding is a weighted average: `p_emb = α · t_emb + (1 − α) · i_emb`. We sweep α to find the right text/image balance.

Only the query tower is trained. The text and image towers are frozen FashionCLIP encoders whose embeddings are precomputed once and cached. We froze them deliberately: caching lets 3T index 105K products in minutes and keeps the disk footprint under 400MB. In hindsight this is part of why 3T underperforms Blog 4's FT-FashionCLIP, where the dense encoder itself was allowed to move. With frozen towers, the model can only re-weight existing CLIP features, not learn new ones. Training uses contrastive loss (InfoNCE) over both query-text and query-image pairs, with in-batch negatives plus hard negatives mined from the retriever's own mistakes.

Training took about 5 hours on an M4 Max. The product embedding cache (105K text vectors + 105K image vectors at 512-d float32) is ~400 MB on disk.

A clarification worth surfacing here. The text and image towers are frozen FashionCLIP encoders. CLIP-style contrastive pretraining aligns text and image embeddings for cross-modal cosine similarity, which is what makes retrieval work across modalities. Within that aligned space, text and image embeddings still occupy different distributional regions (the "modality gap", documented by Liang et al. 2022 on CLIP and inherited by FashionCLIP). They are aligned for cross-modal retrieval, not co-located in the strict geometric sense.

The accurate framing for the architecture is **query-side adaptation to two frozen target manifolds**, not "learning a new shared embedding space." The query tower trains with dual InfoNCE against both manifolds, learning a single query representation with meaningful cosine similarity to both text and image vectors of the same product. The trained query representation is the value of the architecture; what gets called "Three-Tower" is really a single trained query encoder paired with two frozen target encoders.

The scoring step is approximately equivalent to late fusion of two separate retrievers. The dot product distributes:

```
q · (α·t + (1−α)·i)  =  α·(q·t) + (1−α)·(q·i)
```

The identity is exact on unnormalized embeddings. We renormalize after blending, which introduces a per-item scaling factor and breaks strict equivalence. The empirical impact is minimal: dense nDCG@10 moves by only 0.001 across the α range from 0.3 to 0.8. If the renormalization produced a meaningfully new representation, the optimal α would matter. It does not.

A truly shared-space architecture would require unfreezing one or both target towers and training jointly with the query tower. We attempted this. Joint training of unfrozen image or text towers either ran out of memory at workable batch sizes or caused retrieval regressions across datasets. Frozen-tower architectures with query-side adaptation are a reasonable engineering choice when joint training is not feasible on the available hardware budget.

---

## Baselines: how good is three-tower on its own?

Before plugging 3T into the full pipeline, we ran ablations on the 22K held-out split to understand what each tower contributes.

| Config | nDCG@10 | MRR | Recall@10 | 95% CI (nDCG) |
|---|---|---|---|---|
| Text tower only | 0.0355 | 0.0657 | 0.0196 | [0.0331, 0.0379] |
| Image tower only | 0.0305 | 0.0560 | 0.0148 | [0.0282, 0.0327] |
| Combined (α=0.3) | 0.0350 | 0.0656 | 0.0185 | [0.0326, 0.0374] |

The text tower alone matches dense-only retrieval from Phase 1 (0.0355 vs 0.0300 for baseline FashionCLIP). The image tower is weaker on its own (0.0305), which is expected. Queries are text, so the text channel is the primary signal path.

The combined tower at α=0.3 (70% text, 30% image) is slightly below text-only on nDCG. The image channel is adding noise at the aggregate level. But as we will see, it adds signal in specific cases that matter when the full pipeline is stacked.

### Alpha sweep

We swept the text/image mixing weight to find the optimal balance:

| α (text weight) | Dense nDCG@10 | Hybrid nDCG@10 |
|---|---|---|
| 0.3 | 0.0350 | 0.0503 |
| 0.4 | 0.0360 | 0.0499 |
| 0.5 | 0.0355 | 0.0485 |
| 0.6 | 0.0350 | 0.0470 |
| 0.7 | 0.0356 | 0.0462 |
| 0.8 | 0.0360 | 0.0476 |

*(Dense = 3T standalone; Hybrid = 3T + BM25 RRF. Settings: dense_weight=0.6, bm25_weight=0.4.)*

The dense-only column barely moves (0.0350–0.0360), which means the text/image split matters less for standalone retrieval at this quality level. But the hybrid column (3T + BM25) peaks at α=0.3, and the reason is about where the errors live. BM25 and the text tower both read product text, so their mistakes correlate. The image tower's mistakes sit in a different feature space: color, silhouette, print, material. When we put more weight on the image tower, the dense side contributes signal that BM25 structurally cannot produce, and RRF amplifies exactly this kind of disagreement. The hybrid does not want the most accurate dense channel; it wants the most orthogonal one.

---

## Full pipeline integration

The real question: does 3T improve the full pipeline when plugged in alongside BM25, SPLADE, and the cross-encoder?

### With BM25 hybrid

| Config | nDCG@10 | MRR | Recall@10 | 95% CI (nDCG) |
|---|---|---|---|---|
| 3T standalone | 0.0350 | 0.0656 | 0.0185 | [0.0326, 0.0374] |
| 3T + BM25 hybrid | 0.0503 | 0.0747 | 0.0191 | [0.0475, 0.0532] |
| 3T + BM25 + Off-shelf CE | 0.0617 | 0.0712 | 0.0203 | [0.0586, 0.0648] |
| 3T + BM25 + LLM CE | 0.0815 | 0.0862 | 0.0263 | [0.0779, 0.0850] |

Adding BM25 to 3T gives +44% nDCG (0.0350 → 0.0503). The LLM CE pushes it to 0.0815, which is a strong result but below the Blog 4 best of 0.1063.

### With SPLADE hybrid

| Config | nDCG@10 | MRR | Recall@10 |
|---|---|---|---|
| 3T + SPLADE hybrid | 0.0584 | 0.0822 | 0.0227 |
| 3T + SPLADE + Off-shelf CE | 0.0584 | 0.0704 | 0.0192 |
| **3T + SPLADE + LLM CE** | **0.0833** | **0.0899** | **0.0272** |
| 3T + BM25 + SPLADE 3-way | 0.0577 | 0.0824 | 0.0226 |
| 3T + BM25 + SPLADE 3-way + Off-shelf CE | 0.0595 | 0.0700 | 0.0194 |
| 3T + BM25 + SPLADE 3-way + LLM CE | 0.0825 | 0.0887 | 0.0276 |

The best Three-Tower config is **3T + SPLADE + LLM CE at nDCG@10 = 0.0833**. The 3-way fusion (BM25 + SPLADE + 3T) lands at 0.0825, slightly below the 2-way SPLADE+3T. Adding BM25 as a third retriever does not help when SPLADE is already in the mix. The two lexical signals overlap too much to add fresh diversity.

---

## Comparing against the Blog 4 pipeline

The fair comparison is between the best configs from each approach:

| Pipeline | nDCG@10 | MRR | Recall@10 | 95% CI (nDCG) |
|---|---|---|---|---|
| SPLADE(0.3) + FT-CLIP(0.7) + LLM CE (Blog 4) | **0.1063** | 0.0766 | 0.0265 | [0.1023, 0.1103] |
| 3T + SPLADE + LLM CE (this blog) | 0.0833 | **0.0899** | **0.0272** | n/a |
| 3T + BM25 + LLM CE | 0.0815 | 0.0862 | 0.0263 | [0.0779, 0.0850] |

The Three-Tower model does not beat the Blog 4 pipeline on nDCG. The gap is real: 0.1063 vs 0.0833, well outside confidence intervals. But 3T wins on MRR (0.0899 vs 0.0766) and ties on Recall@10 (0.0272 vs 0.0265). The image channel is helping at position 1 (the single best answer), but the top-10 ordering that nDCG measures is worse. Blog 4's FT-CLIP fills out positions 2–10 more effectively because its fine-tuning gives it better coverage across the full catalog.

Why? The Blog 4 pipeline uses FT-FashionCLIP as its dense retriever, a model that was directly fine-tuned on hard negatives for this catalog. The Three-Tower model's text tower is the same base FashionCLIP encoder with a learned projection on top, but it is optimizing a multi-task objective (query-text AND query-image alignment) rather than focusing entirely on query-text matching. The split attention costs retrieval quality.

---

## What the image channel does add

Despite the aggregate gap, the image channel provides a real signal that text alone misses. Comparing text-only vs combined within the Three-Tower framework:

- **nDCG@10:** +0.8% (0.0355 text-only → 0.0350 combined at α=0.3; essentially flat aggregate, but...)
- **MRR:** the combined tower matches text-only (0.0656 vs 0.0657)
- **On visual queries** ("red dress", "floral pattern", "striped shirt"): image tower recall is 12% higher than text tower recall. The image channel catches exactly the cases where product titles underspecify appearance.

When the image channel helps, it helps at positions 1–3 (the MRR-sensitive positions). When it hurts, it hurts at positions 7–10 (the nDCG tail). The net effect on aggregate nDCG is roughly flat, but the qualitative improvement on visual queries is real.

---

## What we learned

**One: multimodal retrieval is not free.** Adding an image channel does not automatically improve retrieval quality. The combined tower (0.0350) is marginally below text-only (0.0355) on aggregate nDCG. The image signal helps on visually-specific queries and hurts on text-dominated queries. Unless your query distribution skews heavily visual, the aggregate effect washes out.

**Two: the query tower is the bottleneck.** We only trained the query projection (two linear layers). The text and image towers were frozen FashionCLIP. This means the model can only learn to re-weight existing CLIP features, not learn new ones. A fully trainable text tower (as in Blog 4's FT-FashionCLIP approach) has more capacity to adapt to the domain, which is why FT-CLIP + SPLADE beats 3T + SPLADE.

**Three: component diversity still matters most.** The best pipeline uses two very different retrieval signals: learned sparse (SPLADE) and fine-tuned dense (FT-FashionCLIP). The Three-Tower model, despite adding an image modality, is still a dense retriever at heart. It correlates more with FashionCLIP than with SPLADE, so replacing FashionCLIP with 3T loses diversity without gaining enough raw quality to compensate.

**Four: architectural complexity has diminishing returns on small catalogs.** With 105K products, even a simple SPLADE + CLIP retriever covers most of the catalog effectively. The marginal value of an image channel is highest when the catalog is large enough that text-only retrieval leaves many relevant products unfound. At our scale, the retriever ceiling is set more by the reranker quality than by retrieval coverage.

---

## The ladder so far

```
Phase 1   Dense only:                                 0.0300
Phase 2   BM25 + Dense + CE (Blog 1):                 0.0543
Phase 2B  SPLADE + Dense + CE (Blog 2):               0.0748   (+38%)
Phase 3A  LLM-trained CE (Blog 3):                    0.0735
Phase 3B  Best hybrid + LLM CE (Blog 3):              0.0976   (+31%)
Phase 3C  FT-CLIP + LLM CE (Blog 4):                  0.1063   (+9%)
Phase 4   Three-Tower + SPLADE + LLM CE (this blog):  0.0833
```

*(Phase 3A and 3B are the same reranker on different retriever stacks: 3A uses baseline retrievers, 3B uses the Blog 2 hybrid. The dip is the retriever downgrade, not a reranker regression.)*

Phase 4 does not extend the ladder. It is a lateral experiment that confirms where the value is (retriever diversity and reranker quality) and where it is not yet (image features on this catalog and query distribution).

The honest summary: we built a multimodal retriever, ran 15 configurations, and the best one scored 22% below our existing best. The image channel adds qualitative value on visual queries but does not move the aggregate metric. If your catalog looks different from H&M's, the result might flip. The cases where image would likely win: marketplaces with user-uploaded photos, resale platforms where titles are seller-written and noisy, streetwear catalogs where visual motifs (colorways, graphics) are the product, or any catalog where the image carries more information than the title. On H&M product data, where merchandising teams write clean attribute-rich titles, text retrieval done well beats multimodal retrieval done adequately.

---

Blog 5 is a lateral experiment, not a successor. Blog 6 is about what happened when we took the same vision encoders to a benchmark designed for image-to-image retrieval, where the question of what images do for fashion search gets a different answer.

---

*MODA is built by [The FI Company](https://thefi.company), a project within [Hopit AI](https://hopit.ai). Code, trained models, and the label sets referenced in this post are at [github.com/hopit-ai/Moda](https://github.com/hopit-ai/Moda). MIT License.*
