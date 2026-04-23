# Training the retriever on its own mistakes

*Blog 4 of the MODA series. We trained FashionCLIP and SPLADE on the candidates they were already getting wrong. The best dense retriever went from 0.0229 to 0.0542 standalone, +137%. The full pipeline landed at 0.1063, our project best.*

---

At the end of [Blog 3](blog_post_phase3a_3b.md) we had a trained cross-encoder and two off-the-shelf retrievers. The pipeline hit nDCG@10 = 0.0976 on our 22K held-out split. That number was produced by a 33M-parameter MiniLM-L12 reranker trained on 194K Sonnet-graded labels, sitting on top of a SPLADE + FashionCLIP hybrid that nobody had touched.

The cross-encoder was doing well. But it was only seeing 100 candidates per query. If those 100 missed the right product entirely, no amount of reranking could find it.

This is the thing we wanted to check. How often was the right product not in the top-100 from the retriever? And if it was happening, was it happening in a pattern we could exploit?

It was. That pattern is where this blog lives.

---

## Quick refresher: the retriever-reranker decomposition

If you are new to this, here is the model.

A retrieval pipeline has two jobs. First, pull a small pool of candidates from a huge index (retrieval). Second, rank that pool as carefully as you can (reranking). The retriever is fast and approximate. The reranker is slow and accurate but only ever sees the candidates the retriever chose.

The implication for error analysis: if the correct answer is not in the retrieved pool, the reranker cannot save you. The ceiling on your final metric is recall at the candidate pool size. If recall@100 is 25%, then even a perfect reranker can only give you nDCG@10 on 25% of queries. The other 75% are lost at retrieval.

In the Blog 3 pipeline, we measured recall@100 on the SPLADE+Dense hybrid: it was about 22.6%. Meaning 77.4% of queries did not have the purchased product in the top-100. Even a godlike reranker had an asymptote.

We had two ways to push that ceiling. Make the retriever better, or make the pool bigger. Blog 3 showed the pool trick makes things worse past ~100 (the cross-encoder's noise floor caps it). So the only real lever left was the retriever itself.

FashionCLIP was designed for fashion. It was trained on Marqo's fashion corpus. It had never been trained on H&M data specifically. SPLADE was even more generic: `splade-cocondenser-ensembledistil` was trained on MS MARCO passages, nothing to do with clothing at all. Both were working out of distribution. The question was how much headroom domain-specific training would unlock.

---

## The training data trick: mine negatives from the retriever's own top-K

The first decision was how to build training data for the retriever. The standard recipes have known failure modes.

**In-batch negatives**: for each query in a batch, use the other queries' positive documents as negatives. Cheap. Also useless, because "navy hoodie" and "red evening gown" are easy to tell apart, so the model learns nothing discriminative.

**Random negatives**: pick random products from the catalog. Same problem, slightly worse.

**Hard negatives**: pick products the current retriever already ranks in its top-K but are actually wrong. These are by construction the examples at the model's decision boundary. They are specifically the mistakes the model is making right now.

We wanted hard negatives. The challenge was labeling. We needed to know which of the top-K candidates were actually wrong versus the ones that just were not the purchased item.

We went back to the LLM labeling setup from Blog 3. Sampled 5,000 queries from our training split, ran the current retriever (SPLADE + FashionCLIP hybrid), took the top-20 per query (100,000 pairs), and sent every pair to GPT-4o-mini with the same 0-3 rubric. Cost: about $3.

Then we built the training set:

- **Positives**: LLM-graded 2 or 3 (good match or exact match).
- **Hard negatives**: LLM-graded 0 but ranked in the retriever's top-20 (the model was confident and wrong).
- **Easy negatives**: we added a handful of in-batch negatives for regularization.

After filtering and deduplication, we had 24,433 contrastive triplets. Not a lot. Each one was a place where the retriever was provably making a specific mistake.

```
Query: "navy slim jeans"

Retriever's top-20 (before training):
  1. Navy Slim Stretch Jeans   <-- LLM score 3 (positive)
  2. Dark Blue Slim Jeans      <-- LLM score 2 (positive)
  3. Black Slim Jeans          <-- LLM score 0 (hard negative, wrong color)
  4. Navy Wide Leg Jeans       <-- LLM score 0 (hard negative, wrong fit)
  5. Dark Blue Chinos          <-- LLM score 1 (ignored, ambiguous)
  ...

Training triplet (positive, hard negative):
  (query, Navy Slim Stretch Jeans, Black Slim Jeans)
```

Every triplet is specifically the model's own confusion. "You ranked Black Slim Jeans high for a navy query. Here is what high-confidence wrong looks like. Don't do that again."

---

## Fine-tuning FashionCLIP

We fine-tuned the text encoder of FashionCLIP with an [InfoNCE contrastive loss](https://arxiv.org/abs/2407.00143). The vision encoder stayed frozen (images come in Blog 5 when we get to multimodal). Only the text side moved.

Standard contrastive setup: for each (query, positive) pair in a batch, use the other positives as in-batch negatives, plus one mined hard negative. Temperature 0.05. Learning rate 1e-6 (small, since FashionCLIP is already pretrained and we did not want to catastrophically forget what it knew). Cosine schedule. Five epochs. About 45 minutes on an M4 Max.

Validation accuracy (fraction of queries where the positive scored higher than all mined negatives) stopped improving at epoch 3 at 0.994. We kept it five epochs to be safe.

**Standalone dense retrieval, before and after**:

| Metric | Baseline FashionCLIP | FT-FashionCLIP | Delta |
|---|---|---|---|
| nDCG@10 | 0.0229 | 0.0542 | +137% |
| MRR | 0.0208 | 0.0505 | +143% |
| Recall@10 | 0.0433 | 0.0811 | +87% |
| Recall@100 | 0.168 | 0.244 | +45% |

The dense retriever more than doubled on every metric. Recall@100 is the one that matters most for the downstream pipeline (it sets the ceiling for the reranker). Going from 16.8% to 24.4% means the cross-encoder now has the right answer in its pool for an additional 7.6% of queries. That is a lot of newly-catchable business.

The shape of the movement is informative. MRR (+143%) moves faster than nDCG@10 (+137%), which is faster than Recall@10 (+87%), which is faster than Recall@100 (+45%). The model is improving most at the very top of the list and least at the bottom. That is what you want from hard-negative fine-tuning. The positive example (the purchased product) is being pulled from rank 30 into the top-5, while the long tail is only slightly reshuffled. Training specifically on the retriever's own confident mistakes is what produces this shape. Random-negative training tends to flatten out across depths; hard-negative training is sharp at the top.

The training data was 24K labeled pairs worth $3 of LLM calls, plus $0 of compute. This is the largest single-component improvement in the project.

---

## Fine-tuning SPLADE

We did the same thing on the sparse side. Different loss (SPLADE uses a combination of InfoNCE and a FLOPS regularizer that keeps the output vectors sparse), same philosophy.

We generated 23,000 labeled pairs from 1,000 queries this time, using Claude Sonnet as the grader (about $4). Trained for three epochs with learning rate 1e-5, FLOPS L1 coefficient tuned to keep average non-zeros per vector at roughly the same level as the base model. About 45 minutes of training.

**Standalone sparse retrieval, before and after**:

| Metric | Baseline SPLADE | FT-SPLADE | Delta |
|---|---|---|---|
| nDCG@10 | 0.0464 | 0.0488 | +5.2% |
| MRR | 0.0695 | 0.0712 | +2.4% |
| Recall@10 | 0.0189 | 0.0195 | +3.2% |

Much less movement than on the dense side. Fine-tuning SPLADE gives us single-digit gains. Fine-tuning FashionCLIP gave us triple-digit gains. The metric shape here is also informative in the opposite direction: everything moves by roughly the same small amount. No structural change at the top vs bottom of the list. The model is learning a mild refinement, not fixing a systematic gap.

This asymmetry surprised us enough that we spent a day trying to understand it. Our best explanation:

SPLADE's pre-training objective is already close to what we want. It was trained to match queries against passages in MS MARCO. The training signal is "does this document answer this query," which is structurally the same task as fashion retrieval. The shift from web passages to product descriptions is real but not huge. There is less to fix.

FashionCLIP's pre-training was on fashion captions, which are written differently from H&M product titles. "A model wearing a navy blue zip-up hoodie with a front kangaroo pocket" (typical CLIP training caption) versus "Ben zip hoodie" (typical H&M title). The domain gap at inference time is large even though both are "fashion text." Hard negative fine-tuning closes that gap directly.

The practical takeaway: if you are resource-constrained, train the dense side first. It will pay back more.

---

## The hybrid sweep, one more time

We now had four retrievers in the matrix: baseline SPLADE, FT-SPLADE, baseline FashionCLIP, FT-FashionCLIP. We also had three rerankers from Blog 3 (off-shelf CE, LLM CE, AttrCE) plus the no-rerank baseline.

This is a 4 × 4 factorial, 16 cells, all on the same 22K held-out split with 95% bootstrap CIs.

We ran all of them. Here is the grid on three metrics. nDCG@10 first:

| Retriever | No rerank | Off-shelf CE | LLM CE | AttrCE |
|---|---|---|---|---|
| Baseline SPLADE + Baseline CLIP | 0.0556 | 0.0748 | 0.0946 | 0.0946 |
| Baseline SPLADE + FT-CLIP | 0.0589 | 0.0755 | **0.1063** | 0.1042 |
| FT-SPLADE + Baseline CLIP | 0.0541 | 0.0747 | 0.0983 | 0.0977 |
| FT-SPLADE + FT-CLIP | 0.0563 | 0.0768 | 0.1017 | 0.0989 |

MRR:

| Retriever | No rerank | Off-shelf CE | LLM CE | AttrCE |
|---|---|---|---|---|
| Baseline SPLADE + Baseline CLIP | 0.0662 | 0.0738 | 0.0660 | 0.0914 |
| Baseline SPLADE + FT-CLIP | 0.0681 | 0.0744 | **0.0766** | 0.0981 |
| FT-SPLADE + Baseline CLIP | 0.0650 | 0.0731 | 0.0925 | 0.0933 |
| FT-SPLADE + FT-CLIP | 0.0667 | 0.0751 | 0.0741 | 0.0940 |

Recall@10:

| Retriever | No rerank | Off-shelf CE | LLM CE | AttrCE |
|---|---|---|---|---|
| Baseline SPLADE + Baseline CLIP | 0.0201 | 0.0215 | 0.0253 | 0.0258 |
| Baseline SPLADE + FT-CLIP | 0.0213 | 0.0218 | **0.0265** | 0.0284 |
| FT-SPLADE + Baseline CLIP | 0.0195 | 0.0214 | 0.0268 | 0.0266 |
| FT-SPLADE + FT-CLIP | 0.0204 | 0.0221 | 0.0258 | 0.0270 |

The project best is **SPLADE(0.3) + FT-FashionCLIP(0.7) + LLM CE = 0.1063 nDCG@10**, 95% CI [0.1023, 0.1103]. Recall@10 = 0.0265, MRR = 0.0766.

Four observations.

**The best combination uses FT-CLIP but not FT-SPLADE.** Fine-tuning the dense retriever helps. Fine-tuning the sparse retriever also helps on its own, but when you combine both, you do worse than combining one fine-tuned with one baseline.

Our interpretation: fine-tuning both retrievers causes them to converge. They both learn to do well on the same hard cases (the ones the LLM grader cared about), which makes their errors correlated. The value of the hybrid comes from diversity, from SPLADE and dense making different mistakes. If you train them both on the same mistakes, they start making the same ones. The hybrid stops adding variety. The pattern is consistent across every column of the matrix: whichever pair is *mismatched in training state* beats the fully-matched pair.

**AttrCE beats LLM CE on MRR in every row.** Row 1: AttrCE = 0.0914 vs LLM CE = 0.0660. Row 2: 0.0981 vs 0.0766. Row 3: 0.0933 vs 0.0925. Row 4: 0.0940 vs 0.0741. On nDCG, LLM CE wins. This is the same split we flagged in Blog 3 and it is stable across retrievers: if you care about position-1 quality, ship AttrCE; if you care about the top-10 ordering, ship LLM CE. The explicit attribute features in AttrCE make the model extra confident about the single best match, at the cost of slight sloppiness in the middle of the list.

**Recall@10 moves with nDCG.** Every cell where nDCG improves, Recall@10 improves too. This is the tell that the CE is not just reshuffling the top-10 but pulling genuinely better candidates into it. If we saw nDCG improve without Recall moving, we would conclude the CE was overfitting to a few dominant features and reranking was all that changed.

**The optimal fusion weight shifted again.** Blog 2 (no training) liked SPLADE(0.5) + Dense(0.5). Blog 3 (LLM CE on baseline retrievers) liked SPLADE(0.4) + Dense(0.6). Blog 4 (LLM CE + FT-FashionCLIP) likes SPLADE(0.3) + FT-Dense(0.7). As the dense side gets better, it deserves more weight. The intuition: RRF fusion is roughly "whichever retriever is more trustworthy on this query type gets more vote share." Baseline dense was not trustworthy enough on H&M titles to carry most of the weight. FT-dense is. The fusion weight is tracking the actual capability delta between the two retrievers.

---

## Two failures we should name

**Mixture of encoders, done properly.**

In Blog 1 we reported a quick experiment on a "Superlinked-style" architecture where we built separate embeddings for color, category, and group, concatenated them with FashionCLIP, and used the resulting 672-dim vector. It failed at -12%. We flagged that experiment as exploratory because we had used the same FashionCLIP model for all four fields, which was obviously wrong.

In Phase 3 we did it properly. We trained three separate small encoders (color 64-dim MLP, category 64-dim MLP, group 32-dim MLP) on pairwise similarity labels from the LLM (is "navy" similar to "dark blue"? is "jeans" similar to "trousers"?). We concatenated these three learned field embeddings with FashionCLIP into a 672-dim vector. This time the architecture was not confused.

Result: nDCG@10 = 0.0382. Compared to 0.0397 for single-encoder FashionCLIP standalone. Within noise. Neutral.

This is a real negative result. The obvious story would be that the field-specific encoders add signal that the text encoder cannot capture. That story is wrong. FashionCLIP's 512-dim text embedding, trained on fashion product text, already encodes color, category, and group implicitly. Concatenating explicit per-field embeddings does not add new information. It adds dimensions the model has to learn to ignore.

Our read is that MoE-style architectures may pay off for modalities that the text encoder genuinely cannot see, like images, numeric price ranges, or behavioral signals. For attributes that are already present in the product text, MoE is duplicative.

**GPT-5-mini labels underperformed GPT-4o-mini labels.**

We ran a separate experiment to see whether cheaper or more recent models would work as graders. Generated 20K retriever labels from GPT-5-mini at about the same cost per label as GPT-4o-mini.

Training on those labels produced nDCG@10 = 0.0503 on the same retrieval task where GPT-4o-mini labels hit 0.0801. The GPT-5-mini labels were systematically noisier on the kind of fashion-specific judgment calls ("is this fit close enough?") that matter for our task. We did not dig deep into why; possibly GPT-5-mini's fashion vocabulary is narrower, possibly it hedges more on ambiguous cases.

The practical takeaway: not all LLM graders are equivalent. Before you spend $25 on labels from any new model, spend $2 validating that the labels move a retrieval metric in the right direction.

---

## Recall analysis: where does this cap?

We can compute the theoretical ceiling. For each query, the purchased product is the one relevant item out of 105,542 in the catalog. Given a candidate pool of size K, recall@K is the probability that the purchased item is in the top K. If it is, a perfect reranker gives it position 1 and nDCG@10 = 1.0 on that query. If it is not, nDCG@10 = 0 on that query regardless of reranker.

So the theoretical max nDCG@10 is just recall@K where K is the pool size. With our best retriever at K=100:

| Retriever | Recall@100 |
|---|---|
| BM25 | 8% |
| FashionCLIP baseline | 18% |
| FT-FashionCLIP | 24% |
| SPLADE + FT-FashionCLIP union | 33% |
| SPLADE + FT-FashionCLIP RRF fusion | 36% |

With a fusion recall of 36%, a perfect reranker could in principle hit nDCG@10 = 0.36. We hit 0.1063. There is a lot of headroom still in the reranker. Our CE is finding the right answer about one-third of the time when the right answer is in the pool.

This is where we think the next lever lives. Not bigger pools (Blog 3 showed they hurt). Not necessarily smarter retrievers (we are already stacking three of them). A better CE.

One in three hits when the right answer is in the pool. That is the number to beat next.

---

## The label economics, final tally

Across the whole project, we spent about $40 on LLM labels:

| Label set | Model | Pairs | Purpose | Cost |
|---|---|---|---|---|
| v1 | GPT-4o-mini | 9.8K | First CE training (pilot) | ~$2 |
| v2 | Sonnet 4.6 | 194K | CE-L12 training | ~$25 |
| v3-bienc | GPT-5-mini | 20K | Bi-encoder FT | ~$3 |
| v3-splade | Sonnet 4.6 | 23K | SPLADE FT | ~$4 |
| v3-img | GPT-5-mini | 20K | (Reserved for image tower, future) | ~$3 |
| v3-hn-r2 | GPT-5-mini | 20K | Hard negative mining round 2 | ~$3 |

Compute was $0 (everything ran on one MacBook). Every model in this series is small enough to fit on a laptop.

The artifact bundle we are releasing with this post:

- FT-FashionCLIP (sentence-transformers compatible)
- FT-SPLADE (uses the standard SPLADE inference loader)
- CE-L6 and CE-L12 (sentence-transformers CrossEncoder)
- All four label sets (CSV, one line per (query, product, score) triple)
- Training scripts for all four models

Everything is MIT licensed. If you run fashion search and you have not tried this flow on your own catalog, the total cost to replicate is about $10 on a small subset, enough to know if it is worth doing at your scale.

---

## What this ladder actually looks like

```
Phase 1 dense only:                                 0.0300
Phase 2 BM25 + Dense + CE (Blog 1):                 0.0543   (+81% over Phase 1)
Phase 2B SPLADE + Dense + CE (Blog 2, no training): 0.0748   (+149%)
Phase 3 LLM-trained CE (Blog 3):                    0.0976   (+225%)
Phase 3C FT retrievers + LLM CE (this post):        0.1063   (+254%)
```

The ladder is steep at the start and flattens out. The easy gains came from architecture (Blog 1). The middle gains came from training data (Blog 3). The last gain came from training on the retriever's own mistakes (this post).

Each stage left open questions. The biggest remaining one, per the recall analysis, is that we are at about 30% of the theoretical ceiling given our candidate pool. The reranker has headroom. Whether that headroom is unlocked by a bigger CE, a smarter training recipe, or an entirely different architectural move, we do not yet know.

We also have not touched images. Every retriever in this series used text only, even though H&M has a product image for every article. There is an obvious untapped signal there.

That is where the next posts are going.

---

*MODA is built by [The FI Company](https://thefi.company), a project within [Hopit AI](https://hopit.ai). Code, trained models, and the label sets referenced in this post are at [github.com/hopit-ai/Moda](https://github.com/hopit-ai/Moda). MIT License.*
