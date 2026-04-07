# $3 of LLM labels beat everything else we tried

*Phase 3 of our fashion search benchmark. We fine-tuned both the retriever and the cross-encoder. The model that won wasn't the one we expected.*

---

In [Phase 2](blog_post.md) we built a zero-shot fashion search pipeline and got nDCG@10 = 0.0553 on 253K real H&M queries. +84% over the dense embedding baseline, no training involved.

Phase 3 was supposed to be about training better models. We did that. We fine-tuned the cross-encoder on H&M purchase data. We fine-tuned FashionCLIP on hard negatives. We ran a full 2x3 factorial evaluation. The final number is 0.0757 nDCG@10, +152% over Phase 1.

But the most interesting finding wasn't a model. It was the training data.

---

## 3A: Fine-tuning the cross-encoder on purchase data

The obvious first move. We had 253K queries with purchase labels (user searched, user bought product X). So we built training pairs: positive = the product the user bought, negative = products shown but not bought. About 1.5M training pairs total.

We fine-tuned the same ms-marco-MiniLM-L-6-v2 cross-encoder from Phase 2 on these pairs. Evaluated on a held-out test set of 22,855 queries (split by unique query text, no leakage).

| Metric | Off-shelf CE | Fine-tuned CE | Delta |
|--------|-------------|---------------|-------|
| nDCG@5 | 0.0442 | 0.0480 | +8.6% |
| nDCG@10 | 0.0646 | 0.0654 | +1.2% |
| MRR | 0.0671 | 0.0644 | -4.0% |
| Recall@10 | 0.0195 | 0.0183 | -6.2% |

A wash. nDCG@5 improved but MRR and Recall got worse. The fine-tuned model learned to rank slightly better at the very top but lost coverage further down. Net effect: barely measurable.

We expected more. 1.5M training pairs should be enough data. The architecture is fine. So what went wrong?

### The problem is the labels

Purchase labels are a lie. When a user searches "black summer dress" and buys one, that doesn't mean the other 19 black summer dresses they scrolled past were bad results. But the training treats every not-purchased product as a negative. The hard negatives are contaminated with relevant products the user just didn't happen to buy.

This is a known problem in e-commerce search. The model isn't learning "what's relevant," it's learning "what this specific user happened to purchase," which is a noisier signal than it looks.

The domain gap is also smaller than we expected. Fashion product text is still natural language. The off-the-shelf MS MARCO model, trained on web search queries and Wikipedia passages, already transfers reasonably well to fashion. There's less room for domain-specific fine-tuning to help than with, say, medical or legal text.

---

## 3B: LLM-judged labels changed everything

Phase 3A pointed at the data, not the model. So we tried fixing the data.

We generated 42,800 relevance labels using GPT-4o-mini (via PaleblueDot API). For each query-product pair, the LLM assigned a graded score: 0 = not relevant, 1 = partial match, 2 = good match, 3 = exact match.

The score distribution was balanced: 27.7% score-0, 21.1% score-1, 25.0% score-2, 26.2% score-3. No skew toward a single label. The LLM can actually distinguish relevance grades in fashion, not just binary yes/no.

Same architecture. Same MiniLM-L6 cross-encoder. Same training loop. Different labels.

| Config | nDCG@10 | MRR | Recall@10 | vs off-shelf CE |
|--------|---------|-----|-----------|-----------------|
| Off-shelf CE | 0.0646 | 0.0671 | 0.0195 | baseline |
| Fine-tuned CE (purchase labels, 3A) | 0.0654 | 0.0644 | 0.0183 | +1.2% |
| LLM-trained CE (GPT-4o-mini labels, 3B) | 0.0747 | 0.0755 | 0.0217 | +15.7% |

+15.7% nDCG@10. Same model, different data. The LLM-trained cross-encoder improved on every single metric where the purchase-trained one struggled.

42,800 clean labels beat 1.5 million noisy ones. The entire LLM labeling cost was about $3.

This confirms something the IR community has been debating: for search relevance training, label quality matters more than label quantity, and both matter more than model architecture. You can spend weeks tuning hyperparameters and architecture, or you can spend $3 on better labels. The $3 wins.

---

## 3C: Fine-tuning the retriever (not just the reranker)

Phases 3A and 3B improved the reranker. But the reranker only sees candidates the retriever surfaced. If the retriever misses a relevant product, no amount of reranking will save it.

So we fine-tuned FashionCLIP itself.

### How we built the training data

We used the retriever's own mistakes as training signal:

1. Sampled 5,000 unique queries from the training set (no overlap with test)
2. Ran FashionCLIP retrieval, took the top-20 candidates per query (100,000 pairs)
3. Sent each pair to GPT-4o-mini for a relevance label (0-3)
4. Products that FashionCLIP ranked highly but the LLM scored 0 = hard negatives (the model's specific failure cases)
5. Products scored 2-3 = positives

This produced 24,433 contrastive triplets: anchor query, positive product, hard negative product. The hard negatives aren't random; they're products the model was confident about but wrong.

### Training

InfoNCE contrastive loss with in-batch negatives plus one mined hard negative per query. Only the text encoder was trainable (vision encoder frozen). 5 epochs, learning rate 1e-6, cosine schedule. All on Apple M4 Max with FP16 mixed precision and gradient accumulation.

### Dense retrieval comparison (22,855 test queries)

| Metric | Baseline FashionCLIP | Fine-tuned FashionCLIP | Delta |
|--------|---------------------|----------------------|-------|
| nDCG@10 | 0.0229 | 0.0444 | +94.2% |
| MRR | 0.0208 | 0.0405 | +94.7% |
| Recall@10 | 0.0433 | 0.0811 | +87.3% |

Nearly doubled across the board. This is the largest single-component improvement in the project. The model learns to avoid its previous mistakes because the training data is literally built from its mistakes.

---

## Putting it together: retriever x reranker (2x3 factorial)

We had two retriever variants (baseline FashionCLIP, fine-tuned FashionCLIP) and three reranker variants (no reranker, off-shelf CE, LLM-trained CE). Six combinations, all evaluated on the same 22,855 held-out test queries with the same BM25-NER hybrid setup.

| | No reranker | Off-shelf CE | LLM-trained CE |
|---|------------|-------------|----------------|
| Baseline FashionCLIP | 0.0422 | 0.0646 | 0.0747 |
| Fine-tuned FashionCLIP | 0.0515 (+22%) | 0.0650 (+0.6%) | 0.0757 (+1.3%) |

A few things stand out.

Without any reranker, the fine-tuned retriever is 22% better. That's a large gap. But once the off-shelf cross-encoder is on top, the retriever improvement compresses to almost nothing (+0.6% nDCG). The CE equalizes retriever quality on nDCG because it re-scores everything anyway.

But nDCG isn't the whole story. The fine-tuned retriever with off-shelf CE improved MRR by 7.7% and Recall@10 by 6.2% compared to baseline with off-shelf CE. Better retrieval surfaces better candidates. The reranker can't fully translate that into nDCG because it already compensates for retriever weaknesses, but it shows up in coverage metrics.

The best combination (B2): fine-tuned retriever + LLM-trained CE = nDCG@10 of 0.0757, MRR of 0.0799. That's +152% over where we started in Phase 1.

---

## The full progression

```
Phase 1: FashionCLIP dense               0.0300
Phase 2: + hybrid + CE rerank            0.0543  (+81%)
Phase 2: + ColBERT-CE cascade            0.0553  (+84%)
Phase 3B: + LLM-trained CE               0.0747  (+149%)
Phase 3: + fine-tuned retriever (B2)     0.0757  (+152%)
```

Total cost: ~$3 for LLM labels. $0 compute (Apple Silicon).

---

## What we took away from Phase 3

The biggest lesson is boring but important: data quality was the bottleneck, not model architecture or pipeline complexity. We spent Phase 2 carefully engineering hybrid fusion weights, testing ColBERT cascades, building NER extractors. Those things helped. Then we spent $3 on LLM labels and it outperformed all of it.

Fine-tuning on purchase data barely moved the needle (+1.2%). Fine-tuning on LLM-judged data moved it significantly (+15.7%). Same model, same training loop. The only difference was which labels went in. If you're training a search ranker on implicit feedback (clicks, purchases, add-to-carts), consider whether the noise in those labels is limiting your model more than the model itself.

Training the retriever on its own mistakes works remarkably well. The hard-negative mining approach (retrieve, have an LLM score the results, train on the failures) is simple and doesn't require any special infrastructure. You need a retriever, an LLM with API access, and a contrastive training loop. The 94% improvement in dense retrieval suggests most off-the-shelf embedding models have a lot of headroom on specific catalogs, and the cheapest way to unlock it is to show them where they're wrong.

The gains are sub-additive on nDCG but additive on Recall. A better retriever and a better reranker don't multiply on nDCG because the reranker already compensates for retriever mistakes. But Recall improves because the retriever surfaces candidates the old retriever missed entirely. If your application cares about coverage (e-commerce usually does), the retriever improvement matters even when nDCG barely moves.

---

## About the cost

| Item | Cost |
|------|------|
| LLM labels for CE training (42.8K pairs) | ~$2 |
| LLM labels for bi-encoder hard negatives (100K pairs) | ~$1 |
| GPU compute | $0 (Apple M4 Max) |
| Total | ~$3 |

We could have rented a cloud GPU and finished faster. We chose not to, partly to test whether Apple Silicon could handle it (it can), and partly because the total training time was about 8 hours overnight. Not everything needs an A100.

---

## What's next

Phase 4 adds multimodal image search. Phase 5 publishes the full benchmark as a preprint. All code, trained models, LLM labels, and evaluation scripts are open source.

---

*MODA is built by [The FI Company](https://thefi.company) which is a project within [Hopit.ai](https://hopit.ai). Code and results: [github.com/hopit-ai/moda](https://github.com/hopit-ai/moda). MIT License.*
