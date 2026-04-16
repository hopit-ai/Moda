# The one swap that beat weeks of tuning

*Blog 2 of the MODA series. We replaced BM25 with SPLADE and got a 38% lift on fashion search. No training, no retraining, no fine-tuning. Just a different retriever.*

---

If you read [Blog 1](blog_post.md), you know where we ended. Our full zero-shot pipeline on 253,685 H&M queries landed at nDCG@10 = 0.0543. BM25 for keywords, FashionCLIP for semantics, reciprocal rank fusion to combine the two, cross-encoder on top for reranking. No model was trained. Everything ran on a MacBook.

That was the story we shipped. Then we looked at the component breakdown and noticed something uncomfortable.

BM25 was pulling its weight in the pipeline but it was also, plainly, the weakest link. On its own it hit 0.0186. Dense retrieval alone hit 0.0265, about 42% better. The hybrid only worked because the two retrievers surface different documents, so even a weak BM25 contributed variety the dense encoder missed.

We kept coming back to one question. What if we replace BM25 with something smarter, but still kept the whole pipeline zero-shot?

This post is about that swap. It turned into the single biggest gain we got without training a thing.

---

## Fast primer: BM25, dense retrieval, and why fashion is weird

If you are comfortable with retrieval, skip this section. If not, here is the shortest useful version.

**BM25** is a keyword scoring function from 1994. It looks at which words in your query appear in each document, weights rare words more heavily (the famous IDF term), and normalizes by document length. It knows nothing about meaning. "Hoodie" and "sweatshirt" are different tokens to BM25, even though any human would call them close enough.

**Dense retrieval** embeds the query and every document into the same vector space using a neural network. Similarity is a cosine between vectors. It captures meaning. "Hoodie" and "sweatshirt" end up close because the model has seen them used interchangeably in training text.

You would expect dense to crush BM25 everywhere, and on most tasks it does. Fashion has a wrinkle. H&M product titles look like this:

```
Ben zip hoodie
Ella summer dress
Max slim chino
```

These are brand-style identifiers. Human first name plus two or three attribute words. Real shoppers do not type "Ben zip hoodie." They type "black zip hoodie" or just "zip hoodie." So the query and the document share some words but not the discriminative ones. BM25 scores them low because the overlap is weak. Dense models score them reasonably because they see past the name token and match on the attribute soup.

That is why in Blog 1 we found dense beating BM25 on H&M. It contradicts the general e-commerce benchmarks where BM25 is competitive. Fashion product text is adversarial to keyword matching in a way that most catalogs are not.

The gap was instructive. It told us the lexical retriever had room. BM25 is roughly the worst thing you can do to a title like "Ben zip hoodie." Anything that knows which tokens carry signal should do better.

---

## Enter SPLADE

[SPLADE](https://arxiv.org/abs/2107.05720) stands for Sparse Lexical and Expansion model. It was published in 2021 and it sits in a strange place architecturally. It uses a BERT-style transformer to produce sparse vectors over the BERT vocabulary, one weight per word piece. You can store and search these vectors with the same inverted index infrastructure you use for BM25. But unlike BM25, the weights are learned, and unlike dense retrieval, every dimension has a real word attached to it.

The useful mental model: SPLADE does learned query expansion and learned term weighting at the same time, then scores like BM25 does.

When you encode the query "zip hoodie" with SPLADE, the model assigns weights not only to "zip" and "hoodie" but also to "sweatshirt", "pullover", "jacket", "cotton", and a long tail of related terms. Some weights are high, most are near zero. The document side does the same thing in reverse: "Ben zip hoodie" expands to include "sweatshirt", "pullover", "casual", and so on.

The scoring step is a dot product between those sparse vectors. It looks like BM25 arithmetic but runs on learned weights.

```
Query: "zip hoodie"
     |
     v
SPLADE encoder (MLM-style BERT)
     |
     v
Sparse vector (vocab size ~30K, non-zeros ~50-200)
{ "zip": 1.4, "hoodie": 2.1, "sweatshirt": 1.2,
  "pullover": 0.9, "jacket": 0.4, "cotton": 0.3, ... }
     |
     v
Inverted index lookup (same backend as BM25)
     |
     v
Dot product with stored document vectors
     |
     v
Ranked results
```

This matters for two reasons we care about in fashion.

First, SPLADE never collapses like manual synonym expansion does. In Blog 1 we reported that adding a hand-curated synonym list to BM25 hurt nDCG by 35%. The expansion flooded the index with false matches, and because IDF is computed on the raw vocabulary, rare terms got diluted. SPLADE avoids this because the expansion weights and the index weights come from the same model. There is no IDF collapse because there is no IDF in the human sense. Everything is learned end-to-end.

Second, SPLADE's expansion is document-aware. For "zip hoodie" the expansion favors clothing-adjacent terms. For "bluetooth speaker" it favors electronics terms. We do not need to curate a fashion synonym list. The model brings its own.

We chose [`naver/splade-cocondenser-ensembledistil`](https://huggingface.co/naver/splade-cocondenser-ensembledistil) as the checkpoint. It was trained on MS MARCO, a passage retrieval task with nothing to do with fashion. Off the shelf. No fine-tuning. That was deliberate. We wanted to know what learned sparse would give us before we did anything else.

---

## Dropping SPLADE in: the first number

We kept the same 253,685-query H&M benchmark from Blog 1. Same products, same queries, same relevance labels. Only the lexical retriever changed. BM25 out, SPLADE in.

| Retriever | nDCG@10 | MRR | Recall@10 | 95% CI |
|---|---|---|---|---|
| BM25 | 0.0186 | 0.0227 | 0.0059 | [.0183, .0190] |
| **SPLADE** | **0.0412** | **0.0695** | **0.0189** | **[.0406, .0417]** |

Gain over BM25 standalone: +121% nDCG@10, +206% MRR, +220% Recall@10.

The MRR jump was the one that actually surprised us. MRR (reciprocal rank of the first correct answer) is dominated by what happens at positions 1 through 3. nDCG@10 smears its weight across the top 10 with a log discount. When MRR moves almost twice as much as nDCG, it means SPLADE is not only finding the right product more often, but when it finds it, it is ranking it close to the top. Recall@10 moving in step with MRR (+220%) confirms this: the right product is showing up, and showing up high.

The intuition here: BM25's scoring is basically "do the query tokens appear in this document." For "zip hoodie" against "Ben zip hoodie," two out of three tokens overlap. Same as for "Ben zip cardigan" or "Ben zip sweater." BM25 cannot separate them. SPLADE's learned weights distinguish the discriminative tokens from the filler. "Hoodie" carries more weight than "zip" for this query, so documents that match "hoodie" rise above documents that only match "zip." That is a top-of-list effect more than a middle-of-list effect, which is why MRR moves more than nDCG.

For context, dense retrieval on its own at 253K was 0.0265. SPLADE alone beats dense alone on fashion. That was the second surprise. Sparse retrieval, done right, is not obsolete even on a task where semantics clearly matter.

---

## The hybrid sweep: where does the new sweet spot live?

In Blog 1 we showed that hybrid (BM25 + Dense) beats either alone because the two retrievers make different mistakes. If we swap BM25 for SPLADE, does hybrid still help, and at what weight?

Reciprocal rank fusion takes two ranked lists and combines them by weight. We swept the SPLADE weight from 0.2 to 0.6 in steps, with Dense getting one minus that.

On the full 253K:

| SPLADE weight | Dense weight | nDCG@10 | MRR | Recall@10 |
|---|---|---|---|---|
| 1.0 | 0.0 | 0.0412 | 0.0695 | 0.0189 |
| 0.3 | 0.7 | 0.0386 | 0.0542 | 0.0168 |
| 0.4 | 0.6 | 0.0411 | 0.0581 | 0.0178 |
| **0.5** | **0.5** | **0.0472** | **0.0662** | **0.0201** |
| 0.0 | 1.0 | 0.0265 | 0.0369 | 0.0106 |

The 50/50 split won on all three metrics. This is a different world from the BM25 era. With BM25, the best hybrid was 0.4/0.6 (dense-heavy) at nDCG = 0.0328. BM25 was weak enough that leaning on dense helped. SPLADE is strong enough on its own that an equal mix does better.

The intuition behind the weight shift: hybrid fusion helps when the two retrievers make *different* mistakes. Recall is where that diversity pays off. A document that dense missed but sparse found (and vice versa) gets a boost from the fusion. With BM25, the lexical retriever was so weak that its "unique catches" were mostly noise, so the sweet spot leaned toward dense. With SPLADE, the lexical retriever has real semantic coverage of its own, and the two retrievers disagree in useful ways more often. Equal weighting extracts more signal from that disagreement.

One thing worth noting: the 0.3/0.7 and 0.4/0.6 splits are *worse* than either retriever alone on this dataset. Unbalanced fusion can hurt when the two retrievers have similar accuracy but are being combined with a weight that does not reflect that.

We cross-checked on our 22,855-query held-out split (the one we use for fine-tuning experiments in Blogs 3 and 4). Five weights this time:

| SPLADE | Dense | nDCG@10 |
|---|---|---|
| 0.2 | 0.8 | 0.0432 |
| 0.3 | 0.7 | 0.0501 |
| 0.4 | 0.6 | 0.0527 |
| **0.5** | **0.5** | **0.0556** |
| 1.0 | 0.0 | 0.0464 |

Same winner, different numbers. The absolute values differ because the query distribution on the 22K split is not identical to the full 253K, but the shape of the weight curve is the same.

**Decision we locked in:** SPLADE(0.5) + Dense(0.5) RRF is the new default retrieval stage. All downstream experiments (this blog and the next two) use it unless stated otherwise.

---

## Adding the reranker back: the full zero-shot pipeline

In Blog 1 the cross-encoder gave us the biggest single jump of any component. A 22M-parameter MiniLM-L6 reranker, off the shelf, pulled the hybrid from 0.0328 to 0.0543. A +66% lift.

We kept the same reranker. Same weights. Same 100-candidate top-k from the hybrid. The only change was that the 100 candidates now came from SPLADE + Dense instead of BM25 + Dense.

On 22K:

| Config | nDCG@10 | MRR | Recall@10 |
|---|---|---|---|
| Blog 1 final (BM25+Dense+CE) | 0.0543 | 0.0569 | 0.0164 |
| SPLADE + off-shelf CE | 0.0740 | 0.0722 | 0.0211 |
| **SPLADE-BestHybrid + off-shelf CE** | **0.0748** | **0.0738** | **0.0215** |

+38% nDCG, +30% MRR, +31% Recall@10 over the Blog 1 number. Still zero training. Same cross-encoder. The only thing that changed was which 100 candidates the reranker saw.

The intuition for why the CE magnifies the retriever upgrade: the cross-encoder re-scores the top-100 candidates. If the purchased product is in those 100, the CE usually finds it. If it is not, no rerank can help. The retriever swap did three things at once: pushed more correct products into the 100, pushed them in at higher starting ranks, and reduced the number of near-misses the CE had to disambiguate. Recall and MRR moving in step confirms that the gain is mostly about *what enters the pool*, not about better reranking of the same pool.

This is the headline of the post. A pure swap at the retrieval stage propagates through the pipeline and lifts the final number by more than a third. The cross-encoder was not getting smarter. It was getting better raw material.

Here is the full pipeline diagram for the post-SPLADE era:

```
Query: "black zip hoodie"
  |
  +---> SPLADE encoder ---> sparse vec ---> inverted index ---> top-1000
  |                                                                |
  +---> FashionCLIP -----> dense vec ----> FAISS --------------> top-1000
  |                                                                |
  |                                                                v
  |                              Reciprocal Rank Fusion (0.5/0.5)
  |                                                                |
  |                                                                v
  |                                                          top-100
  |                                                                |
  |                                                                v
  |                                  Cross-Encoder (MiniLM-L6)
  |                                       scores each (query, doc)
  |                                                                |
  |                                                                v
  +----------------------------------------------------------> top-50
```

Latency on a MacBook for this pipeline:

| Stage | Mean | p50 | p95 |
|---|---|---|---|
| SPLADE encode + retrieve | ~28ms | ~24ms | ~45ms |
| Dense retrieve (FAISS, pre-computed docs) | <1ms | <1ms | <1ms |
| RRF fusion | 0.1ms | 0.1ms | 0.2ms |
| CE rerank (100 candidates) | 50.9ms | 47.7ms | 73.3ms |
| **End-to-end** | **~80ms** | **~73ms** | **~120ms** |

SPLADE costs more than BM25 at query time because the query goes through a transformer. Document vectors are pre-computed once during indexing. If you care about the last 20ms, there are tricks (quantization, smaller models, caching high-frequency queries), but for anything that is not a fast-path search bar, the full pipeline fits in a single API round-trip.

---

## NER on SPLADE does nothing

In Blog 1 we reported that NER attribute boosting on BM25 helped by about +14%. We pull "black", "hoodie", "zip" out of the query with a zero-shot NER model ([GLiNER2](https://github.com/fastino-ai/GLiNER2)), and we use those attributes to boost documents whose matching fields contain them. "black" hits `colour_group_name`, "hoodie" hits `product_type_name`, and so on.

So we asked: does the same NER trick still help when the lexical retriever is SPLADE instead of BM25?

We tested five configurations on the 22K split. For each SPLADE+Dense weight, we compared with and without NER query expansion (where the extracted attributes are added as extra query terms before SPLADE encoding).

| Config | With NER | Without NER | Delta |
|---|---|---|---|
| SPLADE only | 0.0470 | 0.0464 | +0.0006 |
| SPLADE(0.2)+Dense(0.8) | 0.0428 | 0.0432 | −0.0004 |
| SPLADE(0.3)+Dense(0.7) | 0.0493 | 0.0501 | −0.0008 |
| SPLADE(0.4)+Dense(0.6) | 0.0518 | 0.0527 | −0.0009 |
| SPLADE(0.5)+Dense(0.5) | 0.0552 | 0.0556 | −0.0004 |

Four out of five deltas are negative. The remaining one is well within noise. NER on SPLADE does not help. On hybrid configurations it mildly hurts.

Why?

SPLADE's MLM head already does learned expansion. When you encode "black zip hoodie" with SPLADE, the model adds weight to "hoodie" adjacent terms automatically. Appending the NER-extracted attributes to the query re-emphasizes tokens the model has already emphasized, and the extra tokens dilute the query's mass across redundant dimensions. The expansion is already happening inside SPLADE. Doing it again outside adds noise.

This is a clean empirical result and we are recording it explicitly. In the BM25 era, NER was a useful post-processing step because BM25 is blind to meaning. In the SPLADE era, the retriever has eaten the NER trick already.

We still run NER on the query path for other reasons (faceted search, UI filters, analytics), but we no longer use it as a retrieval-time booster.

---

## One engineering fix worth naming

One thing landed between Blog 1 and this blog that does not move the headline number much but is worth flagging because it will matter a lot when we start training things in Blog 3.

During Blog 1 we discovered that the text we indexed in OpenSearch, the text we embedded with FashionCLIP, and the text we sent to the cross-encoder were all slightly different. Different field sets, different truncation lengths, a stray `garment_group_name` in one place and not another. Individually, each difference was minor. Collectively, they meant the reranker was scoring products against a description slightly different from the one the retriever saw.

We fixed it by writing `article_text.py`, a module with one function. Every script that touches product text now imports from it. Before and after on the Blog 1 full pipeline: 0.0538 → 0.0543, a small but real gain and (more importantly) now train-eval consistent. This matters even more once we start fine-tuning, because any train-eval text drift turns into silent label noise at training time.

(If you read Blog 1 closely you already know about GLiNER2. Our NER stage uses `fastino/gliner2-base-v1`, and the 0.0204 BM25+NER number in Blog 1 already reflects the GLiNER2 upgrade. Not repeated here.)

---

## Where this leaves us

The zero-shot ceiling is higher than we thought when we shipped Blog 1. A single component swap (BM25 to SPLADE), keeping everything else fixed, took the pipeline from 0.0543 to 0.0748. That is a 38% lift with zero training, zero labeling, zero infrastructure changes. The only cost is about 25ms of additional query latency and an extra model weight file.

```
Phase 1 dense only:                   0.0300
Phase 2 BM25 + Dense + CE (Blog 1):   0.0543   (+81% over Phase 1)
Phase 2B SPLADE + Dense + CE (this):  0.0748   (+149% over Phase 1)
```

The interesting question now is what happens when we actually start training things. The cross-encoder in this pipeline is still `ms-marco-MiniLM-L-6-v2`, trained on web search passages from 2019. It has never seen a fashion product description. The dense encoder is FashionCLIP, which has seen fashion but was trained on captions from a different corpus than H&M.

Both components are doing reasonably well on our task despite no domain adaptation. That is either very impressive or suggests there is headroom we are leaving on the table.

We spent $25 to find out. Blog 3 is about what happened.

---

*MODA is built by [The FI Company](https://thefi.company), a project within [Hopit AI](https://hopit.ai). Code, trained models, and the label sets referenced across this series are at [github.com/hopit-ai/Moda](https://github.com/hopit-ai/Moda). MIT License.*
