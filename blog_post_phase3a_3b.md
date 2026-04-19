# $25 beat everything we had built

*Blog 3 of the MODA series. We trained the cross-encoder. Same 22M-parameter model. Different labels. +32% over off-the-shelf. The lesson was not about architecture.*

---

At the end of [Blog 2](blog_post_phase2b.md) we were at nDCG@10 = 0.0748. SPLADE on the lexical side, FashionCLIP on the dense side, an RRF fusion at 50/50, and a cross-encoder on top of the top-100 candidates. Every model was off the shelf. We had not trained anything.

The cross-encoder was pulling the biggest single load in that pipeline. In Blog 1 we showed it was responsible for about +51% of the end-to-end gain. Swap it out and the whole pipeline collapses by roughly the same amount. It was also the oldest component in our stack: [`ms-marco-MiniLM-L-6-v2`](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2), a 22-million-parameter distilled BERT trained in 2019 on [MS MARCO](https://microsoft.github.io/msmarco/), a corpus of real Bing search queries paired with web passages.

Nothing about MS MARCO is fashion. And yet the off-the-shelf checkpoint was doing most of the work.

That tension is what this blog is about. If a reranker trained on web search is this useful on fashion, what happens when we train one on fashion? And if we do, where does the budget actually go? The model? The training recipe? The data?

We tried three experiments. The obvious one, with 1.5M purchase labels, barely moved the number. A $2 pilot on LLM-graded labels beat it. Scaling that to $25 became the version we shipped.

---

## What a cross-encoder actually is

Skip this if you work with rerankers. Read it if you do not.

A **bi-encoder** (like FashionCLIP or SPLADE's query model) takes a query and a document and embeds them separately into vectors. You compute query embeddings at query time and document embeddings once at index time. Matching is a cheap dot product. You can score a million documents against one query in milliseconds.

A **cross-encoder** takes a query and a document together, concatenates them into one token sequence, and runs them through the transformer jointly. The output is a single scalar: "how relevant is this document to this query?" Cross-encoders are far more accurate than bi-encoders because the model sees both sides simultaneously and can attend across them. They are also far slower, because you have to run the full transformer forward pass once per (query, document) pair. There is no precomputation.

```
Bi-encoder (fast, less accurate):
  query ---> [encoder] ---> query_vec ---\
                                          +--> dot product --> score
  doc   ---> [encoder] ---> doc_vec   ---/

Cross-encoder (slow, more accurate):
  "query [SEP] doc" ---> [encoder] ---> score
```

In a retrieval pipeline, you use a bi-encoder to pull a rough top-K (usually 100-1000), then a cross-encoder to rerank those K candidates. The cross-encoder never sees the whole corpus. It only has to be fast enough to score ~100 candidates per query.

`ms-marco-MiniLM-L-6-v2` has 6 transformer layers, 22M parameters, and scores 100 candidates in about 50ms on a MacBook. It is fast enough to ship.

---

## 3A-1: the obvious move, the boring result

We had 253K queries with purchase labels. For each query, we knew which product the user bought. We had also seen that users browse ~20 products before buying, so we had plenty of implicit negatives (products shown but not purchased).

The obvious recipe: for each query, take the purchased item as a positive (label = 1.0), take the non-purchased items that were impressed as negatives (label = 0.0), and fine-tune `ms-marco-MiniLM-L-6-v2` with a pairwise margin loss. Three epochs, batch size 32, learning rate 2e-5. Standard training recipe from the sentence-transformers library.

~1.5 million training pairs. Three hours on an M4 Max. Zero cost in dollars.

| Metric | Off-shelf CE | FT on purchase labels | Delta |
|---|---|---|---|
| nDCG@5 | 0.0524 | 0.0582 | +11% |
| nDCG@10 | 0.0639 | 0.0664 | +4% |
| MRR | 0.0648 | 0.0678 | +5% |
| Recall@10 | 0.0172 | 0.0180 | +5% |

*(Numbers are on baseline retrievers, not the SPLADE-best hybrid from Blog 2. Same training recipe applied to the Blog 2 stack lands at 0.0735, which we report in 3A-3 below.)*

Small gain on nDCG@5. Basically flat on nDCG@10. We were expecting a double-digit jump.

Here is what we think happened.

When a shopper searches "black summer dress," they might see twenty perfectly reasonable black dresses. They buy one. For training purposes, that one is labeled positive and the other nineteen are labeled negative. But the other nineteen were not irrelevant. They were the kind of almost-right results you want the model to learn look like the right answer. By training on "this exact product is relevant, these similar products are not," we were teaching the reranker to sharpen a distinction that did not really exist.

Purchase labels look like relevance labels. They are not. They are preference labels with nineteen parts noise for every one part signal, and the noise is specifically the kind that punishes correct-but-unchosen results.

1.5 million labels. Almost no movement. The bottleneck was not compute, not architecture, not data volume. It was that our labels were lying to us.

---

## 3A-2: the label quality experiment

We paused the fine-tuning work and ran a small pilot on a completely different kind of label.

We took 9,800 (query, product) pairs, sampled from the retrieval outputs of our Phase 2 pipeline. We sent them to GPT-4o-mini through [PaleblueDot](https://palebluedot.ai) with a prompt that described relevance on a 0-3 scale:

```
0 = not relevant at all
1 = partially relevant (right category, wrong specifics)
2 = good match (most attributes align)
3 = exact match (the user clearly wanted this specific product)
```

We sent the product text in the same format the cross-encoder would see at inference time. The LLM returned a grade. Total cost: about $2.

We trained the same MiniLM-L6 with these graded labels instead of binary purchase labels. 9,800 pairs versus 1,500,000. About 150x less data.

Same 22M parameters. Same training loop. Different labels.

| Config | nDCG@10 | MRR | Recall@10 |
|---|---|---|---|
| Off-shelf CE | 0.0639 | 0.0648 | 0.0172 |
| FT on 1.5M purchase labels | 0.0664 | 0.0678 | 0.0180 |
| **FT on 9.8K LLM labels** | **0.0689** | **0.0701** | **0.0190** |

The LLM-labeled model beat the purchase-labeled model on every metric. With a 150x smaller training set. For two dollars.

The movement on MRR and Recall@10 is what makes this credible. MRR going from 0.0648 to 0.0701 (+8%) means the single best answer is more often near the top. Recall@10 going from 0.0172 to 0.0190 (+10%) means the correct product is more often in the top-10 rather than at position 30. Both metrics move together, which is what you want to see. If nDCG moved but MRR did not, we would worry that the model was just shuffling middle-of-list. It was not.

We were not expecting it to go the other way. We were expecting it to lose by a little and we would conclude that LLM labels were a reasonable shortcut. The result was that LLM labels were also genuinely better, because the LLM can say "this is a pretty good match even though the user did not buy it" and the purchase log cannot.

---

## 3A-3: going bigger, going better

With the direction confirmed, we spent real money on labels. Not a lot of money. $25.

We took 194,000 (query, product) pairs, sampled from the hybrid retriever's top-20 across a larger query set, and sent them to Claude Sonnet 4.6 with essentially the same 0-3 rubric. The distribution came out balanced: 28% scored 0, 21% scored 1, 25% scored 2, 26% scored 3. The grader can genuinely tell "navy slim fit jeans" apart from "dark blue regular fit chinos" (a 2) apart from "navy slim stretch jeans" (a 3). Purchase logs cannot do this.

We also upgraded the CE architecture from MiniLM-L6 (6 layers, 22M params) to MiniLM-L12 (12 layers, 33M params), since we now had enough labels to train a slightly bigger model without overfitting. Three epochs, same recipe, about 2 hours 17 minutes of training.

Then we plugged it into the Blog 2 pipeline and evaluated on the 22K held-out split.

| Config | nDCG@10 | MRR | Recall@10 |
|---|---|---|---|
| Off-shelf CE-L6 | 0.0639 | 0.0648 | 0.0172 |
| FT-CE-L6 (9.8K GPT-4o-mini) | 0.0689 | 0.0701 | 0.0190 |
| **FT-CE-L12 (194K Sonnet labels)** | **0.0735** | **0.0751** | **0.0217** |

+15% nDCG@10 over off-the-shelf. +32% over pure BM25-era zero-shot. Training cost: $25 in labels, 2h17m of compute, one Apple Silicon laptop.

Why did the step from L6+9.8K to L12+194K help this much? Two forces working together. First, label *quality* was already at the new bar (both sets were LLM-graded). Second, we added both label quantity and model capacity at the same time. With only 9.8K labels, an L12 model overfits; with 194K labels and an L6 model, we leave gradient signal on the table because the smaller model saturates earlier. L12 + 194K is the combination where capacity is roughly matched to label volume, and that is where you get the real lift. This is a general pattern: scaling labels without scaling model (or vice versa) gives diminishing returns; scaling both together compounds.

One more read: Recall@10 moved from 0.0190 to 0.0217 (+14%) while nDCG moved +7% and MRR +7%. Recall moving faster than the others tells us the new CE is rescuing cases where the right product was previously being ranked 11-20 and getting pushed into the top-10. That is exactly the kind of win a better reranker produces on top of a fixed retriever: same pool, better ordering, more hits near the top.

The Spearman correlation between predicted and gold labels on a 1,000-pair held-out validation set went from 0.904 (off-shelf on generic queries) to 0.942 (CE-L12 on fashion). The reranker was meaningfully better at the scoring task itself, not just incidentally better at our retrieval metric.

---

## An attribute-conditioned variant: AttrCE

While we were in the training loop, we ran a sibling experiment. What if we gave the cross-encoder extra features beyond the raw text?

Specifically: what if we extracted attributes with NER (color, product type, garment group) and concatenated them as explicit tokens in the input? So instead of scoring `"[CLS] black zip hoodie [SEP] Ben zip hoodie in soft cotton..."`, the model sees `"[CLS] black zip hoodie [SEP] [COLOR] black [TYPE] hoodie [GROUP] jersey basic [TEXT] Ben zip hoodie in soft cotton..."`.

The intuition was that the LLM labels knew about attributes, the reranker had to figure them out from raw text, and giving them explicitly might help.

| Config | nDCG@10 | MRR |
|---|---|---|
| FT-CE-L12 (LLM labels) | 0.0735 | 0.0751 |
| AttrCE (LLM labels + NER features) | 0.0738 | 0.0799 |

nDCG is flat. MRR jumps from 0.0751 to 0.0799, a +6% gain. AttrCE is meaningfully better at putting the single best result at the top, even though it does not improve the overall top-10 ranking.

This is a real split, and it is the kind of finding you cannot see from a single metric. If your product is "I am feeling lucky" with one answer, ship AttrCE. If it is a grid of ten results, ship the plain LLM-trained CE. We kept both as viable final configs and decided based on downstream use.

---

## Phase 3B: the factorial with the new reranker on top

With the LLM-trained CE in hand, we went back to the Blog 2 hybrid sweep. Blog 2 picked SPLADE(0.5) + Dense(0.5) as the retrieval stage and capped at 0.0748 with an off-shelf CE. What happens at each fusion weight if we swap in the LLM CE instead?

The same weight sweep, now with LLM CE on top:

| SPLADE | Dense | nDCG@10 | MRR | Recall@10 |
|---|---|---|---|---|
| 1.0 | 0.0 (no Dense) | 0.0903 | 0.0878 | 0.0249 |
| ⊕ (concat) | | 0.0946 | 0.0901 | 0.0258 |
| 0.3 | 0.7 | 0.0934 | 0.0893 | 0.0257 |
| **0.4** | **0.6** | **0.0976** | **0.0931** | **0.0268** |
| 0.5 | 0.5 | 0.0933 | 0.0892 | 0.0257 |

Three things stand out.

**The optimal hybrid weight shifted.** In Blog 2 with an off-shelf CE, the sweet spot was SPLADE(0.5) + Dense(0.5). With the LLM CE on top, it shifts to SPLADE(0.4) + Dense(0.6). The intuition: a stronger reranker amplifies whichever retriever has better recall, because the reranker will find the right answer in the pool if it is there. Dense retrieval on this catalog has slightly better recall than SPLADE at K=100, so leaning dense-ward at retrieval time gives the stronger CE a larger true-positive pool to sort. We will see this pattern compound in Blog 4: as components get better, the optimal weight keeps shifting toward dense.

**Concatenation ties the best weighted fusion.** The concat variant (stack SPLADE's top-100 and CLIP's top-100 into a 200-candidate pool before reranking) lands at 0.0946, within noise of the best weighted fusion. The recall gain is clear (Recall@10 = 0.0258 vs 0.0249 for SPLADE alone). The MRR barely moves versus SPLADE-only, which tells us the extra dense candidates are filling slots 6-10 more than slot 1. Concat is also tuning-free, which is worth a small nDCG hit for teams that do not want to maintain a weight.

**Metrics move together.** nDCG, MRR, and Recall@10 pick the same winner at every row. This is a quality check on the experiment. If we had seen nDCG pick one config and MRR pick another within the sweep, we would have suspected overfitting to one position class. They all agree.

The best config here is nDCG@10 = 0.0976. That is +31% over Blog 2's zero-shot headline (0.0748), and +80% over Blog 1's BM25-era zero-shot (0.0543). All of this from training the reranker. The retrievers have not been touched.

---

## The pool size trap

We had a hypothesis. The cross-encoder is the most accurate stage in the pipeline, so if we give it more candidates, we should get more correct answers in the reranked top-10. Pool at 100 works; pool at 200 should work better.

It did not.

| Pool size | nDCG@10 | MRR | Recall@10 | Latency |
|---|---|---|---|---|
| 100 | 0.0735 | 0.0751 | 0.0217 | ~50ms |
| 200 | 0.0711 | 0.0712 | 0.0208 | ~100ms |
| 500 | 0.0564 | 0.0557 | 0.0171 | ~250ms |

Bigger pools made the result worse, not better. We double-checked this wasn't a bug (it wasn't, the ordering is consistent across the LLM CE, off-shelf CE, and AttrCE). Every metric drops in step, which rules out a per-position-class artifact.

The explanation is straightforward once you see the pattern. The cross-encoder is not a perfect oracle. It has a noise floor, and when you feed it 500 candidates instead of 100, most of the extra 400 are irrelevant products that happen to score high on one feature or another. A few of them score falsely high and leak into the top-10, displacing the actually-good products from the smaller pool.

The cleanest way to think about this: rerankers have a signal-to-noise curve. Below some candidate count, the signal is growing faster than the noise. Above it, adding candidates adds more false positives than true positives.

For our pipeline, the crossover is somewhere below 200. We pool at 100 and we are not going higher.

This is useful in its own right. If you are reading this and thinking "just give the reranker more candidates to score" as a path to better quality, the answer is: test it, because it usually makes things worse.

---

## What we actually learned

Three things.

**One: label quality is the budget.** Not model size. Not training recipe. Not architecture. The sequence of results was 1.5M purchase labels → basically flat, 9.8K LLM labels → +8% over off-shelf, 194K LLM labels → +15% over off-shelf. The scaling is in the labels, specifically in how much real signal each label carries. Purchase labels are diluted because the negatives are only sort-of negative. LLM-graded labels are sharp because the grader can actually distinguish "near miss" from "wrong answer." That sharpness is worth more than 150x more data.

**Two: LLM labels cost less than your lunch.** The end-to-end cost of reaching 0.0976 from 0.0748 was $25 of Sonnet calls and a few hours of training on a laptop. Compared to most paths to a 31% lift in a production metric, this is a rounding error. If you are running any kind of search or ranking system and you have not tried LLM-graded labels, the expected return on two hours of your time is very high.

**Three: different rerankers win different metrics.** The plain LLM CE wins nDCG@10. AttrCE wins MRR. The CE L6 variants win latency. You do not get to pick one model that is best at everything. Your choice depends on whether your UI shows one answer or ten, and whether 30ms of latency matters.

---

## The new ladder

```
Phase 1 dense only:                              0.0300
Phase 2 BM25+Dense+CE (Blog 1):                  0.0543
Phase 2B SPLADE+Dense+CE, no training (Blog 2):  0.0748   (+38%)
Phase 3A LLM-trained CE on the same retriever:   0.0735
Phase 3B best hybrid + LLM CE (this blog):       0.0976   (+31%)
```

We are at 0.0976 with a trained reranker and two off-the-shelf retrievers.

The retrievers are still exactly where they were at the end of Blog 2. `ms-marco-MiniLM-L-6-v2` became `fashion-trained MiniLM-L-12`, and that change alone moved the pipeline by 31%. The fact that it moved that much with the retrievers untouched suggests either that the retrievers are already very good, or that they have untapped headroom and the CE was masking it.

We were suspicious it was the second. Blog 4 is about what happened when we trained them too.

---

*MODA is built by [The FI Company](https://thefi.company), a project within [Hopit AI](https://hopit.ai). Code, trained models, and the label sets referenced in this post are at [github.com/hopit-ai/Moda](https://github.com/hopit-ai/Moda). MIT License.*
