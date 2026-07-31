# Where we started, and where we ended up

*Eight posts ago we had an empty benchmark and a laptop. We finish the series having beaten FashionSigLIP, the strongest open fashion model we know of, on both text-to-image and image-to-image search, without losing a single benchmark on either.*

This is the last post in the MODA series. Before the final result, here is the short version of how we got here, with links if you want the full story on any step.

## How we got here

We started by building the benchmark itself, on 253,685 purchase-grounded H&M queries over 105,542 products, and by reproducing Marqo's published FashionSigLIP numbers before we reported anything of our own. That step mattered. If we could not match a known baseline, no later comparison would mean much.

Then we built the search pipeline one component at a time.

| Post | What we changed | Result (H&M text, nDCG@10) |
|---|---|---|
| [Blog 1](blog_post.md) | Zero-shot pipeline: dense retrieval, hybrid fusion, cross-encoder rerank | 0.0543 |
| [Blog 2](blog_post_phase2b.md) | Replaced BM25 with SPLADE, a one-line config change | 0.0748 (+38%) |
| [Blog 3](blog_post_phase3a_3b.md) | Trained the cross-encoder on $25 of LLM-graded labels | 0.0976 |
| [Blog 4](blog_post_phase3c.md) | Fine-tuned the retriever on its own mistakes | 0.1063, our best H&M number |
| [Blog 5](blog_post_phase4.md) | A multimodal three-tower model | 0.0833, below the text-only pipeline |

A few of those still surprise people. Dense retrieval beat BM25 on real fashion queries, which is the opposite of what most e-commerce benchmarks show. Synonym expansion made search worse. Twenty-five dollars of graded labels beat 1.5 million purchase records. And Blog 5 is a loss we published on purpose: the multimodal model scored below the simpler text pipeline, and understanding why it failed was worth more than a win would have been.

Blogs 6 and 7 switched tasks, from text search to image-to-image retrieval on LookBench. We fine-tuned FashionSigLIP's vision encoder on cross-domain pairs and beat the published baseline, moving Fine Recall@1 from 63.84 to 67.68. Then we distilled that into a single model and compressed each vector down to 32 bytes, 96 times smaller than the original, with no measurable quality loss.

That left one open question, the one we promised to answer at the end of Blog 7.

## The last piece: text-to-image, at the same size

We had beaten FashionSigLIP on image-to-image. We had not beaten it on the task it was built for, which is text-to-image retrieval on the standard fashion benchmarks. And we wanted to do it without making the model bigger, because a larger model beating a smaller one is not an interesting result.

So we did not train a new model. We wrapped the frozen FashionSigLIP checkpoint in a multi-view encoding and late-fusion recipe that adds zero parameters. Same 203 million weights, same 768 dimensions. The recipe encodes each query and image from a few complementary views and combines the scores. All of the extra work happens at inference time.

Here is the result: full-corpus MAP@10 on six public fashion benchmarks, with significance from a paired bootstrap over 10,000 resamples.

| Benchmark | FashionSigLIP | MODA | Change |
|---|---|---|---|
| KAGL | 0.2769 | 0.2907 | +5.0%, win |
| Fashion200K | 0.1858 | 0.1951 | +5.0%, win |
| DeepFashion In-Shop | 0.1587 | 0.1637 | +3.2%, win |
| Polyvore | 0.3665 | 0.3719 | +1.5%, win |
| Atlas | 0.1826 | 0.1864 | +2.0%, within noise |
| DeepFashion Multimodal | 0.0148 | 0.0150 | +1.9%, within noise |

We improve on all six. Four clear statistical significance, so we count those as wins. The other two, Atlas and DeepFashion Multimodal, come out ahead on the point estimate but sit inside the confidence interval, so we do not claim them. What we can say plainly is that on this suite the recipe never loses.

The honest note is the same one that runs through the whole series. The win did not come from a bigger or cleverer model. It came from using the model we already had more carefully, at inference time, for free. That is the cheapest trick that worked, and by the eighth post it is a pattern rather than a surprise.

## A closer look at the last piece

FashionSigLIP was trained once and frozen, and we did not change it. What we changed was how we use it. A product photographed on a model, cropped tightly, or padded to a square does not always land in the same place in the embedding space. Encoding each image a few ways and averaging gives a steadier vector than any single view. We do the same on the query side, mixing the raw text with a short "a fashion product photo of" prompt. The final score keeps that steady combined vector as the main signal, and lets a single sharper view nudge the ranking when a crop or a pad happens to show the product more clearly.

We also tried the harder version. A small learned adapter on top of three encoders scored higher, five of six benchmarks instead of four. But it needed roughly three and a half times the parameters, which broke the rule we had set for ourselves: beat FashionSigLIP at FashionSigLIP's size. So we did not ship it and we do not count it, because a bigger model beating a smaller one tells you nothing.

Two honest notes. The blend weight that combines the views was chosen on separate development data, not on the six benchmarks. And we had already seen the aggregate benchmark numbers several times before this run, so it is not a blind evaluation. There is a sealed test set we have not touched. Running the frozen recipe through it once is the next thing we would do to make the result harder to argue with.

## Where that leaves us

Across the two tasks that matter for fashion retrieval, we now beat FashionSigLIP on both.

| Task | FashionSigLIP | MODA |
|---|---|---|
| Image-to-image (LookBench, Fine R@1) | 63.84 | 67.68, and 67.42 as a single 256-dim model |
| Text-to-image (six benchmarks, MAP@10) | baseline | wins 4, loses 0 |

We do not lose a benchmark on either side. As far as we know, that makes MODA the strongest openly available fashion retrieval you can run today, on both text and image queries, at FashionSigLIP's size and license.

## What the series taught us

Eight posts, and a few lessons stand out.

- Dense retrieval beat keyword search on real fashion queries, and the cross-encoder reranker, not any clever retrieval trick, was the single biggest gain ([Blog 1](blog_post.md)). Measuring each component on its own is what told us where to spend effort.
- A million and a half purchase labels barely moved the cross-encoder, and a few thousand LLM-graded ones did ([Blog 3](blog_post_phase3a_3b.md)). Label quality was the budget, not label count or model size.
- The multimodal model scored below the plain text pipeline, and we published it anyway ([Blog 5](blog_post_phase4.md)). Understanding why a method fails was worth more than another point of gain.
- Before claiming we beat FashionSigLIP, we reproduced its published number, and the win came from the training data rather than a new architecture ([Blog 6](blog_post_phase5.md)). The right data mattered more than the right model.

The thread running through all of them: almost every gain came from the cheapest option that fit the constraint, not the most sophisticated one.

## Everything is open

The full pipeline, the training recipes, the evaluation harness, and every result table are in the public repository under an MIT license. The packaged text-to-image recipe and an interactive demo go up on our Hugging Face page this week. Everything in the series ran on a MacBook with no GPU, and the total spend on LLM labels across the whole series was about $40.

## What we are doing next

The retrieval series ends here. The next one has already started, quietly, on a different problem: predicting fashion trends before they peak, rather than searching what already exists. It is a harder question than retrieval, and we will write it up the same way, in the open, with the failures left in.

MODA is built by The FI Company, a project within Hopit AI. Code and models are at github.com/hopit-ai/Moda. MIT License.
