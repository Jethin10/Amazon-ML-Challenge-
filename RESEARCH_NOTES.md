# Entity resolution research notes

The practical next step is to measure *marginal* candidate recall for each
new retrieval route. A route that repeats current candidates gives no recall
gain, while a broad route can add millions of negative pairs. Splink's
[blocking guidance](https://moj-analytical-services.github.io/splink/topic_guides/blocking/blocking_rules.html)
recommends several complementary strict rules and checking block size skew.
Our pilot probes follow that pattern: they count new true links and total new
pairs against the current candidate union.

The [Foursquare Location Matching first-place solution](https://www.kaggle.com/competitions/foursquare-location-matching/writeups/re-waiwai-1st-place-solution)
combined independent retrieval, a fast candidate model, and a more costly
model for selected pairs. Its high private score included a train/test
leakage advantage documented by the authors, so it is not a realistic score
target for this challenge. A second-stage multilingual model could help
ambiguous matches here, but its value must exceed the extra inference cost.

[Ditto](https://github.com/megagonlabs/ditto/blob/master/README.md) is an
example of transformer-based pair matching with hard negative sampling. If
we try such a stage, we should train and evaluate on pairs retrieved by our
actual blocker rather than easy random nonmatches. The RTX 2050 in this
machine has 4 GB VRAM, which constrains batch size and model choice.

The [WDC Products benchmark study](https://arxiv.org/abs/2301.09521)
reports difficulty on unseen entities. Our holdout splits Source 1 IDs while
sharing the Source 2 and 3 search pool. It checks generalization to unseen
Source 1 examples, but cannot measure the France domain shift because the
provided labeled training data contains no France examples. We should treat
the local score and any public leaderboard result as incomplete estimates of
the final private score.

The [multilingual MiniLM cross encoder](https://huggingface.co/cross-encoder/mmarco-mMiniLMv2-L12-H384-v1)
is an Apache 2.0 checkpoint with about 100 million parameters and 15 listed
languages. It is a plausible small reranker for a timed pilot on the 4 GB GPU,
but passage relevance pretraining does not establish entity matching gains.
Before using it, benchmark 10,000 representative pairs and compare macro
F0.5 on a fresh holdout. A full 1–2 million pair rerank could take hours at
this machine's unmeasured throughput. The
[English MiniLM L6 checkpoint](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2)
is a smaller Apache 2.0 speed comparison, but lacks multilingual coverage.
