# What score requires what candidate recall

Source: `src/recall_requirement.py`, output `work/recall_requirement.json`.
Distribution: 43,790 labelled Source 1 entities, 143,941 true links,
mean 3.287 links, 6.09% singletons (audit2 + audit3 pooled).

## Method

Each true link is retrieved independently with probability `recall`, so an
entity with T links retrieves Binomial(T, recall) of them. Expected macro
F0.5 is summed **exactly** over that binomial's support, with singletons
contributing 1.0 (a perfect model predicts nothing for them).

This is the score of a *perfect classifier* on those candidates. No model can
exceed it, so it is a genuine upper bound, not an estimate of what is easy.

## Results

| candidate recall | best possible macro F0.5 |
|---|---|
| 0.90 | 0.96601 |
| 0.93 | 0.97692 |
| 0.94 | 0.98041 |
| 0.9449 (current) | **0.98212** |
| 0.95 | 0.98384 |
| 0.96 | 0.98720 |
| 0.97 | 0.99050 |
| 0.99 | 0.99689 |

Minimum recall for a target score:

| target | min recall |
|---|---|
| 0.960 | 0.8845 |
| 0.970 | 0.9110 |
| 0.980 | 0.9390 |
| 0.985 | 0.9535 |
| **0.988** | **0.9625** |
| 0.990 | 0.9685 |

## Reading this

**A 0.988 score requires ~0.9625 candidate recall.** At the current 0.9449,
the ceiling is 0.98212. So the teams scoring 0.988 are not better
classifiers than us — they are retrieving roughly **2 more percentage points**
of true links, about 1,500 additional links per 76k.

The remaining headroom splits roughly:

- **~0.021** from recall 0.9449 to 1.0 (blocking/retrieval work), and
- **~0.019** from the oracle down to our current 0.960 (classifier work).

That is why V7's +0.012 is real and worth taking, but it cannot by itself
reach 0.988. Retrieval is the larger of the two levers.

## Caveat on the 0.988 figure

This is the user's reported observation of the leaderboard, not something
verified from the portal. If the actual top scores are lower, the required
recall is correspondingly lower: 0.96 needs only 0.8845, which the current
pipeline already exceeds. Confirm the real leaderboard distribution before
committing days to a retrieval rebuild.

## Note on a discarded method

An earlier version of this script modelled per-entity retrieval as
`floor(T * recall)`, which floors every single-link entity to zero retrieved
links and therefore scored those entities 0 regardless of recall. It produced
a flat, obviously wrong 0.85035 across all recall values. The binomial model
above replaced it. Anyone reusing this must not reintroduce the floor.
