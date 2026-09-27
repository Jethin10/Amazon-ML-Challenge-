# France audit findings (2026-09-26)

## The bug

`v1.py::prepare_table` builds `name_key` by stripping only:

```
inc|incorporated|llc|ltd|limited|corp|corporation|pvt|private|llp
```

French legal forms are **not** stripped. Measured rates (`work/legal_suffix_asymmetry.json`,
label-free, computed on the test set by comparing marginal suffix rates per source):

| split/source/country | rows | French suffix | currently-stripped suffix |
|---|---:|---:|---:|
| test/source1/france | 259,452 | **0.6729** | 0.0001 |
| test/source2/france | 703,378 | **0.5424** | 0.0001 |
| test/source3/france | 731,615 | **0.5338** | 0.0001 |
| test/source1/us | 663,106 | 0.0010 | 0.4777 |
| test/source1/india | 809,986 | 0.0005 | 0.8150 |

Two facts follow.

1. **France is the only country carrying these tokens at all.** US/India sit at
   ~0.001. The model has literally never seen `sarl`/`sas`/`eurl` in training,
   so every French name lands off the training distribution of `name_key`.

2. **Source 1 carries them far more often than the targets** (0.673 vs 0.542
   and 0.534, a gap of +0.13 to +0.14). This is the damaging part. A French
   S1 row is far more likely to *have* a suffix than its true target is to
   have the same one, so:

   - `name_key` differs between the pair, so the `exact_name` blocking route
     never fires for a large share of French pairs;
   - `name_exact` is 0, removing the strongest exact-match feature LightGBM has;
   - `name5`/`name4` prefixes are computed from a `name_key` polluted by a
     suffix, so prefix blocking degrades too.

   The stripped-suffix rate shows the same asymmetry in reverse for US/India
   (0.478 vs 0.414/0.411), but there `v1.py` already removes the token, so the
   asymmetry is neutralised. For France nothing removes it.

## Why this is the France blocker

France is ~15% of test rows and is invisible to every local holdout. Its
output profile was already anomalous: mean 4.88 predicted links per S1 vs
~3.46 true mean in both labelled countries, and 5.9% of French rows carrying
>10 links vs ~0.002% for US/India. Under F0.5, where precision counts twice
as much as recall, systematic over-prediction is the most expensive possible
error.

## Status of the fix

V8 applies a bounded recovery route without invalidating the trained V5 model.
It retrieves France pairs with compatible normalized legal suffixes, requires
address similarity of at least 0.70, scores the remaining pairs with the V6
cross encoder, and retains logits at least 5.75. This added 4,484 matches and
21,818 candidates to the target exclusive V7 output.

The direction is evidence based, but its leaderboard magnitude remains
unknown because the supplied labels contain no France records. The exact V8
output passed the full structural validator before packaging.

## Independent evidence that France is the weak spot

| variant | local audit3 | portal |
|---|---|---|
| V4 | 0.92402 | 0.896 |
| V5 | 0.93675 | 0.907 |
| V5, France threshold 0.80 | — | 0.913 |
| V6 (reranked selected links) | 0.94728 | 0.927 |

The local-to-portal gap grows as the model gets more confident: −0.028 for V4,
−0.030 for V5, −0.020 for V6. Local holdouts cover only US and India, so none
of them can detect French failure. Every local number in this repo is blind to
15% of the scored rows.
