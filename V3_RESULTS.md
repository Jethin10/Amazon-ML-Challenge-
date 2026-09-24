# V3 local results

V3 extends V1 blocking with a capped rare name-token route. The classifier
adds script transliteration and address-number overlap features. All text
comes from the supplied challenge files. We tested the route on hash buckets
0–1, fit the model on buckets 2–41, chose its threshold on buckets 42–49,
and checked the fixed model and threshold on buckets 50–59. The target pool is
shared across these Source 1 splits.

| Measure | V1 holdout | V3 holdout |
| --- | ---: | ---: |
| Source 1 records | 22,144 | 22,144 |
| Ground-truth links | 76,681 | 76,681 |
| Candidate pairs | 1,816,215 | 2,666,943 |
| Candidates per Source 1 | 82.0 | 120.4 |
| True links retrieved | 70,240 | 71,153 |
| Candidate recall | 91.60% | 92.79% |
| Macro F0.5 | 0.8892 | 0.9149 |
| Correct true singletons | 1,044 / 1,209 | 1,125 / 1,209 |
| India macro F0.5 | 0.8261 | 0.8834 |
| US macro F0.5 | 0.9311 | 0.9357 |

V3 uses a 0.70 threshold selected on the validation slice and the same
LightGBM early prediction stopping setting as V1. These are local scores,
not leaderboard scores. The test set contains France, which has no labeled
training or holdout examples. Its performance remains unknown.

The holdout candidate recall puts an upper bound on pair recall. It also
limits the macro F0.5 attainable by any classifier using this candidate set.
The user target of 98% recall has not been reached. The new route recovers
913 extra links on the holdout while adding roughly 851,000 candidate pairs.
Additional broad routes tested on the pilot slice had poor true-link yield
per added candidate and were not included.

The original V1 submission files remain in `output/`. V3 predictions go to
`output_v3/` so an ongoing portal upload cannot be changed by this work.
