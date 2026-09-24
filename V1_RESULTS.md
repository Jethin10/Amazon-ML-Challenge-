# Current local results

The current baseline is a blocking pipeline followed by a LightGBM pair
classifier. We split Source 1 IDs with a stable hash. The model fits on hash
buckets 2 through 41, selects its threshold on buckets 42 through 49, and
gets one final check on buckets 50 through 59. Buckets 0 and 1 were used for
small blocking experiments. All splits share the Source 2 and 3 search pool.

| Measure | Development validation | Separate holdout |
| --- | ---: | ---: |
| Source 1 records | 18,057 | 22,144 |
| Ground-truth links | 62,413 | 76,681 |
| Candidate pairs | 1,466,019 | 1,816,215 |
| Candidates per Source 1 | 81.2 | 82.0 |
| True links retrieved | 57,117 | 70,240 |
| Candidate recall | 91.51% | 91.60% |
| Macro F0.5, standard prediction | 0.8902 | 0.8890 |
| Macro F0.5, fast prediction | 0.8903 | 0.8892 |

The fast prediction setting uses LightGBM's `pred_early_stop` with frequency
10 and margin 10. We selected threshold 0.60 on development validation and
kept it fixed on the separate holdout. The holdout has 1,209 true singletons;
fast prediction correctly leaves 1,044 of them empty.

Candidate recall counts how many known matching pairs reach the classifier.
The model cannot recover a pair that blocking missed. Macro F0.5 averages an
F0.5 score over Source 1 records, including true singletons. Pairwise accuracy
would be misleading because most generated pairs are nonmatches.

India remains the hardest part of the labeled data. The separate holdout has
candidate recall 87.64% and macro F0.5 0.8261 for India, compared with 94.24%
and 0.9311 for the US. France appears only in the test data, so neither local
score measures France performance.

These numbers are local validation results, not a leaderboard score.
