# Astralis V8 results

## Scores

| Version | Local audit3 macro F0.5 | Public leaderboard |
| --- | ---: | ---: |
| V5 | 0.93675 | 0.907 |
| V5 France 0.80 | 0.93675 | 0.913 |
| V6 selected rerank | 0.94728 | 0.927 |
| V7 full rerank | 0.95956 | Pending |
| V8 final route | 0.95958 | Pending |
| Retrieval experiment | 0.96561 | Not materialized |

V8's deployable output uses the V7 full reranker plus target exclusivity and
France recovery. Target exclusivity was locally neutral because the labeled
training truth already assigns each target to at most one Source 1 entity.

An additional US and India retrieval experiment reached 0.96561 by selecting
1,064 links, including 1,032 true positives and 32 false positives on audit3.
Those retrieval additions were not materialized across the full test set and
are not part of the final upload file.

## France recovery

France lacks local labels. A controlled suffix sensitivity test reduced the
proxy score by 0.030 to 0.040 when SARL, SAS, or EURL handling was disabled,
which confirmed that legal suffix normalization was a material retrieval path.
The final France stage added 4,484 high confidence matches and 21,818 candidates.

## Output

The final output has 1,732,544 Source 1 rows. The exact leaderboard upload is
`output_v8/matching_results.tsv`; the final review archive is
`submission/Astralis_submission.zip`. A public V8 score has not yet been
observed, so the local score must not be read as a leaderboard result.
