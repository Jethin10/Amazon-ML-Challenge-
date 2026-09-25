# V4 local results

V4 adds a candidate route that matches any shared address number and
address word, then keeps pairs with transliterated name similarity of at
least 0.70 or a missing name. An index over the address word and number
keeps this route practical. V4 retrains LightGBM on the exact expanded
candidate set with the V3 feature schema and a longer boosting schedule.

We selected the route after inspecting hash buckets 50–59. We fit the
classifier on buckets 2–41 and selected its threshold on buckets 42–49.
Buckets 60–69 were held out from route selection, model fitting, and
threshold choice. All splits use the same Source 2 and 3 search pool.

| Measure on buckets 60–69 | V3 | V4 |
| --- | ---: | ---: |
| Source 1 records | 21,921 | 21,921 |
| Ground-truth links | 76,101 | 76,101 |
| Candidate pairs | 2,626,411 | 2,839,414 |
| Candidates per Source 1 | 119.8 | 129.5 |
| True links retrieved | 70,569 | 71,975 |
| Candidate recall | 92.73% | 94.58% |
| Macro F0.5 | 0.9143 | 0.9249 |
| Correct true singletons | 1,138 / 1,224 | 1,142 / 1,224 |
| India macro F0.5 | 0.8854 | 0.9057 |
| US macro F0.5 | 0.9335 | 0.9377 |

The V4 candidate set has an oracle macro F0.5 ceiling of **0.9805** on this
holdout: that is the score if a perfect classifier selected every available
true candidate and no false candidate. The actual model scores 0.9249, so
pair classification still leaves substantial room. This oracle calculation
uses holdout labels for analysis and is not a deployable prediction method.

The V4 classifier uses a threshold of 0.70 selected on buckets 42–49. Its
best iteration was 766 of 800. The longer fit and the new candidate route
both differ from V3, so the table measures the full pipeline change rather
than isolating either component.

These are local scores, not leaderboard scores. France exists only in the
unlabeled test set, so no local France score is available. The 98% candidate
recall goal is not met; V4 reaches 94.58% on the fresh holdout. The private
leaderboard may differ because its entity and country mix is unknown.

## Full test output

V4 inference wrote 1,732,544 Source 1 rows, 201,717,297 candidate pairs,
and 5,994,946 selected matches across Source 2 and Source 3. The full-file
validator checked every row and target ID, found no missing or duplicate
Source 1 rows, and confirmed every selected match is in its candidate list.
The validator reported `ok: true` with no errors. These unlabeled test counts
do not measure accuracy or recall.

## Portal size fallback

`src/make_portal_variant.py --cap 10` makes a separate leaderboard TSV in
`output_v4_portal/` by retaining the first 10 selected IDs on rows with longer
lists. It keeps all 1,732,544 Source 1 rows and passed the organizers' matching
validator with ID checks. Its size is 93,268,009 bytes versus 99,716,074 for
the original. On the labeled audit2 slice, the same list cap changed macro
F0.5 from 0.924916 to 0.924921. Most removed test predictions are in France,
which has no labeled validation data. Therefore the audit result does not
establish the portal variant's France or hidden leaderboard performance.
Keep the original full output as the primary model result; use the fallback
only if the portal cannot process the original file.
