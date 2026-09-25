# ML Challenge 2026: Astralis entity resolution solution

**Team:** Astralis

**Date:** 25 September 2026

| Player | Organisation or college |
| --- | --- |
| Shaurya Singh | Noida Institute of Engineering And Technology (NIET), Greater Noida |
| Dakshanya Maddala | Texas A&M University |
| Jethin kosaraju | Noida Institute of Engineering And Technology (NIET), Greater Noida |

## 1. Task and result

For every Source 1 business, we return zero or more matching records from
Sources 2 and 3. We generate a bounded candidate set, score each candidate
pair with LightGBM, and select pairs with predicted probability at least
0.70. The output includes every test Source 1 record, including France.

On a labeled Source 1 holdout reserved from route selection and model
training, V4 achieved **94.58% candidate recall** and **0.9249 macro F0.5**.
These are local scores. The test set has no labels, and the provided labeled
data has no France examples, so we cannot claim a France or private
leaderboard score from this evaluation.

## 2. Candidate generation

All retrieval routes require equal country labels. The pipeline normalizes
name and address punctuation, case, accents, and spacing. It also uses local
`anyascii` transliteration so names in different scripts can share Latin
features. No external business database or lookup service is used.

We union and deduplicate several complementary routes:

1. Exact normalized name or address.
2. Selective name prefixes combined with city or building number, a city
   plus number, and an address prefix. Broad blocks are capped.
3. Rare address words, optionally combined with a name prefix or number.
4. Rare name words with a number, with target-frequency caps.
5. V4's added route: any shared address number and address word of at least
   five characters. The word occurs in at most 10,000 targets. We keep this
   route's pair when the transliterated names have Jaro-Winkler similarity at
   least 0.70 or either name is missing. A materialized target word-number
   index keeps the join practical.

The deduplicated candidate set is exactly what the pair model scores and is
written to `candidate_pairs.tsv`. On the fresh labeled holdout, it contains
2,839,414 pairs for 21,921 Source 1 entities: 129.5 candidates per S1. It
retrieves 71,975 of 76,101 true links, or 94.58% candidate recall. The
prior V3 route retrieved 70,569 true links from 2,626,411 candidates on the
same holdout.

## 3. Pair model and features

The classifier is LightGBM 4.7.0, trained on pairs produced by these exact
candidate routes. Features compare normalized names and addresses using
Jaro-Winkler and edit similarity, sorted-word similarity, token overlap,
containment and exact-match flags, lengths, city agreement, building-number
agreement or conflict, missing-field flags, and a Source 3 flag. The V4
feature set includes transliterated name and address comparisons, plus
overlap or conflict across **all** address numbers rather than only the first.

The fit uses hash buckets 2–41 of Source 1 IDs. Buckets 42–49 select the
threshold. V4 uses learning rate 0.05, 31 leaves, minimum leaf size 150,
feature fraction 0.85, and early stopping on validation loss. The selected
model's best iteration was 766 of a maximum 800. The threshold is 0.70.
Test inference uses LightGBM prediction early stopping with frequency 10
and margin 10, matching the audited setting.

Buckets 50–59 informed the new retrieval route. Buckets 60–69 were then
held out from route selection, fitting, and threshold selection. All splits
search the same Source 2 and 3 pool. On the fresh holdout, macro F0.5 was
0.9249 for V4 versus 0.9143 for V3. Of 1,224 true singletons, V4 returned
the correct empty list for 1,142. India macro F0.5 was 0.9057 and US macro
F0.5 was 0.9377. The candidate set's oracle macro F0.5 ceiling is 0.9805;
the actual model does not reach that ceiling.

## 4. Full inference, validation, and packaging

V4 wrote 1,732,544 Source 1 rows, 201,717,297 candidate pairs, and
5,994,946 selected matches. `output_v4/matching_results.tsv` is the file
scored on the leaderboard. `output_v4/candidate_pairs.tsv` records the
exact pairs scored by the model and is included in the final package.

Our streaming full-file validator confirmed every required Source 1 row,
valid target IDs, no duplicate rows or IDs, and every selected match within
its candidate list. It reported no errors. The organizers' matching-file
validator also passed with ID checks. These checks establish format and
coverage, not hidden-label accuracy.

The optional `src/make_portal_variant.py --cap 10` produces a smaller
leaderboard TSV if the portal cannot process the original file. It caps only
lists longer than 10. On the labeled holdout, this cap changed macro F0.5
from 0.924916 to 0.924921. Most removed test predictions are in France,
which lacks labels, so the cap's hidden score is uncertain. The original
uncapped file remains our primary model output.

The `code/business_entity_resolution/README.md` in the package gives the
commands from the supplied raw TSVs through training, inference, and
validation. Dependencies are pinned in `requirements.txt`. The trained
`models/v4_lgb.txt` and threshold metadata are included. The pipeline uses
only challenge files for business identities; it does not query external
APIs, registries, search engines, or geocoding services.
