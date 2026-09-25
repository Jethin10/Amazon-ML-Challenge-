# ML Challenge 2026: Astralis V5 entity resolution

**Team:** Astralis

**Date:** 25 September 2026

| Player | Organisation or college |
| --- | --- |
| Shaurya Singh | Noida Institute of Engineering And Technology (NIET), Greater Noida |
| Dakshanya Maddala | Texas A&M University |
| Jethin kosaraju | Noida Institute of Engineering And Technology (NIET), Greater Noida |

## Methodology

For each Source 1 record, we retrieve plausible Source 2 and Source 3 records,
score each pair with LightGBM, and return zero or more target IDs. We keep
singleton rows with an empty match list. Every test Source 1 record is present,
including France. The test output is a tab-separated file with the exact
headers in the challenge statement.

## Candidate generation and blocking

We use the supplied records only. All blocking joins require the same
normalized country string, without restricting countries to a fixed list.
Normalization handles case, punctuation, accents, spacing, legal suffixes,
and local `anyascii` transliteration. Candidate routes include exact
normalized name or address; selective name prefixes with city or number;
rare address and name words; and a shared address number plus address word
screened by transliterated name similarity. The routes are unioned and
deduplicated before matching. `candidate_pairs.tsv` is this exact scored
candidate set, including every final match.

## Model and features

LightGBM 4.7.0 scores each candidate with normalized and transliterated name
and address similarities, sorted-word and token overlap, containment, address
number agreement or conflict, missing-field flags, and source identity. V5
adds the frequency of each name in the target pool and measures each pair's
rank and distance from the strongest name and address candidates for its
Source 1 record. This helps distinguish a distinctive business name from a
common name shared by many unrelated businesses.

Buckets 2-41 of the Source 1 ID hash fit the model. Buckets 42-49 select the
threshold. The model has 63 leaves, minimum leaf size 100, learning rate
0.04, feature fraction 0.9, and 584 best iterations out of a 1,200-round
limit. The selected threshold is 0.625. Inference and audits use LightGBM
prediction early stopping with frequency 10 and margin 10.

## Evaluation

Buckets 70-79 were not used for model fitting, threshold selection, or V5
feature design. On these 21,878 Source 1 entities, the candidate set retrieved
71,966 of 76,167 true links (94.48% recall). V5 scored **0.93675 macro
F0.5**, compared with **0.92402** for V4 on the same candidates. India scored
0.91454 and the US scored 0.95148. These are local scores, not a public or
private leaderboard score. The provided labels contain no France examples,
so France performance cannot be estimated locally.

The full test output passed structural validation: 1,732,544 Source 1 rows,
201,717,297 candidates, 5,886,292 predicted links, and 104,061 rows with
empty predictions. There were no invalid IDs, duplicate IDs, missing rows, or
predicted links absent from the candidate list.

## Reproduction and fair play

`code/business_entity_resolution/README.md` contains exact commands from
the seven supplied raw files through preparation, training, blocking,
inference, validation, and packaging. `requirements.txt` pins all
dependencies, and the package includes the trained V5 model and threshold.
No external business identity lookup, geocoding service, or external data
augmentation is used. The final ZIP contains both output TSVs, the runnable
code, pinned dependencies, and this methodology document.
