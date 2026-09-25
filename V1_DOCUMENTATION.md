# ML Challenge 2026: business entity resolution solution

**Team name:** Astralis

**Team members:**

| Player | Organisation or college |
| --- | --- |
| Shaurya Singh | Noida Institute of Engineering And Technology (NIET), Greater Noida |
| Dakshanya Maddala | Texas A&M University |
| Jethin kosaraju | Noida Institute of Engineering And Technology (NIET), Greater Noida |

**Submission date:** 25 September 2026

## 1. Summary

We match each Source 1 business against records from Sources 2 and 3. A
disk-backed DuckDB pipeline generates candidate pairs, then a LightGBM model
scores them using name and address similarity. On a separate labeled holdout,
the pipeline reaches macro F0.5 of 0.8892 with 91.60% candidate recall. The
test set has no public labels, so this is a local score.

## 2. Problem analysis and strategy

Each Source 1 business can match zero, one, or many Source 2 and 3 records.
Names and addresses contain reordered words, spelling changes, legal suffixes,
trade names, missing fields, address abbreviations, and leading zeros in
building numbers. Some India records use different scripts. The labeled data
contains US and India records; test also contains France records. Our code
treats country as an open string field and includes every test Source 1 row.

Comparing every record pair is infeasible at this scale. We first retrieve a
small candidate set using several keys. The model then weighs agreement and
conflicts across the two fields. We select the final cutoff for macro F0.5,
which rewards precision and includes true singletons.

## 3. Candidate generation

We lowercase text, remove accents and punctuation, and derive normalized
name, address, city, and first building-number fields. Every route requires
the same country. The routes are:

1. Exact normalized name and exact normalized address.
2. First five name characters; first four name characters with city or number;
   city with number; and first eight address characters. We skip target blocks
   larger than 100 for these broad routes.
3. The rarest address word of at least five characters when its target
   frequency is at most 100 in the country.
4. Up to three rare address words paired with the first four name characters
   or a building number. Each word can appear in up to 1,000 targets.
5. Up to two name words paired with a building number, using the same 1,000
   target-frequency limit.

We normalize leading zeros when comparing building numbers in the added
address-word route. The pipeline deduplicates the union of all routes. It
outputs this exact candidate set to `candidate_pairs.tsv` before model
scoring.

On the separate labeled holdout, it generates 1,816,215 pairs for 22,144
Source 1 records, an average of 82.0 candidates each. Those pairs include
70,240 of 76,681 ground-truth links, so candidate recall is 91.60%.

## 4. Matching model

The pair classifier is LightGBM 4.7.0 with 350 boosting rounds, 31 leaves,
learning rate 0.06, and minimum leaf size 150. Inputs include Jaro-Winkler
and edit similarity for names, Jaro-Winkler similarity for addresses, sorted
word similarity for both fields, token overlap for both fields, exact and
containment flags, name and address length ratios, city and building-number
agreement or conflict, missing-field flags, and a Source 3 flag.

Source 1 IDs are split by a stable hash. Buckets 2 through 41 fit the model;
buckets 42 through 49 select the cutoff. Buckets 50 through 59 form a separate
holdout and were not used to fit the model or choose the cutoff. We calculate
F0.5 for each Source 1 record against all its ground-truth links, including
true links missed by blocking, then average across Source 1 records. Correct
empty answers for true singletons score 1. We selected cutoff 0.60 on the
development split and kept it fixed on the holdout and test data.

Inference uses LightGBM's `pred_early_stop` with frequency 10 and margin 10.
On the separate holdout this reduced prediction time and gave F0.5 of 0.8892,
compared with 0.8890 for standard prediction at the same cutoff.

## 5. Results and error analysis

| Measure | Development split | Separate holdout |
| --- | ---: | ---: |
| Source 1 records | 18,057 | 22,144 |
| Candidate recall | 91.51% | 91.60% |
| Average candidates per Source 1 | 81.2 | 82.0 |
| Macro F0.5 with submission inference setting | 0.8903 | 0.8892 |

The holdout includes 1,209 true singletons; 1,044 received correct empty
predictions. By country, India reached candidate recall 87.64% and macro F0.5
0.8261. The US reached 94.24% and 0.9311. The largest remaining limitation
is candidate generation for India, where script changes, missing fields, and
address variants can prevent a true pair from reaching the classifier. France
has no labeled local score because it appears only in the test data.

## 6. Reproduction and data use

All identity information comes from the provided challenge files. We do not
look up businesses through external APIs, registries, search engines, or
geocoding services. `code/business_entity_resolution/README.md` gives the
commands from raw TSVs to both output files. The code uses DuckDB for
disk-backed candidate retrieval and feature calculation, and LightGBM for
model training and prediction. The supplied `models/v1_lgb.txt` also permits
inference without retraining.

The two submitted TSVs are `output/matching_results.tsv` and
`output/candidate_pairs.tsv`. Every final match comes from the submitted
candidate set. The included validator checks row coverage, IDs, duplicates,
and this subset rule before packaging.
