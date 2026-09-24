# Amazon ML Challenge 2026 entity matcher

This directory contains a reproducible blocking and pair-classification
pipeline. It uses only the challenge TSV files. There are no external business
lookups, geocoding calls, or pretrained entity databases.

## Inputs and environment

Place the challenge files at these paths relative to this directory:

```
dataset/train/train_source1.tsv
dataset/train/train_source2.tsv
dataset/train/train_source3.tsv
dataset/train/train_ground_truth.tsv
dataset/test/test_source1.tsv
dataset/test/test_source2.tsv
dataset/test/test_source3.tsv
```

Python 3.14 and the pinned packages in `requirements.txt` were used on Windows
with 16 GB RAM. DuckDB keeps large intermediate tables on disk under `work/`.
The LightGBM model runs on CPU. Run commands from this directory.

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe src/v1.py prepare --part train
.venv\Scripts\python.exe src/v1.py candidates --part train --mode fit
.venv\Scripts\python.exe src/v1.py features --part train --mode fit
.venv\Scripts\python.exe src/v1.py candidates --part train --mode valid
.venv\Scripts\python.exe src/v1.py features --part train --mode valid
.venv\Scripts\python.exe src/train_v1.py
.venv\Scripts\python.exe src/v1.py prepare --part test
.venv\Scripts\python.exe src/predict_v1.py --batches 20
.venv\Scripts\python.exe utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test
```

For the full data, `src/validate_large.py` checks the same format, every
target ID, and the match-to-candidate subset while streaming each output row.
After inference, run `python src/package_submission.py --team Astralis` to
validate the files and build `submission/Astralis_submission.zip`.

`prepare` scans the source TSVs and creates normalized DuckDB tables. The
train stages generate labeled candidate pairs and fit the model. The test
stage writes the required `output/matching_results.tsv` and
`output/candidate_pairs.tsv`. It handles every Source 1 record, including
records with no candidates or predicted matches. Each test batch is recorded
in `work/predict_progress.json`. If a run is interrupted after a completed
batch, resume with `--start-batch N`, where N is the number of completed
batches. Start from zero after changing the model or candidate generator.

The `models/v1_lgb.txt` file is included in the submission package so the
provided output can be reproduced without retraining. The commands above
retrain it from the provided training set.

## Method

The generator joins only records with the same country. It combines exact
normalized name and address keys; short name, city, number, and address
prefixes with a block-size limit; the rarest usable address word; and
selective joins of address word with name prefix or normalized building
number. A final route joins a name word with building number. It removes
duplicate candidate pairs before feature calculation. The pair classifier
uses LightGBM with name and address string similarity, token overlap,
sorted-word similarity, number agreement, missing-value flags, and a Source 3
flag. It uses 350 trees and a 0.60 cutoff. Inference uses LightGBM's early
prediction stopping after validation showed no score loss.

Source 1 hash buckets 2 to 41 fit the model; buckets 42 to 49 select the
threshold. Buckets 50 to 59 form a separate holdout. On that holdout, the
pipeline retrieved 70,240 of 76,681 true links, produced 82.0 candidates
per Source 1 record, and reached macro F0.5 of 0.8892 with the inference
setting used for the submission. These are local results. The test set has
France records, but the labeled training set does not, so no local France
score is available.

`src/audit_v1.py` scores the separate holdout. The `probe_*.py` scripts record
candidate-generation experiments; they are not required to reproduce the
submitted outputs.
