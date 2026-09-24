# Astralis V3 reproduction

The V3 pipeline uses only the supplied challenge data. It compares every
Source 1 record to candidates from Source 2 and Source 3. It treats country
as an open string label, so France records are included in test inference.

Place the four training TSV files under `dataset/train/` and the three test
source TSV files under `dataset/test/`. Use Python 3.14 and the pinned packages in
`requirements.txt`. Run commands from the repository root.

```powershell
.venv\Scripts\python.exe src/v1.py prepare --part train
.venv\Scripts\python.exe src/build_latin.py --part train
.venv\Scripts\python.exe src/v2.py --part train --mode fit --features
.venv\Scripts\python.exe src/v2.py --part train --mode valid --features
.venv\Scripts\python.exe src/v3_features.py --part train --mode fit
.venv\Scripts\python.exe src/v3_features.py --part train --mode valid
.venv\Scripts\python.exe src/train_v3.py
.venv\Scripts\python.exe src/v2.py --part train --mode audit --features
.venv\Scripts\python.exe src/v3_features.py --part train --mode audit
.venv\Scripts\python.exe src/audit_v3.py --early-stop
.venv\Scripts\python.exe src/v1.py prepare --part test
.venv\Scripts\python.exe src/build_latin.py --part test
.venv\Scripts\python.exe src/predict_v3.py --batches 20
```

`predict_v3.py` writes `output_v3/matching_results.tsv` and
`output_v3/candidate_pairs.tsv`. It records completed batches in
`work/predict_v3_progress.json`. Resume an interrupted run with
`--start-batch N` only when both output files contain exactly the first N
complete batches. Starting again at zero replaces the V3 outputs.

The candidate stage unions V1's exact and selective keys with a capped
rare-name-token route. Every route requires the same country. The exact
deduplicated candidate set is passed to the model and written to the
candidate file. The model uses LightGBM on name and address similarity,
token overlap, transliteration, building number overlap, and missing-field
flags. `anyascii` transliteration is local and uses no external lookup.

Hash buckets 2–41 fit the model. Buckets 42–49 choose the threshold of
0.70. Buckets 50–59 are the separate holdout. On that holdout V3 achieved
92.79% candidate recall and 0.9149 macro F0.5, compared with 91.60% and
0.8892 for V1. The training data has no France labels, so this local score
does not establish France performance. See `V3_RESULTS.md` for counts and
per-country results.
