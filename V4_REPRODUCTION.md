# Astralis V4 reproduction

The V4 pipeline uses only the supplied challenge data. It compares every
Source 1 record to candidates from Source 2 and Source 3. It treats country
as an open string label, so France records are included in test inference.

Place the four training TSV files under `dataset/train/` and the three test
source TSV files under `dataset/test/`. Use Python 3.14 and the pinned packages in
`requirements.txt`. Run commands from the repository root.

```powershell
.venv\Scripts\python.exe src/v1.py prepare --part train
.venv\Scripts\python.exe src/build_latin.py --part train
.venv\Scripts\python.exe src/v4.py --part train --mode fit --features
.venv\Scripts\python.exe src/v4.py --part train --mode valid --features
.venv\Scripts\python.exe src/v3_features.py --part train --mode fit --version v4
.venv\Scripts\python.exe src/v3_features.py --part train --mode valid --version v4
.venv\Scripts\python.exe src/train_v4.py
.venv\Scripts\python.exe src/v4.py --part train --mode audit2 --features
.venv\Scripts\python.exe src/v3_features.py --part train --mode audit2 --version v4
.venv\Scripts\python.exe src/audit_v4.py --early-stop --mode audit2
.venv\Scripts\python.exe src/v1.py prepare --part test
.venv\Scripts\python.exe src/build_latin.py --part test
.venv\Scripts\python.exe src/predict_v4.py --batches 20
.venv\Scripts\python.exe src/validate_large.py --output-dir output_v4
```

`predict_v4.py` writes `output_v4/matching_results.tsv` and
`output_v4/candidate_pairs.tsv`. It records completed batches in
`work/predict_v4_progress.json`. Resume an interrupted run with
`--start-batch N` only when both output files contain exactly the first N
complete batches. Starting again at zero replaces the V4 outputs.

The candidate stage unions V1's exact and selective keys, a capped
rare-name-token route, and a route joining any shared address number with
an address word. The last route keeps pairs with transliterated name
similarity at least 0.70 or a missing name. Every route requires the same country. The exact
deduplicated candidate set is passed to the model and written to the
candidate file. The model uses LightGBM on name and address similarity,
token overlap, transliteration, building number overlap, and missing-field
flags. `anyascii` transliteration is local and uses no external lookup.

Hash buckets 2–41 fit the model. Buckets 42–49 choose the threshold of
0.70. Buckets 60–69 are the fresh holdout, after using buckets 50–59 to
select the new candidate route. On the fresh holdout V4 achieved 94.58%
candidate recall and 0.9249 macro F0.5, compared with 92.73% and 0.9143
for V3. The training data has no France labels, so this local score does
not establish France performance. See `V4_RESULTS.md` for counts and
per-country results.
