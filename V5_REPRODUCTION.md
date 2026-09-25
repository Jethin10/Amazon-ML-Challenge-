# Astralis V5 reproduction

V5 uses only the supplied challenge records. All files are tab-separated. The
code supports every country string in the test data, including France. The
training labels cover only India and the US.

Use Python 3.14 and install `requirements.txt`. Place the organizer's four
training TSVs in `dataset/train/` and three test source TSVs in
`dataset/test/`. Run these commands from the repository root:

```powershell
.venv\Scripts\python.exe src/v1.py prepare --part train
.venv\Scripts\python.exe src/build_latin.py --part train
.venv\Scripts\python.exe src/v4.py --part train --mode fit --features
.venv\Scripts\python.exe src/v4.py --part train --mode valid --features
.venv\Scripts\python.exe src/v3_features.py --part train --mode fit --version v4
.venv\Scripts\python.exe src/v3_features.py --part train --mode valid --version v4
.venv\Scripts\python.exe src/v5_features.py --mode fit
.venv\Scripts\python.exe src/v5_features.py --mode valid
.venv\Scripts\python.exe src/train_v5.py
.venv\Scripts\python.exe src/v1.py prepare --part test
.venv\Scripts\python.exe src/build_latin.py --part test
.venv\Scripts\python.exe src/generate_v5_candidates.py --batches 20
.venv\Scripts\python.exe src/run_v5_batches.py
.venv\Scripts\python.exe src/validate_large.py --output-dir output_v5
.venv\Scripts\python.exe src/package_code.py --version v5
.venv\Scripts\python.exe src/package_submission.py --version v5 --team Astralis
```

The packaged `models/v5_lgb.txt` and `models/v5_metrics.json` allow inference
without retraining. In that case, skip the training commands and start at test
preparation. `generate_v5_candidates.py` reproduces the V4 blocking routes
without requiring a V4 classifier. `run_v5_batches.py` scores each candidate
shard in a separate process, using quarters or smaller pieces when memory is
limited. Each Source 1 record and all its candidates stay in one piece, so
candidate ranks and scores are unchanged. Completed pieces are reused on
restart. The runner checks Source 1 row order while merging, then writes all
rows to `output_v5/matching_results.tsv`. The final package includes the exact
candidate set scored in `output_v5/candidate_pairs.tsv`.

For an independent local audit, additionally build `audit3` candidates and
features, then run `src/audit_v5.py --mode audit3`. Buckets 70-79 were not
used to fit V5 or choose its threshold. Their measured F0.5 is recorded in
`V5_RESULTS.md`. France has no labeled holdout, so local performance there is
unknown. A portal variant that changes France lists must be evaluated on the
portal before it can be called an improvement.

Git uses `.gitattributes` to preserve the model file's original line endings
across Windows and Unix checkouts. Converting the model text can make it
unreadable by LightGBM.
