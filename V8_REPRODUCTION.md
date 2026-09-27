# Astralis V8 reproduction

V8 is the final submission route. It starts with V5 LightGBM candidates,
reranks the complete retained candidate set with the multilingual V6 cross
encoder, makes each Source 2 or Source 3 target exclusive to its strongest
Source 1 match, and restores high confidence France links found by legal suffix
and address retrieval.

## Environment

Use Python 3.14, install `requirements-v6.txt`, and place the organizer files
under `dataset/train/` and `dataset/test/`. Inference uses the checked in
`models/v5_lgb.txt` and `models/v6_reranker/` artifacts.

## Final route

Run V5 as described in `V5_REPRODUCTION.md`, then run the V7 full reranker and
the V8 postprocessing stages:

```powershell
.venv\Scripts\python.exe src/predict_v7_full.py
.venv\Scripts\python.exe src/build_v7_collision_pairs.py
.venv\Scripts\python.exe src/apply_target_exclusivity.py
.venv\Scripts\python.exe src/build_france_suffix_pairs.py
.venv\Scripts\python.exe src/filter_france_suffix_pairs.py
.venv\Scripts\python.exe src/score_pair_parquet_streaming.py
.venv\Scripts\python.exe src/merge_retrieval_additions.py
.venv\Scripts\python.exe src/validate_large.py --output-dir output_v8
```

The final leaderboard file is `output_v8/matching_results.tsv`. Package the
same output and its candidate set with:

```powershell
.venv\Scripts\python.exe src/package_code.py --version v8
.venv\Scripts\python.exe src/package_submission.py --version v8 --team Astralis --already-validated
```

The second command writes `submission/Astralis_submission.zip` and verifies
every member by reading it back.

## Thresholds

The full reranker uses logit thresholds 5.75 for the US and 4.25 for India.
France recovery requires normalized legal suffix agreement, address similarity
of at least 0.70, and reranker logit at least 5.75. Target exclusivity keeps the
strongest predicted Source 1 owner when the same target was assigned more than
once.

The supplied training labels contain US and India only. France changes were
therefore checked with suffix sensitivity tests and structural validation; no
France test labels were available for local scoring.
