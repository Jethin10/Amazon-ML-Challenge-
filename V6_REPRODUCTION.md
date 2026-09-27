# Astralis V6 reproduction

V6 is a two-stage matcher. V5 generates and scores the full candidate set.
A multilingual MiniLM cross encoder then reranks V5's retained US and India
pairs with probabilities at most 0.98. France keeps the portal-tested 0.80 V5
threshold. The final files use the same candidate set as V5.

## Environment and data

Use Python 3.14, an NVIDIA GPU with at least 4 GB VRAM, and the organizer's
seven TSVs under `dataset/train/` and `dataset/test/`. Install
`requirements-v6.txt`. The CUDA PyTorch wheel comes from the official PyTorch
index named in that file. The included `models/v5_lgb.txt` and
`models/v6_reranker/` files permit inference without retraining. The
multilingual MiniLM base checkpoint is Apache 2.0 licensed and has about
100 million parameters.

## Build the output

Follow `V5_REPRODUCTION.md` through `src/run_v5_batches.py`. It must produce
all 20 `work/v5_shards/selected_???.parquet` files and the complete V5 output.
Then run:

```powershell
.venv\Scripts\python.exe src/predict_v6_selected.py
.venv\Scripts\python.exe src/validate_large.py --output-dir output_v6_reranker_selected
```

`predict_v6_selected.py` resumes from completed shards. It writes a live
progress file to `work/v6_selected_shards/progress.json` and preserves every
Source 1 row, including rows with no matches. `scripts/watch_v6_progress.ps1`
shows current progress in a PowerShell window.

## Retrain the second stage

Build V5 training features as described in `V5_REPRODUCTION.md`, including
the validation bucket. Then run:

```powershell
.venv\Scripts\python.exe src/build_v6_reranker_pairs.py --mode valid
.venv\Scripts\python.exe src/train_v6_reranker.py --mode train --batch-size 8 --epochs 1 --max-length 128
```

The pair sampler retains difficult V5 predictions and a reproducible sample
of easy pairs. It uses only the supplied training labels and record text. The
reranker input contains each side's business name and address, without entity
IDs. The frozen V5 probability and the reranker logit are combined as:

```text
new_probability = sigmoid(logit(v5_probability) + 0.6 * reranker_logit)
```

For US and India, V5 pairs above 0.98 keep their original score; other saved
V5 pairs use the new score and a 0.60 threshold. France uses the V5 score and
a 0.80 threshold. The weights and thresholds were chosen on the labeled
audit2 split, then checked on the separate audit3 split. Audit3 had been used
for earlier exploratory diagnostics. No test labels or external business
identity service are used.

## Measured limits

The selected-pair method scored 0.94728 macro F0.5 on audit3 versus 0.93675
for V5. Reranking additional lower-scored candidates reached 0.95956 locally,
but that version requires a new full test scoring pass and is not the output
produced by this script. France has no labeled local holdout. The portal
evaluated this complete V6 output at **0.927**, versus **0.913** for the
France-threshold V5 variant.
