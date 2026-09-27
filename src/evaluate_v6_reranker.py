"""Measure entity-level F0.5 after changing only V5's uncertain pair scores."""

from __future__ import annotations

import argparse
import json

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.special import expit, logit

from evaluate_thresholds_streaming import metadata, score
from v1 import DB, ROOT, WORK
from v5_features import ALL_FEATURE_NAMES


ALPHAS = [0.0, .1, .2, .3, .4, .6, .8, 1.0]
THRESHOLDS = [.4, .45, .5, .55, .6, .625, .65, .7, .75, .8, .85, .9]
WINDOWS = {"audit2": (60, 70), "audit3": (70, 80)}


def evaluate(mode: str, selected_only: bool) -> dict:
    lo, hi = WINDOWS[mode]
    con = duckdb.connect(str(DB), read_only=True)
    ordered = [row[0] for row in con.execute(
        "SELECT entity_id FROM train_s1 WHERE hash(entity_id)%1000>=? "
        "AND hash(entity_id)%1000<? ORDER BY entity_id", [lo, hi]
    ).fetchall()]
    con.close()
    position = {sid: i for i, sid in enumerate(ordered)}
    truth, countries = metadata(ordered)
    scored = pd.read_parquet(WORK / f"v6_reranker_{mode}_scored.parquet",
                             columns=["s1_id", "target_id", "reranker_logit"])
    replacement = {(sid, tid): float(value) for sid, tid, value in zip(
        scored.s1_id, scored.target_id, scored.reranker_logit)}
    del scored
    combos = [(alpha, threshold) for alpha in ALPHAS
              for threshold in THRESHOLDS]
    predicted = np.zeros((len(ordered), len(combos)), dtype=np.float64)
    true_positive = np.zeros_like(predicted)
    left = pq.ParquetFile(WORK / f"train_{mode}_v5_features.parquet")
    model = lgb.Booster(model_file=str(ROOT / "models" / "v5_lgb.txt"))
    baseline_batches = left.iter_batches(
        batch_size=50_000,
        columns=["s1_id", "target_id", "label"] + ALL_FEATURE_NAMES)
    total = found = 0
    for ids_batch in baseline_batches:
        ids = ids_batch.to_pandas()
        p = model.predict(ids[ALL_FEATURE_NAMES].to_numpy(dtype=np.float32),
                          num_threads=4, pred_early_stop=True,
                          pred_early_stop_freq=10,
                          pred_early_stop_margin=10).astype(np.float32)
        y = ids.label.to_numpy(dtype=np.int8)
        group = np.fromiter((position[sid] for sid in ids.s1_id),
                            dtype=np.int32, count=len(ids))
        lower = .625 if selected_only else .05
        uncertain = np.flatnonzero((p >= lower) & (p <= .98))
        logits = np.array([replacement[(ids.s1_id.iat[i], ids.target_id.iat[i])]
                           for i in uncertain], dtype=np.float64)
        found += len(logits)
        base_logit = logit(np.clip(p[uncertain], 1e-6, 1 - 1e-6))
        for alpha_index, alpha in enumerate(ALPHAS):
            adjusted = p.copy()
            if alpha:
                adjusted[uncertain] = expit(base_logit + alpha * logits)
            for threshold_index, threshold in enumerate(THRESHOLDS):
                column = alpha_index * len(THRESHOLDS) + threshold_index
                selected = adjusted >= threshold
                if selected_only:
                    selected &= p >= .625
                predicted[:, column] += np.bincount(
                    group[selected], minlength=len(ordered))
                true_positive[:, column] += np.bincount(
                    group[selected & (y == 1)], minlength=len(ordered))
        total += len(ids)
        if total % 500_000 < len(ids):
            print(mode, "evaluated", total, "pairs", flush=True)
    values = score(truth, predicted, true_positive)
    rows = []
    for j, (alpha, threshold) in enumerate(combos):
        rows.append({"alpha": alpha, "threshold": threshold,
                     "macro_f05": float(values[:, j].mean()),
                     "india_f05": float(values[countries == "india", j].mean()),
                     "us_f05": float(values[countries == "us", j].mean()),
                     "predicted": int(predicted[:, j].sum()),
                     "true_positive": int(true_positive[:, j].sum())})
    baseline = next(row for row in rows if row["alpha"] == 0 and
                    row["threshold"] == .625)
    best = max(rows, key=lambda row: row["macro_f05"])
    result = {"mode": mode, "selected_only": selected_only,
              "entities": len(ordered), "pairs": total,
              "reranked": found, "baseline": baseline, "best": best,
              "grid": rows}
    print(json.dumps({key: value for key, value in result.items()
                      if key != "grid"}, indent=2), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=WINDOWS, required=True)
    parser.add_argument("--selected-only", action="store_true",
                        help="Adjust only pairs retained by V5 inference")
    args = parser.parse_args()
    result = evaluate(args.mode, args.selected_only)
    suffix = "_selected_only" if args.selected_only else ""
    path = WORK / f"v6_reranker_{args.mode}{suffix}_evaluation.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("saved", path, flush=True)


if __name__ == "__main__":
    main()
