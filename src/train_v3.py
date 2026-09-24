"""Train a pair classifier with transliteration and numeric address features."""

from __future__ import annotations

import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from v2 import ROOT, WORK, connect, make_work_s1
from v3_features import ALL_FEATURE_NAMES as FEATURE_NAMES


MODELS = ROOT / "models"


def load_frame(path: Path, include_ids: bool) -> tuple[np.ndarray, np.ndarray, pd.DataFrame | None]:
    cols = FEATURE_NAMES + ["label"] + (["s1_id", "target_id"] if include_ids else [])
    frame = pd.read_parquet(path, columns=cols)
    ids = frame[["s1_id", "target_id"]].copy() if include_ids else None
    labels = frame.pop("label").to_numpy(dtype=np.int8, copy=True)
    x = frame[FEATURE_NAMES].to_numpy(dtype=np.float32, copy=True)
    del frame
    return x, labels, ids


def score_at_threshold(
    threshold: float,
    probabilities: np.ndarray,
    labels: np.ndarray,
    group_index: np.ndarray,
    truth_count: np.ndarray,
) -> tuple[float, float, float]:
    selected = probabilities >= threshold
    predicted = np.bincount(group_index, weights=selected, minlength=len(truth_count))
    true_positive = np.bincount(
        group_index, weights=selected & (labels == 1), minlength=len(truth_count)
    )
    false_positive = predicted - true_positive
    false_negative = truth_count - true_positive
    denominator = 1.25 * true_positive + false_positive + 0.25 * false_negative
    scores = np.divide(
        1.25 * true_positive,
        denominator,
        out=np.zeros(len(truth_count), dtype=np.float64),
        where=denominator > 0,
    )
    singleton = truth_count == 0
    scores[singleton] = (predicted[singleton] == 0).astype(np.float64)
    return float(scores.mean()), float((predicted[singleton] == 0).mean()), float(predicted.mean())


def main() -> None:
    MODELS.mkdir(exist_ok=True)
    started = time.monotonic()
    x_fit, y_fit, _ = load_frame(WORK / "train_fit_v3_features.parquet", False)
    x_valid, y_valid, ids = load_frame(WORK / "train_valid_v3_features.parquet", True)
    assert ids is not None
    print(
        f"Fit: {len(y_fit):,} pairs, {int(y_fit.sum()):,} positive. "
        f"Valid: {len(y_valid):,} pairs, {int(y_valid.sum()):,} positive.",
        flush=True,
    )
    train_set = lgb.Dataset(x_fit, label=y_fit, feature_name=FEATURE_NAMES, free_raw_data=True)
    valid_set = lgb.Dataset(x_valid, label=y_valid, reference=train_set, free_raw_data=True)
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": 0.06,
        "num_leaves": 31,
        "min_data_in_leaf": 150,
        "feature_fraction": 0.85,
        "verbosity": -1,
        "num_threads": 8,
        "seed": 2026,
    }
    model = lgb.train(
        params,
        train_set,
        num_boost_round=350,
        valid_sets=[valid_set],
        callbacks=[lgb.early_stopping(30), lgb.log_evaluation(50)],
    )
    probabilities = model.predict(x_valid, num_threads=8)
    con = connect()
    make_work_s1(con, "train", "valid")
    valid_ids = [r[0] for r in con.execute("SELECT entity_id FROM work_s1").fetchall()]
    truth = dict(
        con.execute(
            """
            SELECT g.s1_id, count(*)
            FROM train_truth g JOIN work_s1 s ON g.s1_id=s.entity_id
            GROUP BY g.s1_id
            """
        ).fetchall()
    )
    con.close()
    position = {entity_id: i for i, entity_id in enumerate(valid_ids)}
    group_index = ids["s1_id"].map(position).to_numpy(dtype=np.int32)
    truth_count = np.array([truth.get(entity_id, 0) for entity_id in valid_ids])
    thresholds = np.r_[np.linspace(0.02, 0.98, 49), 0.99, 0.995, 0.999]
    results = [
        (float(th),) + score_at_threshold(th, probabilities, y_valid, group_index, truth_count)
        for th in thresholds
    ]
    best = max(results, key=lambda row: row[1])
    model.save_model(str(MODELS / "v3_lgb.txt"))
    report = {
        "fit_pairs": int(len(y_fit)),
        "fit_positive_pairs": int(y_fit.sum()),
        "valid_pairs": int(len(y_valid)),
        "valid_positive_candidates": int(y_valid.sum()),
        "valid_true_pairs": int(truth_count.sum()),
        "valid_entities": len(valid_ids),
        "valid_singletons": int((truth_count == 0).sum()),
        "candidate_recall": float(y_valid.sum() / truth_count.sum()),
        "best_iteration": model.best_iteration,
        "threshold": best[0],
        "macro_f05": best[1],
        "singleton_accuracy": best[2],
        "predicted_matches_per_entity": best[3],
        "threshold_grid": [
            {"threshold": a, "macro_f05": b, "singleton_accuracy": c, "matches_per_entity": d}
            for a, b, c, d in results
        ],
        "feature_importance_gain": dict(
            sorted(
                zip(FEATURE_NAMES, map(float, model.feature_importance("gain"))),
                key=lambda item: -item[1],
            )
        ),
        "elapsed_seconds": time.monotonic() - started,
    }
    (MODELS / "v3_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "threshold_grid"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
