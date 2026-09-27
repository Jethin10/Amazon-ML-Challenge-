"""Evaluate frozen V5 thresholds with bounded memory and entity-level F0.5."""

from __future__ import annotations

import argparse
import csv
import json

import lightgbm as lgb
import numpy as np
import pyarrow.parquet as pq
import duckdb

from v1 import DATASET, DB, ROOT, WORK
from v5_features import ALL_FEATURE_NAMES


THRESHOLDS = np.round(np.arange(0.4, 0.991, 0.025), 3)


def metadata(ids: list[str]) -> tuple[np.ndarray, np.ndarray]:
    wanted = set(ids)
    truth: dict[str, int] = {}
    countries: dict[str, str] = {}
    with (DATASET / "train" / "train_ground_truth.tsv").open(
        encoding="utf-8", newline=""
    ) as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            sid = row["source1_entity_id"]
            if sid in wanted:
                value = row["matched_entity_ids"]
                truth[sid] = 0 if not value else value.count(",") + 1
    with (DATASET / "train" / "train_source1.tsv").open(
        encoding="utf-8", newline=""
    ) as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            sid = row["entity_id"]
            if sid in wanted:
                countries[sid] = row["country"].strip().lower()
    if len(truth) != len(ids) or len(countries) != len(ids):
        raise ValueError("Missing truth or country metadata for candidate entity")
    return (np.array([truth[sid] for sid in ids], dtype=np.int32),
            np.array([countries[sid] for sid in ids]))


def score(truth: np.ndarray, predicted: np.ndarray, tp: np.ndarray) -> np.ndarray:
    fp = predicted - tp
    fn = truth[:, None] - tp
    denominator = 1.25 * tp + fp + 0.25 * fn
    result = np.divide(1.25 * tp, denominator, out=np.zeros_like(tp),
                       where=denominator > 0)
    singleton = truth == 0
    result[singleton] = (predicted[singleton] == 0).astype(float)
    return result


def evaluate(mode: str, model: lgb.Booster) -> dict:
    file = WORK / f"train_{mode}_v5_features.parquet"
    columns = ["s1_id", "label"] + ALL_FEATURE_NAMES
    reader = pq.ParquetFile(file)
    # Read every Source 1 entity in the split, including zero-candidate rows.
    windows = {"valid": (42, 50), "audit2": (60, 70),
               "audit3": (70, 80)}
    lo, hi = windows[mode]
    con = duckdb.connect(str(DB), read_only=True)
    try:
        ordered = [row[0] for row in con.execute(
            "SELECT entity_id FROM train_s1 WHERE hash(entity_id)%1000>=? "
            "AND hash(entity_id)%1000<? ORDER BY entity_id", [lo, hi]
        ).fetchall()]
    finally:
        con.close()
    position = {sid: i for i, sid in enumerate(ordered)}
    truth, country = metadata(ordered)
    predicted = np.zeros((len(ordered), len(THRESHOLDS)), dtype=np.float64)
    tp = np.zeros_like(predicted)
    pairs = 0
    positives = 0
    for batch in reader.iter_batches(batch_size=50_000, columns=columns):
        frame = batch.to_pandas()
        group = np.fromiter((position[sid] for sid in frame.s1_id),
                            dtype=np.int32, count=len(frame))
        label = frame.label.to_numpy(dtype=np.int8)
        features = frame[ALL_FEATURE_NAMES].to_numpy(dtype=np.float32)
        probability = model.predict(features, num_threads=4,
                                    pred_early_stop=True,
                                    pred_early_stop_freq=10,
                                    pred_early_stop_margin=10)
        for j, threshold in enumerate(THRESHOLDS):
            selected = probability >= threshold
            predicted[:, j] += np.bincount(group[selected], minlength=len(ordered))
            tp[:, j] += np.bincount(group[selected & (label == 1)],
                                    minlength=len(ordered))
        pairs += len(frame)
        positives += int(label.sum())
        if pairs % 500_000 < len(frame):
            print(f"{mode}: {pairs:,} pairs scored", flush=True)
    values = score(truth, predicted, tp)
    result = {"mode": mode, "entities_with_candidates": len(ordered),
              "pairs": pairs, "positives": positives,
              "truth_links": int(truth.sum()),
              "candidate_recall": float(positives / truth.sum()),
              "thresholds": []}
    for j, threshold in enumerate(THRESHOLDS):
        item = {"threshold": float(threshold),
                "macro_f05": float(values[:, j].mean()),
                "selected": int(predicted[:, j].sum()),
                "true_positive": int(tp[:, j].sum()),
                "by_country": {name: float(values[country == name, j].mean())
                               for name in sorted(set(country))}}
        result["thresholds"].append(item)
    print(json.dumps({"mode": mode, "entities": len(ordered),
                      "baseline": next(row for row in result["thresholds"]
                                       if row["threshold"] == 0.625),
                      "best_global": max(result["thresholds"],
                                         key=lambda row: row["macro_f05"])},
                     indent=2), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--modes", nargs="+", choices=["valid", "audit2", "audit3"],
                        default=["valid", "audit3"])
    args = parser.parse_args()
    model = lgb.Booster(model_file=str(ROOT / "models" / "v5_lgb.txt"))
    reports = [evaluate(mode, model) for mode in args.modes]
    path = WORK / "v5_threshold_streaming.json"
    path.write_text(json.dumps(reports, indent=2), encoding="utf-8")
    print(path, flush=True)


if __name__ == "__main__":
    main()
