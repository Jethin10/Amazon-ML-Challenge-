"""Inspect V4 errors on the second audit slice."""

from __future__ import annotations

import json
import csv

import lightgbm as lgb
import numpy as np
import pandas as pd

from v1 import DATASET, ROOT, WORK
from v3_features import ALL_FEATURE_NAMES as FEATURE_NAMES


def main() -> None:
    frame = pd.read_parquet(WORK / "train_audit2_v4_features.parquet")
    model = lgb.Booster(model_file=str(ROOT / "models" / "v4_lgb.txt"))
    x = frame[FEATURE_NAMES].to_numpy(dtype=np.float32, copy=True)
    frame["probability"] = model.predict(x, num_threads=8, pred_early_stop=True,
                                         pred_early_stop_freq=10, pred_early_stop_margin=10)
    bad_positive = frame[(frame.label == 1) & (frame.probability < 0.70)]
    bad_negative = frame[(frame.label == 0) & (frame.probability >= 0.70)]
    picks = pd.concat([
        bad_positive.nlargest(20, "probability"),
        bad_positive.nsmallest(20, "probability"),
        bad_negative.nlargest(40, "probability"),
    ])
    wanted = set(picks.s1_id) | set(picks.target_id)
    records = {}
    for source in ("source1", "source2", "source3"):
        path = DATASET / "train" / f"train_{source}.tsv"
        with path.open(encoding="utf-8", newline="") as handle:
            for record in csv.DictReader(handle, delimiter="\t"):
                if record["entity_id"] in wanted:
                    records[record["entity_id"]] = (
                        record["business_name"], record["business_address"], record["country"]
                    )
    result = {
        "false_negative_candidates": len(bad_positive),
        "false_positive_candidates": len(bad_negative),
        "examples": [
            {"label": int(row.label), "probability": round(float(row.probability), 3),
             "s1_id": row.s1_id, "target_id": row.target_id,
             "s1": records.get(row.s1_id), "target": records.get(row.target_id)}
            for row in picks.itertuples()
        ],
    }
    path = WORK / "v4_error_examples.json"
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(path, len(bad_positive), len(bad_negative))


if __name__ == "__main__":
    main()
