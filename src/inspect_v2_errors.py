"""Inspect high-confidence V2 errors on the untouched audit slice."""

from __future__ import annotations

import json

import lightgbm as lgb
import numpy as np
import pandas as pd

from v2 import FEATURE_NAMES, ROOT, WORK, connect


def main() -> None:
    frame = pd.read_parquet(WORK / "train_audit_features.parquet")
    model = lgb.Booster(model_file=str(ROOT / "models" / "v2_lgb.txt"))
    x = frame[FEATURE_NAMES].to_numpy(dtype=np.float32, copy=True)
    frame["probability"] = model.predict(x, num_threads=8, pred_early_stop=True,
                                         pred_early_stop_freq=10, pred_early_stop_margin=10)
    bad_positive = frame[(frame.label == 1) & (frame.probability < 0.56)]
    bad_negative = frame[(frame.label == 0) & (frame.probability >= 0.56)]
    picks = pd.concat([
        bad_positive.nlargest(20, "probability"),
        bad_positive.nsmallest(20, "probability"),
        bad_negative.nlargest(40, "probability"),
    ])
    wanted = set(picks.s1_id) | set(picks.target_id)
    con = connect()
    records = {}
    for table in ("train_s1", "train_targets"):
        rows = con.execute(
            f"SELECT entity_id, business_name, business_address, country FROM {table} "
            "WHERE entity_id IN (SELECT unnest(?))", [list(wanted)]
        ).fetchall()
        records.update({row[0]: row[1:] for row in rows})
    con.close()
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
    path = WORK / "v2_error_examples.json"
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(path, len(bad_positive), len(bad_negative))


if __name__ == "__main__":
    main()
