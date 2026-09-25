"""Measure whether limiting long prediction lists helps macro F0.5."""

from __future__ import annotations

import json

import lightgbm as lgb
import numpy as np
import pandas as pd

from v4 import ROOT, WORK, connect, make_work_s1
from v3_features import ALL_FEATURE_NAMES as FEATURES


def main() -> None:
    con = connect()
    make_work_s1(con, "train", "audit2")
    ids = [row[0] for row in con.execute("SELECT entity_id FROM work_s1 ORDER BY entity_id").fetchall()]
    truth = dict(con.execute("""
        SELECT g.s1_id, count(*)
        FROM train_truth g JOIN work_s1 s ON g.s1_id=s.entity_id
        GROUP BY g.s1_id
    """).fetchall())
    con.close()
    truth_count = np.array([truth.get(entity_id, 0) for entity_id in ids])
    position = {entity_id: i for i, entity_id in enumerate(ids)}
    frame = pd.read_parquet(WORK / "train_audit2_v4_features.parquet")
    model = lgb.Booster(model_file=str(ROOT / "models" / "v4_lgb.txt"))
    threshold = json.loads((ROOT / "models" / "v4_metrics.json").read_text())["threshold"]
    x = frame[FEATURES].to_numpy(dtype=np.float32, copy=True)
    probability = model.predict(x, num_threads=8, pred_early_stop=True,
                                pred_early_stop_freq=10, pred_early_stop_margin=10)
    selected = frame.loc[probability >= threshold, ["s1_id", "label"]].copy()
    selected["probability"] = probability[probability >= threshold]
    selected["order"] = selected.groupby("s1_id", sort=False).cumcount()
    selected["group"] = selected["s1_id"].map(position)

    def score(chosen: pd.DataFrame) -> tuple[float, int]:
        group = chosen["group"].to_numpy(dtype=np.int32)
        labels = chosen["label"].to_numpy(dtype=np.int8)
        predicted = np.bincount(group, minlength=len(ids))
        tp = np.bincount(group, weights=labels, minlength=len(ids))
        fp = predicted - tp
        fn = truth_count - tp
        denominator = 1.25 * tp + fp + 0.25 * fn
        f = np.divide(1.25 * tp, denominator, out=np.zeros(len(ids)), where=denominator > 0)
        singleton = truth_count == 0
        f[singleton] = (predicted[singleton] == 0).astype(float)
        return float(f.mean()), len(chosen)

    print("uncapped", score(selected), flush=True)
    for cap in (5, 8, 10, 12, 15):
        print(f"first_{cap}", score(selected[selected['order'] < cap]), flush=True)
    ranked = selected.sort_values(["s1_id", "probability"], ascending=[True, False])
    ranked["rank"] = ranked.groupby("s1_id", sort=False).cumcount()
    for cap in (5, 8, 10, 12, 15):
        print(f"top_score_{cap}", score(ranked[ranked['rank'] < cap]), flush=True)


if __name__ == "__main__":
    main()
