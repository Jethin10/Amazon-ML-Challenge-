"""Compare model iteration counts and LightGBM early prediction stopping."""

from __future__ import annotations

import json
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from train_v1 import score_at_threshold
from v1 import FEATURE_NAMES, ROOT, WORK, connect


def main() -> None:
    frame = pd.read_parquet(WORK / "train_valid_features.parquet")
    x = frame[FEATURE_NAMES].to_numpy(dtype=np.float32, copy=True)
    y = frame["label"].to_numpy(dtype=np.int8)
    con = connect()
    ids = [r[0] for r in con.execute("SELECT entity_id FROM work_s1").fetchall()]
    truth = dict(con.execute("""
        SELECT g.s1_id, count(*) FROM train_truth g JOIN work_s1 s
          ON g.s1_id=s.entity_id GROUP BY g.s1_id
    """).fetchall())
    con.close()
    position = {value: i for i, value in enumerate(ids)}
    group = frame["s1_id"].map(position).to_numpy(dtype=np.int32)
    truth_count = np.array([truth.get(value, 0) for value in ids])
    model = lgb.Booster(model_file=str(ROOT / "models" / "v1_lgb.txt"))
    threshold = json.loads((ROOT / "models" / "v1_metrics.json").read_text())["threshold"]
    for name, kwargs in (
        ("full", {}),
        ("250", {"num_iteration": 250}),
        ("200", {"num_iteration": 200}),
        ("early10", {"pred_early_stop": True,
                     "pred_early_stop_freq": 10, "pred_early_stop_margin": 10}),
    ):
        tick = time.monotonic()
        probs = model.predict(x, num_threads=8, **kwargs)
        elapsed = time.monotonic() - tick
        fixed = score_at_threshold(threshold, probs, y, group, truth_count)[0]
        best = max(
            (score_at_threshold(float(t), probs, y, group, truth_count)[0], float(t))
            for t in np.arange(0.30, 0.81, 0.02)
        )
        print(name, "seconds", round(elapsed, 1), "fixed F0.5", round(fixed, 6),
              "best F0.5", round(best[0], 6), "threshold", round(best[1], 2),
              flush=True)


if __name__ == "__main__":
    main()
