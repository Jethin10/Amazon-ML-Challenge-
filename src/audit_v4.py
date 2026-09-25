"""Score the frozen model and threshold on an untouched Source 1 holdout."""

from __future__ import annotations

import argparse
import json

import lightgbm as lgb
import numpy as np
import pandas as pd

from v4 import ROOT, WORK, connect, make_work_s1
from v3_features import ALL_FEATURE_NAMES as FEATURE_NAMES


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--early-stop", action="store_true")
    parser.add_argument("--mode", choices=["audit", "audit2", "audit3"], default="audit2")
    args = parser.parse_args()
    con = connect()
    make_work_s1(con, "train", args.mode)
    rows = con.execute("SELECT entity_id, country FROM work_s1 ORDER BY entity_id").fetchall()
    counts = dict(con.execute("""
        SELECT g.s1_id, count(*)
        FROM train_truth g JOIN work_s1 s ON g.s1_id=s.entity_id
        GROUP BY g.s1_id
    """).fetchall())
    con.close()
    ids = [row[0] for row in rows]
    country = np.array([row[1] for row in rows])
    truth = np.array([counts.get(entity_id, 0) for entity_id in ids])
    position = {entity_id: i for i, entity_id in enumerate(ids)}
    frame = pd.read_parquet(WORK / f"train_{args.mode}_v4_features.parquet")
    model = lgb.Booster(model_file=str(ROOT / "models" / "v4_lgb.txt"))
    threshold = json.loads((ROOT / "models" / "v4_metrics.json").read_text())["threshold"]
    x = frame[FEATURE_NAMES].to_numpy(dtype=np.float32, copy=True)
    predict_options = ({"pred_early_stop": True, "pred_early_stop_freq": 10,
                        "pred_early_stop_margin": 10} if args.early_stop else {})
    probability = model.predict(x, num_threads=8, **predict_options)
    group = frame["s1_id"].map(position).to_numpy(dtype=np.int32)
    labels = frame["label"].to_numpy(dtype=np.int8)
    selected = probability >= threshold
    predicted = np.bincount(group, weights=selected, minlength=len(ids))
    true_positive = np.bincount(group, weights=selected & (labels == 1), minlength=len(ids))
    false_positive = predicted - true_positive
    false_negative = truth - true_positive
    denominator = 1.25 * true_positive + false_positive + 0.25 * false_negative
    score = np.divide(1.25 * true_positive, denominator,
                      out=np.zeros(len(ids)), where=denominator > 0)
    singleton = truth == 0
    score[singleton] = (predicted[singleton] == 0).astype(float)
    report = {
        "entities": len(ids),
        "true_links": int(truth.sum()),
        "candidate_pairs": len(frame),
        "true_links_retrieved": int(labels.sum()),
        "candidate_recall": float(labels.sum() / truth.sum()),
        "candidates_per_entity": float(len(frame) / len(ids)),
        "threshold": threshold,
        "prediction_early_stop": args.early_stop,
        "macro_f05": float(score.mean()),
        "singletons": int(singleton.sum()),
        "correct_singletons": int((predicted[singleton] == 0).sum()),
        "by_country": {},
    }
    for value in sorted(set(country)):
        mask = country == value
        report["by_country"][value] = {
            "entities": int(mask.sum()),
            "macro_f05": float(score[mask].mean()),
            "true_links": int(truth[mask].sum()),
            "candidate_recall": float(labels[country[group] == value].sum() / truth[mask].sum()),
        }
    path = ROOT / "models" / (f"v4_{args.mode}_early_metrics.json" if args.early_stop
                             else f"v4_{args.mode}_metrics.json")
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
