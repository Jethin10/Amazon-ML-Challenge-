"""Measure adding cross-encoder-filtered TF-IDF retrieval to full-band V7."""

from __future__ import annotations

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


ALPHA = 0.6
V7_THRESHOLD = 0.625
RETRIEVAL_THRESHOLDS = np.arange(3.0, 7.01, 0.25)


def main() -> None:
    con = duckdb.connect(str(DB), read_only=True)
    ordered = [row[0] for row in con.execute(
        "SELECT entity_id FROM train_s1 WHERE hash(entity_id)%1000>=70 "
        "AND hash(entity_id)%1000<80 ORDER BY entity_id"
    ).fetchall()]
    con.close()
    position = {sid: i for i, sid in enumerate(ordered)}
    truth, countries = metadata(ordered)

    scored = pd.read_parquet(
        WORK / "v6_reranker_audit3_scored.parquet",
        columns=["s1_id", "target_id", "reranker_logit"],
    )
    replacement = {(sid, tid): float(value) for sid, tid, value in zip(
        scored.s1_id, scored.target_id, scored.reranker_logit
    )}
    del scored

    predicted = np.zeros(len(ordered), dtype=np.float64)
    true_positive = np.zeros(len(ordered), dtype=np.float64)
    reader = pq.ParquetFile(WORK / "train_audit3_v5_features.parquet")
    model = lgb.Booster(model_file=str(ROOT / "models" / "v5_lgb.txt"))
    total = 0
    for batch in reader.iter_batches(
        batch_size=50_000,
        columns=["s1_id", "target_id", "label"] + ALL_FEATURE_NAMES,
    ):
        frame = batch.to_pandas()
        probability = model.predict(
            frame[ALL_FEATURE_NAMES].to_numpy(dtype=np.float32),
            num_threads=4,
            pred_early_stop=True,
            pred_early_stop_freq=10,
            pred_early_stop_margin=10,
        ).astype(np.float64)
        uncertain = np.flatnonzero((probability >= 0.05) & (probability <= 0.98))
        logits = np.array([
            replacement[(frame.s1_id.iat[i], frame.target_id.iat[i])]
            for i in uncertain
        ])
        probability[uncertain] = expit(
            logit(np.clip(probability[uncertain], 1e-6, 1 - 1e-6))
            + ALPHA * logits
        )
        selected = probability >= V7_THRESHOLD
        group = np.fromiter((position[sid] for sid in frame.s1_id),
                            dtype=np.int32, count=len(frame))
        label = frame.label.to_numpy(dtype=np.int8)
        predicted += np.bincount(group[selected], minlength=len(ordered))
        true_positive += np.bincount(
            group[selected & (label == 1)], minlength=len(ordered)
        )
        total += len(frame)
        if total % 500_000 < len(frame):
            print("V7 pairs scored", total, flush=True)

    base_values = score(truth, predicted[:, None], true_positive[:, None])[:, 0]
    additions = {}
    for country in ("us", "india"):
        retrieval = pd.read_parquet(
            WORK / f"v6_retrieval_{country}_audit3_scored.parquet"
        )
        retrieval_group = np.fromiter(
            (position[sid] for sid in retrieval.s1_id),
            dtype=np.int32,
            count=len(retrieval),
        )
        retrieval_label = retrieval.label.to_numpy(dtype=np.int8)
        retrieval_logit = retrieval.reranker_logit.to_numpy()
        country_rows = []
        for threshold in RETRIEVAL_THRESHOLDS:
            selected = retrieval_logit >= threshold
            country_rows.append({
                "threshold": float(threshold),
                "predicted": np.bincount(
                    retrieval_group[selected], minlength=len(ordered)
                ),
                "tp": np.bincount(
                    retrieval_group[selected & (retrieval_label == 1)],
                    minlength=len(ordered),
                ),
                "selected": int(selected.sum()),
                "true_positive": int(retrieval_label[selected].sum()),
            })
        additions[country] = country_rows

    rows = []
    for us in additions["us"]:
        for india in additions["india"]:
            values = score(
                truth,
                (predicted + us["predicted"] + india["predicted"])[:, None],
                (true_positive + us["tp"] + india["tp"])[:, None],
            )[:, 0]
            rows.append({
                "us_logit_threshold": us["threshold"],
                "india_logit_threshold": india["threshold"],
                "macro_f05": float(values.mean()),
                "delta": float(values.mean() - base_values.mean()),
                "us_f05": float(values[countries == "us"].mean()),
                "india_f05": float(values[countries == "india"].mean()),
                "selected": us["selected"] + india["selected"],
                "true_positive": us["true_positive"] + india["true_positive"],
            })
    result = {
        "base_v7_macro_f05": float(base_values.mean()),
        "base_v7_us_f05": float(base_values[countries == "us"].mean()),
        "base_v7_india_f05": float(base_values[countries == "india"].mean()),
        "best": max(rows, key=lambda row: row["macro_f05"]),
        "grid": rows,
    }
    target = WORK / "v8_retrieval_audit3_evaluation.json"
    target.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    print("saved", target, flush=True)


if __name__ == "__main__":
    main()
