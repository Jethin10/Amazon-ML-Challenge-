"""Measure V7 ranking quality under entity-level list decoding."""

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


def main() -> None:
    con = duckdb.connect(str(DB), read_only=True)
    ordered = [row[0] for row in con.execute(
        "SELECT entity_id FROM train_s1 WHERE hash(entity_id)%1000>=70 "
        "AND hash(entity_id)%1000<80 ORDER BY entity_id"
    ).fetchall()]
    con.close()
    position = {sid: i for i, sid in enumerate(ordered)}
    truth, countries = metadata(ordered)

    reranked = pd.read_parquet(
        WORK / "v6_reranker_audit3_scored.parquet",
        columns=["s1_id", "target_id", "reranker_logit"],
    )
    replacement = {(sid, tid): float(value) for sid, tid, value in zip(
        reranked.s1_id, reranked.target_id, reranked.reranker_logit
    )}
    del reranked

    model = lgb.Booster(model_file=str(ROOT / "models" / "v5_lgb.txt"))
    reader = pq.ParquetFile(WORK / "train_audit3_v5_features.parquet")
    parts = []
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
            + 0.6 * logits
        )
        parts.append(pd.DataFrame({
            "group": np.fromiter((position[sid] for sid in frame.s1_id),
                                 dtype=np.int32, count=len(frame)),
            "target_id": frame.target_id.to_numpy(),
            "label": frame.label.to_numpy(dtype=np.int8),
            "probability": probability.astype(np.float32),
        }))
        total += len(frame)
        if total % 500_000 < len(frame):
            print("pairs scored", total, flush=True)

    pairs = pd.concat(parts, ignore_index=True)
    del parts
    pairs.sort_values(["group", "probability"], ascending=[True, False],
                      kind="stable", inplace=True)
    pairs.reset_index(drop=True, inplace=True)
    pairs["rank"] = pairs.groupby("group", sort=False).cumcount().to_numpy() + 1
    group = pairs.group.to_numpy(dtype=np.int32)
    label = pairs.label.to_numpy(dtype=np.int8)
    probability = pairs.probability.to_numpy(dtype=np.float32)
    rank = pairs["rank"].to_numpy(dtype=np.int32)

    rows = []

    def evaluate(name: str, selected: np.ndarray) -> None:
        predicted = np.bincount(
            group[selected], minlength=len(ordered)).astype(np.float64)
        tp = np.bincount(
            group[selected & (label == 1)],
            minlength=len(ordered)).astype(np.float64)
        values = score(truth, predicted[:, None], tp[:, None])[:, 0]
        rows.append({
            "decoder": name,
            "macro_f05": float(values.mean()),
            "india_f05": float(values[countries == "india"].mean()),
            "us_f05": float(values[countries == "us"].mean()),
            "selected": int(selected.sum()),
            "true_positive": int(label[selected].sum()),
        })

    evaluate("threshold_0.625", probability >= 0.625)
    eligible = pairs.loc[probability >= 0.625]
    winners = eligible.sort_values(
        ["target_id", "probability", "group"],
        ascending=[True, False, True], kind="stable",
    ).drop_duplicates("target_id", keep="first")
    exclusive = np.zeros(len(pairs), dtype=bool)
    exclusive[winners.index.to_numpy()] = True
    evaluate("threshold_0.625_target_exclusive", exclusive)
    evaluate("oracle_true_count", rank <= truth[group])
    for cap in range(1, 11):
        evaluate(f"threshold_0.625_cap_{cap}",
                 (probability >= 0.625) & (rank <= cap))

    result = {"pairs": total, "rows": rows}
    target = WORK / "v7_list_decoding_probe.json"
    target.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    print("saved", target, flush=True)


if __name__ == "__main__":
    main()
