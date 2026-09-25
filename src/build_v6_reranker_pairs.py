"""Build a bounded, labeled text-pair pilot from frozen V5 predictions."""

from __future__ import annotations

import argparse
import time

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from v1 import DB, ROOT, WORK, quoted_path
from v5_features import ALL_FEATURE_NAMES


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["valid", "audit2", "audit3"],
                        required=True)
    args = parser.parse_args()
    mode = args.mode
    started = time.monotonic()
    ids_file = pq.ParquetFile(WORK / f"train_{mode}_v5_features.parquet")
    model = lgb.Booster(model_file=str(ROOT / "models" / "v5_lgb.txt"))
    ids_batches = ids_file.iter_batches(
        batch_size=50_000,
        columns=["s1_id", "target_id", "label"] + ALL_FEATURE_NAMES)
    rng = np.random.default_rng(20260926)
    chunks = []
    for batch in ids_batches:
        ids = batch.to_pandas()
        p = model.predict(ids[ALL_FEATURE_NAMES].to_numpy(dtype=np.float32),
                          num_threads=4, pred_early_stop=True,
                          pred_early_stop_freq=10,
                          pred_early_stop_margin=10).astype(np.float32)
        y = ids.label.to_numpy(dtype=np.int8)
        if mode == "valid":
            keep = ((y == 1) & ((p < .95) | (rng.random(len(y)) < .3))) | \
                   ((y == 0) & ((p >= .02) | (rng.random(len(y)) < .002)))
        else:
            keep = ((p >= .05) & (p <= .98)) | \
                   ((p > .98) & (rng.random(len(y)) < .1))
        chunk = ids.loc[keep, ["s1_id", "target_id", "label"]].copy()
        chunk["v5_probability"] = p[keep]
        chunks.append(chunk)
    pairs = pd.concat(chunks, ignore_index=True)
    print("selected", mode, len(pairs), "positive", int(pairs.label.sum()),
          "seconds", round(time.monotonic() - started, 1), flush=True)
    con = duckdb.connect(str(DB), read_only=True)
    con.execute("SET threads=4")
    con.execute("SET memory_limit='2GB'")
    con.execute(f"SET temp_directory={quoted_path(WORK / 'tmp')}")
    con.register("selected", pairs)
    output = WORK / f"v6_reranker_{mode}.parquet"
    con.execute(f"""
        COPY (
            SELECT p.s1_id,p.target_id,p.label,p.v5_probability,
                   s.country,
                   s.business_name AS source_name,
                   s.business_address AS source_address,
                   t.business_name AS target_name,
                   t.business_address AS target_address
            FROM selected p
            JOIN train_s1 s ON p.s1_id=s.entity_id
            JOIN train_targets t ON p.target_id=t.entity_id
        ) TO {quoted_path(output)} (FORMAT PARQUET, COMPRESSION SNAPPY)
    """)
    count = con.execute(
        f"SELECT count(*) FROM read_parquet({quoted_path(output)})").fetchone()[0]
    if count != len(pairs):
        raise ValueError(f"Lost pairs during text join: {count} != {len(pairs)}")
    print("wrote", output, "rows", count,
          "seconds", round(time.monotonic() - started, 1), flush=True)
    con.close()


if __name__ == "__main__":
    main()
