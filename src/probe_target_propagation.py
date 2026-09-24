"""Measure high-confidence target-to-target expansion on the audit split."""

from __future__ import annotations

import argparse
import time

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd
from anyascii import anyascii
from duckdb.sqltypes import VARCHAR

from v1 import DATASET, ROOT, WORK, quoted_path
from v3_features import ALL_FEATURE_NAMES


def prepare_anchors(cutoff: float) -> pd.DataFrame:
    frame = pd.read_parquet(WORK / "train_audit_v3_features.parquet",
                            columns=["s1_id", "target_id", "label"] + ALL_FEATURE_NAMES)
    model = lgb.Booster(model_file=str(ROOT / "models" / "v3_lgb.txt"))
    x = frame[ALL_FEATURE_NAMES].to_numpy(dtype=np.float32, copy=True)
    probability = model.predict(x, num_threads=8, pred_early_stop=True,
                                pred_early_stop_freq=10, pred_early_stop_margin=10)
    chosen = frame.loc[probability >= cutoff, ["s1_id", "target_id", "label"]].copy()
    print("anchors", len(chosen), "true", int(chosen.label.sum()), flush=True)
    return chosen


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cutoff", type=float, default=0.90)
    args = parser.parse_args()
    anchors = prepare_anchors(args.cutoff)
    db = WORK / "propagation_probe.duckdb"
    (WORK / "propagation_tmp").mkdir(exist_ok=True)
    con = duckdb.connect(str(db))
    con.execute("SET threads=4")
    con.execute("SET memory_limit='4GB'")
    con.execute(f"SET temp_directory={quoted_path(WORK / 'propagation_tmp')}")
    con.create_function("romanize", anyascii, [VARCHAR], VARCHAR)
    try:
        if not con.execute("SELECT count(*) FROM information_schema.tables WHERE table_name='target_keys'").fetchone()[0]:
            tick = time.monotonic()
            con.execute(f"""
                CREATE TABLE target_keys AS
                WITH raw AS (
                    SELECT entity_id, country, business_name, business_address
                    FROM read_csv({quoted_path(DATASET / 'train' / 'train_source2.tsv')},
                                  delim='\\t', header=true, all_varchar=true)
                    UNION ALL
                    SELECT entity_id, country, business_name, business_address
                    FROM read_csv({quoted_path(DATASET / 'train' / 'train_source3.tsv')},
                                  delim='\\t', header=true, all_varchar=true)
                ), norm AS (
                    SELECT entity_id, lower(trim(country)) AS country,
                           regexp_replace(lower(romanize(coalesce(business_name,''))),
                                          '[^a-z0-9]+', '', 'g') AS name_key,
                           regexp_replace(lower(romanize(coalesce(business_address,''))),
                                          '[^a-z0-9]+', '', 'g') AS addr_key
                    FROM raw
                )
                SELECT entity_id, country, name_key, left(name_key,5) AS name5,
                       addr_key, left(addr_key,8) AS addr8
                FROM norm
            """)
            print("target keys", con.execute("SELECT count(*) FROM target_keys").fetchone()[0],
                  round(time.monotonic()-tick, 1), flush=True)
        con.register("anchor_frame", anchors)
        con.execute("CREATE OR REPLACE TABLE anchors AS SELECT * FROM anchor_frame")
        con.unregister("anchor_frame")
        con.execute(f"""
            CREATE OR REPLACE TABLE existing AS
            SELECT s1_id, target_id
            FROM read_parquet({quoted_path(WORK / 'train_audit_features.parquet')})
        """)
        con.execute(f"""
            CREATE OR REPLACE TABLE truth AS
            SELECT source1_entity_id AS s1_id,
                   unnest(string_split(matched_entity_ids, ',')) AS target_id
            FROM read_csv({quoted_path(DATASET / 'train' / 'train_ground_truth.tsv')},
                          delim='\\t', header=true, all_varchar=true)
            WHERE matched_entity_ids IS NOT NULL AND matched_entity_ids<>''
        """)
        for field, cap in (("name_key", None), ("name5", 100), ("name5", 300),
                           ("addr8", 100), ("addr8", 300)):
            tick = time.monotonic()
            stats_join = ""
            if cap is not None:
                stats_join = f"""JOIN (
                    SELECT country, {field}, count(*) AS n
                    FROM target_keys GROUP BY country, {field}
                    HAVING count(*)<={cap}
                ) b ON b.country=t.country AND b.{field}=t.{field}"""
            con.execute("DROP TABLE IF EXISTS extra")
            con.execute(f"""
                CREATE TABLE extra AS
                SELECT DISTINCT a.s1_id, t.entity_id AS target_id
                FROM anchors a JOIN target_keys anchor ON a.target_id=anchor.entity_id
                JOIN target_keys t ON anchor.country=t.country
                                  AND anchor.{field}=t.{field}
                {stats_join}
                WHERE anchor.{field}<>''
            """)
            all_pairs, new_pairs, new_true = con.execute("""
                SELECT count(*), count(*) FILTER (WHERE e.target_id IS NULL),
                       count(*) FILTER (WHERE e.target_id IS NULL AND g.target_id IS NOT NULL)
                FROM extra x
                LEFT JOIN existing e USING (s1_id,target_id)
                LEFT JOIN truth g USING (s1_id,target_id)
            """).fetchone()
            print(field, cap, all_pairs, new_pairs, new_true,
                  round(time.monotonic()-tick, 1), flush=True)
    finally:
        con.close()


if __name__ == "__main__":
    main()
