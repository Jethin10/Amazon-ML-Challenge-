"""Add candidate-list competition and name ambiguity to V4 pair features."""

from __future__ import annotations

import argparse
import time

from v1 import WORK, connect, quoted_path
from v3_features import ALL_FEATURE_NAMES as V4_FEATURE_NAMES


EXTRA_FEATURE_NAMES = [
    "s1_name_frequency", "target_name_frequency", "candidate_count",
    "name_rank", "address_rank", "name_gap", "address_gap",
    "strong_name_count", "strong_address_count", "joint_similarity",
]
ALL_FEATURE_NAMES = V4_FEATURE_NAMES + EXTRA_FEATURE_NAMES


def enrich(part: str, mode: str) -> None:
    con=connect()
    started=time.monotonic()
    con.execute(f"""
        CREATE TABLE IF NOT EXISTS {part}_target_name_frequency AS
        SELECT country,name_key,count(*) AS n
        FROM {part}_targets WHERE name_key<>''
        GROUP BY country,name_key
    """)
    source=WORK/f"{part}_{mode}_v4_features.parquet"
    target=WORK/f"{part}_{mode}_v5_features.parquet"
    con.execute(f"""
        COPY (
            WITH augmented AS (
                SELECT f.*,
                       ln(1+coalesce(sf.n,0))::FLOAT AS s1_name_frequency,
                       ln(1+coalesce(tf.n,0))::FLOAT AS target_name_frequency,
                       least(f.latin_name_sorted_jaro,
                             f.latin_address_sorted_jaro)::FLOAT AS joint_similarity
                FROM read_parquet({quoted_path(source)}) f
                JOIN {part}_s1 s ON f.s1_id=s.entity_id
                JOIN {part}_targets t ON f.target_id=t.entity_id
                LEFT JOIN {part}_target_name_frequency sf
                  ON s.country=sf.country AND s.name_key=sf.name_key
                LEFT JOIN {part}_target_name_frequency tf
                  ON t.country=tf.country AND t.name_key=tf.name_key
            ), contextual AS (
                SELECT *,
                       count(*) OVER(PARTITION BY s1_id)::FLOAT AS candidate_count,
                       row_number() OVER(
                           PARTITION BY s1_id
                           ORDER BY latin_name_sorted_jaro DESC,target_id
                       )::FLOAT AS name_rank,
                       row_number() OVER(
                           PARTITION BY s1_id
                           ORDER BY latin_address_sorted_jaro DESC,target_id
                       )::FLOAT AS address_rank,
                       (max(latin_name_sorted_jaro) OVER(PARTITION BY s1_id)
                        -latin_name_sorted_jaro)::FLOAT AS name_gap,
                       (max(latin_address_sorted_jaro) OVER(PARTITION BY s1_id)
                        -latin_address_sorted_jaro)::FLOAT AS address_gap,
                       sum((latin_name_sorted_jaro>=0.90)::INT)
                           OVER(PARTITION BY s1_id)::FLOAT AS strong_name_count,
                       sum((latin_address_sorted_jaro>=0.90)::INT)
                           OVER(PARTITION BY s1_id)::FLOAT AS strong_address_count
                FROM augmented
            )
            SELECT * FROM contextual
        ) TO {quoted_path(target)} (FORMAT PARQUET,COMPRESSION SNAPPY)
    """)
    print(target.name,con.execute(f"SELECT count(*) FROM read_parquet({quoted_path(target)})").fetchone()[0],
          round(time.monotonic()-started,1),flush=True)
    con.close()


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--part",choices=["train","test"],default="train")
    parser.add_argument("--mode",required=True)
    args=parser.parse_args()
    enrich(args.part,args.mode)
