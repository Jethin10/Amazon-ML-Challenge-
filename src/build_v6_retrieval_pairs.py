"""Prepare novel retrieval pairs for an audit3 reranker precision probe."""

from __future__ import annotations

import argparse
import time

import duckdb

from v1 import DB, WORK, quoted_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--country", choices=["india", "us"], required=True)
    args = parser.parse_args()
    input_path = WORK / f"char_tfidf_{args.country}_address_latin_audit3.parquet"
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    started = time.monotonic()
    con = duckdb.connect(str(DB), read_only=True)
    con.execute("SET threads=4")
    con.execute("SET memory_limit='2GB'")
    con.execute(f"SET temp_directory={quoted_path(WORK / 'tmp')}")
    existing = WORK / "train_audit3_v5_features.parquet"
    output = WORK / f"v6_retrieval_{args.country}_audit3.parquet"
    con.execute(f"""
        COPY (
            WITH novel AS (
                SELECT r.s1_id,r.target_id,r.similarity
                FROM read_parquet({quoted_path(input_path)}) r
                ANTI JOIN read_parquet({quoted_path(existing)}) e
                  USING(s1_id,target_id)
            )
            SELECT n.s1_id,n.target_id,
                   (g.target_id IS NOT NULL)::INTEGER AS label,
                   n.similarity,
                   s.country,
                   s.business_name AS source_name,
                   s.business_address AS source_address,
                   t.business_name AS target_name,
                   t.business_address AS target_address
            FROM novel n
            JOIN train_s1 s ON n.s1_id=s.entity_id
            JOIN train_targets t ON n.target_id=t.entity_id
            LEFT JOIN train_truth g USING(s1_id,target_id)
        ) TO {quoted_path(output)} (FORMAT PARQUET, COMPRESSION SNAPPY)
    """)
    count, positives = con.execute(f"""
        SELECT count(*),sum(label) FROM read_parquet({quoted_path(output)})
    """).fetchone()
    print(args.country, "novel pairs", count, "true", positives,
          "seconds", round(time.monotonic() - started, 1),
          "file", output, flush=True)
    con.close()


if __name__ == "__main__":
    main()
