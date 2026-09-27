"""Build France candidates exposed by removing French legal forms.

The original normalizer removes US and Indian legal suffixes but leaves SARL,
SAS, EURL, SASU, SCI and SA in the key.  Source 1 carries these forms much
more often than Sources 2/3, so exact-name blocking misses asymmetric pairs.
This stage creates only those asymmetric pairs and bounds common-name blocks.
"""

from __future__ import annotations

import argparse
import time

import duckdb

from v1 import DB, WORK, quoted_path


FR_FORMS = "sarl|sas|eurl|sasu|sci|sa"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-block", type=int, default=50)
    args = parser.parse_args()
    output = WORK / "france_suffix_pairs.parquet"
    started = time.monotonic()
    con = duckdb.connect(str(DB), read_only=True)
    con.execute("SET threads=4")
    con.execute("SET memory_limit='3GB'")
    con.execute(f"SET temp_directory={quoted_path(WORK / 'tmp')}")
    con.execute(f"""
        COPY (
            WITH src AS (
                SELECT entity_id AS s1_id,business_name,business_address,
                       name_key AS old_key,
                       regexp_replace(regexp_replace(name_norm,
                           '\\b({FR_FORMS})\\b',' ','g'),
                           '[^a-z0-9]+','','g') AS clean_key
                FROM test_s1 WHERE country='france'
            ), dst AS (
                SELECT entity_id AS target_id,business_name,business_address,
                       name_key AS old_key,
                       regexp_replace(regexp_replace(name_norm,
                           '\\b({FR_FORMS})\\b',' ','g'),
                           '[^a-z0-9]+','','g') AS clean_key
                FROM test_targets WHERE country='france'
            ), blocks AS (
                SELECT clean_key,count(*) AS n FROM dst
                WHERE length(clean_key)>=4 GROUP BY clean_key
                HAVING count(*)<={args.max_block}
            )
            SELECT s.s1_id,d.target_id,0::INTEGER AS label,
                   s.business_name AS source_name,
                   s.business_address AS source_address,
                   d.business_name AS target_name,
                   d.business_address AS target_address
            FROM src s JOIN blocks b USING(clean_key)
            JOIN dst d USING(clean_key)
            WHERE s.old_key<>d.old_key
        ) TO {quoted_path(output)} (FORMAT PARQUET,COMPRESSION SNAPPY)
    """)
    pairs, sources = con.execute(f"""
        SELECT count(*),count(DISTINCT s1_id)
        FROM read_parquet({quoted_path(output)})
    """).fetchone()
    print("France suffix candidates", pairs, "Source 1 rows", sources,
          "max block", args.max_block,
          "seconds", round(time.monotonic() - started, 1),
          "file", output, flush=True)
    con.close()


if __name__ == "__main__":
    main()
