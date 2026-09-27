"""Keep fast, high-confidence France suffix candidates for GPU scoring."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import duckdb

from v1 import ROOT, WORK, quoted_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-output", type=Path,
                        default=ROOT / "output_v7_target_exclusive")
    parser.add_argument("--address-threshold", type=float, default=0.70)
    parser.add_argument("--output", type=Path,
                        default=WORK / "france_suffix_unassigned_a70.parquet")
    args = parser.parse_args()
    started = time.monotonic()
    con = duckdb.connect(str(WORK / "v1.duckdb"), read_only=True)
    con.execute("SET threads=4")
    con.execute("SET memory_limit='3GB'")
    con.execute(f"SET temp_directory={quoted_path(WORK / 'tmp')}")
    matching = args.base_output / "matching_results.tsv"
    con.execute(f"""
        COPY (
            WITH assigned AS (
                SELECT DISTINCT unnest(string_split(matched_entity_ids,','))
                  AS target_id
                FROM read_csv({quoted_path(matching)},delim='\\t',header=true,
                              all_varchar=true)
                WHERE matched_entity_ids IS NOT NULL
                  AND matched_entity_ids<>''
            )
            SELECT p.*,
                   jaro_winkler_similarity(
                       lower(strip_accents(p.source_address)),
                       lower(strip_accents(p.target_address)))::FLOAT
                     AS similarity
            FROM read_parquet({quoted_path(WORK / 'france_suffix_pairs.parquet')}) p
            ANTI JOIN assigned a USING(target_id)
            WHERE jaro_winkler_similarity(
                    lower(strip_accents(p.source_address)),
                    lower(strip_accents(p.target_address)))
                  >={args.address_threshold}
        ) TO {quoted_path(args.output)} (FORMAT PARQUET,COMPRESSION SNAPPY)
    """)
    pairs, sources, targets = con.execute(f"""
        SELECT count(*),count(DISTINCT s1_id),count(DISTINCT target_id)
        FROM read_parquet({quoted_path(args.output)})
    """).fetchone()
    print("wrote", args.output, "pairs", pairs, "sources", sources,
          "targets", targets, "seconds", round(time.monotonic()-started, 1),
          flush=True)
    con.close()


if __name__ == "__main__":
    main()
