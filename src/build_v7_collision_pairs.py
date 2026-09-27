"""Extract V7 links assigned to more than one Source 1 record."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import duckdb

from v1 import DB, ROOT, WORK, quoted_path


def parquet_list(paths: list[Path]) -> str:
    return "[" + ",".join(quoted_path(path) for path in paths) + "]"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-output", type=Path,
                        default=ROOT / "output_v7_full_rerank")
    parser.add_argument("--output", type=Path,
                        default=WORK / "v7_collision_pairs.parquet")
    args = parser.parse_args()
    selected = [WORK / "v5_shards" / f"selected_{i:03d}.parquet"
                for i in range(20)]
    extras = sorted((WORK / "v7_uncertain").glob("extra_*.parquet"))
    missing = [path for path in selected if not path.is_file()]
    if missing or not extras:
        raise FileNotFoundError(missing or WORK / "v7_uncertain")
    started = time.monotonic()
    con = duckdb.connect(str(DB), read_only=True)
    con.execute("SET threads=4")
    con.execute("SET memory_limit='3GB'")
    con.execute(f"SET temp_directory={quoted_path(WORK / 'tmp')}")
    matching = args.base_output / "matching_results.tsv"
    con.execute(f"""
        CREATE TEMP TABLE predicted AS
        SELECT source1_entity_id AS s1_id,unnest(
            string_split(matched_entity_ids,',')) AS target_id
        FROM read_csv({quoted_path(matching)},delim='\\t',header=true,
                      all_varchar=true)
        WHERE matched_entity_ids IS NOT NULL AND matched_entity_ids<>''
    """)
    con.execute("""
        CREATE TEMP TABLE collisions AS
        SELECT target_id FROM predicted GROUP BY target_id
        HAVING count(DISTINCT s1_id)>1
    """)
    con.execute(f"""
        CREATE TEMP TABLE score_lookup AS
        SELECT s1_id,target_id,probability::FLOAT AS v5_probability
        FROM read_parquet({parquet_list(selected)})
        UNION ALL
        SELECT s1_id,target_id,v5_probability::FLOAT
        FROM read_parquet({parquet_list(extras)})
    """)
    con.execute(f"""
        COPY (
            SELECT p.s1_id,p.target_id,0::INTEGER AS label,
                   max(l.v5_probability) AS v5_probability,
                   s.business_name AS source_name,
                   s.business_address AS source_address,
                   t.business_name AS target_name,
                   t.business_address AS target_address
            FROM predicted p JOIN collisions c USING(target_id)
            JOIN score_lookup l USING(s1_id,target_id)
            JOIN test_s1 s ON p.s1_id=s.entity_id
            JOIN test_targets t ON p.target_id=t.entity_id
            GROUP BY p.s1_id,p.target_id,s.business_name,s.business_address,
                     t.business_name,t.business_address
        ) TO {quoted_path(args.output)} (FORMAT PARQUET,COMPRESSION SNAPPY)
    """)
    collision_targets, collision_links = con.execute("""
        SELECT count(*),sum(n) FROM (
          SELECT target_id,count(*) n FROM predicted GROUP BY target_id
          HAVING n>1)
    """).fetchone()
    written = con.execute(f"SELECT count(*) FROM read_parquet({quoted_path(args.output)})").fetchone()[0]
    if written != collision_links:
        raise ValueError(f"Only found scores for {written}/{collision_links} links")
    print("collision targets", collision_targets, "links", collision_links,
          "seconds", round(time.monotonic() - started, 1),
          "file", args.output, flush=True)
    con.close()


if __name__ == "__main__":
    main()
