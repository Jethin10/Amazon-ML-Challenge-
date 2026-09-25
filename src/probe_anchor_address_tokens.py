"""Test rare address word propagation from confident audit anchors."""

from __future__ import annotations

import time

import duckdb
from anyascii import anyascii
from duckdb.sqltypes import VARCHAR

from v1 import DATASET, WORK, quoted_path


def main() -> None:
    con = duckdb.connect(str(WORK / "propagation_probe.duckdb"))
    con.execute("SET threads=4")
    con.execute("SET memory_limit='4GB'")
    con.execute(f"SET temp_directory={quoted_path(WORK / 'propagation_tmp')}")
    con.create_function("romanize", anyascii, [VARCHAR], VARCHAR)
    try:
        exists = con.execute(
            "SELECT count(*) FROM information_schema.tables WHERE table_name='target_addr_tokens'"
        ).fetchone()[0]
        if not exists:
            tick = time.monotonic()
            con.execute(f"""
                CREATE TABLE target_addr_tokens AS
                WITH raw AS (
                    SELECT entity_id, lower(trim(country)) AS country, business_address
                    FROM read_csv({quoted_path(DATASET / 'train' / 'train_source2.tsv')},
                                  delim='\\t', header=true, all_varchar=true)
                    UNION ALL
                    SELECT entity_id, lower(trim(country)) AS country, business_address
                    FROM read_csv({quoted_path(DATASET / 'train' / 'train_source3.tsv')},
                                  delim='\\t', header=true, all_varchar=true)
                ), normalized AS (
                    SELECT entity_id, country,
                           trim(regexp_replace(lower(romanize(coalesce(business_address,''))),
                                               '[^a-z0-9]+', ' ', 'g')) AS address
                    FROM raw
                )
                SELECT DISTINCT entity_id, country, u.token
                FROM normalized,
                     unnest(string_split(address,' ')) AS u(token)
                WHERE length(u.token)>=5
            """)
            print("token rows", con.execute("SELECT count(*) FROM target_addr_tokens").fetchone()[0],
                  round(time.monotonic()-tick, 1), flush=True)
        con.execute("""
            CREATE OR REPLACE TABLE token_stats AS
            SELECT country, token, count(*) AS n
            FROM target_addr_tokens GROUP BY country, token
        """)
        con.execute("""
            CREATE OR REPLACE TABLE anchor_tokens AS
            SELECT DISTINCT a.s1_id, a.target_id AS anchor_id,
                   tok.country, tok.token, stats.n,
                   row_number() OVER
                       (PARTITION BY a.s1_id ORDER BY stats.n, tok.token) AS token_rank
            FROM anchors a JOIN target_addr_tokens tok ON a.target_id=tok.entity_id
            JOIN token_stats stats USING (country,token)
        """)
        for cap in (100, 300, 1000):
            for rank in (1, 2, 3):
                tick = time.monotonic()
                con.execute("DROP TABLE IF EXISTS extra")
                con.execute(f"""
                    CREATE TABLE extra AS
                    SELECT DISTINCT a.s1_id, t.entity_id AS target_id
                    FROM anchor_tokens a JOIN target_addr_tokens t
                      ON a.country=t.country AND a.token=t.token
                    WHERE a.n<={cap} AND a.token_rank<={rank}
                """)
                pairs, new, positive = con.execute("""
                    SELECT count(*), count(*) FILTER (WHERE e.target_id IS NULL),
                           count(*) FILTER (WHERE e.target_id IS NULL AND g.target_id IS NOT NULL)
                    FROM extra x LEFT JOIN existing e USING (s1_id,target_id)
                    LEFT JOIN truth g USING (s1_id,target_id)
                """).fetchone()
                print(cap, rank, pairs, new, positive,
                      round(time.monotonic()-tick, 1), flush=True)
    finally:
        con.close()


if __name__ == "__main__":
    main()
