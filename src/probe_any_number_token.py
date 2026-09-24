"""Probe direct blocking on any shared address number plus an address word."""

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
            "SELECT count(*) FROM information_schema.tables WHERE table_name='target_numbers'"
        ).fetchone()[0]
        if not exists:
            tick = time.monotonic()
            con.execute(f"""
                CREATE TABLE target_numbers AS
                WITH raw AS (
                    SELECT entity_id, lower(trim(country)) AS country, business_address
                    FROM read_csv({quoted_path(DATASET / 'train' / 'train_source2.tsv')},
                                  delim='\\t', header=true, all_varchar=true)
                    UNION ALL
                    SELECT entity_id, lower(trim(country)) AS country, business_address
                    FROM read_csv({quoted_path(DATASET / 'train' / 'train_source3.tsv')},
                                  delim='\\t', header=true, all_varchar=true)
                )
                SELECT DISTINCT entity_id, country, u.num
                FROM raw, unnest(regexp_extract_all(coalesce(business_address,''), '[0-9]+')) AS u(num)
                WHERE u.num<>''
            """)
            print("target numbers", con.execute("SELECT count(*) FROM target_numbers").fetchone()[0],
                  round(time.monotonic()-tick,1), flush=True)
        con.execute(f"""
            CREATE OR REPLACE TABLE audit_s1_address AS
            SELECT entity_id, lower(trim(country)) AS country,
                   trim(regexp_replace(lower(romanize(coalesce(business_address,''))),
                                       '[^a-z0-9]+', ' ', 'g')) AS address
            FROM read_csv({quoted_path(DATASET / 'train' / 'train_source1.tsv')},
                          delim='\\t', header=true, all_varchar=true)
            WHERE hash(entity_id)%1000>=50 AND hash(entity_id)%1000<60
        """)
        con.execute("""
            CREATE OR REPLACE TABLE audit_s1_tokens AS
            SELECT DISTINCT s.entity_id AS s1_id, s.country, u.token
            FROM audit_s1_address s, unnest(string_split(s.address,' ')) AS u(token)
            WHERE length(u.token)>=5
        """)
        con.execute("""
            CREATE OR REPLACE TABLE audit_s1_numbers AS
            SELECT DISTINCT s.entity_id AS s1_id, s.country, u.num
            FROM audit_s1_address s, unnest(regexp_extract_all(s.address,'[0-9]+')) AS u(num)
            WHERE u.num<>''
        """)
        for cap in (100, 300, 1000, 10000):
            for minlen in (1, 3):
                tick = time.monotonic()
                con.execute("DROP TABLE IF EXISTS extra")
                con.execute(f"""
                    CREATE TABLE extra AS
                    SELECT DISTINCT st.s1_id, tt.entity_id AS target_id
                    FROM audit_s1_tokens st
                    JOIN token_stats stats ON st.country=stats.country AND st.token=stats.token
                    JOIN target_addr_tokens tt ON st.country=tt.country AND st.token=tt.token
                    JOIN audit_s1_numbers sn ON st.s1_id=sn.s1_id
                    JOIN target_numbers tn ON tt.entity_id=tn.entity_id AND sn.num=tn.num
                    WHERE stats.n<={cap} AND length(sn.num)>={minlen}
                """)
                pairs, new, positive = con.execute("""
                    SELECT count(*), count(*) FILTER (WHERE e.target_id IS NULL),
                           count(*) FILTER (WHERE e.target_id IS NULL AND g.target_id IS NOT NULL)
                    FROM extra x LEFT JOIN existing e USING (s1_id,target_id)
                    LEFT JOIN truth g USING (s1_id,target_id)
                """).fetchone()
                print(cap, minlen, pairs, new, positive,
                      round(time.monotonic()-tick,1), flush=True)
    finally:
        con.close()


if __name__ == "__main__":
    main()
