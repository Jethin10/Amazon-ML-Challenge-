"""Filter broad any-number candidates by name and address agreement."""

from __future__ import annotations

import time

import duckdb
from anyascii import anyascii
from duckdb.sqltypes import VARCHAR

from v1 import DATASET, WORK, quoted_path


def report(con, label: str, table: str) -> None:
    pairs, new, positive = con.execute(f"""
        SELECT count(*), count(*) FILTER (WHERE e.target_id IS NULL),
               count(*) FILTER (WHERE e.target_id IS NULL AND g.target_id IS NOT NULL)
        FROM {table} x LEFT JOIN existing e USING (s1_id,target_id)
        LEFT JOIN truth g USING (s1_id,target_id)
    """).fetchone()
    print(label, pairs, new, positive, flush=True)


def main() -> None:
    con = duckdb.connect(str(WORK / "propagation_probe.duckdb"))
    con.execute("SET threads=4")
    con.execute("SET memory_limit='4GB'")
    con.execute(f"SET temp_directory={quoted_path(WORK / 'propagation_tmp')}")
    con.create_function("romanize", anyascii, [VARCHAR], VARCHAR)
    try:
        tick = time.monotonic()
        con.execute("""
            CREATE OR REPLACE TABLE broad AS
            SELECT DISTINCT st.s1_id, tt.entity_id AS target_id
            FROM audit_s1_tokens st
            JOIN token_stats stats ON st.country=stats.country AND st.token=stats.token
            JOIN target_addr_tokens tt ON st.country=tt.country AND st.token=tt.token
            JOIN audit_s1_numbers sn ON st.s1_id=sn.s1_id
            JOIN target_numbers tn ON tt.entity_id=tn.entity_id AND sn.num=tn.num
            WHERE stats.n<=10000
        """)
        print("broad built", round(time.monotonic()-tick,1), flush=True)
        report(con, "broad", "broad")
        con.execute(f"""
            CREATE OR REPLACE TABLE audit_s1_name AS
            SELECT entity_id,
                   regexp_replace(lower(romanize(coalesce(business_name,''))),
                                  '[^a-z0-9]+', '', 'g') AS name_key
            FROM read_csv({quoted_path(DATASET / 'train' / 'train_source1.tsv')},
                          delim='\\t', header=true, all_varchar=true)
            WHERE hash(entity_id)%1000>=50 AND hash(entity_id)%1000<60
        """)
        con.execute("""
            CREATE OR REPLACE TABLE broad_name AS
            SELECT b.s1_id,b.target_id,
                   jaro_winkler_similarity(s.name_key,t.name_key) AS name_sim,
                   (s.name_key='' OR t.name_key='') AS name_missing
            FROM broad b JOIN audit_s1_name s ON b.s1_id=s.entity_id
            JOIN target_keys t ON b.target_id=t.entity_id
        """)
        for threshold in (0.60, 0.70, 0.80, 0.90):
            con.execute("DROP TABLE IF EXISTS filtered")
            con.execute(f"""
                CREATE TABLE filtered AS
                SELECT s1_id,target_id FROM broad_name
                WHERE name_sim>={threshold} OR name_missing
            """)
            report(con, f"name>={threshold} or missing", "filtered")
        tick = time.monotonic()
        con.execute("""
            CREATE OR REPLACE TABLE shared_tokens AS
            SELECT b.s1_id,b.target_id,count(DISTINCT st.token) AS n
            FROM broad b
            JOIN audit_s1_tokens st ON b.s1_id=st.s1_id
            JOIN target_addr_tokens tt ON b.target_id=tt.entity_id AND st.token=tt.token
            GROUP BY b.s1_id,b.target_id
        """)
        print("shared tokens built", round(time.monotonic()-tick,1), flush=True)
        for minimum in (2, 3):
            con.execute("DROP TABLE IF EXISTS filtered")
            con.execute(f"""
                CREATE TABLE filtered AS
                SELECT s1_id,target_id FROM shared_tokens WHERE n>={minimum}
            """)
            report(con, f"address tokens>={minimum}", "filtered")
    finally:
        con.close()


if __name__ == "__main__":
    main()
