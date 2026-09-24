"""V4 blocking: any shared address number and word, screened by name."""

from __future__ import annotations

import time

from v2 import *  # noqa: F403 - reuse the exact V2 feature and normalization schema
from v2 import build_candidates as build_v2_candidates


def build_candidates(con, part, mode, batch_index=None, batch_count=None):
    build_v2_candidates(con, part, mode, batch_index, batch_count)
    tick = time.monotonic()
    con.execute(f"""
        CREATE TABLE IF NOT EXISTS {part}_latin_addr_tokens AS
        SELECT DISTINCT entity_id, country, u.token
        FROM {part}_targets_latin,
             unnest(string_split(address_latin,' ')) AS u(token)
        WHERE length(u.token)>=5
    """)
    con.execute(f"""
        CREATE TABLE IF NOT EXISTS {part}_latin_addr_token_stats AS
        SELECT country, token, count(*) AS n
        FROM {part}_latin_addr_tokens GROUP BY country, token
    """)
    con.execute(f"""
        CREATE TABLE IF NOT EXISTS {part}_latin_addr_numbers AS
        SELECT DISTINCT entity_id, country, u.num
        FROM {part}_targets_latin,
             unnest(regexp_extract_all(address_latin,'[0-9]+')) AS u(num)
        WHERE u.num<>''
    """)
    con.execute(f"""
        CREATE TABLE IF NOT EXISTS {part}_latin_token_num AS
        SELECT tok.country, tok.token, n.num, tok.entity_id
        FROM {part}_latin_addr_tokens tok
        JOIN {part}_latin_addr_numbers n ON tok.entity_id=n.entity_id
    """)
    con.execute("DROP TABLE IF EXISTS v4_extra")
    con.execute(f"""
        CREATE TABLE v4_extra AS
        WITH s1_keys AS (
            SELECT DISTINCT w.entity_id AS s1_id, s.country,
                   regexp_replace(s.name_latin,'[^a-z0-9]+','','g') AS name_key,
                   u.token, n.num
            FROM work_s1 w JOIN {part}_s1_latin s ON w.entity_id=s.entity_id,
                 unnest(string_split(s.address_latin,' ')) AS u(token),
                 unnest(regexp_extract_all(s.address_latin,'[0-9]+')) AS n(num)
            WHERE length(u.token)>=5 AND n.num<>''
        )
        SELECT DISTINCT st.s1_id, tt.entity_id AS target_id
        FROM s1_keys st
        JOIN {part}_latin_addr_token_stats stats
          ON st.country=stats.country AND st.token=stats.token
        JOIN {part}_latin_token_num tt
          ON st.country=tt.country AND st.token=tt.token AND st.num=tt.num
        JOIN {part}_targets_latin tl ON tt.entity_id=tl.entity_id
        WHERE stats.n<=10000
          AND (st.name_key='' OR tl.name_latin='' OR
               jaro_winkler_similarity(
                   st.name_key,
                   regexp_replace(tl.name_latin,'[^a-z0-9]+','','g'))>=0.70)
    """)
    added = con.execute("SELECT count(*) FROM v4_extra").fetchone()[0]
    con.execute("DROP TABLE IF EXISTS v4_candidates")
    con.execute("""
        CREATE TABLE v4_candidates AS
        SELECT s1_id,target_id FROM candidates
        UNION SELECT s1_id,target_id FROM v4_extra
    """)
    con.execute("DROP TABLE candidates")
    con.execute("ALTER TABLE v4_candidates RENAME TO candidates")
    con.execute("DROP TABLE v4_extra")
    total = con.execute("SELECT count(*) FROM candidates").fetchone()[0]
    print(f"V4 route {added:,}; unique candidates {total:,}; "
          f"{time.monotonic()-tick:.1f}s", flush=True)
    if part == "train":
        positive, retrieved = con.execute("""
            SELECT count(*), count(c.target_id)
            FROM train_truth g JOIN work_s1 s ON g.s1_id=s.entity_id
            LEFT JOIN candidates c ON g.s1_id=c.s1_id AND g.target_id=c.target_id
        """).fetchone()
        print(f"V4 candidate recall {retrieved:,}/{positive:,}={retrieved/positive:.4f}",
              flush=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--part", choices=["train", "test"], required=True)
    parser.add_argument("--mode", required=True)
    parser.add_argument("--features", action="store_true")
    args = parser.parse_args()
    connection = connect()
    try:
        build_candidates(connection, args.part, args.mode)
        if args.features:
            write_features(connection, args.part, args.mode)
    finally:
        connection.close()
