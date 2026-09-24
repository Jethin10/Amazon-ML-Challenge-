"""V2 candidate generation with a capped rare-name-token fallback."""

from __future__ import annotations

import time
import argparse

from v1 import *  # noqa: F403 - preserve the tested normalization and feature schema
from v1 import build_candidates as build_v1_candidates


def build_candidates(con, part, mode, batch_index=None, batch_count=None):
    build_v1_candidates(con, part, mode, batch_index, batch_count)
    started = time.monotonic()
    con.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {part}_name_tokens AS
        SELECT DISTINCT t.entity_id, t.country, token
        FROM {part}_targets t,
             unnest(string_split(t.name_norm, ' ')) AS u(token)
        WHERE length(token) >= 4
        """
    )
    con.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {part}_name_token_stats AS
        SELECT country, token, count(*) AS n
        FROM {part}_name_tokens GROUP BY country, token
        """
    )
    con.execute("DROP TABLE IF EXISTS v2_extra")
    con.execute(
        f"""
        CREATE TABLE v2_extra AS
        WITH ranked AS (
            SELECT s.entity_id AS s1_id, s.country, u.token,
                   row_number() OVER (
                       PARTITION BY s.entity_id ORDER BY b.n, u.token
                   ) AS token_rank
            FROM work_s1 s
            CROSS JOIN LATERAL unnest(list_distinct(string_split(s.name_norm, ' '))) AS u(token)
            JOIN {part}_name_token_stats b
              ON s.country=b.country AND u.token=b.token
            WHERE length(u.token)>=4 AND b.n<=300
        )
        SELECT s.s1_id, t.entity_id AS target_id
        FROM ranked s JOIN {part}_name_tokens t
          ON s.country=t.country AND s.token=t.token
        WHERE s.token_rank<=2
        """
    )
    con.execute("DROP TABLE IF EXISTS v2_candidates")
    con.execute(
        "CREATE TABLE v2_candidates AS "
        "SELECT s1_id, target_id FROM candidates "
        "UNION SELECT s1_id, target_id FROM v2_extra"
    )
    con.execute("DROP TABLE candidates")
    con.execute("ALTER TABLE v2_candidates RENAME TO candidates")
    con.execute("DROP TABLE v2_extra")
    total = con.execute("SELECT count(*) FROM candidates").fetchone()[0]
    print(f"V2 unique candidates: {total:,} in {time.monotonic()-started:.1f}s", flush=True)
    if part == "train":
        positive, retrieved = con.execute(
            """
            SELECT count(*), count(c.target_id)
            FROM train_truth g JOIN work_s1 s ON g.s1_id=s.entity_id
            LEFT JOIN candidates c ON g.s1_id=c.s1_id AND g.target_id=c.target_id
            """
        ).fetchone()
        print(f"V2 candidate recall: {retrieved:,}/{positive:,} = {retrieved/positive:.4f}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--part", choices=["train", "test"], required=True)
    parser.add_argument("--mode", default="pilot")
    parser.add_argument("--features", action="store_true")
    args = parser.parse_args()
    connection = connect()
    try:
        build_candidates(connection, args.part, args.mode)
        if args.features:
            write_features(connection, args.part, args.mode)
    finally:
        connection.close()
