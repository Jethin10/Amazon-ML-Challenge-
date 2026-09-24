"""Measure extra blocking routes on the fixed labeled pilot slice."""

from __future__ import annotations

import time

from v1 import build_candidates, connect


def evaluate(con, label: str, sql: str) -> None:
    started = time.monotonic()
    con.execute("DROP TABLE IF EXISTS probe_pairs")
    con.execute(f"CREATE TABLE probe_pairs AS SELECT DISTINCT s1_id, target_id FROM ({sql})")
    n, new, positive = con.execute(
        """
        SELECT count(*), count(*) FILTER (WHERE c.target_id IS NULL),
               count(*) FILTER (WHERE c.target_id IS NULL AND g.target_id IS NOT NULL)
        FROM probe_pairs p
        LEFT JOIN candidates c USING (s1_id, target_id)
        LEFT JOIN train_truth g USING (s1_id, target_id)
        """
    ).fetchone()
    print(label, "all", n, "new", new, "new true", positive,
          "seconds", round(time.monotonic() - started, 1), flush=True)


def main() -> None:
    con = connect()
    build_candidates(con, "train", "pilot")
    tick = time.monotonic()
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS train_name_tokens AS
        SELECT DISTINCT t.entity_id, t.country, token
        FROM train_targets t,
             unnest(string_split(t.name_norm, ' ')) AS u(token)
        WHERE length(token) >= 4
        """
    )
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS train_name_token_stats AS
        SELECT country, token, count(*) AS n
        FROM train_name_tokens GROUP BY country, token
        """
    )
    print("name token index ready", round(time.monotonic() - tick, 1), flush=True)
    con.execute("DROP TABLE IF EXISTS pilot_name_tokens")
    con.execute(
        """
        CREATE TABLE pilot_name_tokens AS
        SELECT s.entity_id AS s1_id, s.country, s.city5, s.addr_num,
               u.token, b.n,
               row_number() OVER (PARTITION BY s.entity_id ORDER BY b.n, u.token) AS r
        FROM work_s1 s
        CROSS JOIN LATERAL unnest(list_distinct(string_split(s.name_norm, ' '))) AS u(token)
        JOIN train_name_token_stats b
          ON s.country=b.country AND u.token=b.token
        WHERE length(u.token)>=4
        """
    )
    for cap in (30, 100, 300):
        for rank in (1, 2):
            evaluate(con, f"name token cap{cap} top{rank}", f"""
                SELECT s.s1_id, t.entity_id AS target_id
                FROM pilot_name_tokens s JOIN train_name_tokens t
                  ON s.country=t.country AND s.token=t.token
                WHERE s.n<={cap} AND s.r<={rank}
            """)
    evaluate(con, "name token + city", """
        SELECT s.s1_id, t.entity_id AS target_id
        FROM pilot_name_tokens s JOIN train_name_tokens n
          ON s.country=n.country AND s.token=n.token
        JOIN train_targets t ON n.entity_id=t.entity_id
        WHERE s.r<=2 AND s.city5<>'' AND s.city5=t.city5 AND s.n<=1000
    """)
    evaluate(con, "name token + number", """
        SELECT s.s1_id, t.entity_id AS target_id
        FROM pilot_name_tokens s JOIN train_name_tokens n
          ON s.country=n.country AND s.token=n.token
        JOIN train_targets t ON n.entity_id=t.entity_id
        WHERE s.r<=2 AND s.addr_num<>'' AND s.addr_num=t.addr_num AND s.n<=1000
    """)
    con.close()


if __name__ == "__main__":
    main()
