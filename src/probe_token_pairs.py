"""Probe two-address-token blocking against the labeled pilot."""

from __future__ import annotations

import time

from v1 import build_candidates, connect


def main() -> None:
    con = connect()
    build_candidates(con, "train", "pilot")
    con.execute("DROP TABLE IF EXISTS pilot_addr_ranked")
    con.execute("""
        CREATE TABLE pilot_addr_ranked AS
        SELECT s.entity_id AS s1_id, s.country, u.token, b.n,
               row_number() OVER (PARTITION BY s.entity_id ORDER BY b.n, u.token) AS r
        FROM work_s1 s
        CROSS JOIN LATERAL unnest(list_distinct(string_split(s.addr_norm,' '))) AS u(token)
        JOIN train_addr_token_stats b
          ON s.country=b.country AND u.token=b.token
        WHERE length(u.token)>=5 AND b.n<=1000
    """)
    for top in (3, 5):
        tick = time.monotonic()
        con.execute("DROP TABLE IF EXISTS pilot_addr_pairs")
        con.execute(f"""
            CREATE TABLE pilot_addr_pairs AS
            SELECT a.s1_id, a.country, a.token AS token1, b.token AS token2
            FROM pilot_addr_ranked a JOIN pilot_addr_ranked b
              ON a.s1_id=b.s1_id AND a.r<b.r
            WHERE a.r<={top} AND b.r<={top}
        """)
        con.execute("DROP TABLE IF EXISTS probe_pairs")
        con.execute("""
            CREATE TABLE probe_pairs AS
            SELECT DISTINCT s.s1_id, t1.entity_id AS target_id
            FROM pilot_addr_pairs s
            JOIN train_addr_tokens t1
              ON s.country=t1.country AND s.token1=t1.token
            JOIN train_addr_tokens t2
              ON t1.entity_id=t2.entity_id AND s.token2=t2.token
        """)
        n, new, positive = con.execute("""
            SELECT count(*), count(*) FILTER (WHERE c.target_id IS NULL),
                   count(*) FILTER (WHERE c.target_id IS NULL AND g.target_id IS NOT NULL)
            FROM probe_pairs p
            LEFT JOIN candidates c USING (s1_id,target_id)
            LEFT JOIN train_truth g USING (s1_id,target_id)
        """).fetchone()
        print(top, n, new, positive, "seconds", round(time.monotonic()-tick, 1), flush=True)
    con.close()


if __name__ == "__main__":
    main()
