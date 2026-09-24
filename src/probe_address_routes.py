"""Probe selective address-token combinations on a small fixed sample."""

from __future__ import annotations

import time

from v1 import build_candidates, connect


def evaluate(con, label: str, sql: str) -> None:
    tick = time.monotonic()
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
          "seconds", round(time.monotonic()-tick, 1), flush=True)


def main() -> None:
    con = connect()
    build_candidates(con, "train", "pilot")
    con.execute("DROP TABLE IF EXISTS pilot_addr_tokens")
    con.execute(
        """
        CREATE TABLE pilot_addr_tokens AS
        SELECT s.entity_id AS s1_id, s.country, s.name4, s.name5, s.addr_num,
               s.city5, u.token, b.n,
               row_number() OVER (PARTITION BY s.entity_id ORDER BY b.n, u.token) AS r
        FROM work_s1 s
        CROSS JOIN LATERAL unnest(list_distinct(string_split(s.addr_norm, ' '))) AS u(token)
        JOIN train_addr_token_stats b
          ON s.country=b.country AND u.token=b.token
        WHERE length(u.token)>=5
        """
    )
    for cap in (300, 1000):
        for rank in (1, 3):
            evaluate(con, f"address token cap{cap} top{rank}", f"""
                SELECT s.s1_id, t.entity_id AS target_id
                FROM pilot_addr_tokens s JOIN train_addr_tokens t
                  ON s.country=t.country AND s.token=t.token
                WHERE s.n<={cap} AND s.r<={rank}
            """)
    for predicate, label in (
        ("s.name4=t.name4 AND s.name4<>''", "name4"),
        ("s.name5=t.name5 AND s.name5<>''", "name5"),
        ("s.addr_num=t.addr_num AND s.addr_num<>''", "number"),
        ("s.city5=t.city5 AND s.city5<>''", "city"),
    ):
        evaluate(con, f"address token + {label}", f"""
            SELECT s.s1_id, t.entity_id AS target_id
            FROM pilot_addr_tokens s JOIN train_addr_tokens a
              ON s.country=a.country AND s.token=a.token
            JOIN train_targets t ON a.entity_id=t.entity_id
            WHERE s.n<=1000 AND s.r<=3 AND {predicate}
        """)
    con.close()


if __name__ == "__main__":
    main()
