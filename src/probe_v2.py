"""Measure further blocking routes against a fixed pilot, without retraining."""

from __future__ import annotations

import time

from v1 import connect


def evaluate(con, label: str, query: str) -> None:
    tick = time.monotonic()
    con.execute("DROP TABLE IF EXISTS probe_pairs")
    con.execute(f"CREATE TABLE probe_pairs AS SELECT DISTINCT s1_id, target_id FROM ({query})")
    total, new, positive = con.execute(
        """
        SELECT count(*), count(*) FILTER (WHERE c.target_id IS NULL),
               count(*) FILTER (WHERE c.target_id IS NULL AND g.target_id IS NOT NULL)
        FROM probe_pairs p
        LEFT JOIN candidates c USING (s1_id, target_id)
        LEFT JOIN train_truth g USING (s1_id, target_id)
        """
    ).fetchone()
    print(label, total, new, positive, round(time.monotonic()-tick, 1), flush=True)


def main() -> None:
    con = connect()
    try:
        for field, cap in (("name5", 300), ("name5", 1000), ("city5, addr_num", 300),
                           ("city5, addr_num", 1000), ("addr8", 300)):
            keys = [piece.strip() for piece in field.split(",")]
            join = " AND ".join(f"s.{key}=t.{key}" for key in keys)
            key_not_empty = " AND ".join(f"s.{key}<>''" for key in keys)
            target_count = " AND ".join(f"z.{key}=t.{key}" for key in keys)
            group = ", ".join(keys)
            evaluate(con, f"{field} cap{cap}", f"""
                SELECT s.entity_id AS s1_id, t.entity_id AS target_id
                FROM work_s1 s JOIN train_targets t
                  ON s.country=t.country AND {join}
                JOIN (
                    SELECT country, {group}, count(*) AS n
                    FROM train_targets GROUP BY country, {group}
                    HAVING count(*) <= {cap}
                ) z ON z.country=t.country AND {target_count}
                WHERE {key_not_empty}
            """)
        con.execute("DROP TABLE IF EXISTS v2_addr_ranked")
        con.execute("""
            CREATE TABLE v2_addr_ranked AS
            SELECT s.entity_id AS s1_id, s.country, u.token, b.n,
                   row_number() OVER (PARTITION BY s.entity_id ORDER BY b.n, u.token) AS r
            FROM work_s1 s
            CROSS JOIN LATERAL unnest(list_distinct(string_split(s.addr_norm, ' '))) AS u(token)
            JOIN train_addr_token_stats b ON s.country=b.country AND u.token=b.token
            WHERE length(u.token)>=5 AND b.n<=300
        """)
        for rank in (1, 2, 3):
            evaluate(con, f"address token cap300 top{rank}", f"""
                SELECT s.s1_id, t.entity_id AS target_id
                FROM v2_addr_ranked s JOIN train_addr_tokens t
                  ON s.country=t.country AND s.token=t.token
                WHERE s.r<={rank}
            """)
    finally:
        con.close()


if __name__ == "__main__":
    main()
