"""Evaluate rare address-token retrieval on the labeled pilot set."""

from __future__ import annotations

import time

from v1 import build_candidates, connect


def main() -> None:
    con = connect()
    started = time.monotonic()
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS train_addr_tokens AS
        SELECT DISTINCT t.entity_id, t.country, token
        FROM train_targets t,
             unnest(string_split(t.addr_norm, ' ')) AS u(token)
        WHERE length(token) >= 5
        """
    )
    n = con.execute("SELECT count(*) FROM train_addr_tokens").fetchone()[0]
    print("Address token rows", n, "seconds", round(time.monotonic()-started,1), flush=True)
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS train_addr_token_stats AS
        SELECT country, token, count(*) AS n
        FROM train_addr_tokens GROUP BY country, token
        """
    )
    build_candidates(con, "train", "pilot")
    con.execute("DROP TABLE IF EXISTS pilot_s1_tokens")
    con.execute(
        """
        CREATE TABLE pilot_s1_tokens AS
        SELECT s.entity_id AS s1_id, s.country, u.token, b.n,
               row_number() OVER (PARTITION BY s.entity_id ORDER BY b.n, u.token) AS token_rank
        FROM work_s1 s
        CROSS JOIN LATERAL unnest(list_distinct(string_split(s.addr_norm, ' '))) AS u(token)
        JOIN train_addr_token_stats b ON s.country=b.country AND u.token=b.token
        WHERE length(u.token)>=5 AND b.n<=100
        """
    )
    for k in (1, 2, 3, 5):
        con.execute("DROP TABLE IF EXISTS token_candidates")
        con.execute(
            f"""
            CREATE TABLE token_candidates AS
            SELECT DISTINCT s.s1_id, t.entity_id AS target_id
            FROM pilot_s1_tokens s JOIN train_addr_tokens t
              ON s.country=t.country AND s.token=t.token
            WHERE s.token_rank<={k}
            """
        )
        total = con.execute("SELECT count(*) FROM token_candidates").fetchone()[0]
        recovered = con.execute(
            """
            SELECT count(*) FROM train_truth g JOIN work_s1 s ON g.s1_id=s.entity_id
            LEFT JOIN candidates c ON g.s1_id=c.s1_id AND g.target_id=c.target_id
            JOIN token_candidates t ON g.s1_id=t.s1_id AND g.target_id=t.target_id
            WHERE c.target_id IS NULL
            """
        ).fetchone()[0]
        print("top", k, "token pairs", total, "new positives", recovered, flush=True)
    con.close()


if __name__ == "__main__":
    main()
