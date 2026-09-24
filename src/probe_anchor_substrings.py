"""Check whether target-name and address suffixes add useful candidates."""

import time

import duckdb

from v1 import WORK, quoted_path


def main() -> None:
    con = duckdb.connect(str(WORK / "propagation_probe.duckdb"))
    con.execute("SET threads=4")
    con.execute("SET memory_limit='4GB'")
    con.execute(f"SET temp_directory={quoted_path(WORK / 'propagation_tmp')}")
    try:
        for label, expression in (("name_suffix5", "right(name_key,5)"),
                                  ("name_mid5", "substr(name_key,5,5)"),
                                  ("address_suffix8", "right(addr_key,8)")):
            con.execute(f"""
                CREATE OR REPLACE TABLE key_{label} AS
                SELECT entity_id, country, {expression} AS key
                FROM target_keys
            """)
            for cap in (100, 300):
                tick = time.monotonic()
                con.execute("DROP TABLE IF EXISTS extra")
                con.execute(f"""
                    CREATE TABLE extra AS
                    SELECT DISTINCT a.s1_id, t.entity_id AS target_id
                    FROM anchors a
                    JOIN key_{label} anchor ON a.target_id=anchor.entity_id
                    JOIN key_{label} t ON anchor.country=t.country AND anchor.key=t.key
                    JOIN (
                        SELECT country, key, count(*) AS n
                        FROM key_{label} GROUP BY country, key HAVING count(*)<={cap}
                    ) b ON t.country=b.country AND t.key=b.key
                    WHERE anchor.key<>'' AND length(anchor.key)>=5
                """)
                pairs, new, positive = con.execute("""
                    SELECT count(*), count(*) FILTER (WHERE e.target_id IS NULL),
                           count(*) FILTER (WHERE e.target_id IS NULL AND g.target_id IS NOT NULL)
                    FROM extra x LEFT JOIN existing e USING (s1_id,target_id)
                    LEFT JOIN truth g USING (s1_id,target_id)
                """).fetchone()
                print(label, cap, pairs, new, positive,
                      round(time.monotonic()-tick, 1), flush=True)
    finally:
        con.close()


if __name__ == "__main__":
    main()
