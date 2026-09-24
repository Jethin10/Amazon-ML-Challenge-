"""Count true links added by transliterated blocking routes on the pilot slice."""

from __future__ import annotations

from v1 import connect


def main() -> None:
    con = connect()
    try:
        routes = [
            ("latin_exact_name", "name_latin_key", None),
            ("latin_name5_100", "name_latin5", 100),
            ("latin_name5_300", "name_latin5", 300),
            ("latin_exact_address", "address_latin_key", None),
            ("latin_address8_100", "address_latin8", 100),
        ]
        for label, field, cap in routes:
            stats = ""
            if cap is not None:
                stats = f"""JOIN (
                    SELECT country, {field}, count(*) AS n
                    FROM train_targets_latin GROUP BY country, {field}
                    HAVING count(*)<={cap}
                ) b ON b.country=t.country AND b.{field}=t.{field}"""
            con.execute("DROP TABLE IF EXISTS probe_pairs")
            con.execute(f"""
                CREATE TABLE probe_pairs AS
                SELECT DISTINCT s.entity_id AS s1_id, t.entity_id AS target_id
                FROM work_s1 w
                JOIN train_s1_latin s ON w.entity_id=s.entity_id
                JOIN train_targets_latin t
                  ON s.country=t.country AND s.{field}=t.{field}
                {stats}
                WHERE s.{field}<>''
            """)
            total, new, positives = con.execute("""
                SELECT count(*), count(*) FILTER (WHERE c.target_id IS NULL),
                    count(*) FILTER (WHERE c.target_id IS NULL AND g.target_id IS NOT NULL)
                FROM probe_pairs p
                LEFT JOIN candidates c USING (s1_id, target_id)
                LEFT JOIN train_truth g USING (s1_id, target_id)
            """).fetchone()
            print(label, total, new, positives, flush=True)
    finally:
        con.close()


if __name__ == "__main__":
    main()
