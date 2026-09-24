"""Cache script-transliterated text for the existing source tables."""

from __future__ import annotations

import argparse
import time

from anyascii import anyascii
from duckdb.sqltypes import VARCHAR

from v1 import connect


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--part", choices=["train", "test"], required=True)
    args = parser.parse_args()
    con = connect()
    con.create_function("romanize", anyascii, [VARCHAR], VARCHAR)
    try:
        for source in ("s1", "targets"):
            table = f"{args.part}_{source}"
            destination = f"{table}_latin"
            tick = time.monotonic()
            con.execute(f"DROP TABLE IF EXISTS {destination}")
            con.execute(
                f"""
                CREATE TABLE {destination} AS
                WITH romanized AS (
                    SELECT entity_id, country,
                        trim(regexp_replace(lower(romanize(coalesce(business_name,''))),
                                            '[^a-z0-9]+', ' ', 'g')) AS name_latin,
                        trim(regexp_replace(lower(romanize(coalesce(business_address,''))),
                                            '[^a-z0-9]+', ' ', 'g')) AS address_latin
                    FROM {table}
                ), keyed AS (
                    SELECT *,
                        regexp_replace(
                            regexp_replace(name_latin,
                                '\\b(inc|incorporated|llc|ltd|limited|corp|corporation|pvt|private|llp)\\b',
                                ' ', 'g'), '[^a-z0-9]+', '', 'g') AS name_latin_key,
                        regexp_replace(address_latin, '[^a-z0-9]+', '', 'g') AS address_latin_key
                    FROM romanized
                )
                SELECT *, left(name_latin_key, 5) AS name_latin5,
                          left(address_latin_key, 8) AS address_latin8
                FROM keyed
                """
            )
            n = con.execute(f"SELECT count(*) FROM {destination}").fetchone()[0]
            print(destination, n, round(time.monotonic()-tick, 1), flush=True)
    finally:
        con.close()


if __name__ == "__main__":
    main()
