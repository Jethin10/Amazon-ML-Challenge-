"""Disk-backed baseline for Amazon ML Challenge 2026 entity resolution."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import duckdb


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "dataset"
WORK = ROOT / "work"
DB = WORK / "v1.duckdb"


def quoted_path(path: Path) -> str:
    return "'" + path.resolve().as_posix().replace("'", "''") + "'"


def connect() -> duckdb.DuckDBPyConnection:
    WORK.mkdir(exist_ok=True)
    (WORK / "tmp").mkdir(exist_ok=True)
    con = duckdb.connect(str(DB))
    con.execute("SET threads=8")
    con.execute("SET memory_limit='8GB'")
    con.execute(f"SET temp_directory={quoted_path(WORK / 'tmp')}")
    con.execute("SET preserve_insertion_order=false")
    return con


def read_sources(part: str, sources: tuple[str, ...]) -> str:
    queries = []
    for source in sources:
        path = DATASET / part / f"{part}_{source}.tsv"
        if not path.is_file():
            raise FileNotFoundError(path)
        queries.append(
            "SELECT entity_id, business_name, business_address, country "
            f"FROM read_csv({quoted_path(path)}, delim='\\t', header=true, all_varchar=true)"
        )
    return " UNION ALL ".join(queries)


def prepare_table(con: duckdb.DuckDBPyConnection, part: str, role: str) -> None:
    sources = ("source1",) if role == "s1" else ("source2", "source3")
    raw = read_sources(part, sources)
    table = f"{part}_{role}"
    print(f"Preparing {table}", flush=True)
    started = time.monotonic()
    con.execute(f"DROP TABLE IF EXISTS {table}")
    con.execute(
        f"""
        CREATE TABLE {table} AS
        WITH raw AS ({raw}),
        cleaned AS (
            SELECT entity_id, business_name, business_address,
                   lower(trim(coalesce(country, ''))) AS country,
                   trim(regexp_replace(lower(strip_accents(coalesce(business_name, ''))),
                                        '[^a-z0-9]+', ' ', 'g')) AS name_norm,
                   trim(regexp_replace(lower(strip_accents(coalesce(business_address, ''))),
                                        '[^a-z0-9]+', ' ', 'g')) AS addr_norm,
                   trim(regexp_replace(lower(strip_accents(
                        regexp_extract(coalesce(business_address, ''),
                                       ',\\s*([^,]+),\\s*[^,]+$', 1))),
                                       '[^a-z0-9]+', ' ', 'g')) AS city_norm
            FROM raw
        ),
        keyed AS (
            SELECT *,
                   regexp_replace(
                       regexp_replace(name_norm,
                           '\\b(inc|incorporated|llc|ltd|limited|corp|corporation|pvt|private|llp)\\b',
                           ' ', 'g'), '[^a-z0-9]+', '', 'g') AS name_key,
                   regexp_replace(addr_norm, '[^a-z0-9]+', '', 'g') AS addr_key,
                   regexp_replace(city_norm, '[^a-z0-9]+', '', 'g') AS city_key,
                   regexp_extract(addr_norm, '([0-9]+)', 1) AS addr_num
            FROM cleaned
        )
        SELECT *, left(name_key, 5) AS name5,
                  left(name_key, 4) AS name4,
                  left(city_key, 5) AS city5,
                  left(addr_key, 8) AS addr8
        FROM keyed
        """
    )
    n = con.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
    print(f"{table}: {n:,} rows, {time.monotonic() - started:.1f}s", flush=True)


def prepare(con: duckdb.DuckDBPyConnection, part: str) -> None:
    prepare_table(con, part, "s1")
    prepare_table(con, part, "targets")
    if part == "train":
        path = DATASET / "train" / "train_ground_truth.tsv"
        con.execute("DROP TABLE IF EXISTS train_truth")
        con.execute(
            f"""
            CREATE TABLE train_truth AS
            SELECT source1_entity_id AS s1_id,
                   unnest(string_split(matched_entity_ids, ',')) AS target_id
            FROM read_csv({quoted_path(path)}, delim='\\t', header=true, all_varchar=true)
            WHERE matched_entity_ids IS NOT NULL AND matched_entity_ids <> ''
            """
        )
        n = con.execute("SELECT count(*) FROM train_truth").fetchone()[0]
        print(f"train_truth: {n:,} positive pairs", flush=True)


def make_work_s1(
    con: duckdb.DuckDBPyConnection,
    part: str,
    mode: str,
    batch_index: int | None = None,
    batch_count: int | None = None,
) -> None:
    if part == "train":
        windows = {"pilot": (0, 2), "fit": (2, 42), "valid": (42, 50),
                   "audit": (50, 60)}
        if mode not in windows:
            raise ValueError(f"Unsupported training mode: {mode}")
        lo, hi = windows[mode]
        where = f"WHERE hash(entity_id) % 1000 >= {lo} AND hash(entity_id) % 1000 < {hi}"
    else:
        if batch_index is not None and batch_count is not None:
            where = f"WHERE hash(entity_id) % {batch_count} = {batch_index}"
        else:
            where = ""
    con.execute("DROP TABLE IF EXISTS work_s1")
    con.execute(f"CREATE TABLE work_s1 AS SELECT * FROM {part}_s1 {where}")
    n = con.execute("SELECT count(*) FROM work_s1").fetchone()[0]
    print(f"work_s1 {part}/{mode}: {n:,} rows", flush=True)


ROUTES = (
    ("exact_name", ("name_key",), "s.name_key <> ''", None),
    ("exact_address", ("addr_key",), "s.addr_key <> ''", None),
    ("name5", ("name5",), "length(s.name5) = 5", 100),
    ("name4_city", ("name4", "city5"),
     "length(s.name4) = 4 AND length(s.city5) >= 3", 100),
    ("name4_number", ("name4", "addr_num"),
     "length(s.name4) = 4 AND s.addr_num <> ''", 100),
    ("city_number", ("city5", "addr_num"),
     "length(s.city5) >= 3 AND s.addr_num <> ''", 100),
    ("address8", ("addr8",), "length(s.addr8) = 8", 100),
)


def build_candidates(
    con: duckdb.DuckDBPyConnection,
    part: str,
    mode: str,
    batch_index: int | None = None,
    batch_count: int | None = None,
) -> None:
    make_work_s1(con, part, mode, batch_index, batch_count)
    target = f"{part}_targets"
    con.execute("DROP TABLE IF EXISTS candidates_raw")
    con.execute("CREATE TABLE candidates_raw (s1_id VARCHAR, target_id VARCHAR)")
    for label, fields, valid, cap in ROUTES:
        started = time.monotonic()
        join = " AND ".join(f"s.{field} = t.{field}" for field in fields)
        stats_join = ""
        if cap is not None:
            stats = f"{part}_block_{label}"
            keys = ", ".join(fields)
            con.execute(
                f"CREATE TABLE IF NOT EXISTS {stats} AS "
                f"SELECT country, {keys}, count(*) AS n FROM {target} "
                f"GROUP BY country, {keys} HAVING count(*) <= {cap}"
            )
            on = " AND ".join(f"t.{field}=b.{field}" for field in fields)
            stats_join = f"JOIN {stats} b ON t.country=b.country AND {on}"
        con.execute(
            f"""
            INSERT INTO candidates_raw
            SELECT s.entity_id, t.entity_id
            FROM work_s1 s JOIN {target} t
              ON s.country = t.country AND {join}
            {stats_join}
            WHERE {valid}
            """
        )
        n = con.execute("SELECT count(*) FROM candidates_raw").fetchone()[0]
        print(f"{label}: cumulative {n:,} pairs, {time.monotonic()-started:.1f}s", flush=True)
    # The rarest usable address word is a compact fallback for reordered names,
    # trade names, and address variants that miss the prefix-based blocks.
    started = time.monotonic()
    con.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {part}_addr_tokens AS
        SELECT DISTINCT t.entity_id, t.country, token
        FROM {target} t,
             unnest(string_split(t.addr_norm, ' ')) AS u(token)
        WHERE length(token) >= 5
        """
    )
    con.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {part}_addr_token_stats AS
        SELECT country, token, count(*) AS n
        FROM {part}_addr_tokens GROUP BY country, token
        """
    )
    con.execute(
        f"""
        INSERT INTO candidates_raw
        WITH ranked AS (
            SELECT s.entity_id AS s1_id, s.country, u.token,
                   row_number() OVER (
                       PARTITION BY s.entity_id ORDER BY b.n, u.token
                   ) AS token_rank
            FROM work_s1 s
            CROSS JOIN LATERAL unnest(list_distinct(string_split(s.addr_norm, ' '))) AS u(token)
            JOIN {part}_addr_token_stats b
              ON s.country=b.country AND u.token=b.token
            WHERE length(u.token)>=5 AND b.n<=100
        )
        SELECT s.s1_id, t.entity_id
        FROM ranked s JOIN {part}_addr_tokens t
          ON s.country=t.country AND s.token=t.token
        WHERE s.token_rank=1
        """
    )
    n = con.execute("SELECT count(*) FROM candidates_raw").fetchone()[0]
    print(f"rare_address_token: cumulative {n:,} pairs, {time.monotonic()-started:.1f}s", flush=True)
    started = time.monotonic()
    con.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {part}_addr_name4_tokens AS
        SELECT a.country, a.token, t.name4, a.entity_id
        FROM {part}_addr_tokens a
        JOIN {target} t ON a.entity_id=t.entity_id
        JOIN {part}_addr_token_stats b
          ON a.country=b.country AND a.token=b.token
        WHERE b.n<=1000 AND t.name4<>''
        """
    )
    con.execute(
        f"""
        INSERT INTO candidates_raw
        WITH ranked AS (
            SELECT s.entity_id AS s1_id, s.country, s.name4, u.token, b.n,
                   row_number() OVER (
                       PARTITION BY s.entity_id ORDER BY b.n, u.token
                   ) AS token_rank
            FROM work_s1 s
            CROSS JOIN LATERAL unnest(list_distinct(string_split(s.addr_norm, ' '))) AS u(token)
            JOIN {part}_addr_token_stats b
              ON s.country=b.country AND u.token=b.token
            WHERE length(u.token)>=5
        )
        SELECT s.s1_id, t.entity_id
        FROM ranked s JOIN {part}_addr_name4_tokens t
          ON s.country=t.country AND s.token=t.token AND s.name4=t.name4
        WHERE s.token_rank<=3 AND s.n<=1000 AND s.name4<>''
        """
    )
    n = con.execute("SELECT count(*) FROM candidates_raw").fetchone()[0]
    print(f"address_token_name4: cumulative {n:,} pairs, {time.monotonic()-started:.1f}s", flush=True)
    started = time.monotonic()
    con.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {part}_addr_number_tokens AS
        SELECT a.country, a.token, ltrim(t.addr_num,'0') AS num_key, a.entity_id
        FROM {part}_addr_tokens a
        JOIN {target} t ON a.entity_id=t.entity_id
        JOIN {part}_addr_token_stats b
          ON a.country=b.country AND a.token=b.token
        WHERE b.n<=1000 AND t.addr_num<>''
        """
    )
    con.execute(
        f"""
        INSERT INTO candidates_raw
        WITH ranked AS (
            SELECT s.entity_id AS s1_id, s.country,
                   ltrim(s.addr_num,'0') AS num_key, u.token, b.n,
                   row_number() OVER (
                       PARTITION BY s.entity_id ORDER BY b.n, u.token
                   ) AS token_rank
            FROM work_s1 s
            CROSS JOIN LATERAL unnest(list_distinct(string_split(s.addr_norm, ' '))) AS u(token)
            JOIN {part}_addr_token_stats b
              ON s.country=b.country AND u.token=b.token
            WHERE length(u.token)>=5 AND s.addr_num<>''
        )
        SELECT s.s1_id, t.entity_id
        FROM ranked s JOIN {part}_addr_number_tokens t
          ON s.country=t.country AND s.token=t.token AND s.num_key=t.num_key
        WHERE s.token_rank<=3 AND s.n<=1000
        """
    )
    n = con.execute("SELECT count(*) FROM candidates_raw").fetchone()[0]
    print(f"address_token_number: cumulative {n:,} pairs, {time.monotonic()-started:.1f}s", flush=True)
    started = time.monotonic()
    con.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {part}_name_tokens AS
        SELECT DISTINCT t.entity_id, t.country, token
        FROM {target} t,
             unnest(string_split(t.name_norm, ' ')) AS u(token)
        WHERE length(token)>=4
        """
    )
    con.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {part}_name_token_stats AS
        SELECT country, token, count(*) AS n
        FROM {part}_name_tokens GROUP BY country, token
        """
    )
    con.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {part}_name_number_tokens AS
        SELECT n.country, n.token, t.addr_num, n.entity_id
        FROM {part}_name_tokens n
        JOIN {target} t ON n.entity_id=t.entity_id
        JOIN {part}_name_token_stats b
          ON n.country=b.country AND n.token=b.token
        WHERE b.n<=1000 AND t.addr_num<>''
        """
    )
    con.execute(
        f"""
        INSERT INTO candidates_raw
        WITH ranked AS (
            SELECT s.entity_id AS s1_id, s.country, s.addr_num, u.token, b.n,
                   row_number() OVER (
                       PARTITION BY s.entity_id ORDER BY b.n, u.token
                   ) AS token_rank
            FROM work_s1 s
            CROSS JOIN LATERAL unnest(list_distinct(string_split(s.name_norm, ' '))) AS u(token)
            JOIN {part}_name_token_stats b
              ON s.country=b.country AND u.token=b.token
            WHERE length(u.token)>=4
        )
        SELECT s.s1_id, t.entity_id
        FROM ranked s JOIN {part}_name_number_tokens t
          ON s.country=t.country AND s.token=t.token AND s.addr_num=t.addr_num
        WHERE s.token_rank<=2 AND s.n<=1000 AND s.addr_num<>''
        """
    )
    n = con.execute("SELECT count(*) FROM candidates_raw").fetchone()[0]
    print(f"name_token_number: cumulative {n:,} pairs, {time.monotonic()-started:.1f}s", flush=True)
    con.execute("DROP TABLE IF EXISTS candidates")
    con.execute("CREATE TABLE candidates AS SELECT DISTINCT s1_id, target_id FROM candidates_raw")
    con.execute("DROP TABLE candidates_raw")
    n = con.execute("SELECT count(*) FROM candidates").fetchone()[0]
    print(f"Unique candidates: {n:,}", flush=True)
    if part == "train":
        truth = con.execute(
            """
            SELECT count(*) AS positives,
                   count(c.target_id) AS retrieved
            FROM train_truth g JOIN work_s1 s ON g.s1_id = s.entity_id
            LEFT JOIN candidates c ON g.s1_id = c.s1_id AND g.target_id = c.target_id
            """
        ).fetchone()
        print(f"Candidate recall: {truth[1]:,}/{truth[0]:,} = {truth[1]/truth[0]:.4f}", flush=True)
    (WORK / "candidate_summary.json").write_text(
        json.dumps({"part": part, "mode": mode, "candidates": n}, indent=2), encoding="utf-8"
    )


FEATURE_NAMES = [
    "name_jaro", "name_edit", "name_token_overlap", "address_jaro",
    "name_sorted_jaro", "address_sorted_jaro", "address_token_overlap",
    "name_exact", "address_exact", "name_contains", "address_contains",
    "name5_equal", "name4_equal", "city_equal", "number_equal",
    "number_conflict", "name_missing", "address_missing", "name_len_ratio",
    "address_len_ratio", "target_source3",
]


def write_features(con: duckdb.DuckDBPyConnection, part: str, mode: str) -> Path:
    target = f"{part}_targets"
    path = WORK / f"{part}_{mode}_features.parquet"
    truth_join = (
        "LEFT JOIN train_truth g ON c.s1_id=g.s1_id AND c.target_id=g.target_id"
        if part == "train" else ""
    )
    label = "CASE WHEN g.target_id IS NULL THEN 0 ELSE 1 END AS label," if part == "train" else ""
    started = time.monotonic()
    print(f"Writing features to {path.name}", flush=True)
    con.execute(
        f"""
        COPY (
            SELECT c.s1_id, c.target_id, {label}
                jaro_winkler_similarity(s.name_norm,t.name_norm) AS name_jaro,
                1.0 - levenshtein(s.name_key,t.name_key)::DOUBLE /
                    greatest(length(s.name_key),length(t.name_key),1) AS name_edit,
                array_length(list_intersect(string_split(s.name_norm,' '),
                                            string_split(t.name_norm,' ')))::DOUBLE /
                    greatest(array_length(string_split(s.name_norm,' ')),
                             array_length(string_split(t.name_norm,' ')),1) AS name_token_overlap,
                jaro_winkler_similarity(s.addr_norm,t.addr_norm) AS address_jaro,
                jaro_winkler_similarity(
                    array_to_string(list_sort(string_split(s.name_norm,' ')), ' '),
                    array_to_string(list_sort(string_split(t.name_norm,' ')), ' ')
                ) AS name_sorted_jaro,
                jaro_winkler_similarity(
                    array_to_string(list_sort(string_split(s.addr_norm,' ')), ' '),
                    array_to_string(list_sort(string_split(t.addr_norm,' ')), ' ')
                ) AS address_sorted_jaro,
                array_length(list_intersect(string_split(s.addr_norm,' '),
                                            string_split(t.addr_norm,' ')))::DOUBLE /
                    greatest(array_length(string_split(s.addr_norm,' ')),
                             array_length(string_split(t.addr_norm,' ')),1) AS address_token_overlap,
                (s.name_key=t.name_key AND s.name_key<>'')::INTEGER AS name_exact,
                (s.addr_key=t.addr_key AND s.addr_key<>'')::INTEGER AS address_exact,
                (s.name_key<>'' AND t.name_key<>'' AND
                 (contains(s.name_key,t.name_key) OR contains(t.name_key,s.name_key)))::INTEGER AS name_contains,
                (s.addr_key<>'' AND t.addr_key<>'' AND
                 (contains(s.addr_key,t.addr_key) OR contains(t.addr_key,s.addr_key)))::INTEGER AS address_contains,
                (s.name5=t.name5 AND s.name5<>'')::INTEGER AS name5_equal,
                (s.name4=t.name4 AND s.name4<>'')::INTEGER AS name4_equal,
                (s.city5=t.city5 AND s.city5<>'')::INTEGER AS city_equal,
                (s.addr_num<>'' AND t.addr_num<>'' AND
                 ltrim(s.addr_num,'0')=ltrim(t.addr_num,'0'))::INTEGER AS number_equal,
                (s.addr_num<>'' AND t.addr_num<>'' AND
                 ltrim(s.addr_num,'0')<>ltrim(t.addr_num,'0'))::INTEGER AS number_conflict,
                (s.name_norm='' OR t.name_norm='')::INTEGER AS name_missing,
                (s.addr_norm='' OR t.addr_norm='')::INTEGER AS address_missing,
                least(length(s.name_key),length(t.name_key))::DOUBLE /
                    greatest(length(s.name_key),length(t.name_key),1) AS name_len_ratio,
                least(length(s.addr_key),length(t.addr_key))::DOUBLE /
                    greatest(length(s.addr_key),length(t.addr_key),1) AS address_len_ratio,
                starts_with(t.entity_id,'S3-')::INTEGER AS target_source3
            FROM candidates c
            JOIN work_s1 s ON c.s1_id=s.entity_id
            JOIN {target} t ON c.target_id=t.entity_id
            {truth_join}
        ) TO {quoted_path(path)} (FORMAT PARQUET, COMPRESSION SNAPPY)
        """
    )
    n = con.execute(f"SELECT count(*) FROM read_parquet({quoted_path(path)})").fetchone()[0]
    print(f"Features: {n:,} rows, {time.monotonic()-started:.1f}s", flush=True)
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["prepare", "candidates", "features"])
    parser.add_argument("--part", choices=["train", "test"], required=True)
    parser.add_argument("--mode", default="pilot",
                        choices=["pilot", "fit", "valid", "audit", "full"])
    args = parser.parse_args()
    con = connect()
    try:
        if args.command == "prepare":
            prepare(con, args.part)
        elif args.command == "candidates":
            build_candidates(con, args.part, args.mode)
        else:
            write_features(con, args.part, args.mode)
    finally:
        con.close()


if __name__ == "__main__":
    main()
