"""Add transliteration and address-number features to V2 pair features."""

from __future__ import annotations

import argparse
import time

from v1 import FEATURE_NAMES, WORK, connect, quoted_path


EXTRA_FEATURE_NAMES = [
    "latin_name_jaro", "latin_name_sorted_jaro", "latin_name_edit",
    "latin_name_token_overlap", "latin_name_contains",
    "latin_address_sorted_jaro", "latin_address_token_overlap",
    "address_number_overlap", "address_number_disjoint",
]
ALL_FEATURE_NAMES = FEATURE_NAMES + EXTRA_FEATURE_NAMES


def enrich(con, part: str, mode: str, output_version: str = "v3"):
    source = WORK / f"{part}_{mode}_features.parquet"
    destination = WORK / f"{part}_{mode}_{output_version}_features.parquet"
    tick = time.monotonic()
    con.execute(
        f"""
        COPY (
            WITH base AS (
                SELECT f.*, s.name_latin AS sn, t.name_latin AS tn,
                       s.address_latin AS sa, t.address_latin AS ta,
                       regexp_extract_all(s.address_latin, '[0-9]+') AS snums,
                       regexp_extract_all(t.address_latin, '[0-9]+') AS tnums
                FROM read_parquet({quoted_path(source)}) f
                JOIN {part}_s1_latin s ON f.s1_id=s.entity_id
                JOIN {part}_targets_latin t ON f.target_id=t.entity_id
            )
            SELECT * EXCLUDE (sn, tn, sa, ta, snums, tnums),
                jaro_winkler_similarity(sn, tn) AS latin_name_jaro,
                jaro_winkler_similarity(
                    array_to_string(list_sort(string_split(sn, ' ')), ' '),
                    array_to_string(list_sort(string_split(tn, ' ')), ' ')
                ) AS latin_name_sorted_jaro,
                1.0 - levenshtein(sn, tn)::DOUBLE /
                      greatest(length(sn), length(tn), 1) AS latin_name_edit,
                array_length(list_intersect(string_split(sn, ' '), string_split(tn, ' ')))::DOUBLE /
                    greatest(array_length(string_split(sn, ' ')),
                             array_length(string_split(tn, ' ')), 1) AS latin_name_token_overlap,
                (sn<>'' AND tn<>'' AND (contains(sn,tn) OR contains(tn,sn)))::INTEGER
                    AS latin_name_contains,
                jaro_winkler_similarity(
                    array_to_string(list_sort(string_split(sa, ' ')), ' '),
                    array_to_string(list_sort(string_split(ta, ' ')), ' ')
                ) AS latin_address_sorted_jaro,
                array_length(list_intersect(string_split(sa, ' '), string_split(ta, ' ')))::DOUBLE /
                    greatest(array_length(string_split(sa, ' ')),
                             array_length(string_split(ta, ' ')), 1) AS latin_address_token_overlap,
                array_length(list_intersect(snums, tnums))::DOUBLE /
                    greatest(array_length(snums), array_length(tnums), 1)
                    AS address_number_overlap,
                (array_length(snums)>0 AND array_length(tnums)>0 AND
                 array_length(list_intersect(snums, tnums))=0)::INTEGER
                    AS address_number_disjoint
            FROM base
        ) TO {quoted_path(destination)} (FORMAT PARQUET, COMPRESSION SNAPPY)
        """
    )
    n = con.execute(f"SELECT count(*) FROM read_parquet({quoted_path(destination)})").fetchone()[0]
    print(destination.name, n, round(time.monotonic()-tick, 1), flush=True)
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--part", choices=["train", "test"], required=True)
    parser.add_argument("--mode", required=True)
    parser.add_argument("--version", choices=["v3", "v4"], default="v3")
    args = parser.parse_args()
    connection = connect()
    try:
        enrich(connection, args.part, args.mode, args.version)
    finally:
        connection.close()
