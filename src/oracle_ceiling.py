"""Compute macro F0.5 if every available candidate were classified perfectly."""

from __future__ import annotations

import argparse

import duckdb

from v1 import DATASET, WORK, quoted_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["pilot", "valid", "audit"], default="audit")
    args = parser.parse_args()
    bounds = {"pilot": (0, 2), "valid": (42, 50), "audit": (50, 60)}
    lo, hi = bounds[args.mode]
    feature = WORK / f"train_{args.mode}_features.parquet"
    con = duckdb.connect()
    con.execute("SET threads=4")
    con.execute("SET memory_limit='5GB'")
    query = f"""
        WITH entities AS (
            SELECT entity_id FROM read_csv(
                {quoted_path(DATASET / 'train' / 'train_source1.tsv')},
                delim='\\t', header=true, all_varchar=true)
            WHERE hash(entity_id) % 1000 >= {lo} AND hash(entity_id) % 1000 < {hi}
        ), truth AS (
            SELECT source1_entity_id AS s1_id,
                   array_length(string_split(matched_entity_ids, ',')) AS n
            FROM read_csv({quoted_path(DATASET / 'train' / 'train_ground_truth.tsv')},
                          delim='\\t', header=true, all_varchar=true)
        ), retrieved AS (
            SELECT s1_id, count(*) AS k
            FROM read_parquet({quoted_path(feature)}) WHERE label=1
            GROUP BY s1_id
        )
        SELECT count(*) AS entities, sum(coalesce(t.n,0)) AS true_links,
               sum(coalesce(r.k,0)) AS retrieved,
               avg(CASE WHEN coalesce(t.n,0)=0 THEN 1.0
                        ELSE 5.0*coalesce(r.k,0)/(t.n+4.0*coalesce(r.k,0)) END)
                   AS oracle_macro_f05
        FROM entities e LEFT JOIN truth t ON e.entity_id=t.s1_id
        LEFT JOIN retrieved r ON e.entity_id=r.s1_id
    """
    print(con.execute(query).fetchone())
    con.close()


if __name__ == "__main__":
    main()
