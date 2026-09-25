"""Measure whether missed true links have another retrieved true neighbor."""

import duckdb

from v1 import DATASET, WORK, quoted_path


def main() -> None:
    con = duckdb.connect()
    query = f"""
        WITH audit AS (
            SELECT entity_id FROM read_csv(
                {quoted_path(DATASET / 'train' / 'train_source1.tsv')},
                delim='\\t', header=true, all_varchar=true)
            WHERE hash(entity_id) % 1000 >= 50 AND hash(entity_id) % 1000 < 60
        ), truth AS (
            SELECT g.source1_entity_id AS s1_id,
                   unnest(string_split(g.matched_entity_ids, ',')) AS target_id
            FROM read_csv({quoted_path(DATASET / 'train' / 'train_ground_truth.tsv')},
                          delim='\\t', header=true, all_varchar=true) g
            JOIN audit a ON a.entity_id=g.source1_entity_id
            WHERE g.matched_entity_ids IS NOT NULL AND g.matched_entity_ids<>''
        ), retrieved AS (
            SELECT s1_id, target_id
            FROM read_parquet({quoted_path(WORK / 'train_audit_features.parquet')})
            WHERE label=1
        ), covered_s1 AS (
            SELECT DISTINCT s1_id FROM retrieved
        )
        SELECT count(*) AS missed_links,
               count(*) FILTER (WHERE covered_s1.s1_id IS NOT NULL)
                   AS missed_with_other_true_candidate,
               count(*) FILTER (WHERE starts_with(truth.target_id, 'S2-')) AS missed_s2,
               count(*) FILTER (WHERE starts_with(truth.target_id, 'S3-')) AS missed_s3
        FROM truth
        LEFT JOIN retrieved USING (s1_id, target_id)
        LEFT JOIN covered_s1 ON truth.s1_id=covered_s1.s1_id
        WHERE retrieved.target_id IS NULL
    """
    print(con.execute(query).fetchone())
    con.close()


if __name__ == "__main__":
    main()
