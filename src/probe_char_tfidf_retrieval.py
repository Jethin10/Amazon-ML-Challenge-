"""Pilot country-specific character TF-IDF retrieval on untouched audit3."""

from __future__ import annotations

import argparse
import gc
import heapq
import json
import os
import time

import duckdb
import numpy as np
import pandas as pd
import psutil
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn

from v1 import DB, WORK, quoted_path


def ram_gb() -> float:
    return round(psutil.Process(os.getpid()).memory_info().rss / 1e9, 2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--country", choices=["india", "us"], default="india")
    parser.add_argument("--field", choices=["name_latin", "address_latin"],
                        default="name_latin")
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--min-score", type=float, default=.25)
    parser.add_argument("--batch-size", type=int, default=100_000)
    parser.add_argument("--max-targets", type=int)
    args = parser.parse_args()
    started = time.monotonic()
    field = args.field
    con = duckdb.connect(str(DB), read_only=True)
    con.execute("SET threads=3")
    con.execute("SET memory_limit='1GB'")
    con.execute(f"SET temp_directory={quoted_path(WORK / 'tmp')}")
    source = con.execute(
        f"SELECT entity_id,{field} AS value FROM train_s1_latin "
        "WHERE country=? AND hash(entity_id)%1000>=70 "
        "AND hash(entity_id)%1000<80", [args.country]
    ).fetchdf()
    target_count = con.execute(
        "SELECT count(*) FROM train_targets_latin WHERE country=?",
        [args.country]
    ).fetchone()[0]
    sample = con.execute(
        f"SELECT {field} AS value FROM train_targets_latin WHERE country=? "
        "AND hash(entity_id)%40=0", [args.country]
    ).fetchdf()
    print("loaded", args.country, "targets", target_count, "source", len(source),
          "vocabulary_sample", len(sample),
          "ram_gb", ram_gb(), flush=True)
    vectorizer = TfidfVectorizer(
        analyzer="char", ngram_range=(3, 5), min_df=2, max_df=.05,
        max_features=100_000, dtype=np.float32, sublinear_tf=True
    )
    vectorizer.fit(sample.value.fillna(""))
    del sample
    gc.collect()
    query_matrix = vectorizer.transform(source.value.fillna(""))
    print("vectorized queries", query_matrix.shape, "nnz", query_matrix.nnz,
          "ram_gb", ram_gb(), "secs", round(time.monotonic()-started, 1),
          flush=True)
    heaps: list[list[tuple[float, str]]] = [[] for _ in range(len(source))]
    query = f"SELECT entity_id,{field} AS value FROM train_targets_latin WHERE country=?"
    if args.max_targets:
        query += f" LIMIT {int(args.max_targets)}"
    reader = con.execute(query, [args.country]).fetch_record_batch(
        rows_per_batch=args.batch_size
    )
    scanned = 0
    for index, batch in enumerate(reader):
        targets = batch.to_pandas()
        target_matrix = vectorizer.transform(targets.value.fillna(""))
        similarity = sp_matmul_topn(
            query_matrix, target_matrix.T, top_n=args.top_n,
            threshold=args.min_score, sort=True, n_threads=3
        ).tocsr()
        target_ids = targets.entity_id.to_numpy()
        for i, heap in enumerate(heaps):
            for j in range(similarity.indptr[i], similarity.indptr[i + 1]):
                item = (float(similarity.data[j]), target_ids[similarity.indices[j]])
                if len(heap) < args.top_n:
                    heapq.heappush(heap, item)
                elif item > heap[0]:
                    heapq.heapreplace(heap, item)
        scanned += len(targets)
        del targets, target_matrix, similarity, target_ids
        if index % 5 == 0 or scanned == target_count:
            gc.collect()
            print("scanned", scanned, "of", target_count,
                  "ram_gb", ram_gb(), "secs", round(time.monotonic()-started, 1),
                  flush=True)
    records = [(sid, tid, similarity)
               for sid, heap in zip(source.entity_id, heaps)
               for similarity, tid in heap]
    out = pd.DataFrame(records, columns=["s1_id", "target_id", "similarity"])
    print("retrieved", len(out), "ram_gb", ram_gb(),
          "secs", round(time.monotonic()-started, 1), flush=True)
    output_pairs = WORK / f"char_tfidf_{args.country}_{field}_audit3.parquet"
    out.to_parquet(output_pairs, index=False)
    del vectorizer, query_matrix, source, heaps, records
    gc.collect()
    con.register("retrieval", out)
    con.execute("CREATE TEMP TABLE retrieved AS SELECT * FROM retrieval")
    con.unregister("retrieval")
    del out
    gc.collect()
    existing = WORK / "train_audit3_v5_features.parquet"
    result = con.execute(f"""
        SELECT count(*) AS pairs,
               count(*) FILTER (WHERE e.target_id IS NULL) AS new_pairs,
               count(*) FILTER (WHERE e.target_id IS NULL AND g.target_id IS NOT NULL)
                 AS new_true,
               count(*) FILTER (WHERE g.target_id IS NOT NULL) AS total_true
        FROM retrieved r
        LEFT JOIN read_parquet({quoted_path(existing)}) e
          USING(s1_id,target_id)
        LEFT JOIN train_truth g USING(s1_id,target_id)
    """).fetchone()
    report = {"country": args.country, "field": field, "top_n": args.top_n,
              "min_score": args.min_score, "targets_scanned": scanned,
              "pairs": result[0],
              "new_pairs": result[1], "new_true": result[2],
              "total_true": result[3], "elapsed_seconds":
              round(time.monotonic()-started, 1), "peak_end_ram_gb": ram_gb(),
              "pairs_file": str(output_pairs)}
    path = WORK / f"char_tfidf_{args.country}_{field}_audit3.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    con.close()


if __name__ == "__main__":
    main()
