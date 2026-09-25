"""Create a portal experiment from frozen V5 scores for France only."""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict

import pyarrow.parquet as pq

from v1 import DATASET, ROOT, WORK


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, required=True)
    args = parser.parse_args()
    threshold = args.threshold
    if not 0.625 < threshold < 1:
        raise ValueError("France threshold must be above V5's 0.625 baseline")
    tag = f"{round(threshold * 1000):03d}"
    output = ROOT / f"output_v6_france_{tag}"
    output.mkdir(exist_ok=True)
    france: set[str] = set()
    with (DATASET / "test" / "test_source1.tsv").open(
        encoding="utf-8", newline=""
    ) as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if row["country"].strip().lower() == "france":
                france.add(row["entity_id"])
    keep: dict[str, set[str]] = defaultdict(set)
    for index in range(20):
        file = WORK / "v5_shards" / f"selected_{index:03d}.parquet"
        if not file.is_file():
            raise FileNotFoundError(file)
        for batch in pq.ParquetFile(file).iter_batches(
            batch_size=100_000, columns=["s1_id", "target_id", "probability"]
        ):
            for sid, tid, p in zip(*(column.to_pylist() for column in batch.columns)):
                if sid in france and p >= threshold:
                    keep[sid].add(tid)
    source = ROOT / "output_v5" / "matching_results.tsv"
    target = output / "matching_results.tsv"
    temp = output / "matching_results.tsv.tmp"
    rows = changed = removed = kept = 0
    with source.open(encoding="utf-8", newline="") as original, temp.open(
        "w", encoding="utf-8", newline=""
    ) as result:
        reader = csv.DictReader(original, delimiter="\t")
        writer = csv.writer(result, delimiter="\t", lineterminator="\n")
        writer.writerow(["source1_entity_id", "matched_entity_ids"])
        for row in reader:
            sid = row["source1_entity_id"]
            value = row["matched_entity_ids"]
            if sid in france:
                old = value.split(",") if value else []
                allowed = keep.get(sid, set())
                new = [tid for tid in old if tid in allowed]
                if len(new) != len(old):
                    changed += 1
                    removed += len(old) - len(new)
                kept += len(new)
                value = ",".join(new)
            writer.writerow([sid, value])
            rows += 1
    if rows != 1_732_544:
        temp.unlink(missing_ok=True)
        raise ValueError(f"Expected 1,732,544 source rows, found {rows}")
    temp.replace(target)
    source_candidates = ROOT / "output_v5" / "candidate_pairs.tsv"
    link = output / "candidate_pairs.tsv"
    if not link.exists():
        os.link(source_candidates, link)
    report = {"threshold_france": threshold, "other_countries": "unchanged V5",
              "rows": rows, "france_rows": len(france),
              "changed_france_rows": changed,
              "removed_france_links": removed,
              "kept_france_links": kept,
              "bytes": target.stat().st_size,
              "source_file": str(source), "output_file": str(target)}
    (output / "experiment_summary.json").write_text(json.dumps(report, indent=2),
                                                    encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
