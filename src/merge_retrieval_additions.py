"""Add transformer-approved retrieval pairs to predictions and candidates."""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq

from v1 import ROOT, WORK


def append_unique(existing: str, additions: list[str]) -> str:
    if not additions:
        return existing
    values = existing.split(",") if existing else []
    seen = set(values)
    values.extend(value for value in additions
                  if value not in seen and not seen.add(value))
    return ",".join(values)


def rewrite(source: Path, target: Path, column: str,
            additions: dict[str, list[str]]) -> tuple[int, int]:
    rows = added = 0
    temp = target.with_suffix(target.suffix + ".tmp")
    with source.open(encoding="utf-8", newline="") as src, temp.open(
            "w", encoding="utf-8", newline="") as dst:
        reader = csv.DictReader(src, delimiter="\t")
        if reader.fieldnames != ["source1_entity_id", column]:
            raise ValueError(f"Unexpected header in {source}: {reader.fieldnames}")
        writer = csv.writer(dst, delimiter="\t", lineterminator="\n")
        writer.writerow(reader.fieldnames)
        for row in reader:
            sid = row["source1_entity_id"]
            before = row[column]
            after = append_unique(before, additions.get(sid, []))
            added += (after.count(",") + bool(after)) - \
                     (before.count(",") + bool(before))
            writer.writerow([sid, after])
            rows += 1
    temp.replace(target)
    return rows, added


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-output", type=Path,
                        default=ROOT / "output_v7_full_rerank")
    parser.add_argument("--scored", type=Path, required=True)
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "output_v8_retrieval")
    args = parser.parse_args()
    args.output_dir.mkdir(exist_ok=True)
    candidates: dict[str, list[str]] = defaultdict(list)
    matches: dict[str, list[str]] = defaultdict(list)
    reader = pq.ParquetFile(args.scored)
    for batch in reader.iter_batches(
            batch_size=100_000,
            columns=["s1_id", "target_id", "reranker_logit"]):
        frame = batch.to_pandas()
        for sid, tid, value in zip(frame.s1_id, frame.target_id,
                                   frame.reranker_logit):
            candidates[sid].append(tid)
            if value >= args.threshold:
                matches[sid].append(tid)
    matching_rows, matches_added = rewrite(
        args.base_output / "matching_results.tsv",
        args.output_dir / "matching_results.tsv",
        "matched_entity_ids", matches)
    candidate_rows, candidates_added = rewrite(
        args.base_output / "candidate_pairs.tsv",
        args.output_dir / "candidate_pairs.tsv",
        "candidate_entity_ids", candidates)
    report = {
        "base_output": str(args.base_output),
        "scored_pairs": str(args.scored),
        "threshold": args.threshold,
        "matching_rows": matching_rows,
        "candidate_rows": candidate_rows,
        "matches_added": matches_added,
        "candidates_added": candidates_added,
    }
    (args.output_dir / "retrieval_merge_summary.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
