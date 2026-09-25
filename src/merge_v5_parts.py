"""Merge independently scored Source 1 shards and check exact row order."""

from __future__ import annotations

import argparse
import csv

import pandas as pd

from v1 import ROOT


SHARDS = ROOT / "work" / "v5_shards"


def merge(parent: str, children: list[str]) -> None:
    target = SHARDS / f"matches_{parent}.tsv"
    temporary = SHARDS / f"matches_{parent}.tsv.tmp"
    with temporary.open("wb") as dest:
        for child in children:
            path = SHARDS / f"matches_{child}.tsv"
            if not path.is_file() or not (SHARDS / f"selected_{child}.parquet").is_file():
                raise FileNotFoundError(f"Incomplete child {child}")
            with path.open("rb") as source:
                for line in source:
                    if line.startswith(b"source1_entity_id\t"):
                        continue
                    dest.write(line)
    candidates = SHARDS / f"candidate_{parent}.tsv"
    rows = 0
    with candidates.open("r", encoding="utf-8", newline="") as source, temporary.open(
        "r", encoding="utf-8", newline=""
    ) as result:
        candidate_reader = csv.reader(source, delimiter="\t")
        next(candidate_reader)
        result_reader = csv.reader(result, delimiter="\t")
        for (candidate_id, _), (result_id, _) in zip(candidate_reader, result_reader, strict=True):
            if candidate_id != result_id:
                raise ValueError(f"Wrong Source 1 order at row {rows}: {candidate_id} != {result_id}")
            rows += 1
    selected = SHARDS / f"selected_{parent}.parquet"
    selected_temp = SHARDS / f"selected_{parent}.parquet.tmp"
    frame = pd.concat(
        [pd.read_parquet(SHARDS / f"selected_{child}.parquet") for child in children],
        ignore_index=True,
    )
    frame.to_parquet(selected_temp, index=False)
    selected_temp.replace(selected)
    temporary.replace(target)
    print(f"merged {parent}: {rows:,} Source 1 rows, {len(frame):,} selected pairs", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("parent")
    parser.add_argument("children", nargs="+")
    args = parser.parse_args()
    merge(args.parent, args.children)
