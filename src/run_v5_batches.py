"""Resume V5 inference with bounded memory and exact per-entity features."""

from __future__ import annotations

import subprocess
import sys
import time

from merge_v5_parts import merge
from split_eighth import split8
from split_quarter import split_quarter
from v1 import ROOT


SHARDS = ROOT / "work" / "v5_shards"


def run(*args: str) -> None:
    print("RUN", " ".join(args), flush=True)
    subprocess.run([sys.executable, str(ROOT / "src" / "predict_v5.py"), *args],
                   cwd=ROOT, check=True)


def complete(tag: str) -> bool:
    return (SHARDS / f"matches_{tag}.tsv").is_file() and (
        SHARDS / f"selected_{tag}.parquet"
    ).is_file()


def score(tag: str) -> None:
    if complete(tag):
        print("SKIP completed", tag, flush=True)
        return
    run("--batches", "20", "--sub-batch-tag", tag)


def score_small_parts(parent: str) -> None:
    children = split8(parent)
    for child in children:
        if complete(child):
            continue
        for attempt in range(1, 4):
            try:
                score(child)
                break
            except subprocess.CalledProcessError:
                if attempt == 3:
                    raise
                print("RETRY", child, "after memory cooldown", flush=True)
                time.sleep(30)
    merge(parent, children)


def score_batch(index: int) -> None:
    parent = f"{index:03d}"
    if complete(parent):
        print("SKIP completed batch", index, flush=True)
        return
    quarters = split_quarter(index)
    for tag in quarters:
        if complete(tag):
            continue
        try:
            score(tag)
        except subprocess.CalledProcessError:
            print("SPLIT", tag, "after memory failure", flush=True)
            score_small_parts(tag)
    merge(parent, quarters)


def main() -> None:
    for index in range(20):
        score_batch(index)
    run("--batches", "20", "--assemble-only")


if __name__ == "__main__":
    main()
