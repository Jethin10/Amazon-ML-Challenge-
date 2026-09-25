"""Regenerate V5's exact V4 blocking candidates without scoring a V4 model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from predict_v4 import write_grouped_pairs
from v1 import ROOT, connect
from v4 import build_candidates


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--batches",type=int,default=20)
    args=parser.parse_args()
    if args.batches<1:raise ValueError("batches must be positive")
    output=ROOT/"output_v5"
    partial=ROOT/"work"/"v5_candidate_batches"
    output.mkdir(exist_ok=True)
    partial.mkdir(exist_ok=True)
    con=connect()
    expected=con.execute("SELECT count(*) FROM test_s1").fetchone()[0]
    for index in range(args.batches):
        path=partial/f"candidate_{index:03d}.tsv"
        if path.exists():
            print("skip completed candidate batch",index,flush=True)
            continue
        build_candidates(con,"test",f"v5_candidate{index:03d}",index,args.batches)
        temporary=path.with_suffix(".tsv.tmp")
        with temporary.open("w",encoding="utf-8",newline="") as handle:
            rows=write_grouped_pairs(con,"candidates","candidate_entity_ids",handle)
        temporary.replace(path)
        print("candidate batch",index,"rows",rows,flush=True)
    con.close()
    target=output/"candidate_pairs.tsv"
    temporary=output/"candidate_pairs.tsv.tmp"
    rows=0
    with temporary.open("w",encoding="utf-8",newline="") as dst:
        dst.write("source1_entity_id\tcandidate_entity_ids\n")
        for index in range(args.batches):
            path=partial/f"candidate_{index:03d}.tsv"
            with path.open("r",encoding="utf-8",newline="") as src:
                for line in src:
                    dst.write(line)
                    rows+=1
    if rows!=expected:raise RuntimeError(f"Wrong candidate row count: {rows}")
    temporary.replace(target)
    (output/"candidate_generation_summary.json").write_text(json.dumps({
        "rows":rows,"batches":args.batches,"candidate_bytes":target.stat().st_size},indent=2))
    print("generated",target,rows,flush=True)


if __name__=="__main__":
    main()
