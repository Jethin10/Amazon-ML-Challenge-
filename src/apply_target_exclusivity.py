"""Keep only the strongest predicted owner of every duplicated target."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.special import expit, logit

from v1 import ROOT


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-output", type=Path,
                        default=ROOT / "output_v7_full_rerank")
    parser.add_argument("--scored", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "output_v7_target_exclusive")
    parser.add_argument("--alpha", type=float, default=0.6)
    args = parser.parse_args()
    available = pq.read_schema(args.scored).names
    columns = ["s1_id", "target_id", "v5_probability"]
    if "reranker_logit" in available:
        columns.append("reranker_logit")
    frame = pd.read_parquet(args.scored, columns=columns)
    if args.alpha and "reranker_logit" not in frame:
        raise ValueError("Nonzero alpha requires a reranker_logit column")
    if "reranker_logit" in frame and args.alpha:
        frame["score"] = expit(
            logit(np.clip(frame.v5_probability.to_numpy(), 1e-6, 1 - 1e-6))
            + args.alpha * frame.reranker_logit.to_numpy())
    else:
        frame["score"] = frame.v5_probability
    frame.sort_values(["target_id", "score", "s1_id"],
                      ascending=[True, False, True], kind="stable", inplace=True)
    winners = dict(zip(frame.target_id, frame.s1_id))
    args.output_dir.mkdir(exist_ok=True)
    source = args.base_output / "matching_results.tsv"
    target = args.output_dir / "matching_results.tsv"
    temp = target.with_suffix(".tsv.tmp")
    rows = removed = 0
    with source.open(encoding="utf-8", newline="") as src, temp.open(
            "w", encoding="utf-8", newline="") as dst:
        reader = csv.DictReader(src, delimiter="\t")
        writer = csv.writer(dst, delimiter="\t", lineterminator="\n")
        writer.writerow(["source1_entity_id", "matched_entity_ids"])
        for row in reader:
            sid = row["source1_entity_id"]
            values = row["matched_entity_ids"].split(",") \
                if row["matched_entity_ids"] else []
            kept = [tid for tid in values
                    if tid not in winners or winners[tid] == sid]
            removed += len(values) - len(kept)
            writer.writerow([sid, ",".join(kept)])
            rows += 1
    temp.replace(target)
    candidate_source = args.base_output / "candidate_pairs.tsv"
    candidate_target = args.output_dir / "candidate_pairs.tsv"
    if candidate_target.exists():
        candidate_target.unlink()
    try:
        os.link(candidate_source, candidate_target)
    except OSError:
        import shutil
        shutil.copy2(candidate_source, candidate_target)
    report = {"rows": rows, "collision_targets": len(winners),
              "links_removed": removed, "alpha": args.alpha}
    (args.output_dir / "target_exclusivity_summary.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
