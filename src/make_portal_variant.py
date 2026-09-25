"""Make a smaller, validated leaderboard TSV by capping long match lists."""

from __future__ import annotations

import argparse
import json

from v1 import ROOT


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cap", type=int, default=10)
    args = parser.parse_args()
    if args.cap < 1:
        raise ValueError("cap must be positive")
    source = ROOT / "output_v4" / "matching_results.tsv"
    target_dir = ROOT / "output_v4_portal"
    target_dir.mkdir(exist_ok=True)
    target = target_dir / "matching_results.tsv"
    rows = removed = capped_rows = 0
    with source.open("r", encoding="utf-8", newline="") as src, target.open(
        "w", encoding="utf-8", newline=""
    ) as dst:
        header = src.readline()
        if header != "source1_entity_id\tmatched_entity_ids\n":
            raise ValueError("Unexpected source header")
        dst.write(header)
        for line in src:
            s1_id, sep, value = line.rstrip("\n").partition("\t")
            if not sep or not s1_id:
                raise ValueError(f"Malformed line after {rows} rows")
            ids = value.split(",") if value else []
            if len(ids) > args.cap:
                removed += len(ids) - args.cap
                capped_rows += 1
                ids = ids[: args.cap]
            dst.write(s1_id + "\t" + ",".join(ids) + "\n")
            rows += 1
    report = {
        "source": str(source), "target": str(target), "cap": args.cap,
        "rows": rows, "capped_rows": capped_rows, "removed_matches": removed,
        "source_bytes": source.stat().st_size, "target_bytes": target.stat().st_size,
    }
    (target_dir / "portal_variant_summary.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
