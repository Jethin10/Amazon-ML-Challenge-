"""Run the trained matcher over every test S1 and write both required TSVs."""

from __future__ import annotations

import argparse
import csv
import json
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from v1 import FEATURE_NAMES, ROOT, WORK, build_candidates, connect, write_features


OUTPUT = ROOT / "output"


def write_grouped_pairs(con, table: str, field: str, handle) -> int:
    writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
    cursor = con.execute(
        f"""
        SELECT s.entity_id,
               coalesce(string_agg(p.target_id, ','), '')
        FROM work_s1 s LEFT JOIN {table} p ON s.entity_id=p.s1_id
        GROUP BY s.entity_id ORDER BY s.entity_id
        """
    )
    count = 0
    while rows := cursor.fetchmany(1000):
        writer.writerows(rows)
        count += len(rows)
    handle.flush()
    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batches", type=int, default=20)
    parser.add_argument("--start-batch", type=int, default=0)
    args = parser.parse_args()
    if args.batches < 1 or not 0 <= args.start_batch < args.batches:
        raise ValueError("Invalid batch count or starting batch")
    OUTPUT.mkdir(exist_ok=True)
    metrics = json.loads((ROOT / "models" / "v1_metrics.json").read_text(encoding="utf-8"))
    threshold = metrics["threshold"]
    model = lgb.Booster(model_file=str(ROOT / "models" / "v1_lgb.txt"))
    con = connect()
    total_entities = 0
    total_candidates = 0
    total_matches = 0
    started = time.monotonic()
    matching = OUTPUT / "matching_results.tsv"
    candidates = OUTPUT / "candidate_pairs.tsv"
    if args.start_batch:
        if not matching.is_file() or not candidates.is_file():
            raise FileNotFoundError("Cannot resume without both output TSVs")
        with matching.open("rb") as handle:
            handle.readline()
            for line in handle:
                total_entities += 1
                total_matches += line.count(b",") + (not line.endswith(b"\t\n"))
        candidate_rows = 0
        with candidates.open("rb") as handle:
            handle.readline()
            for line in handle:
                candidate_rows += 1
                total_candidates += line.count(b",") + (not line.endswith(b"\t\n"))
        if candidate_rows != total_entities:
            raise RuntimeError("Matching and candidate files have different row counts")
        print(f"Resuming after {total_entities:,} completed S1 rows", flush=True)
    file_mode = "a" if args.start_batch else "w"
    with matching.open(file_mode, encoding="utf-8", newline="") as match_file, candidates.open(
        file_mode, encoding="utf-8", newline=""
    ) as candidate_file:
        if not args.start_batch:
            match_file.write("source1_entity_id\tmatched_entity_ids\n")
            candidate_file.write("source1_entity_id\tcandidate_entity_ids\n")
        for batch in range(args.start_batch, args.batches):
            tick = time.monotonic()
            tag = f"batch{batch:03d}"
            print(f"Batch {batch+1}/{args.batches}", flush=True)
            build_candidates(con, "test", tag, batch, args.batches)
            feature_path = write_features(con, "test", tag)
            scoring_tick = time.monotonic()
            frame = pd.read_parquet(feature_path, columns=["s1_id", "target_id"] + FEATURE_NAMES)
            x = frame[FEATURE_NAMES].to_numpy(dtype=np.float32, copy=True)
            probability = model.predict(
                x, num_threads=8, pred_early_stop=True,
                pred_early_stop_freq=10, pred_early_stop_margin=10,
            )
            chosen = frame.loc[probability >= threshold, ["s1_id", "target_id"]].copy()
            print(f"Scored {len(frame):,} pairs in {time.monotonic()-scoring_tick:.1f}s", flush=True)
            total_candidates += len(frame)
            total_matches += len(chosen)
            del x, probability, frame
            con.register("chosen_frame", chosen)
            con.execute("DROP TABLE IF EXISTS chosen_pairs")
            con.execute("CREATE TABLE chosen_pairs AS SELECT s1_id, target_id FROM chosen_frame")
            con.unregister("chosen_frame")
            del chosen
            writing_tick = time.monotonic()
            n_candidate_rows = write_grouped_pairs(con, "candidates", "candidate_entity_ids", candidate_file)
            n_match_rows = write_grouped_pairs(con, "chosen_pairs", "matched_entity_ids", match_file)
            print(f"Wrote grouped lists in {time.monotonic()-writing_tick:.1f}s", flush=True)
            assert n_candidate_rows == n_match_rows
            total_entities += n_match_rows
            feature_path.unlink()
            (WORK / "predict_progress.json").write_text(
                json.dumps({"completed_batches": batch + 1, "batches": args.batches,
                            "entities": total_entities}, indent=2), encoding="utf-8"
            )
            print(
                f"Batch {batch+1}: {n_match_rows:,} S1 rows, "
                f"{time.monotonic()-tick:.1f}s; total {total_entities:,}",
                flush=True,
            )
    expected = con.execute("SELECT count(*) FROM test_s1").fetchone()[0]
    con.close()
    if total_entities != expected:
        raise RuntimeError(f"Wrote {total_entities} Source 1 rows; expected {expected}")
    report = {
        "test_entities": total_entities,
        "candidate_pairs": total_candidates,
        "predicted_pairs": total_matches,
        "threshold": threshold,
        "prediction_early_stop": True,
        "elapsed_seconds": time.monotonic() - started,
        "matching_bytes": matching.stat().st_size,
        "candidate_bytes": candidates.stat().st_size,
    }
    (OUTPUT / "v1_inference_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
