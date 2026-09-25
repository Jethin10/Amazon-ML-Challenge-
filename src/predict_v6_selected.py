"""Rerank saved V5 matches with the GPU model, preserving France at 0.80."""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from collections import defaultdict

import duckdb
import numpy as np
import pandas as pd
import torch
from scipy.special import expit, logit
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from train_v6_reranker import format_side
from v1 import DATASET, DB, ROOT, WORK, quoted_path


SHARDS = WORK / "v5_shards"
SAVED = WORK / "v6_selected_shards"
OUTPUT = ROOT / "output_v6_reranker_selected"
MODEL = ROOT / "models" / "v6_reranker"
ALPHA = .6
THRESHOLD = .6
FRANCE_THRESHOLD = .8


def progress(**fields) -> None:
    SAVED.mkdir(exist_ok=True)
    fields["updated_unix"] = time.time()
    target = SAVED / "progress.json"
    temp = SAVED / "progress.json.tmp"
    temp.write_text(json.dumps(fields, indent=2), encoding="utf-8")
    temp.replace(target)


def france_ids() -> set[str]:
    found = set()
    with (DATASET / "test" / "test_source1.tsv").open(
        encoding="utf-8", newline=""
    ) as source:
        for row in csv.DictReader(source, delimiter="\t"):
            if row["country"].strip().lower() == "france":
                found.add(row["entity_id"])
    return found


def text_pairs(con, frame: pd.DataFrame) -> pd.DataFrame:
    con.register("rerank_pairs", frame[["row_idx", "s1_id", "target_id"]])
    result = con.execute("""
        SELECT p.row_idx,p.s1_id,p.target_id,
               s.business_name AS source_name,
               s.business_address AS source_address,
               t.business_name AS target_name,
               t.business_address AS target_address
        FROM rerank_pairs p
        JOIN test_s1 s ON p.s1_id=s.entity_id
        JOIN test_targets t ON p.target_id=t.entity_id
    """).fetchdf()
    con.unregister("rerank_pairs")
    if len(result) != len(frame):
        raise ValueError(f"Text join lost pairs: {len(result)} != {len(frame)}")
    return result


def score_text(frame: pd.DataFrame, tokenizer, model, shard: int) -> np.ndarray:
    scores = np.empty(len(frame), dtype=np.float32)
    batch_size = 32
    started = time.monotonic()
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(frame), batch_size):
            end = min(start + batch_size, len(frame))
            part = frame.iloc[start:end]
            left = [format_side(n, a) for n, a in zip(
                part.source_name.fillna(""), part.source_address.fillna(""))]
            right = [format_side(n, a) for n, a in zip(
                part.target_name.fillna(""), part.target_address.fillna(""))]
            tokens = tokenizer(left, right, padding="max_length",
                               truncation=True, max_length=128,
                               return_tensors="pt").to("cuda")
            with torch.autocast("cuda", dtype=torch.float16):
                output = model(**tokens).logits.view(-1)
            scores[start:end] = output.float().cpu().numpy()
            if (start // batch_size + 1) % 100 == 0:
                elapsed = time.monotonic() - started
                print("shard", shard, "scored", end, "of", len(frame),
                      "seconds", round(elapsed, 1), flush=True)
                progress(phase="scoring", shard=shard, shards=20,
                         scored_in_shard=end, total_in_shard=len(frame),
                         seconds_in_shard=round(elapsed, 1))
    return scores


def run_shard(index: int, con, france: set[str], tokenizer, model) -> None:
    selected_path = SAVED / f"selected_{index:03d}.parquet"
    matches_path = SAVED / f"matches_{index:03d}.tsv"
    if selected_path.is_file() and matches_path.is_file():
        print("skip completed shard", index, flush=True)
        return
    started = time.monotonic()
    baseline = pd.read_parquet(SHARDS / f"selected_{index:03d}.parquet")
    baseline.insert(0, "row_idx", np.arange(len(baseline), dtype=np.int32))
    is_france = baseline.s1_id.isin(france).to_numpy()
    probabilities = baseline.probability.to_numpy(dtype=np.float64)
    uncertain = baseline.loc[(~is_france) & (probabilities <= .98)].copy()
    print("shard", index, "selected", len(baseline), "rerank", len(uncertain),
          "France", int(is_france.sum()), flush=True)
    joined = text_pairs(con, uncertain)
    logits = score_text(joined, tokenizer, model, index)
    adjusted = probabilities.copy()
    indices = joined.row_idx.to_numpy(dtype=np.int32)
    adjusted[indices] = expit(logit(np.clip(probabilities[indices], 1e-6,
                                            1 - 1e-6)) + ALPHA * logits)
    keep = np.where(is_france, probabilities >= FRANCE_THRESHOLD,
                    adjusted >= THRESHOLD)
    result = baseline.loc[keep, ["s1_id", "target_id"]].copy()
    result["probability"] = adjusted[keep].astype(np.float32)
    result.to_parquet(selected_path.with_suffix(".parquet.tmp"), index=False)
    selected_path.with_suffix(".parquet.tmp").replace(selected_path)
    mapping: dict[str, list[str]] = defaultdict(list)
    for sid, tid in zip(result.s1_id, result.target_id):
        mapping[sid].append(tid)
    temp = matches_path.with_suffix(".tsv.tmp")
    count = 0
    with (SHARDS / f"candidate_{index:03d}.tsv").open(
        encoding="utf-8", newline=""
    ) as source, temp.open("w", encoding="utf-8", newline="") as target:
        reader = csv.reader(source, delimiter="\t")
        next(reader)
        writer = csv.writer(target, delimiter="\t", lineterminator="\n")
        for sid, _ in reader:
            writer.writerow([sid, ",".join(mapping.get(sid, ()))])
            count += 1
    temp.replace(matches_path)
    print("shard", index, "done", count, "entities", len(result),
          "links", "seconds", round(time.monotonic() - started, 1), flush=True)
    progress(phase="shard_complete", completed_shards=index + 1, shards=20,
             last_shard_seconds=round(time.monotonic() - started, 1))


def assemble() -> None:
    OUTPUT.mkdir(exist_ok=True)
    target = OUTPUT / "matching_results.tsv"
    temp = OUTPUT / "matching_results.tsv.tmp"
    rows = 0
    with temp.open("w", encoding="utf-8", newline="") as output:
        output.write("source1_entity_id\tmatched_entity_ids\n")
        for index in range(20):
            with (SAVED / f"matches_{index:03d}.tsv").open(
                encoding="utf-8", newline=""
            ) as source:
                for line in source:
                    output.write(line)
                    rows += 1
    if rows != 1_732_544:
        raise ValueError(f"Wrong Source 1 row count: {rows}")
    temp.replace(target)
    candidate_link = OUTPUT / "candidate_pairs.tsv"
    if not candidate_link.exists():
        os.link(ROOT / "output_v5" / "candidate_pairs.tsv", candidate_link)
    progress(phase="assembled", completed_shards=20, shards=20, rows=rows,
             output_file=str(target))
    print("assembled", rows, "rows", target, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--first", type=int, default=0)
    parser.add_argument("--last", type=int, default=19)
    args = parser.parse_args()
    if not 0 <= args.first <= args.last < 20:
        parser.error("Expected 0 <= first <= last < 20")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.set_num_threads(4)
    SAVED.mkdir(exist_ok=True)
    checkpoint = MODEL / "model.safetensors"
    signature = {
        "model_bytes": checkpoint.stat().st_size,
        "model_mtime_ns": checkpoint.stat().st_mtime_ns,
        "alpha": ALPHA,
        "threshold": THRESHOLD,
        "france_threshold": FRANCE_THRESHOLD,
        "v5_signature": json.loads((SHARDS / "run_signature.json")
                                   .read_text(encoding="utf-8")),
    }
    signature_path = SAVED / "run_signature.json"
    if signature_path.is_file():
        if json.loads(signature_path.read_text(encoding="utf-8")) != signature:
            raise RuntimeError("Saved V6 shards belong to a different model or V5 run")
    else:
        signature_path.write_text(json.dumps(signature, indent=2),
                                  encoding="utf-8")
    france = france_ids()
    print("France Source 1 records", len(france), flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL).cuda()
    con = duckdb.connect(str(DB), read_only=True)
    con.execute("SET threads=3")
    con.execute("SET memory_limit='2GB'")
    con.execute(f"SET temp_directory={quoted_path(WORK / 'tmp')}")
    for index in range(args.first, args.last + 1):
        run_shard(index, con, france, tokenizer, model)
    con.close()
    if args.first == 0 and args.last == 19:
        assemble()


if __name__ == "__main__":
    main()
