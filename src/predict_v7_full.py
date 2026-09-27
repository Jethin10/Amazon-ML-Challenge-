"""Rerank the full uncertain candidate band, not just V5's retained pairs.

V6 reranked only pairs LightGBM had already kept (p >= 0.625). Local audit
splits show reranking the whole [0.05, 0.625) band as well is worth about
+0.012 macro F0.5 on both audit2 and audit3 at alpha=0.6, threshold=0.625.

France keeps the portal-validated V5 score at its 0.80 threshold, because no
labelled French holdout exists locally to justify moving it.

Writes to a NEW output directory so the validated 0.927 V6 submission is
never overwritten.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from collections import defaultdict
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import torch
from scipy.special import expit, logit
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from train_v6_reranker import format_side
from v1 import DATASET, DB, ROOT, WORK, quoted_path


SHARDS = WORK / "v5_shards"
EXTRA = WORK / "v7_uncertain"
V6_SAVED = WORK / "v6_selected_shards"
SAVED = WORK / "v7_full_shards"
OUTPUT = ROOT / "output_v7_full_rerank"
MODEL = ROOT / "models" / "v6_reranker"

ALPHA = 0.6
THRESHOLD = 0.625
FRANCE_THRESHOLD = 0.8
BATCH_SIZE = 128


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


def extra_for(batch: int) -> pd.DataFrame | None:
    """Concatenate every V7 leaf belonging to one of the 20 V5 batches."""
    prefix = f"{batch:03d}"
    files = sorted(EXTRA.glob(f"extra_{prefix}*.parquet"))
    if not files:
        return None
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)

def text_pairs(con, frame: pd.DataFrame) -> pd.DataFrame:
    con.register("rerank_pairs", frame[["s1_id", "target_id"]].reset_index())
    result = con.execute("""
        SELECT p.index AS row_idx,
               s.business_name AS source_name,
               s.business_address AS source_address,
               t.business_name AS target_name,
               t.business_address AS target_address
        FROM rerank_pairs p
        JOIN test_s1 s ON p.s1_id = s.entity_id
        JOIN test_targets t ON p.target_id = t.entity_id
    """).fetchdf()
    con.unregister("rerank_pairs")
    if len(result) != len(frame):
        raise ValueError(f"Text join lost pairs: {len(result)} != {len(frame)}")
    # DuckDB's hash joins do not preserve the registered frame's row order.
    # Restore it before returning so every transformer logit is combined with
    # the V5 probability for the same candidate pair.
    result = result.sort_values("row_idx", kind="stable").reset_index(drop=True)
    expected = np.arange(len(frame))
    if not np.array_equal(result.row_idx.to_numpy(), expected):
        raise ValueError("Text join returned duplicate or missing pair indexes")
    return result


def score_text(frame: pd.DataFrame, tokenizer, model, batch: int) -> np.ndarray:
    scores = np.empty(len(frame), dtype=np.float32)
    model.eval()
    started = time.monotonic()
    batch_size = BATCH_SIZE
    start = 0
    chunks = 0
    with torch.inference_mode():
        while start < len(frame):
            end = min(start + batch_size, len(frame))
            part = frame.iloc[start:end]
            left = [format_side(n, a) for n, a in zip(
                part.source_name.fillna(""), part.source_address.fillna(""))]
            right = [format_side(n, a) for n, a in zip(
                part.target_name.fillna(""), part.target_address.fillna(""))]
            tokens = tokenizer(left, right, padding="max_length",
                               truncation=True, max_length=128,
                               return_tensors="pt").to("cuda")
            try:
                with torch.autocast("cuda", dtype=torch.float16):
                    out = model(**tokens).logits.view(-1)
            except torch.OutOfMemoryError:
                del tokens
                torch.cuda.empty_cache()
                if batch_size <= 16:
                    raise
                batch_size //= 2
                print(f"batch {batch} CUDA OOM; retrying with batch size "
                      f"{batch_size}", flush=True)
                continue
            scores[start:end] = out.float().cpu().numpy()
            start = end
            chunks += 1
            if chunks % 100 == 0:
                progress(phase="scoring", batch=batch, batches=20,
                         scored_in_batch=end, total_in_batch=len(frame),
                         batch_size=batch_size,
                         seconds_in_batch=round(time.monotonic() - started, 1))
    return scores


def run_batch(index: int, con, france: set[str], tokenizer, model) -> None:
    matches_path = SAVED / f"matches_{index:03d}.tsv"
    if matches_path.is_file():
        print("skip completed batch", index, flush=True)
        return
    started = time.monotonic()

    # V6 already scored every non-France V5 pair in [.625, .98] and kept
    # >.98 untouched. Reuse those adjusted probabilities. At V7's .625
    # threshold, any old-band pair that can survive is present in this file.
    selected = pd.read_parquet(V6_SAVED / f"selected_{index:03d}.parquet")
    selected_is_france = selected.s1_id.isin(france).to_numpy()
    selected_keep = np.where(
        selected_is_france,
        selected.probability.to_numpy() >= FRANCE_THRESHOLD,
        selected.probability.to_numpy() >= THRESHOLD,
    )
    retained = selected.loc[selected_keep, ["s1_id", "target_id", "probability"]]

    extra = extra_for(index)
    if extra is None:
        extra = pd.DataFrame(columns=["s1_id", "target_id", "v5_probability"])
    probs = extra.v5_probability.to_numpy(dtype=np.float64)
    is_france = extra.s1_id.isin(france).to_numpy()
    to_score = pd.DataFrame({
        "s1_id": extra.s1_id[~is_france].to_numpy(),
        "target_id": extra.target_id[~is_france].to_numpy()})
    joined = text_pairs(con, to_score)
    logits = score_text(joined, tokenizer, model, index)

    adjusted = expit(logit(np.clip(probs[~is_france], 1e-6, 1 - 1e-6))
                      + ALPHA * logits)
    extra_result = to_score.loc[adjusted >= THRESHOLD].copy()
    extra_result["probability"] = adjusted[adjusted >= THRESHOLD].astype(np.float32)
    result = pd.concat([retained, extra_result], ignore_index=True)
    result = result.drop_duplicates(subset=["s1_id", "target_id"], keep="first")

    mapping: dict[str, list[str]] = defaultdict(list)
    for sid, tid in zip(result.s1_id, result.target_id):
        mapping[sid].append(tid)
    temp = matches_path.with_suffix(".tsv.tmp")
    rows = 0
    with (SHARDS / f"candidate_{index:03d}.tsv").open(
        encoding="utf-8", newline=""
    ) as source, temp.open("w", encoding="utf-8", newline="") as target:
        reader = csv.reader(source, delimiter="\t")
        next(reader)
        writer = csv.writer(target, delimiter="\t", lineterminator="\n")
        for sid, _ in reader:
            writer.writerow([sid, ",".join(mapping.get(sid, ()))])
            rows += 1
    temp.replace(matches_path)
    print(f"batch {index} done rows={rows} links={len(result)} "
          f"reused={len(retained)} scored={len(to_score)} "
          f"france_extra_skipped={int(is_france.sum())} "
          f"seconds={round(time.monotonic() - started, 1)}", flush=True)
    progress(phase="batch_complete", completed_batches=index + 1,
             batches=20, links=len(result))


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
    link = OUTPUT / "candidate_pairs.tsv"
    if not link.exists():
        os.link(ROOT / "output_v5" / "candidate_pairs.tsv", link)
    print("assembled", rows, "rows ->", target, flush=True)


def check_complete() -> None:
    """Refuse to run unless the uncertain band was extracted for all 20 batches.

    Defined before main() so it exists when the __main__ guard runs.
    """
    missing = [i for i in range(20) if extra_for(i) is None]
    if missing:
        raise SystemExit(
            "V7 extraction incomplete; no extra pairs for batches "
            f"{missing}. Do not run: those Source 1 rows would be left "
            "without their uncertain-band candidates."
        )
    missing_v6 = [i for i in range(20)
                  if not (V6_SAVED / f"selected_{i:03d}.parquet").is_file()]
    if missing_v6:
        raise SystemExit(f"V6 scored shards missing for batches {missing_v6}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--first", type=int, default=0)
    parser.add_argument("--last", type=int, default=19)
    parser.add_argument("--skip-assembly", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.first <= args.last < 20:
        parser.error("Expected 0 <= first <= last < 20")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.set_num_threads(4)
    SAVED.mkdir(exist_ok=True)
    check_complete()
    france = france_ids()
    print("France Source 1 records", len(france), flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL).cuda()
    con = duckdb.connect(str(DB), read_only=True)
    con.execute("SET threads=3")
    con.execute("SET memory_limit='2GB'")
    con.execute(f"SET temp_directory={quoted_path(WORK / 'tmp')}")
    for index in range(args.first, args.last + 1):
        run_batch(index, con, france, tokenizer, model)
    con.close()
    if args.first == 0 and args.last == 19 and not args.skip_assembly:
        assemble()


if __name__ == "__main__":
    main()
