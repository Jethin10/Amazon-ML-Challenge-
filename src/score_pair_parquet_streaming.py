"""Score arbitrary pair parquets with the V6 transformer in bounded memory."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from train_v6_reranker import format_side
from v1 import ROOT


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--row-group-size", type=int, default=25_000)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    output = args.output or args.input.with_name(
        args.input.stem + "_scored.parquet")
    model_path = ROOT / "models" / "v6_reranker"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path).cuda()
    model.eval()
    torch.set_num_threads(4)
    reader = pq.ParquetFile(args.input)
    required = ["s1_id", "target_id", "source_name", "source_address",
                "target_name", "target_address"]
    missing = set(required) - set(reader.schema_arrow.names)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    retain = [name for name in (
        "s1_id", "target_id", "label", "similarity", "v5_probability",
        "anchor_name_similarity", "anchor_address_similarity",
    )
              if name in reader.schema_arrow.names]
    writer = None
    done = 0
    started = time.monotonic()
    try:
        for record_batch in reader.iter_batches(
                batch_size=args.row_group_size, columns=retain + required[2:]):
            frame = record_batch.to_pandas()
            scores = np.empty(len(frame), dtype=np.float32)
            with torch.inference_mode():
                for start in range(0, len(frame), args.batch_size):
                    end = min(start + args.batch_size, len(frame))
                    part = frame.iloc[start:end]
                    left = [format_side(n, a) for n, a in zip(
                        part.source_name.fillna(""),
                        part.source_address.fillna(""))]
                    right = [format_side(n, a) for n, a in zip(
                        part.target_name.fillna(""),
                        part.target_address.fillna(""))]
                    tokens = tokenizer(left, right, padding="max_length",
                                       truncation=True, max_length=128,
                                       return_tensors="pt").to("cuda")
                    with torch.autocast("cuda", dtype=torch.float16):
                        logits = model(**tokens).logits.view(-1)
                    scores[start:end] = logits.float().cpu().numpy()
            table = pa.Table.from_pydict({
                **{name: frame[name].to_numpy() for name in retain},
                "reranker_logit": scores,
            })
            if writer is None:
                writer = pq.ParquetWriter(output, table.schema,
                                          compression="snappy")
            writer.write_table(table)
            done += len(frame)
            print("scored", done, "of", reader.metadata.num_rows,
                  "seconds", round(time.monotonic() - started, 1), flush=True)
    finally:
        if writer is not None:
            writer.close()
    if done != reader.metadata.num_rows:
        raise ValueError(f"Scored {done} of {reader.metadata.num_rows} rows")
    print("wrote", output, flush=True)


if __name__ == "__main__":
    main()
