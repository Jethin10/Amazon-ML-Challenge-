"""Fine tune and score a small multilingual pair reranker on the local GPU."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from v1 import ROOT, WORK


MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"


def format_side(name: str, address: str) -> str:
    return f"Name: {name or ''}; Address: {address or ''}"


class PairDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, tokenizer, max_length: int = 128):
        self.labels = torch.tensor(frame.label.to_numpy(dtype=np.float32))
        left = [format_side(n, a) for n, a in zip(
            frame.source_name.fillna(""), frame.source_address.fillna(""))]
        right = [format_side(n, a) for n, a in zip(
            frame.target_name.fillna(""), frame.target_address.fillna(""))]
        self.tokens = tokenizer(left, right, padding="max_length",
                                truncation=True, max_length=max_length,
                                return_tensors="pt")

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        return {key: value[index] for key, value in self.tokens.items()}, \
               self.labels[index]


def run() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "score"], required=True)
    parser.add_argument("--split", choices=["valid", "audit2", "audit3"],
                        default="audit2")
    parser.add_argument("--input-parquet", type=Path,
                        help="Score an arbitrary labeled pair parquet")
    parser.add_argument("--max-pairs", type=int)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=128)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the reranker pilot")
    torch.set_num_threads(4)
    started = time.monotonic()
    split = "valid" if args.mode == "train" else args.split
    if args.input_parquet is not None and args.mode != "score":
        parser.error("--input-parquet requires --mode score")
    input_path = args.input_parquet or WORK / f"v6_reranker_{split}.parquet"
    frame = pd.read_parquet(input_path)
    if args.max_pairs:
        frame = frame.sample(n=min(args.max_pairs, len(frame)),
                             random_state=2026).reset_index(drop=True)
    print("loaded", split, len(frame), "positive", int(frame.label.sum()),
          "cuda", torch.cuda.get_device_name(0), flush=True)
    model_path = ROOT / "models" / "v6_reranker"
    checkpoint = MODEL if args.mode == "train" or not model_path.is_dir() \
        else str(model_path)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    dataset = PairDataset(frame, tokenizer, args.max_length)
    print("tokenized", len(dataset), "seconds",
          round(time.monotonic() - started, 1), flush=True)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint).cuda()
    if model.config.num_labels != 1:
        raise ValueError(f"Expected one logit, got {model.config.num_labels}")
    if args.mode == "train":
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                            num_workers=0, pin_memory=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5,
                                      weight_decay=.01)
        scaler = torch.amp.GradScaler("cuda")
        model.train()
        step = 0
        for epoch in range(args.epochs):
            loss_total = 0.0
            for features, labels in loader:
                features = {k: v.cuda(non_blocking=True)
                            for k, v in features.items()}
                labels = labels.cuda(non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.float16):
                    logits = model(**features).logits.view(-1)
                    loss = torch.nn.functional.binary_cross_entropy_with_logits(
                        logits, labels)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                step += 1
                loss_total += float(loss.detach())
                if step % 100 == 0:
                    elapsed = time.monotonic() - started
                    print("step", step, "of", len(loader) * args.epochs,
                          "loss", round(loss_total / 100, 4),
                          "seconds", round(elapsed, 1),
                          "steps_per_second", round(step / elapsed, 2),
                          flush=True)
                    loss_total = 0.0
            print("epoch", epoch + 1, "complete", flush=True)
        model.save_pretrained(model_path, safe_serialization=True)
        tokenizer.save_pretrained(model_path)
        (model_path / "training.json").write_text(json.dumps({
            "base_model": MODEL, "train_split": "valid", "rows": len(frame),
            "epochs": args.epochs, "batch_size": args.batch_size,
            "max_length": args.max_length, "steps": step,
            "seconds": round(time.monotonic() - started, 1)
        }, indent=2), encoding="utf-8")
        print("saved", model_path, flush=True)
        return
    loader = DataLoader(dataset, batch_size=args.batch_size * 2,
                        num_workers=0, pin_memory=True)
    model.eval()
    predictions = []
    with torch.inference_mode():
        for index, (features, _) in enumerate(loader):
            features = {k: v.cuda(non_blocking=True)
                        for k, v in features.items()}
            with torch.autocast("cuda", dtype=torch.float16):
                logits = model(**features).logits.view(-1)
            predictions.append(logits.float().cpu().numpy())
            if (index + 1) % 100 == 0:
                print("scored", min((index + 1) * args.batch_size * 2,
                                     len(frame)), "of", len(frame),
                      "seconds", round(time.monotonic() - started, 1),
                      flush=True)
    retain = ["s1_id", "target_id", "label"]
    retain += [column for column in ("v5_probability", "similarity")
               if column in frame.columns]
    frame = frame[retain].copy()
    frame["reranker_logit"] = np.concatenate(predictions)
    output = input_path.with_name(input_path.stem + "_scored.parquet")
    frame.to_parquet(output, index=False)
    print("wrote", output, "seconds", round(time.monotonic() - started, 1),
          flush=True)


if __name__ == "__main__":
    run()
