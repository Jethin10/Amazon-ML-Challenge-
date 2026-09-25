"""Score the validated V4 candidate set with V5 and write complete test output."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from v1 import ROOT, WORK, connect, quoted_path, write_features
from v3_features import enrich as enrich_v4
from v5_features import ALL_FEATURE_NAMES, enrich as enrich_v5


OUTPUT=ROOT/"output_v5"
SHARDS=WORK/"v5_shards"
SOURCE=OUTPUT/"candidate_pairs.tsv"


def expected_rows() -> int:
    con=connect()
    count=con.execute("SELECT count(*) FROM test_s1").fetchone()[0]
    con.close()
    return count


def prepare_shards(batches: int) -> list[Path]:
    SHARDS.mkdir(exist_ok=True)
    paths=[SHARDS/f"candidate_{i:03d}.tsv" for i in range(batches)]
    manifest=SHARDS/"manifest.json"
    if manifest.exists():
        meta=json.loads(manifest.read_text())
        if (meta.get("batches")==batches and meta.get("source_bytes")==SOURCE.stat().st_size
                and meta.get("source_mtime_ns")==SOURCE.stat().st_mtime_ns
                and all(p.exists() for p in paths)):
            return paths
    expected=expected_rows()
    per_batch=math.ceil(expected/batches)
    with SOURCE.open("r",encoding="utf-8",newline="") as source:
        header=source.readline()
        if header!="source1_entity_id\tcandidate_entity_ids\n":
            raise ValueError("Wrong source candidate header")
        rows=0
        for path in paths:
            with path.open("w",encoding="utf-8",newline="") as output:
                output.write(header)
                for _ in range(per_batch):
                    line=source.readline()
                    if not line:break
                    output.write(line)
                    rows+=1
        if source.readline():raise ValueError("More Source 1 rows than expected")
    if rows!=expected:raise ValueError(f"Sharded {rows} rows, expected {expected}")
    manifest.write_text(json.dumps({"batches":batches,"source_bytes":SOURCE.stat().st_size,
                                    "source_mtime_ns":SOURCE.stat().st_mtime_ns,
                                    "rows":rows},indent=2))
    print("sharded",rows,"rows",flush=True)
    return paths


def run_batch(index: int | str, shard: Path, model: lgb.Booster, threshold: float) -> None:
    tick=time.monotonic()
    tag=f"{index:03d}" if isinstance(index,int) else index
    mode=f"v5_batch{tag}"
    out=SHARDS/f"matches_{tag}.tsv"
    out_temp=SHARDS/f"matches_{tag}.tsv.tmp"
    selected_path=SHARDS/f"selected_{tag}.parquet"
    if out.exists() and selected_path.exists():
        print("skip completed batch",index,flush=True)
        return
    con=connect()
    con.execute(f"""
        CREATE TEMP TABLE work_s1 AS
        SELECT s.* FROM test_s1 s
        JOIN read_csv({quoted_path(shard)},delim='\\t',header=true,all_varchar=true) q
          ON s.entity_id=q.source1_entity_id
    """)
    con.execute(f"""
        CREATE TEMP TABLE candidates AS
        SELECT q.source1_entity_id AS s1_id,u.target_id
        FROM read_csv({quoted_path(shard)},delim='\\t',header=true,all_varchar=true) q,
             unnest(string_split(q.candidate_entity_ids,',')) AS u(target_id)
        WHERE q.candidate_entity_ids IS NOT NULL AND q.candidate_entity_ids<>''
    """)
    n=con.execute("SELECT count(*) FROM candidates").fetchone()[0]
    base=write_features(con,"test",mode)
    v4=enrich_v4(con,"test",mode,"v4")
    con.close()
    v5=WORK/f"test_{mode}_v5_features.parquet"
    enrich_v5("test",mode)
    frame=pd.read_parquet(v5,columns=["s1_id","target_id"]+ALL_FEATURE_NAMES)
    if len(frame)!=n:raise ValueError(f"Batch {index} pair count changed")
    x=frame[ALL_FEATURE_NAMES].to_numpy(dtype=np.float32,copy=True)
    probability=model.predict(x,num_threads=8,pred_early_stop=True,
                              pred_early_stop_freq=10,pred_early_stop_margin=10)
    selected=frame.loc[probability>=threshold,["s1_id","target_id"]].copy()
    selected["probability"]=probability[probability>=threshold].astype(np.float32)
    selected.to_parquet(selected_path,index=False)
    del frame,x,probability
    gc.collect()
    mapping:dict[str,list[str]]=defaultdict(list)
    for row in selected.itertuples(index=False):mapping[row.s1_id].append(row.target_id)
    count=0
    with shard.open("r",encoding="utf-8",newline="") as source,out_temp.open(
        "w",encoding="utf-8",newline="") as target:
        reader=csv.reader(source,delimiter="\t")
        next(reader)
        writer=csv.writer(target,delimiter="\t",lineterminator="\n")
        for s1_id,_ in reader:
            writer.writerow([s1_id,",".join(mapping.get(s1_id,()))])
            count+=1
    out_temp.replace(out)
    del selected,mapping
    gc.collect()
    for path in (base,v4,v5):path.unlink(missing_ok=True)
    print(f"batch {index}: {count:,} entities, {n:,} candidates, "
          f"{time.monotonic()-tick:.1f}s",flush=True)


def assemble(paths: list[Path]) -> None:
    OUTPUT.mkdir(exist_ok=True)
    target=OUTPUT/"matching_results.tsv"
    temp=OUTPUT/"matching_results.tsv.tmp"
    rows=0
    with temp.open("w",encoding="utf-8",newline="") as out:
        out.write("source1_entity_id\tmatched_entity_ids\n")
        for index in range(len(paths)):
            shard=SHARDS/f"matches_{index:03d}.tsv"
            with shard.open("r",encoding="utf-8",newline="") as source:
                for line in source:
                    out.write(line)
                    rows+=1
    expected=expected_rows()
    if rows!=expected:raise ValueError(f"Wrong assembled row count: {rows}")
    temp.replace(target)
    print("assembled",rows,"rows",target.stat().st_size,"bytes",flush=True)


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--batches",type=int,default=20)
    parser.add_argument("--batch-index",type=int)
    parser.add_argument("--sub-batch-tag")
    parser.add_argument("--assemble-only",action="store_true")
    args=parser.parse_args()
    if args.batches<1:raise ValueError("batches must be positive")
    if args.batch_index is not None and not 0<=args.batch_index<args.batches:
        raise ValueError("batch index out of range")
    if args.batch_index is not None and args.assemble_only:
        raise ValueError("Choose either a batch index or assemble-only")
    if args.sub_batch_tag is not None:
        import re
        if not re.fullmatch(r"\d{3}[a-d]|\d{3}d?e[0-7]|\d{3}[a-d]e[0-7]",args.sub_batch_tag):
            raise ValueError("sub-batch tag must look like 010a")
        if args.batch_index is not None or args.assemble_only:
            raise ValueError("sub-batch tag cannot be combined with other modes")
    paths=prepare_shards(args.batches)
    if args.assemble_only:
        missing=[i for i in range(args.batches)
                 if not (SHARDS/f"matches_{i:03d}.tsv").exists()]
        if missing:raise RuntimeError(f"Cannot assemble; missing batches: {missing}")
        assemble(paths)
        return
    metrics=json.loads((ROOT/"models"/"v5_metrics.json").read_text())
    model_path=ROOT/"models"/"v5_lgb.txt"
    signature={"model_sha256":hashlib.sha256(model_path.read_bytes()).hexdigest(),
               "threshold":metrics["threshold"],"candidate_bytes":SOURCE.stat().st_size,
               "candidate_mtime_ns":SOURCE.stat().st_mtime_ns,
               "batches":args.batches,"features":ALL_FEATURE_NAMES}
    stamp=SHARDS/"run_signature.json"
    if stamp.exists() and json.loads(stamp.read_text())!=signature:
        raise RuntimeError("Existing V5 batch outputs belong to a different model or candidate file")
    stamp.write_text(json.dumps(signature,indent=2))
    model=lgb.Booster(model_file=str(model_path))
    if args.sub_batch_tag is not None:
        shard=SHARDS/f"candidate_{args.sub_batch_tag}.tsv"
        if not shard.is_file():raise FileNotFoundError(shard)
        run_batch(args.sub_batch_tag,shard,model,metrics["threshold"])
    elif args.batch_index is not None:
        run_batch(args.batch_index,paths[args.batch_index],model,metrics["threshold"])
    else:
        for index,path in enumerate(paths):run_batch(index,path,model,metrics["threshold"])
        assemble(paths)


if __name__=="__main__":
    main()
