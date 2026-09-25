"""Evaluate the frozen V5 model and threshold on a Source 1 holdout."""

from __future__ import annotations

import argparse
import json

import lightgbm as lgb
import numpy as np
import pandas as pd

from train_v4 import score_at_threshold
from v1 import ROOT, WORK, connect, make_work_s1
from v5_features import ALL_FEATURE_NAMES


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--mode",choices=["audit2","audit3"],default="audit3")
    args=parser.parse_args()
    con=connect()
    make_work_s1(con,"train",args.mode)
    entities=con.execute("""
        SELECT s.entity_id,s.country,count(g.target_id) n
        FROM work_s1 s LEFT JOIN train_truth g ON s.entity_id=g.s1_id
        GROUP BY s.entity_id,s.country ORDER BY s.entity_id
    """).df()
    con.close()
    frame=pd.read_parquet(WORK/f"train_{args.mode}_v5_features.parquet",
                          columns=["s1_id","label"]+ALL_FEATURE_NAMES)
    model=lgb.Booster(model_file=str(ROOT/"models"/"v5_lgb.txt"))
    threshold=json.loads((ROOT/"models"/"v5_metrics.json").read_text())["threshold"]
    p=model.predict(frame[ALL_FEATURE_NAMES].to_numpy(dtype=np.float32,copy=True),
                    num_threads=8,pred_early_stop=True,
                    pred_early_stop_freq=10,pred_early_stop_margin=10)
    group=pd.Index(entities.entity_id).get_indexer(frame.s1_id)
    if (group<0).any():raise ValueError("Pair outside audit")
    labels=frame.label.to_numpy(dtype=np.int8)
    truth=entities.n.to_numpy()
    overall=score_at_threshold(threshold,p,labels,group,truth)
    country=entities.country.to_numpy()
    report={"mode":args.mode,"entities":len(entities),"true_links":int(truth.sum()),
            "candidate_pairs":len(frame),"candidate_recall":float(labels.sum()/truth.sum()),
            "threshold":threshold,"macro_f05":overall[0],"by_country":{}}
    for value in sorted(set(country)):
        entity_mask=country==value
        selected=p>=threshold
        pred=np.bincount(group,weights=selected,minlength=len(truth))
        tp=np.bincount(group,weights=selected & (labels==1),minlength=len(truth))
        fp=pred-tp;fn=truth-tp
        den=1.25*tp+fp+0.25*fn
        f=np.divide(1.25*tp,den,out=np.zeros(len(truth)),where=den>0)
        singleton=truth==0
        f[singleton]=(pred[singleton]==0).astype(float)
        report["by_country"][value]={"entities":int(entity_mask.sum()),
            "macro_f05":float(f[entity_mask].mean()),
            "candidate_recall":float(labels[country[group]==value].sum()/truth[entity_mask].sum())}
    out=ROOT/"models"/f"v5_{args.mode}_metrics.json"
    out.write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2),flush=True)


if __name__=="__main__":
    main()
