"""Fit V5 on existing candidates and evaluate a frozen audit2 holdout."""

from __future__ import annotations

import json
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from train_v4 import score_at_threshold
from v1 import ROOT, WORK, connect, make_work_s1
from v5_features import ALL_FEATURE_NAMES


def load(mode: str, with_ids: bool):
    columns=ALL_FEATURE_NAMES+["label"]+(["s1_id","target_id"] if with_ids else [])
    frame=pd.read_parquet(WORK/f"train_{mode}_v5_features.parquet",columns=columns)
    ids=frame[["s1_id","target_id"]].copy() if with_ids else None
    y=frame["label"].to_numpy(dtype=np.int8,copy=True)
    x=frame[ALL_FEATURE_NAMES].to_numpy(dtype=np.float32,copy=True)
    del frame
    return x,y,ids


def groups(mode: str, ids: pd.DataFrame):
    con=connect()
    make_work_s1(con,"train",mode)
    ordered=[r[0] for r in con.execute("SELECT entity_id FROM work_s1 ORDER BY entity_id").fetchall()]
    truth=dict(con.execute("""
        SELECT g.s1_id,count(*) FROM train_truth g
        JOIN work_s1 s ON g.s1_id=s.entity_id GROUP BY g.s1_id
    """).fetchall())
    con.close()
    position=pd.Index(ordered)
    group=position.get_indexer(ids.s1_id)
    if (group<0).any():raise ValueError("Pair outside holdout")
    return group,np.array([truth.get(i,0) for i in ordered]),len(ordered)


def main() -> None:
    started=time.monotonic()
    x_fit,y_fit,_=load("fit",False)
    x_valid,y_valid,id_valid=load("valid",True)
    print("fit",len(y_fit),int(y_fit.sum()),"valid",len(y_valid),int(y_valid.sum()),flush=True)
    train=lgb.Dataset(x_fit,label=y_fit,feature_name=ALL_FEATURE_NAMES,free_raw_data=True)
    valid=lgb.Dataset(x_valid,label=y_valid,reference=train,free_raw_data=True)
    params={"objective":"binary","metric":"binary_logloss","learning_rate":0.04,
            "num_leaves":63,"min_data_in_leaf":100,"feature_fraction":0.9,
            "verbosity":-1,"num_threads":8,"seed":2026}
    model=lgb.train(params,train,num_boost_round=1200,valid_sets=[valid],
                    callbacks=[lgb.early_stopping(80),lgb.log_evaluation(100)])
    del x_fit,y_fit,train,valid
    predict_options={"pred_early_stop":True,"pred_early_stop_freq":10,
                     "pred_early_stop_margin":10}
    p_valid=model.predict(x_valid,num_threads=8,**predict_options)
    del x_valid
    gv,tv,nv=groups("valid",id_valid)
    thresholds=np.r_[np.arange(0.4,0.96,0.025),0.975,0.99]
    table=[(float(t),)+score_at_threshold(t,p_valid,y_valid,gv,tv) for t in thresholds]
    best=max(table,key=lambda row:row[1])
    print("best valid",best,flush=True)
    del p_valid,y_valid,id_valid,gv,tv
    x_audit,y_audit,id_audit=load("audit2",True)
    p_audit=model.predict(x_audit,num_threads=8,**predict_options)
    del x_audit
    ga,ta,na=groups("audit2",id_audit)
    audit=score_at_threshold(best[0],p_audit,y_audit,ga,ta)
    print("audit2",audit,flush=True)
    model.save_model(str(ROOT/"models"/"v5_lgb.txt"))
    report={"fit_pairs":11390093,"valid_entities":nv,"audit_entities":na,
            "best_iteration":model.best_iteration,"threshold":best[0],
            "valid_macro_f05":best[1],"audit_macro_f05":audit[0],
            "audit_candidate_recall":float(y_audit.sum()/ta.sum()),
            "audit_true_links":int(ta.sum()),"feature_names":ALL_FEATURE_NAMES,
            "threshold_grid":[{"threshold":a,"macro_f05":b} for a,b,*_ in table],
            "feature_importance_gain":dict(sorted(zip(ALL_FEATURE_NAMES,
                map(float,model.feature_importance("gain"))),key=lambda item:-item[1])),
            "elapsed_seconds":round(time.monotonic()-started,1)}
    (ROOT/"models"/"v5_metrics.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps({k:v for k,v in report.items() if k not in
                      ("threshold_grid","feature_importance_gain","feature_names")},indent=2),flush=True)


if __name__=="__main__":
    main()
