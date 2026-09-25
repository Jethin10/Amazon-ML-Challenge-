"""Eighth-split fallback for OOM quarters: 8 x ~10.8k S1, then merge to abcd."""
import subprocess
import sys
import time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
SH = ROOT / "work" / "v5_shards"
PY = ROOT / ".venv" / "Scripts" / "python.exe"
LOG = ROOT / "work" / "v5_runner3.log"
def logw(t):
    with LOG.open("a", encoding="utf-8") as f:
        f.write(t)
    print(t, end="", flush=True)
def split8(tag):
    src = SH / f"candidate_{tag}.tsv"
    with src.open("rb") as f:
        hdr = f.readline()
        n = sum(1 for _ in f)
    per = (n + 7) // 8
    subs = [f"{tag}e{i}" for i in range(8)]
    with src.open("rb") as f:
        assert f.readline() == hdr
        for s in subs:
            dst = SH / f"candidate_{s}.tsv"
            if dst.exists():
                for _ in range(per):
                    if not f.readline():
                        break
                continue
            tmp = dst.with_suffix(".tsv.tmp")
            with tmp.open("wb") as d:
                d.write(hdr)
                for _ in range(per):
                    line = f.readline()
                    if not line:
                        break
                    d.write(line)
            tmp.replace(dst)
            logw(f"eighth {s} rows~{per}\n")
    return subs
def run(tag):
    t0 = time.monotonic()
    logw(f"RUN eighth {tag}\n")
    with LOG.open("a", encoding="utf-8") as lf:
        p = subprocess.run([str(PY), "-u", "src/predict_v5.py", "--batches", "20",
                            "--sub-batch-tag", tag], cwd=ROOT,
                           stdout=lf, stderr=subprocess.STDOUT, text=True)
    logw(f"DONE eighth {tag} exit={p.returncode} {time.monotonic()-t0:.1f}s\n")
    if p.returncode != 0:
        raise RuntimeError(tag)
def merge8(tag):
    import pandas as pd
    subs = [f"{tag}e{i}" for i in range(8)]
    out = SH / f"matches_{tag}.tsv"
    tmp = out.with_suffix(".tsv.tmp")
    rows = 0
    with tmp.open("wb") as d:
        for s in subs:
            with (SH / f"matches_{s}.tsv").open("rb") as f:
                first = f.readline()
                if not first.startswith(b"source1_entity_id"):
                    d.write(first)
                    rows += 1
                for line in f:
                    if line.startswith(b"source1_entity_id"):
                        continue
                    d.write(line)
                    rows += 1
    tmp.replace(out)
    fr = [pd.read_parquet(SH / f"selected_{s}.parquet") for s in subs]
    m = pd.concat(fr, ignore_index=True)
    st = (SH / f"selected_{tag}.parquet").with_suffix(".parquet.tmp")
    m.to_parquet(st, index=False)
    st.replace(SH / f"selected_{tag}.parquet")
    logw(f"MERGE eighth {tag} rows={rows} sel={len(m)}\n")
if __name__ == "__main__":
    tag = sys.argv[1]
    subs = split8(tag)
    for s in subs:
        if (SH / f"matches_{s}.tsv").exists() and (SH / f"selected_{s}.parquet").exists():
            logw(f"SKIP eighth {s}\n")
            continue
        t0 = time.monotonic()
        while True:
            try:
                run(s)
                break
            except RuntimeError:
                if time.monotonic() - t0 > 1800:
                    raise
                logw("COOLDOWN 120s\n")
                time.sleep(120)
    merge8(tag)
    logw(f"EIGHTH {tag} DONE\n")
