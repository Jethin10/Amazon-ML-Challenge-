"""Split a candidate shard into quarters to fit DuckDB memory (OOM fix)."""
from __future__ import annotations
from pathlib import Path
SHARDS = Path(__file__).resolve().parents[1] / "work" / "v5_shards"
def split_quarter(index: int) -> list[str]:
    original = SHARDS / f"candidate_{index:03d}.tsv"
    with original.open("rb") as source:
        header = source.readline()
        lines = sum(1 for _ in source)
    per = (lines + 3) // 4
    tags = [f"{index:03d}{p}" for p in "abcd"]
    with original.open("rb") as source:
        assert source.readline() == header
        for tag in tags:
            shard = SHARDS / f"candidate_{tag}.tsv"
            if shard.exists():
                for _ in range(per):
                    if not source.readline():
                        break
                continue
            tmp = shard.with_suffix(".tsv.tmp")
            with tmp.open("wb") as dest:
                dest.write(header)
                for _ in range(per):
                    line = source.readline()
                    if not line:
                        break
                    dest.write(line)
            tmp.replace(shard)
            print("wrote", shard.name, flush=True)
    return tags
if __name__ == "__main__":
    import sys
    print(split_quarter(int(sys.argv[1])), flush=True)
