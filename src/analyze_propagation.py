"""Test target-to-target retrieval potential using labeled audit anchors."""

from __future__ import annotations

import csv
import re
from collections import Counter, defaultdict

import duckdb
import pandas as pd
from anyascii import anyascii
from rapidfuzz.distance import JaroWinkler

from v1 import DATASET, WORK, quoted_path


PUNCT = re.compile(r"[^a-z0-9]+")
LEGAL = re.compile(r"\b(inc|incorporated|llc|ltd|limited|corp|corporation|pvt|private|llp)\b")


def normalize(value: str) -> str:
    return PUNCT.sub(" ", anyascii(value or "").lower()).strip()


def get_fields(record: dict[str, str]) -> dict[str, object]:
    name = normalize(record["business_name"])
    address = normalize(record["business_address"])
    return {
        "name": name,
        "address": address,
        "name_key": PUNCT.sub("", LEGAL.sub(" ", name)),
        "addr_key": PUNCT.sub("", address),
        "addr_tokens": {v for v in address.split() if len(v) >= 5},
        "numbers": set(re.findall(r"[0-9]+", address)),
    }


def main() -> None:
    con = duckdb.connect()
    audit_ids = {row[0] for row in con.execute(
        f"SELECT entity_id FROM read_csv({quoted_path(DATASET / 'train' / 'train_source1.tsv')}, "
        "delim='\\t', header=true, all_varchar=true) "
        "WHERE hash(entity_id)%1000 >= 50 AND hash(entity_id)%1000 < 60"
    ).fetchall()}
    con.close()
    frame = pd.read_parquet(WORK / "train_audit_features.parquet",
                            columns=["s1_id", "target_id", "label"])
    anchors: dict[str, list[str]] = defaultdict(list)
    for s1, target in frame.loc[frame.label == 1, ["s1_id", "target_id"]].itertuples(index=False):
        anchors[s1].append(target)
    del frame
    misses = []
    with (DATASET / "train" / "train_ground_truth.tsv").open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            s1 = row["source1_entity_id"]
            if s1 not in audit_ids or not row["matched_entity_ids"]:
                continue
            found = set(anchors[s1])
            for target in row["matched_entity_ids"].split(","):
                if target not in found:
                    misses.append((s1, target))
    wanted = {target for group in anchors.values() for target in group}
    wanted.update(target for _, target in misses)
    records = {}
    for source in ("source2", "source3"):
        with (DATASET / "train" / f"train_{source}.tsv").open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                if row["entity_id"] in wanted:
                    records[row["entity_id"]] = get_fields(row)
        print(source, len(records), flush=True)
    counts = Counter()
    for s1, target in misses:
        counts["missed"] += 1
        if not anchors[s1]:
            continue
        counts["has_anchor"] += 1
        t = records[target]
        flags = set()
        best_name = best_address = 0.0
        for anchor_id in anchors[s1]:
            a = records[anchor_id]
            if a["name_key"] and a["name_key"] == t["name_key"]:
                flags.add("exact_name")
            if a["addr_key"] and a["addr_key"] == t["addr_key"]:
                flags.add("exact_address")
            if len(a["name_key"]) >= 5 and a["name_key"][:5] == t["name_key"][:5]:
                flags.add("name5")
            if len(a["addr_key"]) >= 8 and a["addr_key"][:8] == t["addr_key"][:8]:
                flags.add("address8")
            if a["addr_tokens"] & t["addr_tokens"]:
                flags.add("address_token")
            if a["numbers"] & t["numbers"]:
                flags.add("number")
            best_name = max(best_name, JaroWinkler.normalized_similarity(a["name"], t["name"]))
            best_address = max(best_address, JaroWinkler.normalized_similarity(a["address"], t["address"]))
        for flag in flags:
            counts[flag] += 1
        counts["name_jaro_085"] += best_name >= 0.85
        counts["address_jaro_085"] += best_address >= 0.85
        counts["name_or_address_085"] += best_name >= 0.85 or best_address >= 0.85
    for label, count in counts.most_common():
        print(label, count)


if __name__ == "__main__":
    main()
