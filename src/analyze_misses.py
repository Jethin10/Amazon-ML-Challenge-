"""Classify true validation links that v1 blocking failed to retrieve."""

from __future__ import annotations

import csv
import json
import re
import unicodedata
from collections import Counter

import duckdb
import pandas as pd
from rapidfuzz.distance import JaroWinkler

from v1 import DATASET, WORK, quoted_path


LEGAL = re.compile(r"\b(inc|incorporated|llc|ltd|limited|corp|corporation|pvt|private|llp)\b")
SPACES = re.compile(r"[^a-z0-9]+")
CITY = re.compile(r",\s*([^,]+),\s*[^,]+$")
NUMBER = re.compile(r"[0-9]+")


def normalize(value: str) -> str:
    value = unicodedata.normalize("NFKD", value or "")
    value = value.encode("ascii", "ignore").decode("ascii").lower()
    return SPACES.sub(" ", value).strip()


def key(value: str) -> str:
    return SPACES.sub("", value)


def fields(row: dict[str, str]) -> dict[str, str]:
    name = normalize(row["business_name"])
    address = normalize(row["business_address"])
    city_match = CITY.search(row["business_address"] or "")
    city = normalize(city_match.group(1)) if city_match else ""
    number = NUMBER.search(address)
    return {
        "name": name,
        "address": address,
        "name_key": key(LEGAL.sub(" ", name)),
        "addr_key": key(address),
        "city5": key(city)[:5],
        "addr_num": number.group() if number else "",
        "country": (row["country"] or "").strip().lower(),
    }


def main() -> None:
    con = duckdb.connect()
    path = DATASET / "train" / "train_source1.tsv"
    valid_ids = {
        row[0]
        for row in con.execute(
            f"SELECT entity_id FROM read_csv({quoted_path(path)}, delim='\\t', "
            "header=true, all_varchar=true) "
            "WHERE hash(entity_id) % 1000 >= 42 AND hash(entity_id) % 1000 < 50"
        ).fetchall()
    }
    con.close()
    covered_frame = pd.read_parquet(
        WORK / "train_valid_features.parquet", columns=["s1_id", "target_id", "label"]
    )
    covered = set(
        zip(covered_frame.loc[covered_frame.label == 1, "s1_id"],
            covered_frame.loc[covered_frame.label == 1, "target_id"])
    )
    del covered_frame
    missed: list[tuple[str, str]] = []
    true_links = 0
    with (DATASET / "train" / "train_ground_truth.tsv").open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            s1_id = row["source1_entity_id"]
            if s1_id not in valid_ids or not row["matched_entity_ids"]:
                continue
            for target_id in row["matched_entity_ids"].split(","):
                true_links += 1
                if (s1_id, target_id) not in covered:
                    missed.append((s1_id, target_id))
    print(f"Validation links: {true_links:,}; missed: {len(missed):,}", flush=True)
    wanted = {entity_id for pair in missed for entity_id in pair}
    records: dict[str, dict[str, str]] = {}
    for source in ("source1", "source2", "source3"):
        path = DATASET / "train" / f"train_{source}.tsv"
        with path.open(encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f, delimiter="\t"):
                if row["entity_id"] in wanted:
                    records[row["entity_id"]] = row
        print(f"Scanned {source}; found {len(records):,} relevant records", flush=True)
    summary = Counter()
    examples = []
    for s1_id, target_id in missed:
        s = fields(records[s1_id])
        t = fields(records[target_id])
        same_country = s["country"] == t["country"]
        name_similarity = JaroWinkler.normalized_similarity(s["name"], t["name"])
        address_similarity = JaroWinkler.normalized_similarity(s["address"], t["address"])
        name5 = len(s["name_key"]) >= 5 and s["name_key"][:5] == t["name_key"][:5]
        name4_city = (
            len(s["name_key"]) >= 4 and len(s["city5"]) >= 3
            and s["name_key"][:4] == t["name_key"][:4]
            and s["city5"] == t["city5"]
        )
        name4_number = (
            len(s["name_key"]) >= 4 and s["addr_num"] != ""
            and s["name_key"][:4] == t["name_key"][:4]
            and s["addr_num"] == t["addr_num"]
        )
        city_number = (
            len(s["city5"]) >= 3 and s["addr_num"] != ""
            and s["city5"] == t["city5"] and s["addr_num"] == t["addr_num"]
        )
        address8 = len(s["addr_key"]) >= 8 and s["addr_key"][:8] == t["addr_key"][:8]
        any_capped_route = name5 or name4_city or name4_number or city_number or address8
        shared_name_tokens = {
            token for token in s["name"].split()
            if len(token) >= 4 and not LEGAL.fullmatch(token)
        } & {
            token for token in t["name"].split()
            if len(token) >= 4 and not LEGAL.fullmatch(token)
        }
        shared_address_tokens = {
            token for token in s["address"].split() if len(token) >= 5
        } & {
            token for token in t["address"].split() if len(token) >= 5
        }
        summary["total"] += 1
        summary["country_mismatch"] += not same_country
        summary["missing_name_either"] += not s["name"] or not t["name"]
        summary["missing_address_either"] += not s["address"] or not t["address"]
        summary["name_similarity_ge_085"] += name_similarity >= 0.85
        summary["address_similarity_ge_085"] += address_similarity >= 0.85
        summary["name_similarity_lt_05"] += name_similarity < 0.5
        summary["address_similarity_lt_05"] += address_similarity < 0.5
        summary["same_name5_but_capped"] += same_country and name5
        summary["same_name4_city_but_capped"] += same_country and name4_city
        summary["same_name4_number_but_capped"] += same_country and name4_number
        summary["same_city_number_but_capped"] += same_country and city_number
        summary["same_address8_but_capped"] += same_country and address8
        summary["no_current_route_key"] += same_country and not any_capped_route
        summary["strong_address_weak_name"] += address_similarity >= 0.85 and name_similarity < 0.65
        summary["strong_name_weak_address"] += name_similarity >= 0.85 and address_similarity < 0.65
        summary["both_weak"] += name_similarity < 0.65 and address_similarity < 0.65
        summary["shared_name_token"] += bool(shared_name_tokens)
        summary["shared_name_token_city"] += bool(shared_name_tokens) and s["city5"] != "" and s["city5"] == t["city5"]
        summary["shared_name_token_number"] += bool(shared_name_tokens) and s["addr_num"] != "" and s["addr_num"] == t["addr_num"]
        summary["shared_address_token"] += bool(shared_address_tokens)
        summary["shared_address_token_name4"] += bool(shared_address_tokens) and s["name_key"][:4] == t["name_key"][:4]
        if len(examples) < 12 and (summary["total"] % 700 == 0):
            examples.append({
                "s1_id": s1_id, "target_id": target_id,
                "s1_name": records[s1_id]["business_name"],
                "target_name": records[target_id]["business_name"],
                "s1_address": records[s1_id]["business_address"],
                "target_address": records[target_id]["business_address"],
                "name_similarity": round(name_similarity, 3),
                "address_similarity": round(address_similarity, 3),
            })
    report = {"counts": dict(summary), "examples": examples}
    (WORK / "missed_analysis.json").write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
