"""Size the effect of French legal forms using labelled data as a proxy.

France has no labels, so its true score is unmeasurable. What we CAN measure
is the model's sensitivity to a French-style legal suffix: take real
labelled US/India pairs, append French legal forms to the Source 1 name, and
observe how the same features and LightGBM react.

This is a sensitivity probe, NOT a France score estimate. It answers "how much
does a French suffix cost per pair", not "what will France score".
"""

from __future__ import annotations

import json
import time

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd

from evaluate_thresholds_streaming import metadata
from v1 import DB, ROOT, WORK, quoted_path
from v5_features import ALL_FEATURE_NAMES


FRENCH_SUFFIXES = ["SARL", "SAS", "EURL", "SASU", "SCI", "SA"]
US_SUFFIXES = ["Inc", "LLC", "Ltd", "Corp", "LLP"]

BASE = WORK / "train_audit3_v5_features.parquet"


def macro_f05(truth, pred, tp) -> float:
    fp = pred - tp
    fn = truth - tp
    den = 1.25 * tp + fp + 0.25 * fn
    out = np.divide(1.25 * tp, den, out=np.zeros_like(tp), where=den > 0)
    out[truth == 0] = (pred[truth == 0] == 0).astype(float)
    return float(out.mean())


FEATURE_SQL = """
    WITH recalculated AS (
    SELECT f.* EXCLUDE (
            name_jaro, name_edit, name_token_overlap, name_sorted_jaro,
            name_exact, name_contains, name5_equal, name4_equal,
            name_missing, name_len_ratio,
            latin_name_jaro, latin_name_sorted_jaro, latin_name_edit,
            latin_name_token_overlap, latin_name_contains,
            name_rank, name_gap, strong_name_count, joint_similarity
        ),
        jaro_winkler_similarity(s.name_norm,t.name_norm) AS name_jaro,
        1.0 - levenshtein(s.name_key,t.name_key)::DOUBLE /
              greatest(length(s.name_key),length(t.name_key),1) AS name_edit,
        array_length(list_intersect(string_split(s.name_norm,' '),
            string_split(t.name_norm,' ')))::DOUBLE /
            greatest(array_length(string_split(s.name_norm,' ')),
                     array_length(string_split(t.name_norm,' ')),1) AS name_token_overlap,
        jaro_winkler_similarity(s.addr_norm,t.addr_norm) AS address_jaro,
        jaro_winkler_similarity(
            array_to_string(list_sort(string_split(s.name_norm,' ')),' '),
            array_to_string(list_sort(string_split(t.name_norm,' ')),' ')) AS name_sorted_jaro,
        jaro_winkler_similarity(
            array_to_string(list_sort(string_split(s.addr_norm,' ')),' '),
            array_to_string(list_sort(string_split(t.addr_norm,' ')),' ')) AS address_sorted_jaro,
        array_length(list_intersect(string_split(s.addr_norm,' '),
            string_split(t.addr_norm,' ')))::DOUBLE /
            greatest(array_length(string_split(s.addr_norm,' ')),
                     array_length(string_split(t.addr_norm,' ')),1) AS address_token_overlap,
        (s.name_key=t.name_key AND s.name_key<>'')::INTEGER AS name_exact,
        (s.addr_key=t.addr_key AND s.addr_key<>'')::INTEGER AS address_exact,
        (s.name_key<>'' AND t.name_key<>'' AND
         (contains(s.name_key,t.name_key) OR contains(t.name_key,s.name_key)))::INTEGER AS name_contains,
        (s.addr_key<>'' AND t.addr_key<>'' AND
         (contains(s.addr_key,t.addr_key) OR contains(t.addr_key,s.addr_key)))::INTEGER AS address_contains,
        (s.name5=t.name5 AND s.name5<>'')::INTEGER AS name5_equal,
        (s.name4=t.name4 AND s.name4<>'')::INTEGER AS name4_equal,
        (s.city5=t.city5 AND s.city5<>'')::INTEGER AS city_equal,
        (s.addr_num<>'' AND t.addr_num<>'' AND
         ltrim(s.addr_num,'0')=ltrim(t.addr_num,'0'))::INTEGER AS number_equal,
        (s.name_norm='' OR t.name_norm='')::INTEGER AS name_missing,
        least(length(s.name_key),length(t.name_key))::DOUBLE /
            greatest(length(s.name_key),length(t.name_key),1) AS name_len_ratio,
        jaro_winkler_similarity(s.name_latin,tl.name_latin) AS latin_name_jaro,
        jaro_winkler_similarity(
            array_to_string(list_sort(string_split(s.name_latin,' ')),' '),
            array_to_string(list_sort(string_split(tl.name_latin,' ')),' ')
        ) AS latin_name_sorted_jaro,
        1.0 - levenshtein(s.name_latin,tl.name_latin)::DOUBLE /
              greatest(length(s.name_latin),length(tl.name_latin),1)
              AS latin_name_edit,
        array_length(list_intersect(string_split(s.name_latin,' '),
            string_split(tl.name_latin,' ')))::DOUBLE /
            greatest(array_length(string_split(s.name_latin,' ')),
                     array_length(string_split(tl.name_latin,' ')),1)
              AS latin_name_token_overlap,
        (s.name_latin<>'' AND tl.name_latin<>'' AND
         (contains(s.name_latin,tl.name_latin) OR
          contains(tl.name_latin,s.name_latin)))::INTEGER AS latin_name_contains
    FROM read_parquet({base}) f
    JOIN {tbl} s ON f.s1_id = s.entity_id
    JOIN train_targets t ON f.target_id = t.entity_id
    JOIN train_targets_latin tl ON f.target_id=tl.entity_id
    ), contextual AS (
        SELECT recalculated.*,
               row_number() OVER(
                   PARTITION BY s1_id
                   ORDER BY latin_name_sorted_jaro DESC,target_id
               )::FLOAT AS name_rank,
               (max(latin_name_sorted_jaro) OVER(PARTITION BY s1_id)
                -latin_name_sorted_jaro)::FLOAT AS name_gap,
               sum((latin_name_sorted_jaro>=0.90)::INT)
                   OVER(PARTITION BY s1_id)::FLOAT AS strong_name_count,
               least(latin_name_sorted_jaro,
                     latin_address_sorted_jaro)::FLOAT AS joint_similarity
        FROM recalculated
    )
    SELECT * FROM contextual
"""


def build_side(con, suffix, tbl) -> None:
    """Materialise the Source 1 side, optionally with a suffix appended."""
    con.execute(f"DROP TABLE IF EXISTS {tbl}")
    if suffix is None:
        con.execute(f"""
            CREATE TEMP TABLE {tbl} AS
            SELECT s.entity_id, s.country, s.name_norm, s.addr_norm,
                   s.name_key, s.addr_key, s.city_key, s.city_norm,
                   s.addr_num, s.name5, s.name4, s.city5, s.addr8,
                   l.name_latin, l.address_latin
            FROM train_s1 s JOIN ents e ON s.entity_id = e.entity_id
            JOIN train_s1_latin l ON s.entity_id=l.entity_id
        """)
    else:
        con.execute(f"""
            CREATE TEMP TABLE {tbl} AS
            WITH injected AS (
                SELECT s.entity_id, s.country, s.addr_norm, s.addr_key,
                       s.city_key, s.city_norm, s.addr_num, s.city5, s.addr8,
                       l.address_latin,
                       trim(regexp_replace(lower(strip_accents(
                           coalesce(s.business_name,'') || ' {suffix}')),
                           '[^a-z0-9]+', ' ', 'g')) AS name_norm
                FROM train_s1 s JOIN ents e ON s.entity_id = e.entity_id
                JOIN train_s1_latin l ON s.entity_id=l.entity_id
            ), keyed AS (
                SELECT *, regexp_replace(regexp_replace(name_norm,
                    '\\b(inc|incorporated|llc|ltd|limited|corp|corporation|pvt|private|llp)\\b',
                    ' ', 'g'), '[^a-z0-9]+', '', 'g') AS name_key
                FROM injected
            )
            SELECT *, left(name_key,5) AS name5, left(name_key,4) AS name4,
                   name_norm AS name_latin
            FROM keyed
        """)


def main() -> None:
    started = time.monotonic()
    con = duckdb.connect(str(DB), read_only=True)
    con.execute("SET threads=4")
    con.execute("SET memory_limit='3GB'")
    con.execute(f"SET temp_directory={quoted_path(WORK / 'tmp')}")
    lo, hi = 70, 80
    entities = [r[0] for r in con.execute(
        "SELECT entity_id FROM train_s1 WHERE hash(entity_id)%1000>=? "
        "AND hash(entity_id)%1000<? ORDER BY entity_id", [lo, hi]).fetchall()]
    position = {s: i for i, s in enumerate(entities)}
    con.register("ents", pd.DataFrame({"entity_id": entities}))

    truth, _ = metadata(entities)
    print("entities", len(entities), "truth links", int(truth.sum()),
          flush=True)

    model = lgb.Booster(model_file=str(ROOT / "models" / "v5_lgb.txt"))

    def variant(suffix, tag) -> dict:
        tbl = f"sens_{tag}"
        build_side(con, suffix, tbl)
        out = con.execute(FEATURE_SQL.format(base=quoted_path(BASE),
                                             tbl=tbl)).fetchdf()
        p = model.predict(out[ALL_FEATURE_NAMES].to_numpy(dtype=np.float32),
                          num_threads=4, pred_early_stop=True,
                          pred_early_stop_freq=10, pred_early_stop_margin=10)
        group = np.fromiter((position[s] for s in out.s1_id), dtype=np.int32,
                            count=len(out))
        y = out.label.to_numpy(dtype=np.int8)
        best = (0.0, 0.625)
        for thr in np.round(np.arange(0.3, 0.951, 0.025), 3):
            sel = p >= thr
            value = macro_f05(
                truth.astype(np.float64),
                np.bincount(group[sel], minlength=len(entities)).astype(np.float64),
                np.bincount(group[sel & (y == 1)],
                            minlength=len(entities)).astype(np.float64))
            if value > best[0]:
                best = (value, float(thr))
        con.execute(f"DROP TABLE IF EXISTS {tbl}")
        del out
        return {"variant": tag, "suffix": suffix or "none",
                "best_macro_f05": round(best[0], 5), "threshold": best[1]}

    results = [variant(None, "baseline")]
    print(results[-1], round(time.monotonic() - started, 1), flush=True)
    for suffix in FRENCH_SUFFIXES[:3]:
        results.append(variant(suffix, "french_suffix"))
        print(results[-1], round(time.monotonic() - started, 1), flush=True)
    for suffix in US_SUFFIXES[:2]:
        results.append(variant(suffix, "us_suffix_control"))
        print(results[-1], round(time.monotonic() - started, 1), flush=True)
    con.close()

    base = results[0]["best_macro_f05"]
    for item in results:
        item["delta_vs_baseline"] = round(item["best_macro_f05"] - base, 5)
    out_path = WORK / "french_suffix_sensitivity.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("\n=== legal-suffix sensitivity (labelled audit3 proxy) ===")
    print(f"baseline (no injection)   {base:.5f}")
    for item in results[1:]:
        print(f"{item['variant']:20s} {item['suffix']:5s} "
              f"{item['best_macro_f05']:.5f}  delta {item['delta_vs_baseline']:+.5f}")
    print("saved", out_path)


if __name__ == "__main__":
    main()
