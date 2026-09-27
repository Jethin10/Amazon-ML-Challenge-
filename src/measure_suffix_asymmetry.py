"""Measure French legal-suffix asymmetry between Source 1 and target sources.

v1.py strips `inc|llc|ltd|limited|corp|corporation|pvt|private|llp` from
name_key but NOT French forms (sarl, sas, eurl, sasu, sci, sa).

If Source 1 carries a French suffix that the target source omits, then:
  * name_key differs  -> the `exact_name` blocking route never fires, and
  * name_exact = 0    -> LightGBM loses its strongest exact-match signal.

This is measurable on the TEST set with NO labels: it only compares the
marginal rate of suffix presence per source. A large S1-vs-target gap is
direct evidence of systematic blocking failure for France.
"""

from __future__ import annotations

import csv
import json
import re
import time
from collections import Counter
from pathlib import Path

from v1 import DATASET

# The exact list v1.py currently removes.
STRIPPED = {"inc", "incorporated", "llc", "ltd", "limited", "corp",
            "corporation", "pvt", "private", "llp"}
# French forms that are NOT stripped today.
FRENCH = {"sarl", "sas", "sasu", "eurl", "sci", "sa", "snc", "gie", "sca",
          "scs", "sel", "srl", "spa"}

PUNCT = re.compile(r"[^a-z0-9]+")


def norm(value: str) -> str:
    return PUNCT.sub(" ", (value or "").lower()).strip()


def rate(path: Path, country: str) -> dict:
    total = 0
    with_suffix = 0
    stripped_present = 0
    tokens = Counter()
    last = None
    for _ in range(6):
        try:
            with path.open(encoding="utf-8", newline="") as handle:
                for row in csv.DictReader(handle, delimiter="\t"):
                    if row["country"].strip().lower() != country:
                        continue
                    total += 1
                    toks = set(norm(row["business_name"]).split())
                    for t in toks & FRENCH:
                        tokens[t] += 1
                    if toks & FRENCH:
                        with_suffix += 1
                    if toks & STRIPPED:
                        stripped_present += 1
            break
        except OSError as exc:  # transient Windows sharing violation
            last = exc
            time.sleep(3)
    else:
        raise RuntimeError(f"could not read {path}: {last}")
    n = max(total, 1)
    return {
        "rows": total,
        "has_french_legal_suffix_rate": round(with_suffix / n, 4),
        "has_stripped_legal_suffix_rate": round(stripped_present / n, 4),
        "top_tokens": tokens.most_common(8),
    }


def main() -> None:
    out = {}
    for part in ("test", "train"):
        for source in ("source1", "source2", "source3"):
            path = DATASET / part / f"{part}_{source}.tsv"
            if not path.is_file():
                continue
            for country in ("france", "us", "india"):
                if part == "train" and country == "france":
                    continue
                print(f"{part}/{source}/{country} ...", flush=True)
                out[f"{part}/{source}/{country}"] = rate(path, country)

    print("\n=== French legal-suffix rate by source ===")
    print(f"{'split/source/country':32s} {'rows':>10s} {'FR-suffix':>10s} {'stripped':>9s}")
    for key, item in out.items():
        print(f"{key:32s} {item['rows']:>10,} "
              f"{item['has_french_legal_suffix_rate']:>10.4f} "
              f"{item['has_stripped_legal_suffix_rate']:>9.4f}")

    s1 = out.get("test/source1/france")
    tgt = [v for k, v in out.items()
           if k.startswith("test/source2/france") or k.startswith("test/source3/france")]
    if s1 and tgt:
        print("\nFrance Source 1 vs targets (blocking-relevant):")
        for label, item in zip(("source2", "source3"), tgt):
            gap = s1["has_french_legal_suffix_rate"] - item["has_french_legal_suffix_rate"]
            print(f"  S1 {s1['has_french_legal_suffix_rate']:.4f} vs "
                  f"{label} {item['has_french_legal_suffix_rate']:.4f}  "
                  f"gap {gap:+.4f}")
        print("  -> a positive gap means name_key mismatches systematically")
    path = Path(__file__).resolve().parents[1] / "work" / "legal_suffix_asymmetry.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("\nsaved", path)


if __name__ == "__main__":
    main()
