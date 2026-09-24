"""Stream both full test outputs through exact format and ID checks."""

from __future__ import annotations

import csv
import json
from itertools import zip_longest
from pathlib import Path

from v1 import DATASET, ROOT


def read_source_ids(path: Path) -> set[str]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        next(reader)
        return {row[0] for row in reader}


def main() -> None:
    expected = read_source_ids(DATASET / "test" / "test_source1.tsv")
    target_ids = read_source_ids(DATASET / "test" / "test_source2.tsv")
    target_ids.update(read_source_ids(DATASET / "test" / "test_source3.tsv"))
    seen: set[str] = set()
    errors: list[str] = []
    candidate_count = match_count = empty_count = 0
    matching = ROOT / "output" / "matching_results.tsv"
    candidates = ROOT / "output" / "candidate_pairs.tsv"
    with matching.open(encoding="utf-8", newline="") as fm, candidates.open(
        encoding="utf-8", newline=""
    ) as fc:
        mr = csv.reader(fm, delimiter="\t")
        cr = csv.reader(fc, delimiter="\t")
        if next(mr) != ["source1_entity_id", "matched_entity_ids"]:
            errors.append("Wrong matching_results.tsv header")
        if next(cr) != ["source1_entity_id", "candidate_entity_ids"]:
            errors.append("Wrong candidate_pairs.tsv header")
        for row_number, (m, c) in enumerate(zip_longest(mr, cr), start=2):
            if m is None or c is None:
                errors.append(f"Output files have different row counts at line {row_number}")
                break
            if len(m) != 2 or len(c) != 2:
                errors.append(f"Expected two columns at line {row_number}")
                continue
            s1_id = m[0]
            if s1_id != c[0]:
                errors.append(f"Files disagree on Source 1 ID at line {row_number}")
            if s1_id in seen:
                errors.append(f"Duplicate Source 1 row: {s1_id}")
            seen.add(s1_id)
            if s1_id not in expected:
                errors.append(f"Unknown Source 1 ID: {s1_id}")
            candidate_list = c[1].split(",") if c[1] else []
            match_list = m[1].split(",") if m[1] else []
            candidate_set = set(candidate_list)
            match_set = set(match_list)
            candidate_count += len(candidate_list)
            match_count += len(match_list)
            empty_count += not match_list
            if len(candidate_set) != len(candidate_list):
                errors.append(f"Duplicate candidate for {s1_id}")
            if len(match_set) != len(match_list):
                errors.append(f"Duplicate match for {s1_id}")
            if not match_set <= candidate_set:
                errors.append(f"Match absent from candidate list for {s1_id}")
            for entity_id in candidate_set:
                if not entity_id.startswith(("S2-", "S3-")) or entity_id not in target_ids:
                    errors.append(f"Invalid candidate ID {entity_id} for {s1_id}")
            if len(errors) >= 20:
                break
    missing = expected - seen
    if missing:
        errors.append(f"Missing {len(missing)} Source 1 rows, e.g. {min(missing)}")
    report = {
        "ok": not errors,
        "source1_rows": len(seen),
        "expected_source1_rows": len(expected),
        "candidate_pairs": candidate_count,
        "matched_pairs": match_count,
        "empty_predictions": empty_count,
        "errors": errors,
    }
    (ROOT / "output" / "validation_summary.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
