"""Refresh a readable training, evaluation, and inference dashboard."""

from __future__ import annotations

import json
import time
from datetime import datetime

from v1 import ROOT


SHARDS = ROOT / "work" / "v5_shards"
OUTPUT = ROOT / "output_v5"
PROGRESS = OUTPUT / "progress.txt"
TRAIN = json.loads((ROOT / "models" / "v5_metrics.json").read_text())
AUDIT = json.loads((ROOT / "models" / "v5_audit3_metrics.json").read_text())
STAGES = [
    ("features.parquet", "building pair features"),
    ("v4_features.parquet", "building name and address features"),
    ("v5_features.parquet", "adding candidate context"),
]


def batch_counts(path) -> tuple[int, int, int]:
    rows = matched_rows = predictions = 0
    with path.open("rb") as stream:
        for line in stream:
            rows += 1
            ids = line.rstrip(b"\r\n").split(b"\t", 1)[1]
            if ids:
                matched_rows += 1
                predictions += ids.count(b",") + 1
    return rows, matched_rows, predictions


def snapshot(cache: dict[str, tuple[int, int, int]]) -> str:
    completed = sorted(SHARDS.glob("matches_???.tsv"))
    count = len(completed)
    for path in completed:
        if path.name not in cache:
            cache[path.name] = batch_counts(path)
    rows = sum(cache[p.name][0] for p in completed)
    matched = sum(cache[p.name][1] for p in completed)
    predictions = sum(cache[p.name][2] for p in completed)
    finalization = OUTPUT / "finalization_status.json"
    phase = json.loads(finalization.read_text()).get("phase") if finalization.exists() else "not started"
    now = datetime.now().astimezone()
    lines = [
        "ASTRALIS V5 - TRAINING, EVALUATION, AND INFERENCE",
        f"Updated: {now:%Y-%m-%d %H:%M:%S %Z}  |  refreshes every 3 seconds",
        "",
        "TRAINING - COMPLETED",
        f"  Fitted candidate pairs: {TRAIN['fit_pairs']:,}",
        f"  Best trees: {TRAIN['best_iteration']}  |  selected threshold: {TRAIN['threshold']:.3f}",
        f"  Training and threshold search time: {TRAIN['elapsed_seconds'] / 60:.1f} minutes",
        "",
        "EVALUATION - LABELED DATA",
        f"  Validation macro F0.5: {TRAIN['valid_macro_f05']:.5f}",
        f"  Untouched holdout macro F0.5: {AUDIT['macro_f05']:.5f}",
        "  Previous V4 on same holdout: 0.92402",
        f"  Holdout candidate recall: {AUDIT['candidate_recall']:.2%}",
        f"  India F0.5: {AUDIT['by_country']['india']['macro_f05']:.5f}"
        f"  |  US F0.5: {AUDIT['by_country']['us']['macro_f05']:.5f}",
        "  Portal: V4 = 0.896  |  V5 = pending upload and evaluation",
        "",
        "LIVE TEST INFERENCE",
        f"  Completed batches: {count}/20 ({count / 20:.0%})",
        f"  Source 1 rows written: {rows:,} / 1,732,544",
        f"  Rows with matches: {matched:,}  |  predicted links: {predictions:,}",
    ]
    if count < 20:
        tag=f"{count:03d}"
        half=""
        piece_label=""
        if count == 19 and (SHARDS / "candidate_019c.tsv").exists():
            quarter = "c" if not (SHARDS / "matches_019c.tsv").exists() else "d"
            for number in range(8):
                candidate_tag=f"019{quarter}e{number}"
                if not (SHARDS / f"matches_{candidate_tag}.tsv").exists():
                    tag=candidate_tag
                    piece_label=f" (quarter {quarter.upper()}, piece {number + 1}/8)"
                    break
            else:
                tag=f"019{quarter}"
                piece_label=f" (merging quarter {quarter.upper()})"
        elif (SHARDS/f"candidate_{tag}a.tsv").exists():
            half="b" if (SHARDS/f"matches_{tag}a.tsv").exists() else "a"
            tag+=half
        prefix = f"test_v5_batch{tag}_"
        stage = "preparing candidates"
        size = 0
        for suffix, description in STAGES:
            path = ROOT / "work" / f"{prefix}{suffix}"
            if path.exists():
                stage, size = description, path.stat().st_size
        if (SHARDS / f"selected_{tag}.parquet").exists():
            stage = "scoring model and writing batch output"
        half_label=f" (half {half.upper()} of 2)" if half else ""
        lines.append(f"  Current: batch {count + 1}/20{piece_label or half_label} - {stage}")
        if size:
            lines.append(f"  Current feature file: {size / 1_000_000:.0f} MB")
    if completed:
        last = completed[-1]
        lines.append(f"  Last completed: {last.name} at "
                     f"{datetime.fromtimestamp(last.stat().st_mtime):%H:%M:%S}")
        if count > 1:
            average = (last.stat().st_mtime - completed[0].stat().st_mtime) / (count - 1)
            remaining_minutes = round((20 - count) * average / 60)
            lines.append(f"  Approximate inference time left: {remaining_minutes // 60}h "
                         f"{remaining_minutes % 60}m (varies by batch)")
    lines.extend(["", f"FINALIZATION: {phase}",
                  "Upload only when finalization says ready. Local scores do not predict France."])
    return "\n".join(lines) + "\n"


def main() -> None:
    OUTPUT.mkdir(exist_ok=True)
    cache: dict[str, tuple[int, int, int]] = {}
    while True:
        content = snapshot(cache)
        temporary = PROGRESS.with_suffix(".txt.tmp")
        temporary.write_text(content, encoding="utf-8")
        while True:
            try:
                temporary.replace(PROGRESS)
                break
            except PermissionError:
                # Windows readers can briefly hold the destination open.
                time.sleep(0.2)
        if "FINALIZATION: ready" in content or "FINALIZATION: failed" in content:
            break
        time.sleep(3)


if __name__ == "__main__":
    main()
