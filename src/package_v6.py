"""Package the selected-pair V6 pipeline, trained models, and validated output."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import zipfile
from pathlib import Path

from package_code import V5_SOURCES
from v1 import ROOT


OUTPUT = ROOT / "output_v6_reranker_selected"
CODE_ROOT = Path("code") / "business_entity_resolution"
V6_SOURCES = (
    "build_v6_reranker_pairs.py", "train_v6_reranker.py",
    "evaluate_v6_reranker.py", "predict_v6_selected.py",
    "evaluate_thresholds_streaming.py",
    "make_france_threshold_variant.py", "package_v6.py",
)


def manifest(full: bool) -> list[tuple[Path, Path]]:
    files = [
        (ROOT / "V6_DOCUMENTATION.md", Path("Documentation_template.md")),
        (ROOT / "V6_REPRODUCTION.md", CODE_ROOT / "README.md"),
        (ROOT / "V5_REPRODUCTION.md", CODE_ROOT / "V5_REPRODUCTION.md"),
        (ROOT / "RESEARCH_NOTES.md", CODE_ROOT / "RESEARCH_NOTES.md"),
        (ROOT / "requirements-v6.txt", CODE_ROOT / "requirements.txt"),
        (ROOT / "requirements-v6.txt", CODE_ROOT / "requirements-v6.txt"),
        (ROOT / "models" / "v5_lgb.txt", CODE_ROOT / "models" / "v5_lgb.txt"),
        (ROOT / "models" / "v5_metrics.json", CODE_ROOT / "models" / "v5_metrics.json"),
        (ROOT / "models" / "v5_audit3_metrics.json",
         CODE_ROOT / "models" / "v5_audit3_metrics.json"),
        (ROOT / "utils" / "validate_submission.py",
         CODE_ROOT / "utils" / "validate_submission.py"),
        (ROOT / "scripts" / "watch_v6_progress.ps1",
         CODE_ROOT / "scripts" / "watch_v6_progress.ps1"),
    ]
    files.extend((ROOT / "models" / "v6_reranker" / name,
                  CODE_ROOT / "models" / "v6_reranker" / name)
                 for name in ("config.json", "model.safetensors",
                              "tokenizer.json", "tokenizer_config.json",
                              "training.json"))
    files.extend((ROOT / "src" / name, CODE_ROOT / "src" / name)
                 for name in sorted(set(V5_SOURCES + V6_SOURCES)))
    if full:
        files = [
            (OUTPUT / "matching_results.tsv", Path("output") / "matching_results.tsv"),
            (OUTPUT / "candidate_pairs.tsv", Path("output") / "candidate_pairs.tsv"),
        ] + files
    return files


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--team", default="Astralis")
    parser.add_argument("--full", action="store_true",
                        help="Include validated output TSVs")
    parser.add_argument("--already-validated", action="store_true")
    args = parser.parse_args()
    if args.full:
        if not args.already_validated:
            subprocess.run([sys.executable, str(ROOT / "src" / "validate_large.py"),
                            "--output-dir", str(OUTPUT)], cwd=ROOT, check=True)
        else:
            summary = json.loads((OUTPUT / "validation_summary.json")
                                 .read_text(encoding="utf-8"))
            if not summary.get("ok") or summary.get("errors"):
                raise ValueError("Output validation did not pass")
    files = manifest(args.full)
    for source, _ in files:
        if not source.is_file():
            raise FileNotFoundError(source)
    submission = ROOT / "submission"
    submission.mkdir(exist_ok=True)
    slug = "".join(c if c.isalnum() or c in "-_" else "_" for c in args.team)
    suffix = "submission" if args.full else "code"
    archive = submission / f"{slug}_v6_{suffix}.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=3, allowZip64=True) as output:
        for source, destination in files:
            print("packing", destination, flush=True)
            output.write(source, destination.as_posix())
    with zipfile.ZipFile(archive) as result:
        bad = result.testzip()
        if bad is not None:
            raise RuntimeError(f"Corrupt ZIP member: {bad}")
    print("created", archive, "bytes", archive.stat().st_size, flush=True)


if __name__ == "__main__":
    main()
