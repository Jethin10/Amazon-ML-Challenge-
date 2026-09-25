"""Build a small code-only archive for the leaderboard's code upload field."""

import argparse
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

from v1 import ROOT


V5_SOURCES = (
    "v1.py", "v2.py", "v3_features.py", "v4.py", "v5_features.py",
    "build_latin.py", "predict_v4.py", "generate_v5_candidates.py",
    "train_v4.py", "train_v5.py", "audit_v5.py", "predict_v5.py",
    "run_v5_batches.py", "split_quarter.py", "split_eighth.py",
    "merge_v5_parts.py", "validate_large.py", "package_code.py",
    "package_submission.py", "watch_v5_progress.py",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", choices=["v1", "v3", "v4", "v5"], default="v1")
    parser.add_argument("--france-threshold", type=float,
                        help="Include the France postprocessing step for V5")
    args = parser.parse_args()
    if args.france_threshold is not None and args.version != "v5":
        parser.error("--france-threshold requires --version v5")
    if args.france_threshold is not None and not 0.625 < args.france_threshold < 1:
        parser.error("France threshold must be between 0.625 and 1")
    tag = (f"_france_{round(args.france_threshold * 1000):03d}"
           if args.france_threshold is not None else "")
    name = ("Astralis_code.zip" if args.version == "v1"
            else f"Astralis_{args.version}{tag}_code.zip")
    output = ROOT / "submission" / name
    code_root = Path("code") / "business_entity_resolution"
    method = ("V1_DOCUMENTATION.md" if args.version == "v1"
              else "V3_REPRODUCTION.md" if args.version == "v3"
              else "V5_DOCUMENTATION.md" if args.version == "v5"
              else "Documentation_template.md")
    files = [
        (ROOT / method, Path("Documentation_template.md")),
        (ROOT / ("SOLUTION_README.md" if args.version == "v1"
                 else f"{args.version.upper()}_REPRODUCTION.md"),
         code_root / "README.md"),
        (ROOT / "requirements.txt", code_root / "requirements.txt"),
        (ROOT / "models" / f"{args.version}_lgb.txt", code_root / "models" / f"{args.version}_lgb.txt"),
        (ROOT / "models" / f"{args.version}_metrics.json", code_root / "models" / f"{args.version}_metrics.json"),
        (ROOT / "utils" / "validate_submission.py", code_root / "utils" / "validate_submission.py"),
    ]
    if args.version != "v1":
        results = f"{args.version.upper()}_RESULTS.md"
        files.append((ROOT / results, code_root / results))
    if args.version == "v5":
        files.append((ROOT / "models" / "v5_audit3_metrics.json",
                      code_root / "models" / "v5_audit3_metrics.json"))
        sources = [ROOT / "src" / name for name in V5_SOURCES]
        if args.france_threshold is not None:
            sources.append(ROOT / "src" / "make_france_threshold_variant.py")
    else:
        sources = list((ROOT / "src").glob("*.py"))
    files.extend((source, code_root / "src" / source.name) for source in sources)
    for source, _ in files:
        if not source.is_file():
            raise FileNotFoundError(source)
    output.parent.mkdir(exist_ok=True)
    with ZipFile(output, "w", compression=ZIP_DEFLATED, compresslevel=6) as archive:
        for source, destination in files:
            archive.write(source, destination.as_posix())
    with ZipFile(output) as archive:
        bad = archive.testzip()
        if bad:
            raise RuntimeError(f"Corrupt ZIP member: {bad}")
    print(f"{output} ({output.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
