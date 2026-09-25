"""Validate the outputs and build the challenge's requested ZIP layout."""

from __future__ import annotations

import argparse
import subprocess
import sys
import zipfile
from pathlib import Path

from v1 import ROOT
from package_code import V5_SOURCES


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--team", default="Astralis")
    parser.add_argument("--version", choices=["v1", "v3", "v4", "v5"], default="v1")
    parser.add_argument("--already-validated", action="store_true",
                        help="Use only after validate_large.py passed for these exact output files")
    args = parser.parse_args()
    output_dir = ROOT / ("output" if args.version == "v1" else f"output_{args.version}")
    if args.already_validated:
        if not (output_dir / "validation_summary.json").is_file():
            raise FileNotFoundError("Expected validation_summary.json before packaging")
    else:
        subprocess.run([sys.executable, str(ROOT / "src" / "validate_large.py"),
                        "--output-dir", str(output_dir)],
                       cwd=ROOT, check=True)
    submission = ROOT / "submission"
    submission.mkdir(exist_ok=True)
    slug = "".join(c if c.isalnum() or c in "-_" else "_" for c in args.team)
    suffix = "" if args.version == "v1" else f"_{args.version}"
    archive = submission / f"{slug}{suffix}_submission.zip"
    code_root = Path("code") / "business_entity_resolution"
    methodology = (ROOT / "V1_DOCUMENTATION.md" if args.version == "v1"
                   else ROOT / "V3_REPRODUCTION.md" if args.version == "v3"
                   else ROOT / "V5_DOCUMENTATION.md" if args.version == "v5"
                   else ROOT / "Documentation_template.md")
    readme = (ROOT / "SOLUTION_README.md" if args.version == "v1"
              else ROOT / f"{args.version.upper()}_REPRODUCTION.md")
    manifest = [
        (output_dir / "matching_results.tsv", Path("output") / "matching_results.tsv"),
        (output_dir / "candidate_pairs.tsv", Path("output") / "candidate_pairs.tsv"),
        (methodology, Path("Documentation_template.md")),
        (readme, code_root / "README.md"),
        (ROOT / "requirements.txt", code_root / "requirements.txt"),
        (ROOT / "models" / f"{args.version}_lgb.txt", code_root / "models" / f"{args.version}_lgb.txt"),
        (ROOT / "models" / f"{args.version}_metrics.json", code_root / "models" / f"{args.version}_metrics.json"),
        (ROOT / "utils" / "validate_submission.py", code_root / "utils" / "validate_submission.py"),
    ]
    if args.version != "v1":
        results = f"{args.version.upper()}_RESULTS.md"
        manifest.append((ROOT / results, code_root / results))
    if args.version == "v5":
        manifest.append((ROOT / "models" / "v5_audit3_metrics.json",
                         code_root / "models" / "v5_audit3_metrics.json"))
        sources = [ROOT / "src" / name for name in V5_SOURCES]
    else:
        sources = list((ROOT / "src").glob("*.py"))
    for source in sources:
        manifest.append((source, code_root / "src" / source.name))
    for source, _ in manifest:
        if not source.is_file():
            raise FileNotFoundError(source)
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=3, allowZip64=True) as output:
        for source, dest in manifest:
            print(f"Packing {dest}", flush=True)
            output.write(source, dest.as_posix())
    with zipfile.ZipFile(archive) as result:
        bad = result.testzip()
        if bad is not None:
            raise RuntimeError(f"Corrupt ZIP member: {bad}")
    print(f"Created {archive} ({archive.stat().st_size:,} bytes)", flush=True)


if __name__ == "__main__":
    main()
