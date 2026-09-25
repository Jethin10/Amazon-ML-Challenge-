"""Build a small code-only archive for the leaderboard's code upload field."""

import argparse
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

from v1 import ROOT


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", choices=["v1", "v3", "v4"], default="v1")
    args = parser.parse_args()
    name = ("Astralis_code.zip" if args.version == "v1"
            else f"Astralis_{args.version}_code.zip")
    output = ROOT / "submission" / name
    code_root = Path("code") / "business_entity_resolution"
    method = ("V1_DOCUMENTATION.md" if args.version == "v1"
              else "V3_REPRODUCTION.md" if args.version == "v3"
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
    files.extend((source, code_root / "src" / source.name) for source in (ROOT / "src").glob("*.py"))
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
