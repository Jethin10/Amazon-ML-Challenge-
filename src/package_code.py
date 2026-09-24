"""Build a small code-only archive for the leaderboard's code upload field."""

import argparse
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

from v1 import ROOT


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", choices=["v1", "v3"], default="v1")
    args = parser.parse_args()
    name = "Astralis_code.zip" if args.version == "v1" else "Astralis_v3_code.zip"
    output = ROOT / "submission" / name
    code_root = Path("code") / "business_entity_resolution"
    method = "Documentation_template.md" if args.version == "v1" else "V3_REPRODUCTION.md"
    files = {
        ROOT / method: Path("Documentation_template.md"),
        ROOT / ("SOLUTION_README.md" if args.version == "v1" else "V3_REPRODUCTION.md"):
            code_root / "README.md",
        ROOT / "requirements.txt": code_root / "requirements.txt",
        ROOT / "models" / f"{args.version}_lgb.txt": code_root / "models" / f"{args.version}_lgb.txt",
        ROOT / "models" / f"{args.version}_metrics.json": code_root / "models" / f"{args.version}_metrics.json",
        ROOT / "utils" / "validate_submission.py": code_root / "utils" / "validate_submission.py",
    }
    if args.version == "v3":
        files[ROOT / "V3_RESULTS.md"] = code_root / "V3_RESULTS.md"
    files.update({source: code_root / "src" / source.name for source in (ROOT / "src").glob("*.py")})
    for source in files:
        if not source.is_file():
            raise FileNotFoundError(source)
    output.parent.mkdir(exist_ok=True)
    with ZipFile(output, "w", compression=ZIP_DEFLATED, compresslevel=6) as archive:
        for source, destination in files.items():
            archive.write(source, destination.as_posix())
    with ZipFile(output) as archive:
        bad = archive.testzip()
        if bad:
            raise RuntimeError(f"Corrupt ZIP member: {bad}")
    print(f"{output} ({output.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
