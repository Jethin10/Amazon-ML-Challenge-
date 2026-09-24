"""Validate the outputs and build the challenge's requested ZIP layout."""

from __future__ import annotations

import argparse
import subprocess
import sys
import zipfile
from pathlib import Path

from v1 import ROOT


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--team", default="Astralis")
    args = parser.parse_args()
    subprocess.run([sys.executable, str(ROOT / "src" / "validate_large.py")],
                   cwd=ROOT, check=True)
    submission = ROOT / "submission"
    submission.mkdir(exist_ok=True)
    slug = "".join(c if c.isalnum() or c in "-_" else "_" for c in args.team)
    archive = submission / f"{slug}_submission.zip"
    code_root = Path("code") / "business_entity_resolution"
    manifest = {
        ROOT / "output" / "matching_results.tsv": Path("output") / "matching_results.tsv",
        ROOT / "output" / "candidate_pairs.tsv": Path("output") / "candidate_pairs.tsv",
        ROOT / "Documentation_template.md": Path("Documentation_template.md"),
        ROOT / "SOLUTION_README.md": code_root / "README.md",
        ROOT / "requirements.txt": code_root / "requirements.txt",
        ROOT / "models" / "v1_lgb.txt": code_root / "models" / "v1_lgb.txt",
        ROOT / "models" / "v1_metrics.json": code_root / "models" / "v1_metrics.json",
        ROOT / "utils" / "validate_submission.py": code_root / "utils" / "validate_submission.py",
    }
    for source in (ROOT / "src").glob("*.py"):
        manifest[source] = code_root / "src" / source.name
    for source in manifest:
        if not source.is_file():
            raise FileNotFoundError(source)
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=3, allowZip64=True) as output:
        for source, dest in manifest.items():
            print(f"Packing {dest}", flush=True)
            output.write(source, dest.as_posix())
    with zipfile.ZipFile(archive) as result:
        bad = result.testzip()
        if bad is not None:
            raise RuntimeError(f"Corrupt ZIP member: {bad}")
    print(f"Created {archive} ({archive.stat().st_size:,} bytes)", flush=True)


if __name__ == "__main__":
    main()
