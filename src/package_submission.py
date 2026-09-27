"""Validate the outputs and build the challenge's requested ZIP layout."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import zipfile
from pathlib import Path

from v1 import ROOT
from package_code import V5_SOURCES, V8_SOURCES


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--team", default="Astralis")
    parser.add_argument("--version", choices=["v1", "v3", "v4", "v5", "v8"], default="v1")
    parser.add_argument("--france-threshold", type=float,
                        help="Package a validated V5 France threshold variant")
    parser.add_argument("--already-validated", action="store_true",
                        help="Use only after validate_large.py passed for these exact output files")
    args = parser.parse_args()
    if args.france_threshold is not None and args.version != "v5":
        parser.error("--france-threshold requires --version v5")
    if args.france_threshold is not None and not 0.625 < args.france_threshold < 1:
        parser.error("France threshold must be between 0.625 and 1")
    variant = (f"_france_{round(args.france_threshold * 1000):03d}"
               if args.france_threshold is not None else "")
    output_dir = ROOT / ("output" if args.version == "v1"
                         else f"output_{args.version}" if not variant
                         else f"output_v6{variant}")
    if args.already_validated:
        summary_file = output_dir / "validation_summary.json"
        if not summary_file.is_file():
            raise FileNotFoundError("Expected validation_summary.json before packaging")
        summary = json.loads(summary_file.read_text(encoding="utf-8"))
        if not summary.get("ok") or summary.get("errors"):
            raise ValueError("Output validation summary reports errors")
    else:
        subprocess.run([sys.executable, str(ROOT / "src" / "validate_large.py"),
                        "--output-dir", str(output_dir)],
                       cwd=ROOT, check=True)
    if args.france_threshold is not None:
        experiment = json.loads((output_dir / "experiment_summary.json")
                                .read_text(encoding="utf-8"))
        if experiment["threshold_france"] != args.france_threshold:
            raise ValueError("Output directory has a different France threshold")
    submission = ROOT / "submission"
    submission.mkdir(exist_ok=True)
    slug = "".join(c if c.isalnum() or c in "-_" else "_" for c in args.team)
    suffix = "" if args.version == "v1" else f"_{args.version}{variant}"
    archive = submission / (f"{slug}_submission.zip" if args.version == "v8"
                            else f"{slug}{suffix}_submission.zip")
    code_root = Path("code") / "business_entity_resolution"
    methodology = (ROOT / "V1_DOCUMENTATION.md" if args.version == "v1"
                   else ROOT / "V3_REPRODUCTION.md" if args.version == "v3"
                   else ROOT / "V5_DOCUMENTATION.md" if args.version == "v5"
                   else ROOT / "V8_DOCUMENTATION.md" if args.version == "v8"
                   else ROOT / "Documentation_template.md")
    readme = (ROOT / "SOLUTION_README.md" if args.version == "v1"
              else ROOT / f"{args.version.upper()}_REPRODUCTION.md")
    manifest = [
        (output_dir / "matching_results.tsv", Path("output") / "matching_results.tsv"),
        (output_dir / "candidate_pairs.tsv", Path("output") / "candidate_pairs.tsv"),
        (methodology, Path("Documentation_template.md")),
        (readme, code_root / "README.md"),
        (ROOT / ("requirements-v6.txt" if args.version == "v8" else "requirements.txt"),
         code_root / "requirements.txt"),
        (ROOT / "utils" / "validate_submission.py", code_root / "utils" / "validate_submission.py"),
    ]
    if args.version == "v8":
        manifest.extend([
            (ROOT / "models" / "v5_lgb.txt", code_root / "models" / "v5_lgb.txt"),
            (ROOT / "models" / "v5_metrics.json", code_root / "models" / "v5_metrics.json"),
            (ROOT / "models" / "v5_audit3_metrics.json", code_root / "models" / "v5_audit3_metrics.json"),
        ])
        manifest.extend(
            (source, code_root / "models" / "v6_reranker" / source.name)
            for source in sorted((ROOT / "models" / "v6_reranker").glob("*"))
            if source.is_file()
        )
    else:
        manifest.extend([
            (ROOT / "models" / f"{args.version}_lgb.txt", code_root / "models" / f"{args.version}_lgb.txt"),
            (ROOT / "models" / f"{args.version}_metrics.json", code_root / "models" / f"{args.version}_metrics.json"),
        ])
    if args.version != "v1":
        results = f"{args.version.upper()}_RESULTS.md"
        manifest.append((ROOT / results, code_root / results))
    if args.version == "v5":
        manifest.append((ROOT / "models" / "v5_audit3_metrics.json",
                         code_root / "models" / "v5_audit3_metrics.json"))
        sources = [ROOT / "src" / name for name in V5_SOURCES]
        if args.france_threshold is not None:
            sources.append(ROOT / "src" / "make_france_threshold_variant.py")
    elif args.version == "v8":
        sources = [ROOT / "src" / source for source in dict.fromkeys(V8_SOURCES)]
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
