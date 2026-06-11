from __future__ import annotations

import argparse
import sys

from pegasus.source_artifacts.contracts import (
    source_manifest_summary,
    validate_source_artifact_manifest,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit Slice 12A source artifact manifest reality gates.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--require-materialized-external", action="store_true")
    args = parser.parse_args()

    validation = validate_source_artifact_manifest(
        manifest_path=args.manifest,
        require_materialized_external=args.require_materialized_external,
    )
    if not validation["ok"]:
        for error in validation["errors"]:
            print(f"ERROR: {error}")
        return 1

    summary = source_manifest_summary(manifest_path=args.manifest)
    if summary["artifact_count"] < 1:
        print("ERROR: source artifact manifest is empty")
        return 1

    print(
        "AUDIT PASSED: Slice 12A source artifact manifest reality gate validated "
        f"mode={summary['compile_source_mode']} artifacts={summary['artifact_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
