from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit Slice 13A SHE substrate boundary artifacts.")
    parser.add_argument("--manifest", type=Path, required=False)
    parser.add_argument("--run", type=Path, required=False)
    parser.add_argument("--require-exclusions", action="store_true")
    args = parser.parse_args()

    if args.manifest is None and args.run is None:
        raise SystemExit("Provide --manifest or --run.")

    if args.manifest is not None:
        payload = load(args.manifest)
    else:
        path = args.run / "Tables" / "substrate_manifest.json"
        if not path.exists():
            raise SystemExit(f"substrate manifest missing from run: {path}")
        payload = load(path)

    errors: list[str] = []
    if payload.get("schema_version") != "1.0":
        errors.append("schema_version must be 1.0")
    if not payload.get("substrate_id"):
        errors.append("substrate_id missing")
    if payload.get("admissible_candidate_count", 0) < 0:
        errors.append("admissible_candidate_count invalid")
    if payload.get("excluded_field_count", 0) < 0:
        errors.append("excluded_field_count invalid")
    if args.require_exclusions and payload.get("excluded_field_count", 0) <= 0:
        errors.append("expected at least one substrate exclusion")

    for candidate in payload.get("candidates", []):
        if candidate.get("column") in payload.get("all_missing_columns", []):
            errors.append(f"all-missing column admitted: {candidate.get('column')}")
        if not candidate.get("registry_hash"):
            errors.append(f"candidate missing registry_hash: {candidate.get('column')}")

    for exclusion in payload.get("exclusions", []):
        reason = exclusion.get("reason")
        if reason in {"all_missing", "zero_variance_constant"} and exclusion.get("column") == "*":
            errors.append("wildcard zero-variance/all-missing exclusion is invalid")

    if errors:
        print(json.dumps({"ok": False, "errors": errors}, indent=2, sort_keys=True))
        return 1

    print(
        "AUDIT PASSED: Slice 13A SHE substrate boundary validated "
        f"candidates={payload.get('admissible_candidate_count')} "
        f"exclusions={payload.get('excluded_field_count')} "
        f"zero_variance={payload.get('zero_variance_exclusion_count')} "
        f"all_missing={payload.get('all_missing_exclusion_count')}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
