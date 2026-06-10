from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

from pegasus.output.validate import validate_output_bundle


def fail(payload: dict) -> None:
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2))
    sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    args = parser.parse_args()

    run = Path(args.run)
    failures: list[dict] = []

    positive = validate_output_bundle(run_dir=str(run))
    if not positive.ok:
        failures.append({"kind": "positive_validation", "errors": positive.errors})

    with tempfile.TemporaryDirectory(prefix="pegasus_slice2e_validator_") as tmp:
        tmp_root = Path(tmp)

        extra = tmp_root / "extra"
        shutil.copytree(run, extra)
        (extra / "extra_artifact.txt").write_text("poison", encoding="utf-8")
        result = validate_output_bundle(run_dir=str(extra))
        if result.ok or not any("extra first-class artifact" in error for error in result.errors):
            failures.append({"kind": "negative_extra_artifact", "ok": result.ok, "errors": result.errors})

        bad_manifest = tmp_root / "bad_manifest"
        shutil.copytree(run, bad_manifest)
        manifest_path = bad_manifest / "ReproducibilityManifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["telemetry"]["total_wall_seconds"] = -1
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8")
        result = validate_output_bundle(run_dir=str(bad_manifest))
        if result.ok or not any("total_wall_seconds" in error for error in result.errors):
            failures.append({"kind": "negative_bad_telemetry", "ok": result.ok, "errors": result.errors})

    if failures:
        fail({"status": "failed", "failures": failures})

    print("AUDIT PASSED: Slice 2E output validator accepts valid compile bundle and rejects poisoned bundles.")


if __name__ == "__main__":
    main()
