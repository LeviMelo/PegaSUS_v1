from __future__ import annotations

import argparse
import json
from pathlib import Path

from pegasus.acceptance.contracts import summarize_run
from pegasus.source_artifacts.compile_policy import resolve_compile_source_reality


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Slice 12B compile source-reality integration.")
    parser.add_argument("--run", required=True)
    parser.add_argument("--expected-mode", default="fixture_only")
    args = parser.parse_args()

    run_dir = Path(args.run)
    run_config = json.loads((run_dir / "RunConfig.json").read_text(encoding="utf-8"))
    p_vector = json.loads((run_dir / "P_vector.json").read_text(encoding="utf-8"))
    user_intent = json.loads((run_dir / "UserIntent.json").read_text(encoding="utf-8"))
    manifest = json.loads((run_dir / "ReproducibilityManifest.json").read_text(encoding="utf-8"))
    summary = summarize_run(run_dir).as_manifest()

    mode = run_config.get("compile_source_mode")
    assert mode == args.expected_mode, f"RunConfig compile_source_mode mismatch: {mode!r}"
    assert p_vector.get("source_artifact_reality", {}).get("compile_source_mode") == args.expected_mode
    assert user_intent.get("compile_source_reality", {}).get("compile_source_mode") == args.expected_mode
    assert manifest.get("compile_source_mode") == args.expected_mode
    assert summary.get("compile_source_mode") == args.expected_mode

    if args.expected_mode == "fixture_only":
        assert run_config.get("source_reality_production_candidate") is False
        assert run_config.get("source_artifact_manifest_present") in {False, True}

    missing = resolve_compile_source_reality().as_manifest()
    assert missing["compile_source_mode"] == "fixture_only"
    assert missing["source_artifact_manifest_present"] is False

    print(
        "AUDIT PASSED: Slice 12B compile source-reality integration validated "
        f"mode={args.expected_mode} run={run_dir}"
    )


if __name__ == "__main__":
    main()
