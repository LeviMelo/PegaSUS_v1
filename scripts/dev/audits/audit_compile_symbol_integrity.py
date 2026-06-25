from __future__ import annotations

import json
import inspect
from pathlib import Path


REQUIRED_COMPILE_SYMBOLS = [
    "_intent_municipality_filter_cod6",
    "_geo_scope_from_intent",
    "_context_policy_enabled",
    "_compile_population_tensor_mode",
    "_compiler_architecture_metadata",
    "_external_compile_inputs",
    "_run_compile_impl",
    "run_compile",
]


def main() -> int:
    import pegasus.workflows.compile as compile_workflow
    import pegasus.workflows.pirs_pipeline as pirs_pipeline

    errors: list[str] = []

    for name in REQUIRED_COMPILE_SYMBOLS:
        if not hasattr(compile_workflow, name):
            errors.append(f"missing compile symbol: {name}")

    source = Path("src/pegasus/workflows/compile.py").read_text(encoding="utf-8")
    forbidden = [
        "fixture_only",
        "quarantined_fixture_only",
        "fixture_compatibility_modules",
        "compile_smoke_manifest",
        "compile_smoke_population",
        "tests/fixtures",
        "create_empty_output_bundle",
    ]
    for token in forbidden:
        if token in source:
            errors.append(f"compile.py contains forbidden token: {token}")

    impl = inspect.getsource(compile_workflow._run_compile_impl)
    if "external_inputs = _external_compile_inputs" not in impl:
        errors.append("_run_compile_impl does not assign source paths from _external_compile_inputs")
    for name in ("sim_events_path", "sinasc_events_path", "sidra_facts_path"):
        if f'{name} = external_inputs[' not in impl:
            errors.append(f"_run_compile_impl does not bind {name} from source manifest")

    summary_source = inspect.getsource(pirs_pipeline.pirs_planning_pipeline_summary)
    if "payload_map = _as_mapping(payload)" not in summary_source:
        errors.append("pirs_planning_pipeline_summary is not Pylance-safe around optional payloads")

    payload = {
        "ok": not errors,
        "errors": errors,
        "contract": "compile.py must expose all production helpers and bind source paths from materialized source manifest; pirs_pipeline must avoid optional .get chains.",
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
