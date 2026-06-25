from __future__ import annotations

import json
from pathlib import Path


def main() -> int:
    path = Path("src/pegasus/workflows/compile.py")
    text = path.read_text(encoding="utf-8")

    errors: list[str] = []

    impl_start = text.find("def _run_compile_impl(")
    impl_end = text.find("\ndef run_compile(", impl_start)
    if impl_start == -1 or impl_end == -1:
        errors.append("could_not_isolate__run_compile_impl")
    else:
        impl = text[impl_start:impl_end]
        flush_pos = impl.find("bundle_manager.flush_to_disk")
        if flush_pos == -1:
            errors.append("missing_bundle_manager_flush_to_disk_in__run_compile_impl")

        bad_stage_pos = impl.find('telemetry.stage("output_validation")')
        if bad_stage_pos != -1:
            errors.append("preflush_telemetry_output_validation_block_present")

        validate_pos = impl.find("validation = validate_output_bundle(run_dir=str(run_dir))")
        if validate_pos == -1:
            errors.append("missing_final_validate_output_bundle")
        elif flush_pos != -1 and validate_pos < flush_pos:
            errors.append("validate_output_bundle_occurs_before_bundle_manager_flush")

    payload = {
        "ok": not errors,
        "errors": errors,
        "checked": str(path),
        "contract": "compile.py may validate the 17-key bundle only after OutputBundleManager.flush_to_disk().",
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
