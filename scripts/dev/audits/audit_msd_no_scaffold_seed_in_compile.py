from __future__ import annotations

import json
from pathlib import Path


CHECKS = [
    ("src/pegasus/workflows/compile.py", "create_empty_output_bundle"),
    ("src/pegasus/workflows/compile.py", "pegasus.output.bundle import"),
    ("src/pegasus/cli.py", "create_empty_output_bundle"),
    ("src/pegasus/cli.py", "pegasus.output.bundle import"),
]


def main() -> int:
    errors: list[str] = []
    for rel, token in CHECKS:
        path = Path(rel)
        if path.exists() and token in path.read_text(encoding="utf-8"):
            errors.append(f"{rel} still contains forbidden production scaffold initializer token: {token}")

    payload = {
        "ok": not errors,
        "errors": errors,
        "contract": "Production compile/CLI must use schema_seed, not Slice-0 scaffold bundle initialization.",
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
