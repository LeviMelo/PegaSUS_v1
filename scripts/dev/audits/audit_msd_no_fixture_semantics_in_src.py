from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path("src/pegasus")

ALLOWED = {
    "src/pegasus/output/bundle.py",
    "src/pegasus/output/source_reality_guard.py",
}

TOKENS = [
    "tests/fixtures",
    "fixture_only",
    "quarantined_fixture_only",
    "fixture_compatibility_modules",
    "create_empty_output_bundle",
    "sim_raw_fixture",
    "sinasc_raw_fixture",
    "cnes_raw_fixture",
    "sih_raw_fixture",
    "compile_smoke_manifest",
    "compile_smoke_population",
    "slice0_empty",
    "scaffold run",
]

PATTERNS = [re.compile(re.escape(token)) for token in TOKENS]


def main() -> int:
    errors: list[str] = []
    for path in sorted(ROOT.rglob("*.py")):
        rel = path.as_posix()
        if rel in ALLOWED:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for token, pattern in zip(TOKENS, PATTERNS):
            if pattern.search(text):
                errors.append(f"{rel} contains forbidden production token: {token}")
    payload = {
        "ok": not errors,
        "errors": errors,
        "contract": "src/pegasus production code must not expose development-source fallback semantics.",
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
