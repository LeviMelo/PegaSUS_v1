
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path.cwd()

PRODUCTION_STORAGE_CLEAN = (
    Path("src/pegasus/efg/promotion_apply.py"),
    Path("src/pegasus/output/maternal_child_compile_attach.py"),
    Path("src/pegasus/output/population_tensor_compile_attach.py"),
    Path("src/pegasus/output/race_bridge_attach.py"),
    Path("src/pegasus/dashboard/read_only.py"),
)

FORBIDDEN = (
    "import pyarrow.parquet as pq",
    "pq.read_table",
    "pq.write_table",
    "pl.read_parquet",
    ".write_parquet(",
)


def main() -> None:
    errors: list[str] = []
    findings: list[str] = []
    for rel in PRODUCTION_STORAGE_CLEAN:
        path = ROOT / rel
        if not path.exists():
            errors.append(f"missing expected file: {rel}")
            continue
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            for token in FORBIDDEN:
                if token in line:
                    findings.append(f"{rel}:{line_no}: {line.strip()}")
    if findings:
        errors.extend(f"storage boundary bypass remains: {item}" for item in findings)
    payload = {
        "audit": "slice28z_storage_boundary_adoption",
        "status": "failed" if errors else "passed",
        "errors": errors,
        "checked_files": [str(path) for path in PRODUCTION_STORAGE_CLEAN],
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
