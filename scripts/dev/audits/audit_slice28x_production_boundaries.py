from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Any

ROOT = Path(__file__).resolve().parents[3]

STORAGE_FORBIDDEN = (
    "pyarrow.parquet",
    "pq.read_table(",
    "pq.write_table(",
    "pl.read_parquet(",
    ".write_parquet(",
)

COMPUTE_FORBIDDEN = (
    "generator.manual_seed(",
)

# Slice 28X-28ZC is a production-boundary regression gate for modules that
# were explicitly brought under the output.table_io / compute.random contracts.
# It is not a repository-wide ban on Polars/Arrow inside unfinished SHE/adaptor code.
GUARDED_STORAGE_FILES = {
    "src/pegasus/efg/promotion_apply.py",
    "src/pegasus/output/maternal_child_compile_attach.py",
    "src/pegasus/output/population_tensor_compile_attach.py",
    "src/pegasus/output/race_bridge_attach.py",
    "src/pegasus/output/sidra_denominator_anchor.py",
    "src/pegasus/dashboard/read_only.py",
}

GUARDED_COMPUTE_FILES = {
    "src/pegasus/pirs/hsic.py",
}

MAX_REPORTED_ERRORS = 20


@dataclass(frozen=True)
class BoundaryViolation:
    kind: str
    path: str
    line: int
    pattern: str
    text: str


def _line_hits(text: str, patterns: tuple[str, ...]) -> Iterable[tuple[int, str, str]]:
    for lineno, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        for pattern in patterns:
            if pattern in line:
                yield lineno, pattern, stripped


def _read(rel: str) -> str:
    path = ROOT / rel
    if not path.exists():
        return ""
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8", errors="replace")


def collect_boundary_violations() -> list[BoundaryViolation]:
    violations: list[BoundaryViolation] = []
    for rel in sorted(GUARDED_STORAGE_FILES):
        text = _read(rel)
        for lineno, pattern, line in _line_hits(text, STORAGE_FORBIDDEN):
            violations.append(BoundaryViolation("storage", rel, lineno, pattern, line))
    for rel in sorted(GUARDED_COMPUTE_FILES):
        text = _read(rel)
        for lineno, pattern, line in _line_hits(text, COMPUTE_FORBIDDEN):
            violations.append(BoundaryViolation("compute", rel, lineno, pattern, line))
    return violations


def _summarize(violations: list[BoundaryViolation]) -> list[dict[str, object]]:
    return [asdict(violation) for violation in violations[:MAX_REPORTED_ERRORS]]


def run_audit() -> dict[str, Any]:
    """Return the Slice 28X production-boundary audit payload.

    This function is the stable import API used by earlier Slice 28X tests.
    ``main()`` is only the CLI wrapper and must not be the sole entry point.
    """
    violations = collect_boundary_violations()
    storage = [violation for violation in violations if violation.kind == "storage"]
    compute = [violation for violation in violations if violation.kind == "compute"]
    return {
        "audit": "slice28x_production_boundaries",
        "status": "passed" if not violations else "failed",
        "storage_bypass_count": len(storage),
        "compute_bypass_count": len(compute),
        "error_count": len(violations),
        "errors": _summarize(violations),
        "errors_truncated": max(0, len(violations) - MAX_REPORTED_ERRORS),
        "warnings": [],
        "policy": {
            "scope": "guarded production modules refactored by slices 28Z-28ZB",
            "guarded_storage_files": sorted(GUARDED_STORAGE_FILES),
            "guarded_compute_files": sorted(GUARDED_COMPUTE_FILES),
            "storage_forbidden": list(STORAGE_FORBIDDEN),
            "compute_forbidden": list(COMPUTE_FORBIDDEN),
            "max_reported_errors": MAX_REPORTED_ERRORS,
        },
    }


def main() -> None:
    payload = run_audit()
    print(json.dumps(payload, indent=2, sort_keys=True))
    if payload["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
