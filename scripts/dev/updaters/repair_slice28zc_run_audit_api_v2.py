from __future__ import annotations

import ast
import shutil
import zipfile
from pathlib import Path

ROOT = Path.cwd()
UPDATER_REL = Path("scripts/dev/updaters/repair_slice28zc_run_audit_api_v2.py")

AUDIT_28X = r'''
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
'''.lstrip()

TEST_API = r'''
from __future__ import annotations

from scripts.dev.audits.audit_slice28x_production_boundaries import run_audit


def test_slice28zc_slice28x_audit_preserves_run_audit_import_api() -> None:
    result = run_audit()
    assert result["audit"] == "slice28x_production_boundaries"
    assert result["status"] == "passed"
    assert result["storage_bypass_count"] == 0
    assert result["compute_bypass_count"] == 0
    assert result["error_count"] == 0
'''.lstrip()

DOC_NOTE = """

## Slice 28ZC repair — audit import API compatibility

`audit_slice28x_production_boundaries.py` exposes `run_audit()` as its stable import API. CLI execution must call this function rather than duplicating payload construction inside `main()`. This preserves compatibility with earlier Slice 28X integration tests while keeping the stricter scoped boundary policy introduced in Slice 28ZC.
"""


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def _backup(path: Path, suffix: str) -> None:
    if path.exists():
        backup = path.with_name(path.name + suffix)
        if not backup.exists():
            shutil.copy2(path, backup)


def _append_doc_note() -> None:
    path = ROOT / "docs" / "production_boundaries.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    text = path.read_text(encoding="utf-8") if path.exists() else "# Production Boundaries\n"
    if "Slice 28ZC repair — audit import API compatibility" not in text:
        path.write_text(text.rstrip() + DOC_NOTE, encoding="utf-8", newline="\n")


def _replace_top_level_string_assignment(path: Path, name: str, new_value: str, suffix: str) -> None:
    if not path.exists():
        return
    text = path.read_text(encoding="utf-8")
    module = ast.parse(text)
    target: ast.Assign | ast.AnnAssign | None = None
    for node in module.body:
        if isinstance(node, ast.Assign):
            if any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
                target = node
                break
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == name:
            target = node
            break
    if target is None or target.lineno is None or target.end_lineno is None:
        return
    _backup(path, suffix)
    lines = text.splitlines()
    replacement = f"{name} = r'''\n{new_value.rstrip()}\n'''.lstrip()"
    new_lines = lines[: target.lineno - 1] + replacement.splitlines() + lines[target.end_lineno :]
    path.write_text("\n".join(new_lines) + "\n", encoding="utf-8", newline="\n")


def _self_check() -> None:
    audit = ROOT / "scripts/dev/audits/audit_slice28x_production_boundaries.py"
    text = audit.read_text(encoding="utf-8")
    required = [
        "def run_audit()",
        "collect_boundary_violations()",
        "GUARDED_STORAGE_FILES",
        "GUARDED_COMPUTE_FILES",
        "MAX_REPORTED_ERRORS",
        "errors_truncated",
        "guarded production modules refactored by slices 28Z-28ZB",
    ]
    missing = [token for token in required if token not in text]
    forbidden = ["SRC_ROOT.rglob", "src/pegasus/she/maternal_child_linkage.py"]
    present = [token for token in forbidden if token in text]
    if missing or present:
        raise RuntimeError(f"slice28zc run_audit API repair self-check failed; missing={missing}; forbidden_present={present}")


def main() -> None:
    audit_path = ROOT / "scripts/dev/audits/audit_slice28x_production_boundaries.py"
    test_path = ROOT / "tests/unit/test_slice28zc_boundary_audit_api.py"
    _backup(audit_path, ".slice28zc_run_audit_api_v2.bak")
    _backup(test_path, ".slice28zc_run_audit_api_v2.bak")
    _write(audit_path, AUDIT_28X)
    _write(test_path, TEST_API)
    _append_doc_note()

    # Keep generated updaters from reintroducing the missing run_audit API if rerun.
    _replace_top_level_string_assignment(
        ROOT / "scripts/dev/updaters/apply_slice28zc_boundary_audit_hardening.py",
        "AUDIT_28X",
        AUDIT_28X,
        ".slice28zc_run_audit_api_v2.bak",
    )
    _replace_top_level_string_assignment(
        ROOT / "scripts/dev/updaters/repair_slice28zc_boundary_audit_scope_and_verbosity_v1.py",
        "AUDIT_28X",
        AUDIT_28X,
        ".slice28zc_run_audit_api_v2.bak",
    )

    _write(ROOT / UPDATER_REL, Path(__file__).read_text(encoding="utf-8"))
    _self_check()
    print("slice28zc run_audit API compatibility repair applied")


if __name__ == "__main__":
    main()
