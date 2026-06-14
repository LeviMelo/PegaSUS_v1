from __future__ import annotations

from pathlib import Path
from textwrap import dedent
import shutil
import zipfile

ROOT = Path.cwd()
UPDATER_REL = Path("scripts/dev/updaters/repair_slice28zc_boundary_audit_scope_and_verbosity_v1.py")

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

AUDIT_CLOSURE = r'''
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]

AUDIT_SCRIPTS = [
    "scripts/dev/audits/audit_slice28x_production_boundaries.py",
    "scripts/dev/audits/audit_slice28z_storage_boundary_adoption.py",
    "scripts/dev/audits/audit_slice28za_sidra_anchor_storage_boundary.py",
    "scripts/dev/audits/audit_slice28zb_compute_rng_boundary.py",
]

MAX_STDOUT_TAIL = 800
MAX_STDERR_TAIL = 800


def _load_last_json(stdout: str) -> dict[str, Any]:
    stripped = stdout.strip()
    if not stripped:
        return {"status": "failed", "error_count": 1, "errors": ["audit produced no stdout"]}
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        # Keep diagnostics bounded. Do not embed unbounded audit output in the
        # aggregate audit payload.
        return {
            "status": "failed",
            "error_count": 1,
            "errors": ["audit stdout was not a single JSON document"],
            "stdout_tail": stripped[-MAX_STDOUT_TAIL:],
        }


def _result_summary(script: str, proc: subprocess.CompletedProcess[str]) -> dict[str, Any]:
    payload = _load_last_json(proc.stdout)
    status = payload.get("status", "failed")
    if proc.returncode != 0 and status == "passed":
        status = "failed"
    summary: dict[str, Any] = {
        "script": script,
        "status": status,
        "returncode": proc.returncode,
        "storage_bypass_count": payload.get("storage_bypass_count"),
        "compute_bypass_count": payload.get("compute_bypass_count"),
        "error_count": payload.get("error_count", len(payload.get("errors", [])) if isinstance(payload.get("errors"), list) else None),
        "errors_truncated": payload.get("errors_truncated", 0),
    }
    errors = payload.get("errors", [])
    if errors:
        summary["errors"] = errors[:5] if isinstance(errors, list) else [str(errors)]
    if proc.stderr.strip():
        summary["stderr_tail"] = proc.stderr.strip()[-MAX_STDERR_TAIL:]
    if "stdout_tail" in payload:
        summary["stdout_tail"] = payload["stdout_tail"]
    return summary


def _run_audit(script: str) -> dict[str, Any]:
    path = ROOT / script
    if not path.exists():
        return {
            "script": script,
            "status": "failed",
            "returncode": None,
            "error_count": 1,
            "errors": [f"audit script missing: {script}"],
        }
    proc = subprocess.run(
        [sys.executable, str(path)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return _result_summary(script, proc)


def main() -> None:
    results = [_run_audit(script) for script in AUDIT_SCRIPTS]
    failures = [result for result in results if result.get("status") != "passed" or result.get("returncode") not in (0, None)]
    payload = {
        "audit": "boundary_closure",
        "status": "passed" if not failures else "failed",
        "scripts": AUDIT_SCRIPTS,
        "result_count": len(results),
        "failure_count": len(failures),
        "results": results,
        "errors": failures,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
'''.lstrip()

TEST_AUDIT_CLOSURE = r'''
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_slice28zc_boundary_closure_audit_passes() -> None:
    proc = subprocess.run(
        [sys.executable, "scripts/dev/audits/audit_boundary_closure.py"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-2000:]
    payload = json.loads(proc.stdout)
    assert payload["status"] == "passed"
    assert payload["failure_count"] == 0
    assert not payload["errors"]


def test_slice28zc_slice28x_audit_is_strict_for_guarded_files() -> None:
    proc = subprocess.run(
        [sys.executable, "scripts/dev/audits/audit_slice28x_production_boundaries.py"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-2000:]
    payload = json.loads(proc.stdout)
    assert payload["status"] == "passed"
    assert payload["storage_bypass_count"] == 0
    assert payload["compute_bypass_count"] == 0
    assert payload["error_count"] == 0
    assert payload["errors"] == []
'''.lstrip()

TEST_POLICY_SOURCE = r'''
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
AUDIT_28X = ROOT / "scripts" / "dev" / "audits" / "audit_slice28x_production_boundaries.py"
AUDIT_CLOSURE = ROOT / "scripts" / "dev" / "audits" / "audit_boundary_closure.py"


def test_slice28zc_slice28x_audit_is_scoped_to_closed_boundary_modules() -> None:
    text = AUDIT_28X.read_text(encoding="utf-8")
    assert "GUARDED_STORAGE_FILES" in text
    assert "GUARDED_COMPUTE_FILES" in text
    assert "src/pegasus/output/sidra_denominator_anchor.py" in text
    assert "src/pegasus/pirs/hsic.py" in text
    assert "src/pegasus/output/table_io.py" not in text
    assert "src/pegasus/she/maternal_child_linkage.py" not in text
    assert "src/pegasus/she/population/sidra_anchor.py" not in text
    assert "src/pegasus/she/sih_costs.py" not in text


def test_slice28zc_audits_keep_failure_output_bounded() -> None:
    text_28x = AUDIT_28X.read_text(encoding="utf-8")
    text_closure = AUDIT_CLOSURE.read_text(encoding="utf-8")
    assert "MAX_REPORTED_ERRORS" in text_28x
    assert "errors_truncated" in text_28x
    assert "MAX_STDOUT_TAIL" in text_closure
    assert "stdout_tail" in text_closure
    assert "proc.stdout[-2000:]" not in text_closure
'''.lstrip()

DOC_NOTE = """

## Slice 28ZC repair — scoped and terse boundary closure

The Slice 28ZC hardening gate is intentionally scoped to production modules that were explicitly migrated to the canonical `output.table_io` and `compute.random` boundaries in Slices 28Z through 28ZB. It is not a repository-wide ban on Polars/Arrow inside unfinished SHE/adaptor surfaces. Audit output must remain bounded: violations are summarized and truncated instead of dumping the full scanner result into the terminal.
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
    if "Slice 28ZC repair — scoped and terse boundary closure" not in text:
        text = text.rstrip() + DOC_NOTE
        path.write_text(text, encoding="utf-8", newline="\n")


def _self_check() -> None:
    audit = ROOT / "scripts/dev/audits/audit_slice28x_production_boundaries.py"
    text = audit.read_text(encoding="utf-8")
    required = [
        "GUARDED_STORAGE_FILES",
        "GUARDED_COMPUTE_FILES",
        "MAX_REPORTED_ERRORS",
        "errors_truncated",
        "src/pegasus/output/sidra_denominator_anchor.py",
        "src/pegasus/pirs/hsic.py",
    ]
    forbidden = [
        "SRC_ROOT.rglob",
        "src/pegasus/she/maternal_child_linkage.py",
        "src/pegasus/she/population/sidra_anchor.py",
        "src/pegasus/she/sih_costs.py",
    ]
    missing = [token for token in required if token not in text]
    present = [token for token in forbidden if token in text]
    if missing or present:
        raise RuntimeError(f"slice28zc repair self-check failed; missing={missing}; forbidden_present={present}")


def main() -> None:
    targets = [
        ROOT / "scripts/dev/audits/audit_slice28x_production_boundaries.py",
        ROOT / "scripts/dev/audits/audit_boundary_closure.py",
        ROOT / "tests/integration/test_slice28zc_boundary_closure.py",
        ROOT / "tests/unit/test_slice28zc_boundary_policy_source.py",
    ]
    for target in targets:
        _backup(target, ".slice28zc_scope_verbosity_v1.bak")

    _write(ROOT / "scripts/dev/audits/audit_slice28x_production_boundaries.py", AUDIT_28X)
    _write(ROOT / "scripts/dev/audits/audit_boundary_closure.py", AUDIT_CLOSURE)
    _write(ROOT / "tests/integration/test_slice28zc_boundary_closure.py", TEST_AUDIT_CLOSURE)
    _write(ROOT / "tests/unit/test_slice28zc_boundary_policy_source.py", TEST_POLICY_SOURCE)
    _append_doc_note()
    _write(ROOT / UPDATER_REL, Path(__file__).read_text(encoding="utf-8"))
    _self_check()
    print("slice28zc boundary audit scope/verbosity repair applied")


if __name__ == "__main__":
    main()
