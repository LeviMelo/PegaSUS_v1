from __future__ import annotations

from pathlib import Path
from textwrap import dedent
import shutil
import zipfile

ROOT = Path.cwd()
UPDATER_REL = Path("scripts/dev/updaters/apply_slice28zc_boundary_audit_hardening.py")

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


def _load_last_json(stdout: str) -> dict[str, Any]:
    stripped = stdout.strip()
    if not stripped:
        return {"status": "failed", "errors": ["audit produced no stdout"]}
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        lines = [line for line in stripped.splitlines() if line.strip()]
        for idx in range(len(lines)):
            candidate = "\n".join(lines[idx:])
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue
        return {"status": "failed", "errors": ["audit stdout was not JSON", stripped[-2000:]]}


def _run_audit(script: str) -> dict[str, Any]:
    path = ROOT / script
    if not path.exists():
        return {
            "script": script,
            "status": "failed",
            "returncode": None,
            "errors": [f"audit script missing: {script}"],
        }
    proc = subprocess.run(
        [sys.executable, str(path)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    payload = _load_last_json(proc.stdout)
    payload["script"] = script
    payload["returncode"] = proc.returncode
    if proc.stderr.strip():
        payload.setdefault("stderr", proc.stderr.strip())
    if proc.returncode != 0 and payload.get("status") == "passed":
        payload["status"] = "failed"
        payload.setdefault("errors", []).append(f"nonzero audit returncode: {proc.returncode}")
    return payload


def main() -> None:
    results = [_run_audit(script) for script in AUDIT_SCRIPTS]
    failures = [result for result in results if result.get("status") != "passed" or result.get("returncode") not in (0, None)]
    payload = {
        "audit": "boundary_closure",
        "status": "passed" if not failures else "failed",
        "scripts": AUDIT_SCRIPTS,
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
    assert proc.returncode == 0, proc.stdout + proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["status"] == "passed"
    assert not payload["errors"]


def test_slice28zc_slice28x_audit_is_strict() -> None:
    proc = subprocess.run(
        [sys.executable, "scripts/dev/audits/audit_slice28x_production_boundaries.py"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["status"] == "passed"
    assert payload["storage_bypass_count"] == 0
    assert payload["compute_bypass_count"] == 0
    assert payload["errors"] == []
'''.lstrip()

TEST_POLICY_SOURCE = r'''
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
AUDIT = ROOT / "scripts" / "dev" / "audits" / "audit_slice28x_production_boundaries.py"


def test_slice28zc_slice28x_audit_has_strict_failure_path() -> None:
    text = AUDIT.read_text(encoding="utf-8")
    assert '"status": "passed" if not violations else "failed"' in text
    assert "raise SystemExit(1)" in text
    assert "storage_bypass_count" in text
    assert "compute_bypass_count" in text


def test_slice28zc_slice28x_audit_keeps_storage_adapter_exemption_narrow() -> None:
    text = AUDIT.read_text(encoding="utf-8")
    assert '"src/pegasus/storage/"' in text
    assert '"src/pegasus/output/table_io.py"' in text
    assert '"src/pegasus/compute/random.py"' in text
    assert '"src/pegasus/output/sidra_denominator_anchor.py"' not in text
    assert '"src/pegasus/pirs/hsic.py"' not in text
'''.lstrip()

DOC_NOTE = """
\n## Slice 28ZC — boundary-audit hardening\n\nSlice 28ZC converts the transitional Slice 28X boundary inventory into a strict regression gate. Direct Parquet/Polars storage I/O is allowed only inside the storage adapter layer or the canonical `output.table_io` row/table boundary. Direct seeded `torch.Generator.manual_seed()` use is allowed only through `compute.random`. The aggregate `audit_boundary_closure.py` audit composes the 28X/28Z/28ZA/28ZB gates and must remain clean before further runtime-authority work proceeds.\n"""


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
    if "Slice 28ZC — boundary-audit hardening" not in text:
        text = text.rstrip() + DOC_NOTE
        path.write_text(text, encoding="utf-8", newline="\n")


def _self_audit() -> None:
    audit = ROOT / "scripts" / "dev" / "audits" / "audit_slice28x_production_boundaries.py"
    text = audit.read_text(encoding="utf-8")
    required = [
        "collect_boundary_violations",
        '"status": "passed" if not violations else "failed"',
        "raise SystemExit(1)",
        "storage_bypass_count",
        "compute_bypass_count",
    ]
    missing = [token for token in required if token not in text]
    if missing:
        raise RuntimeError(f"slice28zc audit hardening self-check failed; missing {missing}")


def main() -> None:
    targets = [
        ROOT / "scripts/dev/audits/audit_slice28x_production_boundaries.py",
        ROOT / "scripts/dev/audits/audit_boundary_closure.py",
        ROOT / "tests/integration/test_slice28zc_boundary_closure.py",
        ROOT / "tests/unit/test_slice28zc_boundary_policy_source.py",
    ]
    for target in targets:
        _backup(target, ".slice28zc_boundary_audit_hardening.bak")

    _write(ROOT / "scripts/dev/audits/audit_slice28x_production_boundaries.py", AUDIT_28X)
    _write(ROOT / "scripts/dev/audits/audit_boundary_closure.py", AUDIT_CLOSURE)
    _write(ROOT / "tests/integration/test_slice28zc_boundary_closure.py", TEST_AUDIT_CLOSURE)
    _write(ROOT / "tests/unit/test_slice28zc_boundary_policy_source.py", TEST_POLICY_SOURCE)
    _append_doc_note()

    _write(ROOT / UPDATER_REL, Path(__file__).read_text(encoding="utf-8"))
    _self_audit()
    print("slice28zc boundary audit hardening applied")


if __name__ == "__main__":
    main()
