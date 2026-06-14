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
