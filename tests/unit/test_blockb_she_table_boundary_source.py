from __future__ import annotations

from scripts.dev.audits.audit_blockb_she_table_boundary import run_audit


def test_blockb_she_table_boundary_source_is_clean() -> None:
    result = run_audit()
    assert result["status"] == "passed", result
    assert result["error_count"] == 0, result
