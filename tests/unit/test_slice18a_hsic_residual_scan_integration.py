
from __future__ import annotations

from pegasus.pirs.hsic_run import _covariate_specs, _fdr_bh, hsic_residual_scan_summary


def test_slice18a_fdr_is_monotone_and_bounded() -> None:
    q = _fdr_bh([0.01, 0.20, None, 0.03])
    assert q[0] is not None and 0.0 <= q[0] <= 1.0
    assert q[2] is None
    assert q[0] <= q[3] <= q[1]


def test_slice18a_covariate_specs_respect_design_manifest() -> None:
    rows = [{"response": 1.0, "covariate_001": 2.0, "covariate_002": 3.0}]
    manifest = {"field_specs": [{"field_id": "capacity", "role": "covariate", "column": "covariate_001"}]}
    assert _covariate_specs(manifest, rows) == [{"field_id": "capacity", "column": "covariate_001"}]


def test_slice18a_summary_reports_blocked_manifest_path() -> None:
    summary = hsic_residual_scan_summary({"status": "blocked", "blocking_reasons": ["x"], "manifest_path": "m.json"})
    assert summary["status"] == "blocked"
    assert summary["blocking_reason_count"] == 1
    assert summary["manifest_path"] == "m.json"
