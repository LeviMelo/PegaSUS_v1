from __future__ import annotations

from pegasus.pirs.hsic_report import HSICEvidenceItem, render_hsic_markdown_report


def test_slice18c_markdown_report_is_read_only_and_contains_evidence_table() -> None:
    item = HSICEvidenceItem(
        rank=1,
        hypothesis_id="h1",
        residual_field_id="resid",
        covariate_field_id="cov_a",
        statistic=0.42,
        p_value=0.01,
        q_value=0.02,
        n_eff=33.0,
        state="exploratory",
        evidence_tier="supported_descriptive",
        decision="retain_for_review",
        warnings=[],
        hsic_mode="exact_linear",
        residual_mode="in_sample",
        fold_scheme="none",
    )
    text = render_hsic_markdown_report([item], generated_at="2026-06-13T00:00:00+00:00", source="manifest.json")
    assert "read-only descriptive export" in text
    assert "| Rank | Hypothesis | Covariate" in text
    assert "h1" in text
    assert "cov_a" in text
    assert "retain_for_review" in text


def test_slice18c_evidence_item_serializes_dashboard_safe_read_only() -> None:
    item = HSICEvidenceItem(
        rank=2,
        hypothesis_id="h2",
        residual_field_id=None,
        covariate_field_id="cov_b",
        statistic=None,
        p_value=None,
        q_value=None,
        n_eff=None,
        state="fragile",
        evidence_tier="fragile_descriptive",
        decision="show_as_fragile",
        warnings=["small_support"],
    )
    payload = item.as_json()
    assert payload["dashboard_safe"] is True
    assert payload["read_only"] is True
    assert payload["warnings"] == ["small_support"]
