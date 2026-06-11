from __future__ import annotations

from pathlib import Path

import pytest

from pegasus.output.pirs_bundle import candidates_from_fixture
from pegasus.pirs.crossfit import assert_standard_deep_not_in_sample, fold_scheme_for_budget, residual_mode_for_budget
from pegasus.pirs.families import exposure_offset_source, family_for_outcome
from pegasus.pirs.field_selection import select_fields_for_pirs

FIXTURE = Path("tests/fixtures/pirs/pirs_slice8a_fixture.json")


def test_pirs_selection_excludes_zero_variance_and_illegal_fields() -> None:
    result = select_fields_for_pirs(candidates_from_fixture(FIXTURE), budget="standard")
    assert result.selected_outcome is not None
    assert result.selected_outcome.field_id == "pirs_fixture_deaths_outcome"
    assert [c.field_id for c in result.selected_covariates] == ["pirs_fixture_cnes_capacity_covariate"]
    rejected = {r["field_id"]: r["reason"] for r in result.rejected}
    assert rejected["pirs_fixture_zero_variance_covariate"] == "zero_variance_field_excluded_from_design_matrix"
    assert rejected["pirs_fixture_illegal_covariate"] == "q_state_illegal_excluded_not_model_eligible"


def test_pirs_family_and_exposure_offset_from_carrier_registry_semantics() -> None:
    candidates = {c.field_id: c for c in candidates_from_fixture(FIXTURE)}
    family = family_for_outcome(outcome=candidates["pirs_fixture_deaths_outcome"], offset=candidates["pirs_fixture_population_offset"])
    assert family == "poisson_count_with_log_offset"
    assert exposure_offset_source(family=family, offset=candidates["pirs_fixture_population_offset"]) == "pirs_fixture_population_offset"


def test_standard_and_deep_do_not_use_in_sample_residuals() -> None:
    assert residual_mode_for_budget("fast") == "in_sample"
    assert residual_mode_for_budget("standard") == "cross_fitted"
    assert residual_mode_for_budget("deep") == "parametric_bootstrap"
    standard = fold_scheme_for_budget(budget="standard")
    deep = fold_scheme_for_budget(budget="deep")
    assert standard.preserves_spatial_blocks is True
    assert standard.preserves_temporal_blocks is True
    assert deep.bootstrap_count > 0
    with pytest.raises(ValueError):
        assert_standard_deep_not_in_sample("standard", "in_sample")
