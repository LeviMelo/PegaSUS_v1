from __future__ import annotations

import pytest

from pegasus.pirs.hsic import select_hsic_mode, residual_mode_for_hsic, validate_residual_mode_for_hsic, linear_hsic_statistic, run_hsic_scan
from pegasus.pirs.nulls import select_null_regime, assert_monthly_null_preserves_season, descriptive_only_when_insufficient_blocks
from pegasus.pirs.fdr import correct_p_values
from pegasus.pirs.nystrom import nystrom_diagnostics
from pegasus.pirs.rff import rff_diagnostics


def test_hsic_mode_selection_and_cuda_abort() -> None:
    assert select_hsic_mode(n_eff=99, budget="standard") == "disabled"
    assert select_hsic_mode(n_eff=6000, budget="standard") == "nystrom"
    assert select_hsic_mode(n_eff=6000, budget="fast") == "rff"
    assert select_hsic_mode(n_eff=200, budget="standard", cuda_required=True, cuda_available=False) == "cuda_unavailable_abort"


def test_standard_deep_residual_modes_reject_in_sample() -> None:
    assert residual_mode_for_hsic(budget="standard") == "cross_fitted"
    with pytest.raises(ValueError):
        validate_residual_mode_for_hsic(budget="standard", residual_mode="in_sample")


def test_null_regimes_preserve_monthly_season_and_block_guard() -> None:
    monthly = select_null_regime("monthly_seasonal_panel")
    assert "season_preserving" in monthly.null_strategy
    assert_monthly_null_preserves_season(monthly)
    assert descriptive_only_when_insufficient_blocks(spatial_blocks=4, temporal_blocks=12) is True
    assert descriptive_only_when_insufficient_blocks(spatial_blocks=5, temporal_blocks=5) is False


def test_hsic_scan_and_fdr_emit_diagnostics() -> None:
    residuals = [float(i) for i in range(120)]
    covariate = [float(i) * 0.5 for i in range(120)]
    out = run_hsic_scan(
        outcome_residual_field_id="e_y",
        covariate_field_id="x",
        residuals=residuals,
        covariate=covariate,
        support_intersection={"n_eff": 120},
        budget="standard",
        null_strategy="spatial_block_cyclic_time_shift",
        fdr_method="BY",
        permutations=1000,
    )
    assert out.hsic_mode == "exact"
    assert out.statistic is not None and out.statistic > 0.9
    fdr = correct_p_values([out.p_value], method="BY")
    assert fdr.q_values[0] is not None
    assert linear_hsic_statistic([1, 2, 3], [1, 2, 3]) > 0.9


def test_approximation_diagnostics_contracts() -> None:
    assert nystrom_diagnostics(n_eff=6000, budget="standard").as_manifest()["approximation"] == "nystrom"
    assert rff_diagnostics(n_eff=6000, budget="fast").as_manifest()["approximation"] == "rff"
