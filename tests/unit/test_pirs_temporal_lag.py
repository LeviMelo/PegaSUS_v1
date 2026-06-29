from __future__ import annotations

from pegasus.pirs.design_matrix import MatrixFieldSpec, _build_lagged_covariates


def test_temporal_lag_aligns_covariate_to_prior_year_per_municipality() -> None:
    # Two municipalities, three years each; positional cell order matches the
    # value vectors produced by the Q-tensor / design matrix.
    support_rows = [
        {"row_id": 0, "municipality_cod6": "270010", "year": 2015},
        {"row_id": 1, "municipality_cod6": "270010", "year": 2016},
        {"row_id": 2, "municipality_cod6": "270010", "year": 2017},
        {"row_id": 3, "municipality_cod6": "270430", "year": 2015},
        {"row_id": 4, "municipality_cod6": "270430", "year": 2016},
        {"row_id": 5, "municipality_cod6": "270430", "year": 2017},
    ]
    specs = [MatrixFieldSpec(field_id="arbovirus_share", role="covariate", column="arbovirus_share")]
    vectors: dict[str, list[object]] = {"arbovirus_share": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]}

    new_specs, warnings = _build_lagged_covariates(
        specs=specs, vectors=vectors, support_rows=support_rows, lags=[1]
    )

    assert [s.field_id for s in new_specs] == ["arbovirus_share@lag1"]
    assert new_specs[0].column == "arbovirus_share_lag1"
    assert new_specs[0].role == "covariate"
    assert new_specs[0].required is False
    # lag-1: value at (m,t) is the covariate at (m, t-1); earliest year has no
    # antecedent -> None (covariate missingness), never cross-municipality leakage.
    assert vectors["arbovirus_share@lag1"] == [None, 10.0, 20.0, None, 40.0, 50.0]
    assert "arbovirus_share@lag1" not in warnings


def test_temporal_lag_no_antecedent_emits_warning_and_outcome_not_lagged() -> None:
    support_rows = [
        {"row_id": 0, "municipality_cod6": "270010", "year": 2016},
        {"row_id": 1, "municipality_cod6": "270430", "year": 2016},
    ]
    specs = [
        MatrixFieldSpec(field_id="micro", role="outcome", column="micro"),
        MatrixFieldSpec(field_id="arbo", role="covariate", column="arbo"),
    ]
    vectors: dict[str, list[object]] = {"micro": [0.1, 0.2], "arbo": [1.0, 2.0]}

    new_specs, warnings = _build_lagged_covariates(
        specs=specs, vectors=vectors, support_rows=support_rows, lags=[1]
    )

    # Only covariates are lagged; the outcome is never lagged.
    assert [s.field_id for s in new_specs] == ["arbo@lag1"]
    # Single year -> no antecedent cell exists -> all None + warning.
    assert vectors["arbo@lag1"] == [None, None]
    assert warnings.get("arbo@lag1") == "lagged_covariate_no_antecedent_cells"
