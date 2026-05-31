from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeAlias

import polars as pl


Schema: TypeAlias = Mapping[str, Any]


def empty_df(schema: Schema) -> pl.DataFrame:
    return pl.DataFrame(
        {name: pl.Series(name=name, values=[], dtype=dtype) for name, dtype in schema.items()}
    )


V_FIELDS_SCHEMA: Schema = {
    "field_id": pl.Utf8,
    "name": pl.Utf8,
    "kind": pl.Utf8,
    "carrier": pl.Utf8,
    "unit": pl.Utf8,
    "aggregation": pl.Utf8,
    "role": pl.List(pl.Utf8),
    "source": pl.List(pl.Utf8),
    "support_json": pl.Utf8,
    "axes_json": pl.Utf8,
    "operator": pl.Utf8,
    "provenance": pl.List(pl.Utf8),
    "state": pl.Utf8,
    "dashboard_safe": pl.Utf8,
    "warnings": pl.List(pl.Utf8),
    "lineage_hash": pl.Utf8,
    "registry_hash": pl.Utf8,
    "materialization_state": pl.Utf8,
    "path": pl.Utf8,
}

E_DAG_SCHEMA: Schema = {
    "edge_id": pl.Utf8,
    "parent_field_id": pl.Utf8,
    "child_field_id": pl.Utf8,
    "operator": pl.Utf8,
    "operator_params_json": pl.Utf8,
    "registry_versions_json": pl.Utf8,
    "created_at": pl.Utf8,
}

Q_TENSOR_SCHEMA: Schema = {
    "field_id": pl.Utf8,
    "n_events": pl.Float64,
    "n_denom": pl.Float64,
    "n_eff": pl.Float64,
    "cov_S": pl.Float64,
    "cov_T": pl.Float64,
    "missingness": pl.Float64,
    "zero_inflation": pl.Float64,
    "denom_fragility": pl.Float64,
    "cv": pl.Float64,
    "moran_i": pl.Float64,
    "temporal_roughness": pl.Float64,
    "spatial_entropy": pl.Float64,
    "provenance_risk": pl.Float64,
    "race_bridge_cv": pl.Float64,
    "sensitivity_width": pl.Float64,
    "state": pl.Utf8,
    "dashboard_safe": pl.Utf8,
    "warnings": pl.List(pl.Utf8),
    "computed_at": pl.Utf8,
    "q_schema_version": pl.Utf8,
}

WARNINGS_SCHEMA: Schema = {
    "warning_id": pl.Utf8,
    "field_id": pl.Utf8,
    "source": pl.Utf8,
    "severity": pl.Utf8,
    "code": pl.Utf8,
    "message": pl.Utf8,
    "parent_warning_ids": pl.List(pl.Utf8),
    "created_at": pl.Utf8,
}

MODEL_ASSOCIATIONS_SCHEMA: Schema = {
    "model_id": pl.Utf8,
    "outcome_field_id": pl.Utf8,
    "family": pl.Utf8,
    "status": pl.Utf8,
    "covariate_field_ids": pl.List(pl.Utf8),
    "offset_field_id": pl.Utf8,
    "diagnostics_json": pl.Utf8,
    "warnings": pl.List(pl.Utf8),
}

RESIDUAL_ASSOCIATIONS_SCHEMA: Schema = {
    "residual_field_id": pl.Utf8,
    "parent_model_id": pl.Utf8,
    "outcome_field_id": pl.Utf8,
    "residual_type": pl.Utf8,
    "support_json": pl.Utf8,
    "provenance": pl.List(pl.Utf8),
    "state": pl.Utf8,
    "warnings": pl.List(pl.Utf8),
}

HYPOTHESES_SCHEMA: Schema = {
    "hypothesis_id": pl.Utf8,
    "outcome_field_id": pl.Utf8,
    "covariate_field_id": pl.Utf8,
    "residual_field_id": pl.Utf8,
    "statistic": pl.Float64,
    "p_value": pl.Float64,
    "q_value": pl.Float64,
    "hsic_mode": pl.Utf8,
    "null_strategy": pl.Utf8,
    "fdr_method": pl.Utf8,
    "n_eff": pl.Float64,
    "state": pl.Utf8,
    "warnings": pl.List(pl.Utf8),
    "approximation_diagnostics_json": pl.Utf8,
}

VARIABLE_DICTIONARY_SCHEMA: Schema = {
    "field_id": pl.Utf8,
    "display_name": pl.Utf8,
    "technical_name": pl.Utf8,
    "definition": pl.Utf8,
    "estimand_label": pl.Utf8,
    "source_systems": pl.List(pl.Utf8),
    "carrier": pl.Utf8,
    "unit": pl.Utf8,
    "support_description": pl.Utf8,
    "axis_description": pl.Utf8,
    "provenance_description": pl.Utf8,
    "state": pl.Utf8,
    "dashboard_safe": pl.Utf8,
    "interpretation_warning": pl.Utf8,
}

FAILED_BRANCHES_SCHEMA: Schema = {
    "failed_branch_id": pl.Utf8,
    "attempted_operator": pl.Utf8,
    "parent_field_ids": pl.List(pl.Utf8),
    "reason_code": pl.Utf8,
    "delta_result_json": pl.Utf8,
    "recoverable": pl.Boolean,
    "suggested_route": pl.Utf8,
}

QUARANTINED_FIELDS_SCHEMA: Schema = {
    "field_id": pl.Utf8,
    "state": pl.Utf8,
    "reason_code": pl.Utf8,
    "warnings": pl.List(pl.Utf8),
}

FORCED_FIELDS_SCHEMA: Schema = {
    "field_id": pl.Utf8,
    "force_selector": pl.Utf8,
    "state": pl.Utf8,
    "warnings": pl.List(pl.Utf8),
}

PARQUET_SCHEMAS: dict[str, Schema] = {
    "V_fields": V_FIELDS_SCHEMA,
    "E_DAG": E_DAG_SCHEMA,
    "Q_tensor": Q_TENSOR_SCHEMA,
    "Warnings": WARNINGS_SCHEMA,
    "ModelAssociations": MODEL_ASSOCIATIONS_SCHEMA,
    "ResidualAssociations": RESIDUAL_ASSOCIATIONS_SCHEMA,
    "Hypotheses": HYPOTHESES_SCHEMA,
    "VariableDictionary": VARIABLE_DICTIONARY_SCHEMA,
    "FailedBranches": FAILED_BRANCHES_SCHEMA,
    "QuarantinedFields": QUARANTINED_FIELDS_SCHEMA,
    "ForcedFields": FORCED_FIELDS_SCHEMA,
}

PARQUET_FILENAMES: dict[str, str] = {
    "V_fields": "V_fields.parquet",
    "E_DAG": "E_DAG.parquet",
    "Q_tensor": "Q_tensor.parquet",
    "Warnings": "Warnings.parquet",
    "ModelAssociations": "ModelAssociations.parquet",
    "ResidualAssociations": "ResidualAssociations.parquet",
    "Hypotheses": "Hypotheses.parquet",
    "VariableDictionary": "VariableDictionary.parquet",
    "FailedBranches": "FailedBranches.parquet",
    "QuarantinedFields": "QuarantinedFields.parquet",
    "ForcedFields": "ForcedFields.parquet",
}