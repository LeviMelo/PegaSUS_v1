"""Production-safe 17-key run-directory schema seed.

This writes canonical empty schemas and neutral JSON envelopes only.
It does not create analytic rows or development-source semantics.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


def _write(path: Path, rows: list[dict[str, Any]], schema: pa.Schema) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows, schema=schema)
    pq.write_table(table, path)


V_FIELDS_SCHEMA = pa.schema([
    ("field_id", pa.string()),
    ("name", pa.string()),
    ("kind", pa.string()),
    ("carrier", pa.string()),
    ("unit", pa.string()),
    ("aggregation", pa.string()),
    ("role", pa.string()),
    ("source", pa.string()),
    ("support_json", pa.string()),
    ("axes_json", pa.string()),
    ("operator", pa.string()),
    ("provenance", pa.string()),
    ("state", pa.string()),
    ("dashboard_safe", pa.string()),
    ("warnings", pa.string()),
    ("lineage_hash", pa.string()),
    ("registry_hash", pa.string()),
    ("materialization_state", pa.string()),
    ("path", pa.string()),
])

E_DAG_SCHEMA = pa.schema([
    ("edge_id", pa.string()),
    ("parent_field_id", pa.string()),
    ("child_field_id", pa.string()),
    ("operator", pa.string()),
    ("operator_params_json", pa.string()),
    ("registry_versions_json", pa.string()),
    ("created_at", pa.string()),
])

Q_TENSOR_SCHEMA = pa.schema([
    ("field_id", pa.string()),
    ("n_events", pa.float64()),
    ("n_denom", pa.float64()),
    ("n_eff", pa.float64()),
    ("cov_S", pa.float64()),
    ("cov_T", pa.float64()),
    ("missingness", pa.float64()),
    ("zero_inflation", pa.float64()),
    ("denom_fragility", pa.float64()),
    ("cv", pa.float64()),
    ("moran_i", pa.float64()),
    ("temporal_roughness", pa.float64()),
    ("spatial_entropy", pa.float64()),
    ("provenance_risk", pa.float64()),
    ("race_axis_source", pa.string()),
    ("race_axis_target", pa.string()),
    ("missing_race_share", pa.float64()),
    ("emission_prior_strength", pa.float64()),
    ("race_bridge_cv", pa.float64()),
    ("sensitivity_width", pa.float64()),
    ("bridge_mode", pa.string()),
    ("state", pa.string()),
    ("dashboard_safe", pa.string()),
    ("warnings", pa.string()),
    ("computed_at", pa.string()),
    ("q_schema_version", pa.string()),
])

WARNINGS_SCHEMA = pa.schema([
    ("warning_id", pa.string()),
    ("field_id", pa.string()),
    ("source", pa.string()),
    ("severity", pa.string()),
    ("code", pa.string()),
    ("message", pa.string()),
    ("inherited_from", pa.string()),
    ("created_at", pa.string()),
])

MODEL_ASSOC_SCHEMA = pa.schema([
    ("id", pa.string()),
    ("model_id", pa.string()),
    ("status", pa.string()),
    ("family", pa.string()),
    ("outcome_field_id", pa.string()),
    ("covariate_field_id", pa.string()),
    ("covariate_field_ids", pa.string()),
    ("offset_field_id", pa.string()),
    ("residual_field_id", pa.string()),
    ("diagnostics_json", pa.string()),
    ("created_at", pa.string()),
    ("warnings", pa.string()),
])

RESIDUAL_ASSOC_SCHEMA = pa.schema([
    ("id", pa.string()),
    ("residual_association_id", pa.string()),
    ("residual_field_id", pa.string()),
    ("model_id", pa.string()),
    ("parent_model_id", pa.string()),
    ("outcome_field_id", pa.string()),
    ("residual_type", pa.string()),
    ("status", pa.string()),
    ("created_at", pa.string()),
    ("warnings", pa.string()),
])

HYPOTHESES_SCHEMA = pa.schema([
    ("hypothesis_id", pa.string()),
    ("outcome_field_id", pa.string()),
    ("covariate_field_id", pa.string()),
    ("residual_field_id", pa.string()),
    ("statistic", pa.float64()),
    ("p_value", pa.float64()),
    ("q_value", pa.float64()),
    ("hsic_mode", pa.string()),
    ("residual_mode", pa.string()),
    ("fold_scheme", pa.string()),
    ("bootstrap_count", pa.int64()),
    ("residual_uncertainty", pa.string()),
    ("null_strategy", pa.string()),
    ("fdr_method", pa.string()),
    ("n_eff", pa.float64()),
    ("state", pa.string()),
    ("warnings", pa.string()),
    ("approximation_diagnostics_json", pa.string()),
])

VARIABLE_DICTIONARY_SCHEMA = pa.schema([
    ("field_id", pa.string()),
    ("display_name", pa.string()),
    ("technical_name", pa.string()),
    ("definition", pa.string()),
    ("estimand_label", pa.string()),
    ("source_systems", pa.string()),
    ("carrier", pa.string()),
    ("unit", pa.string()),
    ("support_description", pa.string()),
    ("axis_description", pa.string()),
    ("provenance_description", pa.string()),
    ("state", pa.string()),
    ("dashboard_safe", pa.string()),
    ("interpretation_warning", pa.string()),
])

FAILED_BRANCH_SCHEMA = pa.schema([
    ("failed_branch_id", pa.string()),
    ("attempted_operator", pa.string()),
    ("parent_field_ids", pa.string()),
    ("failure_stage", pa.string()),
    ("failed_terms", pa.string()),
    ("reason", pa.string()),
    ("warnings", pa.string()),
    ("created_at", pa.string()),
])

QUARANTINED_SCHEMA = pa.schema([
    ("field_id", pa.string()),
    ("state", pa.string()),
    ("reason", pa.string()),
    ("warnings", pa.string()),
])

TABLE_SCHEMAS = {
    "V_fields.parquet": V_FIELDS_SCHEMA,
    "E_DAG.parquet": E_DAG_SCHEMA,
    "Q_tensor.parquet": Q_TENSOR_SCHEMA,
    "Warnings.parquet": WARNINGS_SCHEMA,
    "ModelAssociations.parquet": MODEL_ASSOC_SCHEMA,
    "ResidualAssociations.parquet": RESIDUAL_ASSOC_SCHEMA,
    "Hypotheses.parquet": HYPOTHESES_SCHEMA,
    "VariableDictionary.parquet": VARIABLE_DICTIONARY_SCHEMA,
    "FailedBranches.parquet": FAILED_BRANCH_SCHEMA,
    "QuarantinedFields.parquet": QUARANTINED_SCHEMA,
    "ForcedFields.parquet": QUARANTINED_SCHEMA,
}


def _json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )


def create_schema_seed_output_bundle(run_dir: str | Path) -> Path:
    root = Path(run_dir)
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)

    for name, schema in TABLE_SCHEMAS.items():
        _write(root / name, [], schema)

    (root / "Tables").mkdir(parents=True, exist_ok=True)
    (root / "Maps").mkdir(parents=True, exist_ok=True)

    _json(root / "UserIntent.json", {
        "frozen": True,
        "intent_source": "compile_request",
        "schema_version": "1.0",
    })
    _json(root / "P_vector.json", {
        "schema_version": "1.0",
        "provenance": {},
    })
    _json(root / "RunConfig.json", {
        "schema_version": "1.0",
        "workflow_mode": "compile",
    })
    _json(root / "ReproducibilityManifest.json", {
        "schema_version": "1.0",
        "workflow_mode": "compile",
        "source_hashes": {},
        "registry_hashes": {},
        "telemetry": {},
    })
    return root


__all__ = ["create_schema_seed_output_bundle", "TABLE_SCHEMAS"]
