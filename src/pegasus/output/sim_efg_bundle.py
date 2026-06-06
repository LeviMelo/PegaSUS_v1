from __future__ import annotations

import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

from pegasus.core.enums import FieldState, MaterializationState
from pegasus.core.schemas import FailedBranch, FieldNode, QState, WarningRecord
from pegasus.efg.declaration import OperatorSpec
from pegasus.efg.legality import evaluate_delta, make_failed_branch
from pegasus.efg.lineage import make_lineage
from pegasus.efg.node import make_field_node
from pegasus.efg.q_tensor import compute_q_state
from pegasus.output.schemas import OUTPUT_BUNDLE_FILES


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _write_table(path: Path, rows: list[dict[str, Any]], schema: pa.Schema) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows, schema=schema)
    pq.write_table(table, path)


def _field_row(field: FieldNode) -> dict[str, Any]:
    return {
        "field_id": field.id,
        "name": field.name,
        "kind": field.kind,
        "carrier": field.carrier,
        "unit": field.unit,
        "aggregation": field.aggregation,
        "role": _json(field.role),
        "source": _json(field.source),
        "support_json": _json(field.support),
        "axes_json": _json(field.axes),
        "operator": field.operator,
        "provenance": _json(field.provenance),
        "state": field.state.value,
        "dashboard_safe": str(field.dashboard_safe),
        "warnings": _json(field.warnings),
        "lineage_hash": field.id,
        "registry_hash": _json(field.lineage.registry_versions),
        "materialization_state": field.materialization_state.value,
        "path": field.path,
    }


def _q_row(q: QState) -> dict[str, Any]:
    return q.model_dump(mode="json")


def _warning_row(w: WarningRecord) -> dict[str, Any]:
    return {
        "warning_id": w.warning_id,
        "field_id": w.field_id,
        "source": w.source,
        "severity": w.severity,
        "code": w.code,
        "message": w.message,
        "inherited_from": _json(w.inherited_from),
        "created_at": w.created_at,
    }


def _failed_branch_row(f: FailedBranch) -> dict[str, Any]:
    return {
        "failed_branch_id": f.failed_branch_id,
        "attempted_operator": f.attempted_operator,
        "parent_field_ids": _json(f.parent_field_ids),
        "failure_stage": f.failure_stage,
        "failed_terms": _json(f.failed_terms),
        "reason": f.reason,
        "warnings": _json(f.warnings),
        "created_at": f.created_at,
    }


def _variable_dictionary_row(field: FieldNode) -> dict[str, Any]:
    axes = field.axes or {}
    diagnostic_role = axes.get("diagnostic_role")
    topology = axes.get("topology")
    race_axis = axes.get("race_axis_type") or axes.get("race_axis")

    if field.name == "SIMAdministrativeRaceDeaths":
        estimand = "raw_administrative_race_count"
        warning = "Administrative SIM death-declaration race distribution; not self-declared race-specific risk."
    elif field.name == "IBGESelfDeclaredPopulationPlaceholder":
        estimand = "synthetic_fixture_denominator_placeholder"
        warning = "Placeholder used only to exercise declaration gate; not an official denominator."
    else:
        estimand = "event_count"
        warning = "Fixture-derived metadata field; not a production epidemiological estimate."

    return {
        "field_id": field.id,
        "display_name": field.name,
        "technical_name": field.name,
        "definition": f"{field.name} generated from normalized SIM fixture metadata.",
        "estimand_label": estimand,
        "source_systems": _json(field.source),
        "carrier": field.carrier,
        "unit": field.unit,
        "support_description": _json(field.support),
        "axis_description": _json(
            {
                "axes": axes,
                "race_axis": race_axis,
                "diagnostic_role": diagnostic_role,
                "topology": topology,
            }
        ),
        "provenance_description": _json(field.provenance),
        "state": field.state.value,
        "dashboard_safe": str(field.dashboard_safe),
        "interpretation_warning": warning,
    }


def build_sim_fixture_fields(sim_events_path: str | Path) -> tuple[list[FieldNode], list[FailedBranch], list[WarningRecord]]:
    sim_events_path = Path(sim_events_path)
    df = pl.read_parquet(sim_events_path)

    n_events = int(df.height)
    n_missing_race = int(df.filter(pl.col("race_missingness_state").is_in(["missing", "unknown"])).height)
    missing_race_share = float(n_missing_race / n_events) if n_events else 1.0

    years = sorted([int(x) for x in df["year"].drop_nulls().unique().to_list()])
    municipalities = sorted([str(x) for x in df["mun_residence_cod6"].drop_nulls().unique().to_list()])

    source_hashes = sorted(set(df["source_manifest_hash"].drop_nulls().to_list()))
    registry_versions = {
        "registry_set": "v1.0",
        "source_fields": "v1.0",
        "carrier_registry": "v1.0",
        "unit_registry": "v1.0",
        "quality_permissions": "v1.0",
        "race_axis_registry": "v1.0",
    }

    all_deaths_lineage = make_lineage(
        parent_ids=[],
        operator_type="sigma_C",
        operator_params={"source": "SIM-DO", "selection": "all_deaths"},
        registry_versions=registry_versions,
        source_manifest_hashes=source_hashes,
    )
    all_deaths = make_field_node(
        name="SIMDeathsAll",
        kind="extensive_measure",
        carrier="Deaths",
        unit="counts",
        support={
            "support": "municipality_year",
            "years": years,
            "municipalities": municipalities,
            "n_events": float(n_events),
            "n_eff": float(n_events),
            "cov_S": float(len(municipalities)),
            "cov_T": float(len(years)),
            "missingness": 0.0,
            "denom_fragility": 1.0,
        },
        axes={
            "time": "year",
            "geography": "mun_residence_cod6",
            "diagnostic_role": "all_deaths",
            "topology": "none",
            "race_axis_type": None,
        },
        aggregation="additive",
        role=["outcome", "demographic"],
        source=["SIM-DO"],
        operator="sigma_C",
        provenance=["official"],
        state=FieldState.quarantined_descriptive,
        warnings=["fixture_small_n"],
        lineage=all_deaths_lineage,
        materialization_state=MaterializationState.metadata_only,
        path=str(sim_events_path),
        dashboard_safe=False,
    )

    race_lineage = make_lineage(
        parent_ids=[all_deaths.id],
        operator_type="sigma_C",
        operator_params={"source": "SIM-DO", "selection": "race_color_admin"},
        registry_versions=registry_versions,
        source_manifest_hashes=source_hashes,
    )
    admin_race_deaths = make_field_node(
        name="SIMAdministrativeRaceDeaths",
        kind="extensive_measure",
        carrier="Deaths",
        unit="counts",
        support={
            "support": "municipality_year_admin_race",
            "years": years,
            "municipalities": municipalities,
            "n_events": float(n_events),
            "n_eff": float(n_events),
            "cov_S": float(len(municipalities)),
            "cov_T": float(len(years)),
            "missingness": missing_race_share,
            "missing_race_share": missing_race_share,
            "denom_fragility": 1.0,
        },
        axes={
            "time": "year",
            "geography": "mun_residence_cod6",
            "race": "race_color_admin",
            "race_axis_type": "administrative_death_declaration",
            "diagnostic_role": "all_deaths",
            "topology": "none",
        },
        aggregation="additive",
        role=["observer", "demographic", "exploratory"],
        source=["SIM-DO"],
        operator="sigma_C",
        provenance=["official"],
        state=FieldState.quarantined_descriptive,
        warnings=["fixture_small_n", "administrative_race_axis_not_self_declared"],
        lineage=race_lineage,
        materialization_state=MaterializationState.metadata_only,
        path=str(sim_events_path),
        dashboard_safe=False,
    )

    pop_lineage = make_lineage(
        parent_ids=[],
        operator_type="fixture_placeholder",
        operator_params={"source": "IBGE", "selection": "self_declared_population_placeholder"},
        registry_versions=registry_versions,
        source_manifest_hashes=[],
    )
    ibge_population = make_field_node(
        name="IBGESelfDeclaredPopulationPlaceholder",
        kind="extensive_measure",
        carrier="Population",
        unit="person_years",
        support={
            "support": "municipality_year_self_declared_race",
            "years": years,
            "municipalities": municipalities,
            "n_denom": 10000.0,
            "n_eff": 10000.0,
            "cov_S": float(len(municipalities)),
            "cov_T": float(len(years)),
            "missingness": 0.0,
            "denom_fragility": 0.0,
        },
        axes={
            "time": "year",
            "geography": "municipality",
            "race": "IBGE_self_declared_race",
            "race_axis_type": "self_declared",
        },
        aggregation="additive",
        role=["demographic", "exposure_offset"],
        source=["IBGE", "fixture"],
        operator="fixture_placeholder",
        provenance=["synthetic"],
        state=FieldState.quarantined_descriptive,
        warnings=["synthetic_fixture_denominator", "not_allowed_for_production_rates"],
        lineage=pop_lineage,
        materialization_state=MaterializationState.metadata_only,
        path=None,
        dashboard_safe=False,
    )

    operator = OperatorSpec(name="RN", role="mortality_rate", output_kind="intensive_density")
    delta = evaluate_delta(
        parents=[admin_race_deaths, ibge_population],
        operator=operator,
        intent=None,
        registries=None,
        alignment=None,
    )
    failed_branch = make_failed_branch(
        parents=[admin_race_deaths, ibge_population],
        operator=operator,
        delta=delta,
        reason="Direct SIM administrative race numerator over IBGE self-declared race denominator is illegal without Bridge_R.",
    )

    warnings = [
        WarningRecord(
            warning_id="fixture_small_n",
            field_id=all_deaths.id,
            source="pegasus.output.sim_efg_bundle",
            severity="info",
            code="fixture_small_n",
            message="SIM fixture has small n and is not an analytic production field.",
            inherited_from=[],
            created_at=_now(),
        ),
        WarningRecord(
            warning_id="race_axis_declaration_incommensurable",
            field_id=admin_race_deaths.id,
            source="pegasus.efg.declaration",
            severity="abort",
            code="race_axis_declaration_incommensurable",
            message="Direct race-specific SIM/IBGE division failed because declaration processes differ and Bridge_R was not applied.",
            inherited_from=[],
            created_at=_now(),
        ),
    ]

    return [all_deaths, admin_race_deaths, ibge_population], [failed_branch], warnings


def write_sim_fixture_efg_bundle(
    *,
    sim_events_path: str | Path,
    run_dir: str | Path,
) -> Path:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "Tables").mkdir(exist_ok=True)
    (run_dir / "Maps").mkdir(exist_ok=True)

    fields, failed_branches, warnings = build_sim_fixture_fields(sim_events_path)
    q_states = [
        compute_q_state(field=f, provenance=f.provenance, warnings=warnings)
        for f in fields
    ]

    v_schema = pa.schema([
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
    _write_table(run_dir / "V_fields.parquet", [_field_row(f) for f in fields], v_schema)

    edge_schema = pa.schema([
        ("edge_id", pa.string()),
        ("parent_field_id", pa.string()),
        ("child_field_id", pa.string()),
        ("operator", pa.string()),
        ("operator_params_json", pa.string()),
        ("registry_versions_json", pa.string()),
        ("created_at", pa.string()),
    ])
    edges = []
    for field in fields:
        for parent_id in field.lineage.parent_ids:
            edges.append(
                {
                    "edge_id": f"{parent_id}->{field.id}",
                    "parent_field_id": parent_id,
                    "child_field_id": field.id,
                    "operator": field.lineage.operator_type,
                    "operator_params_json": _json(field.lineage.operator_params),
                    "registry_versions_json": _json(field.lineage.registry_versions),
                    "created_at": _now(),
                }
            )
    _write_table(run_dir / "E_DAG.parquet", edges, edge_schema)

    q_schema = pa.schema([
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
    q_rows = []
    for q in q_states:
        row = _q_row(q)
        row["warnings"] = _json(row["warnings"])
        row["dashboard_safe"] = str(row["dashboard_safe"])
        q_rows.append(row)
    _write_table(run_dir / "Q_tensor.parquet", q_rows, q_schema)

    warning_schema = pa.schema([
        ("warning_id", pa.string()),
        ("field_id", pa.string()),
        ("source", pa.string()),
        ("severity", pa.string()),
        ("code", pa.string()),
        ("message", pa.string()),
        ("inherited_from", pa.string()),
        ("created_at", pa.string()),
    ])
    _write_table(run_dir / "Warnings.parquet", [_warning_row(w) for w in warnings], warning_schema)

    failed_schema = pa.schema([
        ("failed_branch_id", pa.string()),
        ("attempted_operator", pa.string()),
        ("parent_field_ids", pa.string()),
        ("failure_stage", pa.string()),
        ("failed_terms", pa.string()),
        ("reason", pa.string()),
        ("warnings", pa.string()),
        ("created_at", pa.string()),
    ])
    _write_table(run_dir / "FailedBranches.parquet", [_failed_branch_row(f) for f in failed_branches], failed_schema)

    vd_schema = pa.schema([
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
    _write_table(run_dir / "VariableDictionary.parquet", [_variable_dictionary_row(f) for f in fields], vd_schema)

    empty_assoc_schema = pa.schema([
        ("id", pa.string()),
        ("status", pa.string()),
        ("warnings", pa.string()),
    ])
    for name in ["ModelAssociations", "ResidualAssociations"]:
        _write_table(run_dir / f"{name}.parquet", [], empty_assoc_schema)

    hyp_schema = pa.schema([
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
    _write_table(run_dir / "Hypotheses.parquet", [], hyp_schema)

    qf_schema = pa.schema([
        ("field_id", pa.string()),
        ("state", pa.string()),
        ("reason", pa.string()),
        ("warnings", pa.string()),
    ])
    _write_table(
        run_dir / "QuarantinedFields.parquet",
        [
            {
                "field_id": q.field_id,
                "state": q.state.value,
                "reason": "fixture_or_q_state_limit",
                "warnings": _json(q.warnings),
            }
            for q in q_states
            if q.state.value != "verified"
        ],
        qf_schema,
    )
    _write_table(run_dir / "ForcedFields.parquet", [], qf_schema)

    (run_dir / "P_vector.json").write_text(
        _json(
            {
                "schema_version": "1.0",
                "provenance": {f.id: f.provenance for f in fields},
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "UserIntent.json").write_text(
        _json(
            {
                "frozen": True,
                "intent_source": "sim_fixture_efg_slice1d",
                "population_mode": "synthetic_test_fixture",
                "warning": "fixture_only_not_production_rate",
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "RunConfig.json").write_text(
        _json(
            {
                "frozen": True,
                "slice": "1D",
                "workflow": "sim_fixture_efg",
                "sim_events_path": str(sim_events_path),
            }
        ),
        encoding="utf-8",
    )

    stages = [
        "config_load",
        "registry_validation",
        "datasus_acquire",
        "datasus_profile",
        "datasus_normalize",
        "sidra_metadata",
        "sidra_plan",
        "sidra_fetch",
        "sidra_normalize",
        "geo_support",
        "she_build",
        "population_solver",
        "stdfm",
        "efg_build",
        "q_tensor",
        "pirs_model",
        "pirs_hsic",
        "output_serialization",
        "output_validation",
    ]

    manifest = {
        "run_id": run_dir.name,
        "created_at": _now(),
        "completed_at": _now(),
        "status": "success",
        "code_version": {
            "package_version": "0.1.0",
            "git_commit": "uninitialized",
            "git_dirty": False,
        },
        "environment": {
            "python_version": sys.version,
            "os": platform.platform(),
            "duckdb_version": None,
            "polars_version": pl.__version__,
            "pyarrow_version": pa.__version__,
            "torch_version": None,
            "torch_cuda_available": False,
            "cuda_device_name": None,
            "r_version": None,
            "microdatasus_version": None,
            "read_dbc_version": None,
        },
        "registry_hashes": {f.name: "v1.0" for f in fields},
        "source_manifest_hashes": [],
        "random_seeds": {},
        "telemetry": {
            "total_wall_seconds": 0.0,
            "stage_wall_seconds": {f"{s}_seconds": 0.0 for s in stages},
            "stage_status": {
                s: (
                    "success"
                    if s in {"config_load", "datasus_normalize", "she_build", "efg_build", "q_tensor", "output_serialization", "output_validation"}
                    else "skipped"
                )
                for s in stages
            },
            "resource_summary": {
                "peak_rss_mb": None,
                "peak_vram_mb": None,
                "duckdb_temp_bytes": None,
                "rows_read": {"sim_events": len(fields)},
                "rows_written": {"V_fields": len(fields), "FailedBranches": len(failed_branches)},
                "parquet_bytes_written": 0,
            },
        },
    }
    (run_dir / "ReproducibilityManifest.json").write_text(_json(manifest), encoding="utf-8")

    expected = set(OUTPUT_BUNDLE_FILES.values())
    found = {p.name for p in run_dir.iterdir()}
    extra = found - expected
    missing = expected - found
    if extra or missing:
        raise RuntimeError(f"Invalid first-class bundle keys. extra={sorted(extra)} missing={sorted(missing)}")

    return run_dir
