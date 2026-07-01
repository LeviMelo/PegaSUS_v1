from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import polars as pl

from pegasus.efg.dag import EFGResult, build_efg
from pegasus.efg.materialization_manifest import load_substrate_bundle_for_efg
from pegasus.output.bundle import create_empty_output_bundle
from pegasus.output.table_io import append_replace_rows, write_rows_like
from pegasus.she.substrate import SubstrateBundle


def _infer_datasus_uf_prefix(events_path: Path) -> str:
    df = pl.read_parquet(events_path, columns=["mun_residence_cod6"])
    prefixes = sorted(
        {
            str(value)[:2]
            for value in df["mun_residence_cod6"].drop_nulls().to_list()
            if len(str(value)) >= 2
        }
    )
    if len(prefixes) != 1:
        raise ValueError(f"Cannot infer a single DATASUS UF prefix from SIM events: {prefixes}")
    return prefixes[0]


def build_sim_compiler_run(*args, **kwargs):
    raise RuntimeError("Manual SIM EFG bundler is retired. Use build_autonomous_efg.")


def _q_row(field_id: str, *, n_events: float = 0.0, state: str = "verified") -> dict[str, Any]:
    return {
        "field_id": field_id,
        "n_events": n_events,
        "n_denom": 0.0,
        "n_eff": n_events,
        "cov_S": 1.0,
        "cov_T": 1.0,
        "missingness": 0.0,
        "zero_inflation": 0.0,
        "denom_fragility": 0.0,
        "cv": 0.0,
        "moran_i": 0.0,
        "temporal_roughness": 0.0,
        "spatial_entropy": 0.0,
        "provenance_risk": 0.0,
        "race_axis_source": "",
        "race_axis_target": "",
        "missing_race_share": 0.0,
        "emission_prior_strength": 0.0,
        "race_bridge_cv": 0.0,
        "sensitivity_width": 0.0,
        "bridge_mode": "",
        "state": state,
        "dashboard_safe": "False",
        "warnings": "[]",
        "computed_at": "fixture",
        "q_schema_version": "1.0",
    }


def _vd_row(field: dict[str, Any]) -> dict[str, Any]:
    return {
        "field_id": field["field_id"],
        "display_name": field["name"],
        "technical_name": field["name"],
        "definition": f"Fixture field {field['name']}.",
        "estimand_label": field.get("kind", "fixture_field"),
        "source_systems": field.get("source", "[]"),
        "carrier": field.get("carrier", ""),
        "unit": field.get("unit", ""),
        "support_description": field.get("support_json", "{}"),
        "axis_description": field.get("axes_json", "{}"),
        "provenance_description": field.get("provenance", "[]"),
        "state": field.get("state", "verified"),
        "dashboard_safe": field.get("dashboard_safe", "False"),
        "interpretation_warning": field.get("warnings", "[]"),
    }


def _field_row(
    *,
    field_id: str,
    name: str,
    carrier: str,
    unit: str,
    support: dict[str, Any],
    axes: dict[str, Any],
    kind: str = "fixture_field",
    role: list[str] | None = None,
    source: list[str] | None = None,
    state: str = "verified",
) -> dict[str, Any]:
    return {
        "field_id": field_id,
        "name": name,
        "kind": kind,
        "carrier": carrier,
        "unit": unit,
        "aggregation": "additive",
        "role": json.dumps(role or ["source_field"], sort_keys=True),
        "source": json.dumps(source or ["SIM-DO"], sort_keys=True),
        "support_json": json.dumps(support, sort_keys=True),
        "axes_json": json.dumps(axes, sort_keys=True),
        "operator": "fixture_efg",
        "provenance": json.dumps(["fixture", "source_normalized"], sort_keys=True),
        "state": state,
        "dashboard_safe": "False",
        "warnings": "[]",
        "lineage_hash": field_id,
        "registry_hash": "fixture",
        "materialization_state": "materialized",
        "path": "",
    }


def _write_profile_json(run_dir: Path) -> None:
    for name in ("UserIntent.json", "RunConfig.json", "ReproducibilityManifest.json", "P_vector.json"):
        path = run_dir / name
        payload = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        payload.setdefault("schema_version", "1.0")
        payload["run_profile"] = "core_vital"
        if name in {"RunConfig.json", "ReproducibilityManifest.json"}:
            payload["source_hashes"] = {"fixture": "fixture"}
            payload["registry_hashes"] = {"fixture": "fixture"}
        if name == "ReproducibilityManifest.json":
            payload["telemetry"] = {
                "total_wall_seconds": 0.0,
                "stage_status": {"fixture_efg": "success"},
                "stage_wall_seconds": {"fixture_efg": 0.0},
            }
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_sim_fixture_efg_run(
    *,
    sim_events_path: str | Path,
    run_dir: str | Path,
    municipality_cod6: str | None = None,
) -> Path:
    """Create a canonical SIM fixture run bundle for compatibility tests."""
    run = create_empty_output_bundle(run_dir)
    _write_profile_json(run)
    df = pl.read_parquet(sim_events_path)
    if municipality_cod6 is not None and "mun_residence_cod6" in df.columns:
        df = df.filter(pl.col("mun_residence_cod6").cast(pl.Utf8) == str(municipality_cod6))
    municipalities = []
    if "mun_residence_cod6" in df.columns:
        municipalities = sorted(str(x) for x in df["mun_residence_cod6"].drop_nulls().unique().to_list())
    years = []
    if "year" in df.columns:
        years = sorted(int(x) for x in df["year"].drop_nulls().unique().to_list())
    support = {"support": "municipality_year", "municipalities": municipalities, "years": years}
    support["n_events"] = df.height
    axes = {"geography": "mun_residence_cod6", "time": "year"}
    field = _field_row(
        field_id="sim_deaths_all",
        name="SIMDeathsAll",
        kind="event_count",
        carrier="Deaths",
        unit="deaths",
        support=support,
        axes=axes,
        role=["outcome", "source_field"],
    )
    edge = {
        "edge_id": "edge_fixture_sim_deaths_all",
        "parent_field_id": "slice0_scaffold_field",
        "child_field_id": "sim_deaths_all",
        "operator": "fixture_efg",
        "operator_params_json": "{}",
        "registry_versions_json": "{}",
        "created_at": "fixture",
    }
    append_replace_rows(run / "V_fields.parquet", [field], id_column="field_id")
    append_replace_rows(run / "Q_tensor.parquet", [_q_row("sim_deaths_all", n_events=float(df.height))], id_column="field_id")
    append_replace_rows(run / "VariableDictionary.parquet", [_vd_row(field)], id_column="field_id")
    write_rows_like(run / "E_DAG.parquet", [edge])
    return run


def build_autonomous_efg(
    *,
    substrate: SubstrateBundle,
    registry_root: str | Path = "config/registries",
    intent: Any = None,
    intent_constraints: dict[str, Any] | None = None,
    operator_budget: int = 256,
    operator_mode: str = "standard",
) -> EFGResult:
    """Workflow boundary for the Macro-Slice 19A autonomous EFG core."""

    return build_efg(
        substrate=substrate,
        registry_root=registry_root,
        intent=intent,
        intent_constraints=intent_constraints,
        operator_budget=operator_budget,
        operator_mode=operator_mode,
    )


def build_autonomous_efg_from_manifest(
    *,
    substrate_manifest: str | Path,
    output_path: str | Path | None = None,
    registry_root: str | Path = "config/registries",
    intent: Any = None,
    intent_constraints: dict[str, Any] | None = None,
    operator_budget: int = 256,
    operator_mode: str = "standard",
) -> EFGResult:
    substrate = load_substrate_bundle_for_efg(substrate_manifest)
    result = build_autonomous_efg(
        substrate=substrate,
        registry_root=registry_root,
        intent=intent,
        intent_constraints=intent_constraints,
        operator_budget=operator_budget,
        operator_mode=operator_mode,
    )
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(result.as_manifest(), indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    return result
