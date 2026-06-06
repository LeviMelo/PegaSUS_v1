from __future__ import annotations

from pathlib import Path
import textwrap

ROOT = Path.cwd()


def write(path: str, content: str) -> None:
    target = ROOT / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(textwrap.dedent(content).lstrip(), encoding="utf-8", newline="\n")


def main() -> None:
    write("src/pegasus/efg/lineage.py", r'''
    from __future__ import annotations

    from typing import Any

    from pegasus.core.hashing import content_hash
    from pegasus.core.schemas import Lineage


    def make_lineage(
        *,
        parent_ids: list[str],
        operator_type: str,
        operator_params: dict[str, Any],
        registry_versions: dict[str, str] | None = None,
        source_manifest_hashes: list[str] | None = None,
        code_version: str = "0.1.0",
    ) -> Lineage:
        return Lineage(
            parent_ids=parent_ids,
            operator_type=operator_type,
            operator_params=operator_params,
            registry_versions=registry_versions or {"registry_set": "v1.0"},
            source_manifest_hashes=source_manifest_hashes or [],
            code_version=code_version,
        )


    def lineage_hash(lineage: Lineage) -> str:
        return content_hash(lineage.model_dump(mode="json"))


    def field_id_from_lineage(lineage: Lineage) -> str:
        return lineage_hash(lineage)
    ''')

    write("src/pegasus/efg/node.py", r'''
    from __future__ import annotations

    from typing import Literal

    from pegasus.core.enums import FieldState, MaterializationState
    from pegasus.core.schemas import FieldNode, Lineage
    from pegasus.efg.lineage import field_id_from_lineage


    def make_field_node(
        *,
        name: str,
        kind: Literal[
            "extensive_measure",
            "intensive_density",
            "marked_functional",
            "context_gradient",
            "bridge_divergence",
            "bridge_module",
            "observer_proxy",
            "latent_context",
            "model_residual",
        ],
        carrier: str,
        unit: str,
        support: dict,
        axes: dict,
        aggregation: Literal[
            "additive",
            "weighted_mean",
            "statistical_functional",
            "compositional",
            "non_aggregable",
        ],
        role: list[str],
        source: list[str],
        operator: str | None,
        provenance: list[str],
        state: FieldState | str,
        warnings: list[str],
        lineage: Lineage,
        materialization_state: MaterializationState | str,
        path: str | None = None,
        dashboard_safe: bool | Literal["warning"] = False,
    ) -> FieldNode:
        return FieldNode(
            id=field_id_from_lineage(lineage),
            name=name,
            kind=kind,
            carrier=carrier,
            unit=unit,
            support=support,
            axes=axes,
            aggregation=aggregation,
            role=role,
            source=source,
            operator=operator,
            provenance=provenance,
            state=FieldState(state) if isinstance(state, str) else state,
            warnings=warnings,
            lineage=lineage,
            materialization_state=(
                MaterializationState(materialization_state)
                if isinstance(materialization_state, str)
                else materialization_state
            ),
            path=path,
            dashboard_safe=dashboard_safe,
        )
    ''')

    write("src/pegasus/efg/declaration.py", r'''
    from __future__ import annotations

    from typing import Literal

    from pydantic import BaseModel, ConfigDict

    from pegasus.core.schemas import FieldNode


    class OperatorSpec(BaseModel):
        model_config = ConfigDict(extra="forbid")

        name: str
        role: str
        output_kind: str | None = None
        params: dict = {}


    class DeclarationResult(BaseModel):
        model_config = ConfigDict(extra="forbid")

        ok: bool
        failed_terms: list[str]
        warnings: list[str]
        reason: str | None = None


    def _race_axis(field: FieldNode | None) -> str | None:
        if field is None:
            return None
        axes = field.axes or {}
        return (
            axes.get("race_axis_type")
            or axes.get("race_axis")
            or axes.get("declaration_process")
        )


    def _bridge_applied(field: FieldNode) -> bool:
        provenance = set(field.provenance or [])
        operators = {field.operator} if field.operator else set()
        warnings = set(field.warnings or [])
        return bool(
            provenance
            & {
                "bayesian_race_axis_bridge",
                "bridge_derived",
                "Bridge_R",
            }
        ) or bool(
            operators
            & {
                "Bridge_R",
                "Bridge_R_fixedC_dynamicW",
                "Bridge_R_posteriorC",
            }
        ) or "race_bridge_applied" in warnings


    def evaluate_declaration_compatibility(
        *,
        numerator: FieldNode,
        denominator: FieldNode | None,
        operator: OperatorSpec,
        registries=None,
    ) -> DeclarationResult:
        if operator.name != "RN" or denominator is None:
            return DeclarationResult(ok=True, failed_terms=[], warnings=[])

        numerator_race = _race_axis(numerator)
        denominator_race = _race_axis(denominator)

        if numerator_race is None or denominator_race is None:
            return DeclarationResult(ok=True, failed_terms=[], warnings=[])

        if numerator_race == denominator_race:
            return DeclarationResult(ok=True, failed_terms=[], warnings=[])

        if _bridge_applied(numerator):
            return DeclarationResult(
                ok=True,
                failed_terms=[],
                warnings=["race_axis_aligned_by_bridge"],
            )

        return DeclarationResult(
            ok=False,
            failed_terms=["declaration"],
            warnings=["race_axis_declaration_incommensurable"],
            reason=(
                f"Race-axis mismatch: numerator={numerator_race}; "
                f"denominator={denominator_race}; Bridge_R not applied."
            ),
        )
    ''')

    write("src/pegasus/efg/legality.py", r'''
    from __future__ import annotations

    from datetime import datetime, timezone

    from pegasus.core.schemas import DeltaResult, FailedBranch, FieldNode
    from pegasus.efg.declaration import OperatorSpec, evaluate_declaration_compatibility


    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()


    def _quality_ok(field: FieldNode) -> bool:
        return field.state.value not in {"illegal_excluded", "quarantined_nochildren"}


    def evaluate_delta(
        *,
        parents: list[FieldNode],
        operator: OperatorSpec,
        intent=None,
        registries=None,
        alignment=None,
    ) -> DeltaResult:
        numerator = parents[0] if parents else None
        denominator = parents[1] if len(parents) > 1 else None

        delta_support = 1
        delta_axes = 1
        delta_carrier = 1
        delta_unit = 1
        delta_aggregation = 1
        delta_provenance = 1
        delta_quality = 1
        delta_declaration = 1
        warnings: list[str] = []
        failed_terms: list[str] = []

        if operator.name == "RN":
            if numerator is None or denominator is None:
                delta_support = 0
                failed_terms.append("support")
            else:
                if not (numerator.carrier == "Deaths" and denominator.carrier == "Population"):
                    delta_carrier = 0
                    failed_terms.append("carrier")
                if not (numerator.unit == "counts" and denominator.unit == "person_years"):
                    delta_unit = 0
                    failed_terms.append("unit")
                if numerator.aggregation != "additive" or denominator.aggregation != "additive":
                    delta_aggregation = 0
                    failed_terms.append("aggregation")

        for parent in parents:
            if not _quality_ok(parent):
                delta_quality = 0
                if "quality" not in failed_terms:
                    failed_terms.append("quality")

        if numerator is not None:
            declaration = evaluate_declaration_compatibility(
                numerator=numerator,
                denominator=denominator,
                operator=operator,
                registries=registries,
            )
            if not declaration.ok:
                delta_declaration = 0
                failed_terms.extend(x for x in declaration.failed_terms if x not in failed_terms)
                warnings.extend(declaration.warnings)

        legal = all(
            [
                delta_support,
                delta_axes,
                delta_carrier,
                delta_unit,
                delta_aggregation,
                delta_provenance,
                delta_quality,
                delta_declaration,
            ]
        )

        failed_branch_id = None
        if not legal:
            failed_branch_id = "failed_" + "_".join(failed_terms) if failed_terms else "failed_unknown"

        return DeltaResult(
            legal=legal,
            delta_support=delta_support,
            delta_axes=delta_axes,
            delta_carrier=delta_carrier,
            delta_unit=delta_unit,
            delta_aggregation=delta_aggregation,
            delta_provenance=delta_provenance,
            delta_quality=delta_quality,
            delta_declaration=delta_declaration,
            failed_terms=failed_terms,
            warnings=warnings,
            failed_branch_id=failed_branch_id,
        )


    def make_failed_branch(
        *,
        parents: list[FieldNode],
        operator: OperatorSpec,
        delta: DeltaResult,
        reason: str,
    ) -> FailedBranch:
        failure_stage = "declaration" if "declaration" in delta.failed_terms else (
            delta.failed_terms[0] if delta.failed_terms else "output_validation"
        )
        return FailedBranch(
            failed_branch_id=delta.failed_branch_id or "failed_unknown",
            attempted_operator=operator.name,
            parent_field_ids=[p.id for p in parents],
            failure_stage=failure_stage,
            failed_terms=delta.failed_terms,
            reason=reason,
            warnings=delta.warnings,
            created_at=_now(),
        )
    ''')

    write("src/pegasus/efg/q_tensor.py", r'''
    from __future__ import annotations

    from datetime import datetime, timezone

    from pegasus.core.enums import FieldState
    from pegasus.core.schemas import FieldNode, QState, WarningRecord


    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()


    def provenance_risk(provenance: list[str]) -> float:
        p = set(provenance)
        if "synthetic" in p:
            return 1.0
        if "latent" in p:
            return 0.8
        if "SIM_informed_denominator_prior" in p or "facility_linkage_filtered" in p:
            return 0.5
        if "reconstructed" in p or "geneallocated" in p:
            return 0.4
        if "classification_projected" in p or "longitudinally_stitched" in p:
            return 0.3
        if "harmonized" in p or "deflated" in p or "AMC_contracted" in p:
            return 0.2
        return 0.0


    def classify_q_state(
        *,
        n_eff: float | None,
        denom_fragility: float | None,
        missingness: float | None,
        risk: float,
    ) -> FieldState:
        n_eff = 0.0 if n_eff is None else n_eff
        denom_fragility = 1.0 if denom_fragility is None else denom_fragility
        missingness = 1.0 if missingness is None else missingness

        if n_eff >= 100 and denom_fragility < 0.05 and missingness < 0.10 and risk < 0.5:
            return FieldState.verified
        if n_eff >= 30 and denom_fragility < 0.20:
            return FieldState.fragile
        return FieldState.quarantined_descriptive


    def compute_q_state(
        *,
        field: FieldNode,
        tensor=None,
        denominator=None,
        provenance: list[str] | None = None,
        warnings: list[WarningRecord] | None = None,
        registries=None,
    ) -> QState:
        provenance = provenance or field.provenance
        risk = provenance_risk(provenance)

        n_events = field.support.get("n_events")
        n_denom = field.support.get("n_denom")
        n_eff = field.support.get("n_eff", n_events)
        missingness = field.support.get("missingness", 0.0)
        denom_fragility = field.support.get("denom_fragility", 1.0 if n_denom is None else 0.0)
        zero_inflation = field.support.get("zero_inflation", 0.0)

        state = classify_q_state(
            n_eff=n_eff,
            denom_fragility=denom_fragility,
            missingness=missingness,
            risk=risk,
        )

        race_axis = field.axes.get("race_axis_type") or field.axes.get("race_axis")

        return QState(
            field_id=field.id,
            n_events=n_events,
            n_denom=n_denom,
            n_eff=n_eff,
            cov_S=field.support.get("cov_S"),
            cov_T=field.support.get("cov_T"),
            missingness=missingness,
            zero_inflation=zero_inflation,
            denom_fragility=denom_fragility,
            cv=field.support.get("cv"),
            moran_i=field.support.get("moran_i"),
            temporal_roughness=field.support.get("temporal_roughness"),
            spatial_entropy=field.support.get("spatial_entropy"),
            provenance_risk=risk,
            race_axis_source=race_axis,
            race_axis_target=None,
            missing_race_share=field.support.get("missing_race_share"),
            emission_prior_strength=None,
            race_bridge_cv=None,
            sensitivity_width=None,
            bridge_mode=None,
            state=state,
            dashboard_safe=False if state != FieldState.verified else field.dashboard_safe,
            warnings=field.warnings,
            computed_at=_now(),
            q_schema_version="1.0",
        )
    ''')

    write("src/pegasus/output/sim_efg_bundle.py", r'''
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
    ''')

    write("src/pegasus/workflows/build_efg.py", r'''
    from __future__ import annotations

    from pathlib import Path

    from pegasus.output.sim_efg_bundle import write_sim_fixture_efg_bundle


    def build_sim_fixture_efg_run(
        *,
        sim_events_path: str | Path,
        run_dir: str | Path,
    ) -> Path:
        return write_sim_fixture_efg_bundle(
            sim_events_path=sim_events_path,
            run_dir=run_dir,
        )
    ''')

    write("src/pegasus/cli.py", r'''
    from __future__ import annotations

    import importlib.util
    import shutil
    import sys
    from pathlib import Path

    import typer
    from rich import print

    from pegasus.core.config import validate_config_tree
    from pegasus.core.paths import ensure_data_lake
    from pegasus.datasus.cache import DatasusCache
    from pegasus.datasus.manifests import (
        build_datasus_manifests,
        load_datasus_config,
        read_request_manifest,
        write_request_manifest,
    )
    from pegasus.datasus.normalize import normalize_sim_do_events
    from pegasus.datasus.profile import profile_table
    from pegasus.datasus.schema_compare import compare_profiles
    from pegasus.datasus.subprocess import DatasusConfig, fetch_datasus_chunk
    from pegasus.output.bundle import create_empty_output_bundle
    from pegasus.output.validate import validate_output_bundle
    from pegasus.registries.validators import validate_registry_tree
    from pegasus.workflows.build_efg import build_sim_fixture_efg_run

    app = typer.Typer(no_args_is_help=True)
    registries_app = typer.Typer(no_args_is_help=True)
    sidra_app = typer.Typer(no_args_is_help=True)
    datasus_app = typer.Typer(no_args_is_help=True)
    efg_app = typer.Typer(no_args_is_help=True)

    app.add_typer(registries_app, name="registries")
    app.add_typer(sidra_app, name="sidra")
    app.add_typer(datasus_app, name="datasus")
    app.add_typer(efg_app, name="efg")


    def _fail(errors: list[str]) -> None:
        for error in errors:
            print(f"[red]ERROR[/red] {error}")
        raise typer.Exit(1)


    @app.command()
    def init() -> None:
        ensure_data_lake(".")
        run_dir = create_empty_output_bundle(Path("data/runs/slice0_empty"))
        print(f"[green]initialized[/green] data lake and scaffold run: {run_dir}")


    @app.command("validate-config")
    def validate_config() -> None:
        errors = validate_config_tree(".")
        if errors:
            _fail(errors)
        print("[green]config valid[/green]")


    @registries_app.command("validate")
    def validate_registries() -> None:
        errors = validate_registry_tree("config/registries")
        if errors:
            _fail(errors)
        print("[green]registries valid[/green]")


    @app.command("validate-run")
    def validate_run(run: Path = typer.Option(..., "--run")) -> None:
        result = validate_output_bundle(run_dir=str(run))
        if not result.ok:
            _fail(result.errors)
        print("[green]run bundle valid[/green]")


    @app.command()
    def doctor() -> None:
        checks: dict[str, str] = {}
        checks["python"] = sys.version.split()[0]

        for mod in ["duckdb", "polars", "pyarrow", "pydantic", "typer", "yaml"]:
            checks[mod] = "ok" if importlib.util.find_spec(mod) else "missing"

        torch_spec = importlib.util.find_spec("torch")
        if torch_spec:
            import torch

            checks["torch"] = getattr(torch, "__version__", "ok")
            checks["torch_cuda_available"] = str(torch.cuda.is_available())
            checks["cuda_device_name"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
        else:
            checks["torch"] = "missing"
            checks["torch_cuda_available"] = "False"
            checks["cuda_device_name"] = "none"

        checks["Rscript"] = shutil.which("Rscript") or "missing"
        checks["microdatasus"] = "unchecked_without_rscript" if checks["Rscript"] == "missing" else "check_with_R_script"
        checks["read.dbc"] = "unchecked_without_rscript" if checks["Rscript"] == "missing" else "check_with_R_script"

        checks["data_write"] = "ok"
        try:
            ensure_data_lake(".")
        except Exception as exc:
            checks["data_write"] = f"failed: {exc}"

        reg_errors = validate_registry_tree("config/registries")
        checks["registry_schema"] = "ok" if not reg_errors else f"{len(reg_errors)} errors"

        for key, value in checks.items():
            color = "green" if value not in {"missing", "False"} and not str(value).startswith("failed") else "yellow"
            print(f"[{color}]{key}[/] {value}")

        print("[yellow]doctor is still light: no DATASUS/SIDRA ingestion is executed.[/yellow]")


    @datasus_app.command("ingest")
    def datasus_ingest(
        system: str = typer.Option(..., "--system"),
        uf: str = typer.Option(..., "--uf"),
        years: str = typer.Option(..., "--years"),
        dry_run: bool = typer.Option(False, "--dry-run", help="Plan and persist manifests without invoking R."),
    ) -> None:
        cfg_payload = load_datasus_config()
        cfg = DatasusConfig.from_mapping(cfg_payload)
        manifests = build_datasus_manifests(system=system, uf=uf, years=years, config=cfg_payload)

        cache = DatasusCache()
        blocked = False
        failed = False

        for manifest in manifests:
            planned_path = write_request_manifest(manifest)
            print(f"[cyan]planned[/cyan] {manifest.system} {manifest.uf} {manifest.year_start}: {planned_path}")

            if dry_run:
                continue

            executed = fetch_datasus_chunk(
                manifest,
                config=cfg,
                cache=cache,
                timeout_seconds=cfg.r_timeout_seconds,
                heartbeat_timeout_seconds=cfg.heartbeat_timeout_seconds,
            )
            executed_path = write_request_manifest(executed)

            if executed.status == "success":
                print(f"[green]success[/green] {executed.system} {executed.uf} {executed.year_start}: {executed_path}")
            elif executed.status == "blocked":
                blocked = True
                print(f"[yellow]blocked[/yellow] {executed.system} {executed.uf} {executed.year_start}: {executed.error_message}")
            else:
                failed = True
                print(f"[red]{executed.status}[/red] {executed.system} {executed.uf} {executed.year_start}: {executed.error_message}")

        if failed:
            raise typer.Exit(1)
        if blocked:
            raise typer.Exit(2)


    @datasus_app.command("profile")
    def datasus_profile(manifest: Path = typer.Option(..., "--manifest")) -> None:
        request = read_request_manifest(manifest)

        raw_path = Path(request.raw_path)
        processed_path = Path(request.processed_path)

        if not raw_path.exists() or not processed_path.exists():
            print("[yellow]blocked[/yellow] raw/processed artifacts are missing; cannot profile this manifest.")
            raise typer.Exit(2)

        raw_profile_path = Path("data/metadata/datasus/profiles") / request.system / request.request_hash / "raw_profile.json"
        processed_profile_path = Path("data/metadata/datasus/profiles") / request.system / request.request_hash / "processed_profile.json"
        compare_path = Path("data/metadata/datasus/schema_compare") / request.system / request.request_hash / "schema_compare.json"

        raw_profile = profile_table(raw_path, output_path=raw_profile_path)
        processed_profile = profile_table(processed_path, output_path=processed_profile_path)
        compare_profiles(raw_profile, processed_profile, output_path=compare_path)

        print(f"[green]raw profile[/green] {raw_profile_path}")
        print(f"[green]processed profile[/green] {processed_profile_path}")
        print(f"[green]schema comparison[/green] {compare_path}")


    @datasus_app.command("normalize-sim")
    def datasus_normalize_sim(
        input_path: Path = typer.Option(..., "--input"),
        output_path: Path = typer.Option(..., "--output"),
        source_manifest_hash: str = typer.Option("fixture", "--source-manifest-hash"),
    ) -> None:
        result = normalize_sim_do_events(
            input_path=input_path,
            output_path=output_path,
            source_manifest_hash=source_manifest_hash,
        )
        print(f"[green]sim normalized[/green] rows={result['row_count']} output={result['output_path']}")


    @efg_app.command("build-sim-fixture")
    def efg_build_sim_fixture(
        sim_events: Path = typer.Option(..., "--sim-events"),
        run_dir: Path = typer.Option(..., "--run-dir"),
    ) -> None:
        output = build_sim_fixture_efg_run(
            sim_events_path=sim_events,
            run_dir=run_dir,
        )
        result = validate_output_bundle(run_dir=str(output))
        if not result.ok:
            _fail(result.errors)
        print(f"[green]sim fixture EFG bundle valid[/green] {output}")


    @sidra_app.command("metadata")
    def sidra_metadata(tables: Path = typer.Option(..., "--tables")) -> None:
        print(f"[yellow]blocked[/yellow] SIDRA metadata is Slice 2. Seed received: {tables}")
        raise typer.Exit(2)


    @sidra_app.command("plan")
    def sidra_plan(view: str = typer.Option(..., "--view")) -> None:
        print(f"[yellow]blocked[/yellow] SIDRA planning is Slice 2. View received: {view}")
        raise typer.Exit(2)


    @sidra_app.command("extract")
    def sidra_extract(plan: Path = typer.Option(..., "--plan")) -> None:
        print(f"[yellow]blocked[/yellow] SIDRA extraction is Slice 2. Plan received: {plan}")
        raise typer.Exit(2)


    @app.command()
    def compile(intent: Path = typer.Option(..., "--intent")) -> None:
        print(f"[yellow]blocked[/yellow] compile workflow requires later slices. Intent: {intent}")
        raise typer.Exit(2)
    ''')

    write("tests/unit/test_efg_declaration.py", r'''
    from pegasus.core.enums import FieldState, MaterializationState
    from pegasus.efg.declaration import OperatorSpec, evaluate_declaration_compatibility
    from pegasus.efg.lineage import make_lineage
    from pegasus.efg.node import make_field_node


    def _node(name: str, carrier: str, unit: str, race_axis: str):
        lineage = make_lineage(
            parent_ids=[],
            operator_type="fixture",
            operator_params={"name": name},
        )
        return make_field_node(
            name=name,
            kind="extensive_measure",
            carrier=carrier,
            unit=unit,
            support={},
            axes={"race_axis_type": race_axis},
            aggregation="additive",
            role=["fixture"],
            source=["fixture"],
            operator="fixture",
            provenance=["official"],
            state=FieldState.fragile,
            warnings=[],
            lineage=lineage,
            materialization_state=MaterializationState.metadata_only,
            dashboard_safe=False,
        )


    def test_race_axis_mismatch_fails_without_bridge():
        numerator = _node("sim_admin", "Deaths", "counts", "administrative_death_declaration")
        denominator = _node("ibge_self", "Population", "person_years", "self_declared")

        result = evaluate_declaration_compatibility(
            numerator=numerator,
            denominator=denominator,
            operator=OperatorSpec(name="RN", role="mortality_rate"),
        )

        assert not result.ok
        assert "declaration" in result.failed_terms
        assert "Bridge_R not applied" in (result.reason or "")


    def test_race_axis_mismatch_passes_after_bridge_provenance():
        numerator = _node("sim_bridge", "Deaths", "counts", "administrative_death_declaration").model_copy(
            update={"provenance": ["official", "bayesian_race_axis_bridge"]}
        )
        denominator = _node("ibge_self", "Population", "person_years", "self_declared")

        result = evaluate_declaration_compatibility(
            numerator=numerator,
            denominator=denominator,
            operator=OperatorSpec(name="RN", role="mortality_rate"),
        )

        assert result.ok
    ''')

    write("tests/unit/test_sim_fixture_efg_bundle.py", r'''
    from pathlib import Path

    import polars as pl

    from pegasus.datasus.normalize import normalize_sim_do_events
    from pegasus.output.validate import validate_output_bundle
    from pegasus.workflows.build_efg import build_sim_fixture_efg_run


    def test_sim_fixture_efg_bundle_validates(tmp_path: Path):
        sim_events = tmp_path / "sim_events.parquet"
        run_dir = tmp_path / "run"

        normalize_sim_do_events(
            input_path="tests/fixtures/datasus/sim_do_fixture.csv",
            output_path=sim_events,
            source_manifest_hash="fixture_manifest_hash",
        )

        build_sim_fixture_efg_run(
            sim_events_path=sim_events,
            run_dir=run_dir,
        )

        result = validate_output_bundle(run_dir=str(run_dir))
        assert result.ok, result.errors

        v = pl.read_parquet(run_dir / "V_fields.parquet")
        assert {"SIMDeathsAll", "SIMAdministrativeRaceDeaths", "IBGESelfDeclaredPopulationPlaceholder"} <= set(v["name"].to_list())

        q = pl.read_parquet(run_dir / "Q_tensor.parquet")
        assert q.height == v.height

        failed = pl.read_parquet(run_dir / "FailedBranches.parquet")
        assert failed.height == 1
        assert failed.row(0, named=True)["failure_stage"] == "declaration"
        assert "declaration" in failed.row(0, named=True)["failed_terms"]
    ''')

    print("Applied Slice 1D: minimal EFG lineage/declaration/Q-state and SIM fixture 17-key bundle.")


if __name__ == "__main__":
    main()